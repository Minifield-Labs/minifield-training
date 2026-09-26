"""Selection, training, and dense effective export contracts."""

from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import load_file

from minifield_training.checkpoints import inference_output
from minifield_training.checkpoints import training_state
from minifield_training.core import parameters
from minifield_training.kernels import quantization as kernels
from minifield_training.models.lfm2_5 import model
from minifield_training.optimizers import adamw
from minifield_training.strategies import classification
from minifield_training.strategies import quantization
from minifield_training.strategies import sft


def _strategy(names: frozenset[str]) -> quantization.NamedQuantization:
    return quantization.NamedQuantization(
        kernels.Group128Quantizer("ternary-g128-absmax-f16-v1"), names
    )


class _AlternatePlan:
    """Small external plan proving the consumer accepts the protocol."""

    names = frozenset({"projection.weight"})
    identity = "sign-ste-fixture-v1"

    def select(
        self,
        inventory: parameters.FullParameterInventory,
        roles: Mapping[str, str],
    ) -> frozenset[str]:
        assert roles["projection.weight"] == "projection"
        assert self.names.issubset(inventory.names)
        return self.names

    def effective(self, weight: jax.Array) -> jax.Array:
        decoded = jnp.sign(weight)
        return jax.lax.stop_gradient(decoded) + (
            weight - jax.lax.stop_gradient(weight)
        )


def test_protocol_accepts_alternate_numerical_strategy() -> None:
    """The shared consumer calls an injected plan, not one concrete class."""
    inventory = parameters.build_inventory(
        {"projection.weight": (1, 3)},
        format_id="fixture/1",
        decayed_names=frozenset(),
        source_dtype="float32",
        quantization_profile=_AlternatePlan.identity,
        quantized_names=_AlternatePlan.names,
    )
    strategy: quantization.QuantizationPlan = _AlternatePlan()
    quantization.validate_plan(
        inventory, strategy, {"projection.weight": "projection"}
    )
    master = jnp.full((1, 3), -0.2, dtype=jnp.float32)
    result = quantization.apply(
        {"projection.weight": master}, inventory, strategy
    )
    np.testing.assert_array_equal(
        result["projection.weight"], np.full((1, 3), -1.0)
    )
    np.testing.assert_array_equal(
        jax.grad(lambda value: jnp.sum(strategy.effective(value)))(master),
        np.ones((1, 3)),
    )


def test_exact_selection_rejects_protected_and_frozen() -> None:
    """Only named projections can enter the plan, including future heads."""
    cfg = model.Config(128, 128, 1, 1, 8, ("conv",))
    allowed = (True, True)
    dense = classification.parameter_inventory(cfg, allowed)
    names = classification.projection_names(cfg)
    qat = classification.parameter_inventory(cfg, allowed, _strategy(names))
    assert frozenset(s.name for s in qat.specs if s.quantized) == names
    assert qat.sha256 != dense.sha256
    for protected in ("model.embed_tokens.weight", classification.HEAD_NAME):
        with pytest.raises(ValueError, match="Ineligible"):
            classification.parameter_inventory(
                cfg, allowed, _strategy(frozenset({protected}))
            )
    with pytest.raises(ValueError, match="unknown"):
        classification.parameter_inventory(
            cfg, allowed, _strategy(frozenset({"typo.weight"}))
        )


def test_dense_effective_export_roundtrip(tmp_path: object) -> None:
    """Strided matrices survive FP32 safetensors with lineage metadata."""
    root = Path(str(tmp_path))
    inventory = parameters.build_inventory(
        {"projection.weight": (2, 128), "head.weight": (2, 2)},
        format_id="fixture/1",
        decayed_names=frozenset(),
        source_dtype="float32",
        quantization_profile="ternary-g128-absmax-f16-v1",
        quantized_names=frozenset({"projection.weight"}),
    )
    matrix = np.arange(256, dtype=np.float32).reshape(128, 2).T
    masters = {
        "projection.weight": jnp.asarray(matrix),
        "head.weight": jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
    }
    strategy = _strategy(frozenset({"projection.weight"}))
    effective = quantization.apply(masters, inventory, strategy)
    path = root / "weights.safetensors"
    inference_output.DenseEffectiveOutput().write(
        path,
        effective,
        inventory,
        source_model="fixture",
        source_revision="v1",
    )
    loaded = load_file(str(path))
    for name in inventory.names:
        np.testing.assert_array_equal(loaded[name], effective[name])
    with safe_open(path, framework="numpy") as handle:
        assert handle.metadata()["format"] == "pt"
        assert handle.metadata()["quantization_profile"] == strategy.identity
    activation = np.arange(128, dtype=np.float32)
    np.testing.assert_array_equal(
        loaded["projection.weight"] @ activation,
        np.asarray(effective["projection.weight"]) @ activation,
    )
    with pytest.raises(FileExistsError):
        inference_output.DenseEffectiveOutput().write(
            path,
            effective,
            inventory,
            source_model="fixture",
            source_revision="v1",
        )


def test_classifier_qat_update_and_resume_identity(tmp_path: object) -> None:
    """A real model update keeps masters and rejects changed quantizer state."""
    cfg = model.Config(128, 128, 1, 1, 8, ("conv",))
    allowed = (True, True)
    rng = np.random.default_rng(7)
    backbone = {
        name: jnp.asarray(rng.normal(0, 0.02, shape), dtype=jnp.float32)
        for name, shape in model.expected_shapes(cfg).items()
    }
    masters = classification.initialize_from_backbone(
        backbone, cfg, allowed, head_seed=3
    )
    strategy = _strategy(classification.projection_names(cfg))
    inventory = classification.parameter_inventory(cfg, allowed, strategy)
    optimizer = adamw.AdamWConfig(0.001)
    initial = adamw.initialize_state(masters, inventory)
    batch = {
        "input_ids": jnp.asarray([[[1, 2, 3]]], dtype=jnp.int32),
        "attention_mask": jnp.asarray([[[1, 1, 1]]], dtype=jnp.int32),
        "labels": jnp.asarray([[1]], dtype=jnp.int32),
        "valid_rows": jnp.asarray([[True]]),
    }
    update = jax.jit(
        classification.make_lfm2_5_step(
            cfg,
            allowed,
            inventory,
            optimizer,
            dtype=jnp.float32,
            quantization_strategy=strategy,
        )
    )
    result = update(initial, batch, jnp.asarray([True]))
    assert bool(result.committed)
    assert np.isfinite(float(result.loss))
    assert any(
        not np.array_equal(
            result.state["params"][name], initial["params"][name]
        )
        for name in strategy.names
    )
    output = Path(str(tmp_path)) / "effective.safetensors"
    effective = quantization.apply(result.state["params"], inventory, strategy)
    inference_output.DenseEffectiveOutput().write(
        output,
        effective,
        inventory,
        source_model="fixture",
        source_revision="v1",
    )
    reloaded = {
        name: jnp.asarray(value)
        for name, value in load_file(str(output)).items()
    }
    original_logits = classification.logits(
        result.state["params"],
        batch["input_ids"][0],
        batch["attention_mask"][0],
        cfg,
        allowed,
        dtype=jnp.float32,
        inventory=inventory,
        quantization_strategy=strategy,
    )
    restored_logits = classification.logits(
        reloaded,
        batch["input_ids"][0],
        batch["attention_mask"][0],
        cfg,
        allowed,
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(original_logits, restored_logits)
    np.testing.assert_array_equal(
        result.state["params"]["model.embed_tokens.weight"],
        initial["params"]["model.embed_tokens.weight"],
    )
    cursor = training_state.Cursor("qat", "data", "source", 1)
    directory = Path(str(tmp_path)) / "step-00000001"
    training_state.save(
        directory,
        result.state,
        inventory,
        optimizer_id=optimizer.implementation_identity,
        cursor=cursor,
    )
    restored, restored_cursor = training_state.load(
        directory,
        inventory,
        optimizer_id=optimizer.implementation_identity,
        run_id="qat",
        data_sha256="data",
        source_id="source",
    )
    assert restored_cursor == cursor
    second = update(result.state, batch, jnp.asarray([True]))
    resumed = update(restored, batch, jnp.asarray([True]))
    for name in inventory.names:
        np.testing.assert_array_equal(
            second.state["params"][name], resumed.state["params"][name]
        )
    with pytest.raises(ValueError, match="identity"):
        training_state.load(
            directory,
            classification.parameter_inventory(cfg, allowed),
            optimizer_id=optimizer.implementation_identity,
            run_id="qat",
            data_sha256="data",
            source_id="source",
        )


def test_sft_qat_step_uses_shared_update() -> None:
    """Causal SFT trains FP32 masters through fake quantization."""
    cfg = model.Config(128, 128, 1, 1, 8, ("conv",))
    names = model.projection_names(cfg)
    strategy = _strategy(names)
    inventory = model.parameter_inventory(
        cfg, quantization_profile=strategy.identity, quantized_names=names
    )
    rng = np.random.default_rng(11)
    masters = {
        name: jnp.asarray(rng.normal(0, 0.02, shape), dtype=jnp.float32)
        for name, shape in model.expected_shapes(cfg).items()
    }
    current = adamw.initialize_state(masters, inventory)
    ids = jnp.asarray([[[1, 2, 3]]], dtype=jnp.int32)
    batch = {
        "input_ids": ids,
        "attention_mask": jnp.ones_like(ids),
        "loss_mask": jnp.asarray([[[0, 1, 1]]], dtype=jnp.int32),
    }
    update = jax.jit(
        sft.make_lfm2_5_step(
            cfg,
            inventory,
            adamw.AdamWConfig(0.001),
            dtype=jnp.float32,
            quantization_strategy=strategy,
        )
    )
    result = update(current, batch, jnp.asarray([True]))
    assert bool(result.committed)
    assert int(result.state["step"]) == 1
    assert np.isfinite(float(result.loss))


def test_qat_plan_cannot_be_omitted_or_select_sft_embedding() -> None:
    """Both consumers fail before JIT when QAT metadata lacks a valid plan."""
    cfg = model.Config(128, 128, 1, 1, 8, ("conv",))
    allowed = (True, True)
    strategy = _strategy(model.projection_names(cfg))
    class_inventory = classification.parameter_inventory(cfg, allowed, strategy)
    with pytest.raises(ValueError, match="identity"):
        classification.make_lfm2_5_step(
            cfg,
            allowed,
            class_inventory,
            adamw.AdamWConfig(0.001),
            dtype=jnp.float32,
        )
    with pytest.raises(ValueError, match="requires quantization plan"):
        quantization.apply({}, class_inventory, None)
    sft_inventory = model.parameter_inventory(
        cfg,
        quantization_profile=strategy.identity,
        quantized_names=strategy.names,
    )
    with pytest.raises(ValueError, match="requires quantization plan"):
        sft.make_lfm2_5_step(
            cfg,
            sft_inventory,
            adamw.AdamWConfig(0.001),
            dtype=jnp.float32,
        )
    embedding = "model.embed_tokens.weight"
    invalid = model.parameter_inventory(
        cfg,
        quantization_profile=strategy.identity,
        quantized_names=frozenset({embedding}),
    )
    with pytest.raises(ValueError, match="Ineligible"):
        sft.make_lfm2_5_step(
            cfg,
            invalid,
            adamw.AdamWConfig(0.001),
            dtype=jnp.float32,
            quantization_strategy=_strategy(frozenset({embedding})),
        )


def test_dense_checkpoint_warm_start_allows_new_optimizer(
    tmp_path: Path,
) -> None:
    """Warm start takes verified masters and resets moments under a new LR."""
    inventory = parameters.build_inventory(
        {"weight": (2, 2)},
        format_id="fixture/1",
        decayed_names=frozenset(),
        source_dtype="float32",
    )
    master = {"weight": jnp.asarray([[1.0, 2.0], [3.0, 4.0]])}
    old = adamw.AdamWConfig(0.01)
    directory = tmp_path / "dense"
    training_state.save(
        directory,
        adamw.initialize_state(master, inventory),
        inventory,
        optimizer_id=old.implementation_identity,
        cursor=training_state.Cursor("dense", "data", "source", 0),
    )
    warm = training_state.load_warm_start_masters(
        directory,
        inventory,
        run_id="dense",
        data_sha256="data",
        source_id="source",
    )
    new = adamw.initialize_state(warm, inventory)
    np.testing.assert_array_equal(new["params"]["weight"], master["weight"])
    np.testing.assert_array_equal(new["m"]["weight"], np.zeros((2, 2)))
    assert (
        adamw.AdamWConfig(0.02).implementation_identity
        != old.implementation_identity
    )
