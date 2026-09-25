"""Eight-device cases invoked explicitly by test_data_parallel's child."""

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from minifield_training.batching import contracts
from minifield_training.checkpoints import training_state
from minifield_training.core import parameters
from minifield_training.engine import step
from minifield_training.engine import training_run
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import model
from minifield_training.optimizers import adamw
from minifield_training.optimizers import state
from minifield_training.strategies import classification


def _mesh() -> jax.sharding.Mesh:
    """Admit exactly 8 local CPU devices."""
    return jax.sharding.Mesh(
        np.asarray(training_run.require_devices(8, "cpu")), ("data",)
    )


def _replicate(value: state.State, mesh: jax.sharding.Mesh) -> state.State:
    """Place the logical tree unchanged on each replica."""
    # JAX 0.7.2's PartitionSpec constructor has no type annotations.
    spec = jax.sharding.PartitionSpec()  # type: ignore[no-untyped-call]
    return cast(
        state.State,
        jax.device_put(value, jax.sharding.NamedSharding(mesh, spec)),
    )


def _setup() -> tuple[parameters.FullParameterInventory, state.State]:
    """Use a scalar affine model with a frozen intercept."""
    inventory = parameters.build_inventory(
        {"weight": (), "bias": ()},
        format_id="parallel/1",
        decayed_names=frozenset(),
        frozen_names=frozenset({"bias"}),
    )
    return inventory, adamw.initialize_state(
        {"weight": jnp.float32(2), "bias": jnp.float32(1)}, inventory
    )


def _terms(
    params: dict[str, jax.Array], batch: dict[str, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """Compute squared residuals with explicit per-row supervision weights."""
    residual = params["weight"] * batch["x"] + params["bias"]
    return (
        jnp.sum(batch["mask"] * residual**2, dtype=jnp.float32),
        jnp.sum(batch["mask"], dtype=jnp.float32),
    )


def _batch() -> dict[str, jax.Array]:
    """Give replicas unequal target counts, including entirely padded ones."""
    values = np.zeros((3, 16), dtype=np.float32)
    masks = np.zeros_like(values)
    values[0, 0], values[0, 2] = 1, 2
    values[1, 0], values[1, 1] = 3, 4
    masks[0, (0, 2)] = 1
    masks[1, :2] = 1
    values[2] = np.nan  # The inactive slot must never be evaluated.
    return {"x": jnp.asarray(values), "mask": jnp.asarray(masks)}


@pytest.mark.parametrize("fused", [False, True])
def test_global_weighting_and_replicated_state(fused: bool) -> None:
    """Independent affine-model math checks reduction and one Adam commit."""
    inventory, initial = _setup()
    mesh = _mesh()
    update = step.make_streaming_step(
        _terms,
        inventory,
        adamw.AdamWConfig(0.1, clip_norm=100),
        mesh=mesh,
        fuse_accumulation=fused,
    )
    result = update(
        _replicate(initial, mesh), _batch(), np.array([True, True, False])
    )
    # Residuals 3, 5, 7, 9: mean loss 41; mean gradient 35.
    # Adam m = 0.1*35, v = 0.05*35^2, first parameter update = -0.1.
    assert bool(result.committed)
    assert int(result.state["step"]) == 1
    np.testing.assert_allclose(result.loss, 41, rtol=1e-6)
    np.testing.assert_allclose(result.state["m"]["weight"], 3.5, rtol=1e-6)
    np.testing.assert_allclose(result.state["v"]["weight"], 61.25, rtol=1e-6)
    np.testing.assert_allclose(result.state["params"]["weight"], 1.9, rtol=1e-6)
    for group, expected in (
        (result.state["params"], 1),
        (result.state["m"], 0),
        (result.state["v"], 0),
    ):
        np.testing.assert_array_equal(group["bias"], expected)
    for leaf in jax.tree.leaves(result.state):
        assert leaf.sharding.is_fully_replicated
        assert len(leaf.addressable_shards) == 8
        for shard in leaf.addressable_shards:
            np.testing.assert_array_equal(shard.data, leaf)


@pytest.mark.parametrize("failure", ["negative", "nan", "loss", "empty"])
def test_bad_replica_rejects_entire_update(failure: str) -> None:
    """One invalid shard cannot hide behind another shard's positive count."""
    inventory, initial = _setup()
    expected = jax.tree.map(np.asarray, initial)
    update = step.make_streaming_step(
        _terms, inventory, adamw.AdamWConfig(0.1), mesh=_mesh()
    )
    batch = _batch()
    if failure == "negative":
        batch["mask"] = batch["mask"].at[0, 4].set(-0.5)
    elif failure == "nan":
        batch["mask"] = batch["mask"].at[0, 4].set(jnp.nan)
    elif failure == "loss":
        batch["x"] = batch["x"].at[0, 2].set(jnp.inf)
    else:
        batch["mask"] = jnp.zeros_like(batch["mask"])
    result = update(
        _replicate(initial, _mesh()), batch, np.array([True, False, False])
    )
    assert not bool(result.committed)
    for actual, saved in zip(
        jax.tree.leaves(result.state), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual, saved)


def test_shape_and_device_admission() -> None:
    """Reject incompatible physical rows, axis names, and requested devices."""
    inventory, initial = _setup()
    with pytest.raises(RuntimeError, match="Expected one cpu"):
        training_run.require_single_device("cpu")
    with pytest.raises(RuntimeError, match="Expected 8 tpu"):
        training_run.require_devices(8, "tpu")
    with pytest.raises(ValueError, match="positive"):
        training_run.require_devices(0)
    with pytest.raises(ValueError, match="data mesh"):
        step.make_streaming_step(
            _terms,
            inventory,
            adamw.AdamWConfig(0.1),
            mesh=jax.sharding.Mesh(np.asarray(jax.devices()), ("wrong",)),
        )
    update = step.make_streaming_step(
        _terms, inventory, adamw.AdamWConfig(0.1), mesh=_mesh()
    )
    with pytest.raises(ValueError, match="divisible"):
        update(initial, {"x": jnp.ones((1, 3))}, np.array([True]))


def test_runner_checkpoint_resume(tmp_path: Path) -> None:
    """Restored replicated training equals uninterrupted next-update state."""
    inventory, initial = _setup()
    optimizer = adamw.AdamWConfig(0.1, clip_norm=100)
    update = step.make_streaming_step(
        _terms, inventory, optimizer, mesh=_mesh()
    )
    cursor = training_state.Cursor("parallel", "data", "source", 0)
    starts: list[int] = []

    def source(
        start: int, unused_deadline: float | None
    ) -> Iterator[contracts.PhysicalUpdate]:
        """Record the restored cursor while producing deterministic updates."""
        starts.append(start)
        for index in range(start, 2):
            yield contracts.PhysicalUpdate(
                _batch(), np.array([True, True, False]), (str(index),)
            )

    def run(
        current: state.State, position: training_state.Cursor, root: Path
    ) -> tuple[state.State, training_state.Cursor]:
        """Exercise shared persistence and mesh placement."""
        return training_run.run(
            None,
            current,
            update,
            inventory,
            training_run.RunConfig(1, 1, 1, max_steps=1),
            checkpoint_root=root,
            optimizer_id=optimizer.implementation_identity,
            cursor=position,
            required_platform="cpu",
            batch_source=source,
        )

    live, position = run(initial, cursor, tmp_path / "first")
    restored, restored_cursor = training_state.load(
        tmp_path / "first/step-00000001",
        inventory,
        optimizer_id=optimizer.implementation_identity,
        run_id=cursor.run_id,
        data_sha256=cursor.data_sha256,
        source_id=cursor.source_id,
    )
    resumed, _ = run(restored, restored_cursor, tmp_path / "resumed")
    continued, _ = run(live, position, tmp_path / "continued")
    assert starts == [0, 1, 1]
    assert int(resumed["step"]) == 2
    for actual, expected in zip(
        jax.tree.leaves(resumed), jax.tree.leaves(continued), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_tiny_classifier_matches_single_device(dtype: types.DType) -> None:
    """Real conv/attention gradients reduce safely with padded replicas."""
    cfg = model.Config(4, 8, 1, 1, 8, ("conv", "attention"))
    rng = np.random.default_rng(6)
    backbone = {
        name: jnp.asarray(rng.normal(0, 0.1, shape), dtype=jnp.float32)
        for name, shape in model.expected_shapes(cfg).items()
    }
    allowed = (True, True, False)
    inventory = classification.parameter_inventory(cfg, allowed)
    params = classification.initialize_from_backbone(
        backbone, cfg, allowed, head_seed=6
    )
    physical = {
        "input_ids": jnp.tile(jnp.array([[1, 2, 3, 0]]), (16, 1)),
        "attention_mask": jnp.tile(jnp.array([[1, 1, 1, 0]]), (16, 1)),
        "labels": jnp.array([0, 1, 1] + [2] * 13),
        "valid_rows": jnp.arange(16) < 3,
    }
    physical["attention_mask"] = (
        physical["attention_mask"] * physical["valid_rows"][:, None]
    )
    optimizer = adamw.AdamWConfig(0.01)
    ordinary = classification.make_lfm2_5_streaming_step(
        cfg, allowed, inventory, optimizer, dtype=dtype
    )
    parallel = classification.make_lfm2_5_streaming_step(
        cfg, allowed, inventory, optimizer, dtype=dtype, mesh=_mesh()
    )
    expected_loss, expected_count, expected_grad = ordinary.gradient(
        params, physical
    )
    actual_loss, actual_count, actual_grad = parallel.gradient(params, physical)
    np.testing.assert_array_equal(actual_count, 3)
    np.testing.assert_array_equal(actual_count, expected_count)
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-6)
    tolerance = 1e-6 if dtype == jnp.float32 else 0.025
    for name, expected in expected_grad.items():
        error = np.linalg.norm(
            np.asarray(actual_grad[name]) - np.asarray(expected)
        )
        scale = max(float(np.linalg.norm(np.asarray(expected))), 1e-6)
        assert error / scale <= tolerance, name
