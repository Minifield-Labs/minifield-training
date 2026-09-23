"""Tiny public-model integration for causal SFT composition."""

import jax
import jax.numpy as jnp
import numpy as np

from minifield_training.models.lfm2_5 import model
from minifield_training.objectives import loss
from minifield_training.optimizers import adamw
from minifield_training.strategies import sft


def test_real_model_step_reduces_supervised_loss() -> None:
    """A deterministic dense LFM2.5 fixture learns through public APIs."""
    cfg = model.Config(
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=8,
        layer_types=("conv",),
        conv_kernel=3,
    )
    rng = np.random.default_rng(7)
    params = {
        name: jnp.asarray(rng.normal(0, 0.1, shape), dtype=jnp.float32)
        for name, shape in model.expected_shapes(cfg).items()
    }
    inventory = model.parameter_inventory(cfg)
    current = adamw.initialize_state(params, inventory)
    ids = jnp.asarray([[[1, 2, 3, 4]], [[2, 3, 4, 5]]], dtype=jnp.int32)
    attention = jnp.ones_like(ids)
    mask = jnp.asarray([[[0, 1, 1, 1]], [[0, 1, 1, 0]]], dtype=jnp.int32)
    batch = {"input_ids": ids, "attention_mask": attention, "loss_mask": mask}
    active = jnp.asarray([True, True])

    def measured_loss(parameters: dict[str, jax.Array]) -> float:
        """Measure token-normalized NLL independently of the engine."""
        total = 0.0
        count = 0.0
        for index in range(2):
            logits = model.forward(
                parameters, ids[index], attention[index], cfg, dtype=jnp.float32
            )
            part, targets = loss.causal_loss_terms(
                logits, ids[index], mask[index], attention[index]
            )
            total += float(part)
            count += float(targets)
        return total / count

    before = measured_loss(current["params"])
    update = sft.make_lfm2_5_step(
        cfg, inventory, adamw.AdamWConfig(learning_rate=0.03), dtype=jnp.float32
    )
    for _ in range(6):
        result = update(current, batch, active)
        assert bool(result.committed)
        current = result.state
    after = measured_loss(current["params"])
    assert after < before - 0.05
    assert int(current["step"]) == 6
    assert any(
        not np.array_equal(
            np.asarray(current["params"][name]), np.asarray(value)
        )
        for name, value in params.items()
    )


def test_bfloat16_compute_keeps_float32_masters() -> None:
    """The adapter's default compute dtype leaves optimizer state in FP32."""
    cfg = model.Config(
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=8,
        layer_types=("conv",),
    )
    params = {
        name: jnp.full(shape, 0.05, dtype=jnp.float32)
        for name, shape in model.expected_shapes(cfg).items()
    }
    inventory = model.parameter_inventory(cfg)
    current = adamw.initialize_state(params, inventory)
    ids = jnp.asarray([[[1, 2, 3]]], dtype=jnp.int32)
    batch = {
        "input_ids": ids,
        "attention_mask": jnp.ones_like(ids),
        "loss_mask": jnp.asarray([[[0, 1, 1]]], dtype=jnp.int32),
    }
    result = jax.jit(
        sft.make_lfm2_5_step(cfg, inventory, adamw.AdamWConfig(0.01))
    )(current, batch, jnp.asarray([True]))
    assert bool(result.committed)
    for group in ("params", "m", "v"):
        assert all(
            value.dtype == jnp.float32 for value in result.state[group].values()
        )
