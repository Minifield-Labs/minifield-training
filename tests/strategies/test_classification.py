"""Small full-model decision update through the shared engine."""

import jax
import jax.numpy as jnp
import numpy as np

from minifield_training.models.lfm2_5 import model
from minifield_training.optimizers import adamw
from minifield_training.strategies import classification


def test_only_head_is_new_and_embeddings_stay_frozen() -> None:
    """Warm start preserves each pretrained leaf, then learns a decision."""
    cfg = model.Config(4, 8, 1, 1, 8, ("conv",))
    rng = np.random.default_rng(12)
    backbone = {
        name: jnp.asarray(rng.normal(0, 0.1, shape), dtype=jnp.float32)
        for name, shape in model.expected_shapes(cfg).items()
    }
    allowed = (True, True, False)
    params = classification.initialize_from_backbone(
        backbone, cfg, allowed, head_seed=6
    )
    assert set(params) - set(backbone) == {classification.HEAD_NAME}
    for name in backbone:
        np.testing.assert_array_equal(params[name], backbone[name])
    inventory = classification.parameter_inventory(cfg, allowed)
    assert inventory.frozen_names == ("model.embed_tokens.weight",)
    state = adamw.initialize_state(params, inventory)
    ids = jnp.asarray([[[1, 2, 3], [2, 3, 0]]], dtype=jnp.int32)
    batch = {
        "input_ids": ids,
        "attention_mask": jnp.asarray([[[1, 1, 1], [1, 1, 0]]]),
        "labels": jnp.asarray([[1, 2]], dtype=jnp.int32),
        "valid_rows": jnp.asarray([[True, False]]),
    }
    update = jax.jit(
        classification.make_lfm2_5_step(
            cfg,
            allowed,
            inventory,
            adamw.AdamWConfig(0.01),
            dtype=jnp.float32,
        )
    )
    result = update(state, batch, jnp.asarray([True]))
    assert bool(result.committed)
    assert np.isfinite(float(result.loss))
    assert int(result.state["step"]) == 1
    np.testing.assert_array_equal(
        result.state["params"]["model.embed_tokens.weight"],
        backbone["model.embed_tokens.weight"],
    )
    assert not np.array_equal(
        result.state["params"][classification.HEAD_NAME],
        params[classification.HEAD_NAME],
    )
    assert int(
        classification.predict(
            result.state["params"],
            ids[0, :1],
            batch["attention_mask"][0, :1],
            cfg,
            allowed,
            dtype=jnp.float32,
        )[0]
    ) in (0, 1)


def test_streaming_classifier_accepts_plain_block_autodiff() -> None:
    """The strategy threads the plain-block policy to its gradient program."""
    cfg = model.Config(4, 8, 1, 1, 8, ("conv",))
    rng = np.random.default_rng(8)
    backbone = {
        name: jnp.asarray(rng.normal(0, 0.1, shape), dtype=jnp.float32)
        for name, shape in model.expected_shapes(cfg).items()
    }
    allowed = (True, True, False)
    parameters = classification.initialize_from_backbone(
        backbone, cfg, allowed, head_seed=6
    )
    inventory = classification.parameter_inventory(cfg, allowed)
    batch = {
        "input_ids": jnp.asarray([[1, 2, 3]], dtype=jnp.int32),
        "attention_mask": jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
        "labels": jnp.asarray([1], dtype=jnp.int32),
        "valid_rows": jnp.asarray([True]),
    }
    optimizer = adamw.AdamWConfig(0.01)
    checkpointed = classification.make_lfm2_5_streaming_step(
        cfg, allowed, inventory, optimizer, dtype=jnp.float32
    )
    plain = classification.make_lfm2_5_streaming_step(
        cfg,
        allowed,
        inventory,
        optimizer,
        dtype=jnp.float32,
        rematerialize_blocks=False,
    )
    fused = classification.make_lfm2_5_streaming_step(
        cfg,
        allowed,
        inventory,
        optimizer,
        dtype=jnp.float32,
        fuse_accumulation=True,
    )
    assert checkpointed.accumulate is None
    assert fused.accumulate is not None
    old_loss, old_count, old_gradient = checkpointed.gradient(parameters, batch)
    new_loss, new_count, new_gradient = plain.gradient(parameters, batch)
    np.testing.assert_array_equal(old_loss, new_loss)
    np.testing.assert_array_equal(old_count, new_count)
    assert old_gradient.keys() == new_gradient.keys()
    for name in old_gradient:
        np.testing.assert_allclose(
            old_gradient[name], new_gradient[name], rtol=1e-6, atol=1e-6
        )
