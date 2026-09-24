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
