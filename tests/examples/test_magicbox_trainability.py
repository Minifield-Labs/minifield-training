"""Frozen embeddings stay outside differentiation, decay, and updates."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from examples.magicbox import composition
from examples.magicbox import smoke
from minifield_training.batching import schema_fields as batching
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.magicbox import model
from minifield_training.objectives import schema_fields as objective
from minifield_training.optimizers import adamw
from minifield_training.strategies import schema_fields as strategy


def test_pretrained_trainability_inventory() -> None:
    """The real recipe freezes exactly the 67.1M embedding parameters."""
    config = (
        Path(__file__).parents[1] / "models/lfm2_5/fixtures/encoder_config.json"
    )
    cfg = encoder.Adapter().parse_config(json.loads(config.read_text()))
    inventory = composition.inventory(cfg, model.Config())
    assert inventory.frozen_names == ("lfm2.embed_tokens.weight",)
    assert inventory.parameter_count == 356_390_916
    assert inventory.trainable_parameter_count == 289_282_052
    embedding = next(spec for spec in inventory.specs if not spec.trainable)
    assert embedding.shape == (65536, 1024)
    assert not embedding.decayed


def test_embeddings_unchanged_while_encoder_and_heads_learn() -> None:
    """A real update omits embedding gradients and preserves their bits."""
    cfg, fusion, params = smoke.tiny()
    inventory = composition.inventory(cfg, fusion)
    packed = batching.build(
        [smoke.fixture()],
        batching.Shape(1, 1, 16, 64, 8, 128, 0),
        seed=17,
        update=0,
        weighting=objective.balance_types,
    )
    update = strategy.make_step(
        composition.bind(cfg, fusion, training=True, bf16=False),
        inventory,
        adamw.AdamWConfig(learning_rate=0.003, weight_decay=0.1),
    )
    batch = {
        key: jnp.asarray(value[0]) for key, value in packed.microbatches.items()
    }
    _, _, gradients = update.gradient(params, batch)
    assert set(gradients) == set(inventory.trainable_names)
    original = {name: np.array(value) for name, value in params.items()}
    current = adamw.initialize_state(params, inventory)
    result = update(current, packed.microbatches, packed.active)
    assert bool(result.committed)
    embedding = "lfm2.embed_tokens.weight"
    np.testing.assert_array_equal(
        result.state["params"][embedding], original[embedding]
    )
    np.testing.assert_array_equal(result.state["m"][embedding], 0)
    np.testing.assert_array_equal(result.state["v"][embedding], 0)
    for prefix in ("lfm2.layers.", "magicbox."):
        assert any(
            not np.array_equal(result.state["params"][name], original[name])
            for name in inventory.trainable_names
            if name.startswith(prefix)
        )
