"""Compare MagicBox physical gradients on one and eight CPU devices."""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np

from examples.magicbox import composition as magicbox
from examples.magicbox import smoke
from minifield_training.batching import schema_fields as batching
from minifield_training.objectives import schema_fields as objective
from minifield_training.optimizers import adamw
from minifield_training.strategies import schema_fields as strategy


def main() -> None:
    """Compare sharded loss, mass, and gradients."""
    if len(jax.devices()) != 8:
        raise ValueError(
            "Set XLA_FLAGS=--xla_force_host_platform_device_count=8"
        )
    cfg, fusion, params = smoke.tiny()
    record = smoke.fixture()
    records = [
        dataclasses.replace(
            record,
            id=str(index),
            fields=record.fields if index % 2 else record.fields[:2],
        )
        for index in range(8)
    ]
    packed = batching.build(
        records,
        batching.Shape(1, 8, 16, 64, 8, 128, 0),
        seed=0,
        update=0,
        weighting=objective.balance_types,
    )
    batch = {
        key: jnp.asarray(value[0]) for key, value in packed.microbatches.items()
    }
    optimizer = adamw.AdamWConfig(learning_rate=0.001)
    single = strategy.make_step(
        magicbox.bind(cfg, fusion, training=True, bf16=False),
        magicbox.inventory(cfg, fusion),
        optimizer,
    )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    parallel = strategy.make_step(
        magicbox.bind(cfg, fusion, training=True, bf16=False),
        magicbox.inventory(cfg, fusion),
        optimizer,
        mesh=mesh,
    )
    expected = single.gradient(params, batch)
    actual = parallel.gradient(params, batch)
    differences = []
    for left, right in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(left), np.asarray(right), rtol=2e-4, atol=2e-6
        )
        differences.append(
            float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
        )
    print(
        json.dumps(
            {
                "devices": 8,
                "platform": "cpu",
                "dtype": "float32",
                "loss": float(actual[0]),
                "mass": float(actual[1]),
                "maximum_absolute_gradient_difference": max(differences),
            }
        )
    )


if __name__ == "__main__":
    main()
