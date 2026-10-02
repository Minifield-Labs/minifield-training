"""Compose schema-field supervision with any compatible model forward."""

from collections.abc import Callable

import jax

from minifield_training.core import parameters as core_parameters
from minifield_training.engine import step
from minifield_training.kernels import types
from minifield_training.objectives import schema_fields as objective
from minifield_training.optimizers import adamw


def make_step(
    forward: Callable[[types.Parameters, types.DeviceBatch], types.DeviceBatch],
    inventory: core_parameters.FullParameterInventory,
    optimizer: adamw.AdamWConfig,
    *,
    mesh: jax.sharding.Mesh | None = None,
) -> step.JitStep:
    """Bind task loss to a supplied training forward and parameter inventory."""

    def loss_terms(
        params: types.Parameters, batch: types.DeviceBatch
    ) -> tuple[jax.Array, jax.Array]:
        return objective.terms(forward(params, batch), batch)

    return step.make_jit_step(loss_terms, inventory, optimizer, mesh=mesh)
