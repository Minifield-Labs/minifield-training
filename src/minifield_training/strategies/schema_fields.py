"""Compose schema-field supervision with any compatible model forward."""

from collections.abc import Callable

import jax

from minifield_training.core import parameters as core_parameters
from minifield_training.engine import step
from minifield_training.kernels import types
from minifield_training.objectives import schema_fields as objective
from minifield_training.optimizers import adamw

type Terms = Callable[
    [types.DeviceBatch, types.DeviceBatch], tuple[jax.Array, jax.Array]
]


def make_step(
    forward: Callable[[types.Parameters, types.DeviceBatch], types.DeviceBatch],
    inventory: core_parameters.FullParameterInventory,
    optimizer: adamw.AdamWConfig,
    *,
    mesh: jax.sharding.Mesh | None = None,
    terms: Terms = objective.terms,
    transaction: step.Transaction = adamw.make_transaction,
) -> step.JitStep:
    """Bind a task loss to a supplied training forward and inventory.

    ``terms(outputs, batch)`` returns summed loss and weight mass. The default
    is per-row schema-field supervision; ``objectives.pointer.terms`` binds
    the joint pointer formulation. ``transaction`` selects the optimizer
    commit, such as ``optimizers.optax_adamw.make_transaction``.
    """

    def loss_terms(
        params: types.Parameters, batch: types.DeviceBatch
    ) -> tuple[jax.Array, jax.Array]:
        return terms(forward(params, batch), batch)

    return step.make_jit_step(
        loss_terms, inventory, optimizer, mesh=mesh, transaction=transaction
    )
