"""One token-normalized optimizer update over fixed-shape microbatches."""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from minifield_training.core import parameters as core_parameters
from minifield_training.kernels import types
from minifield_training.optimizers import adamw
from minifield_training.optimizers import state

type LossTerms = Callable[
    [types.Parameters, types.DeviceBatch], tuple[jax.Array, jax.Array]
]
type LogicalStep = Callable[
    [state.State, types.DeviceBatch, jax.Array], adamw.CommitResult
]


def make_step(
    loss_terms: LossTerms,
    inventory: core_parameters.FullParameterInventory,
    config: adamw.AdamWConfig,
) -> LogicalStep:
    """Build a pure logical update from unaveraged loss and target count.

    ``microbatches`` is a dictionary of arrays with a common leading
    microbatch axis. ``active`` is a boolean vector on that axis. Inactive
    slots skip the objective and contribute zeros. Only trainable inventory
    leaves are differentiated; the full FP32 tree reaches the forward call.
    The returned function is compatible with ``jax.jit``.
    """
    if not inventory.trainable_names:
        raise ValueError("Logical updates need a trainable parameter")
    transition = adamw.make_transaction(inventory, config)
    trainable_names = inventory.trainable_names

    def step(
        full_state: state.State,
        microbatches: types.DeviceBatch,
        active: jax.Array,
    ) -> adamw.CommitResult:
        """Accumulate sums, normalize once, and attempt one AdamW commit."""
        adamw.validate_full_weight_state_structure(full_state, inventory)
        if active.ndim != 1 or np.dtype(active.dtype) != np.dtype(bool):
            raise ValueError("active must be a boolean microbatch vector")
        if not microbatches or any(
            value.ndim < 1 or value.shape[0] != active.shape[0]
            for value in microbatches.values()
        ):
            raise ValueError("microbatches need a common leading axis")
        trainable = {
            name: full_state["params"][name] for name in trainable_names
        }
        frozen = {
            name: full_state["params"][name] for name in inventory.frozen_names
        }
        zeros = jax.tree.map(jnp.zeros_like, trainable)

        def objective(
            selected: types.Parameters, batch: types.DeviceBatch
        ) -> tuple[jax.Array, jax.Array]:
            """Rebuild the full forward tree while tracing selected leaves."""
            total, count = loss_terms({**frozen, **selected}, batch)
            if total.shape or count.shape:
                raise ValueError("loss and count must be scalars")
            if np.dtype(total.dtype) != np.dtype(np.float32) or np.dtype(
                count.dtype
            ) != np.dtype(np.float32):
                raise ValueError("loss and count must be float32")
            return total, count

        def accumulate(
            carry: tuple[jax.Array, jax.Array, types.Parameters, jax.Array],
            item: tuple[types.DeviceBatch, jax.Array],
        ) -> tuple[
            tuple[jax.Array, jax.Array, types.Parameters, jax.Array], None
        ]:
            """Skip inactive slots without evaluating their objective."""
            batch, enabled = item

            def evaluate(
                _: None,
            ) -> tuple[jax.Array, jax.Array, types.Parameters]:
                """Differentiate the summed loss of an active slot."""
                (loss, count), gradients = jax.value_and_grad(
                    objective, has_aux=True
                )(trainable, batch)
                return loss, count, gradients

            def skip(_: None) -> tuple[jax.Array, jax.Array, types.Parameters]:
                """Supply finite neutral values for an inactive slot."""
                return jnp.float32(0), jnp.float32(0), zeros

            loss, count, gradients = jax.lax.cond(enabled, evaluate, skip, None)
            old_loss, old_count, old_gradients, counts_valid = carry
            return (
                old_loss + loss,
                old_count + count,
                jax.tree.map(jnp.add, old_gradients, gradients),
                counts_valid & jnp.isfinite(count) & (count >= 0),
            ), None

        initial = (jnp.float32(0), jnp.float32(0), zeros, jnp.asarray(True))
        (loss_sum, count_sum, gradient_sum, counts_valid), _ = jax.lax.scan(
            accumulate, initial, (microbatches, active)
        )
        valid_count = counts_valid & jnp.isfinite(count_sum) & (count_sum > 0)
        denominator = jnp.where(valid_count, count_sum, jnp.float32(1))
        gradients = cast(
            types.Parameters,
            jax.tree.map(lambda value: value / denominator, gradient_sum),
        )
        adamw.validate_gradient_tree(gradients, inventory)
        return transition(
            full_state, gradients, loss_sum / denominator, valid_count
        )

    return step
