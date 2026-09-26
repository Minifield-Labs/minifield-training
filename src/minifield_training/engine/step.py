"""Token-normalized optimizer updates over fixed-shape microbatches."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

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
type Accumulation = tuple[jax.Array, jax.Array, types.Parameters, jax.Array]
type PhysicalGradient = Callable[
    [types.Parameters, types.DeviceBatch],
    tuple[jax.Array, jax.Array, types.Parameters],
]


def _parallel_gradient(
    gradient: PhysicalGradient, mesh: jax.sharding.Mesh
) -> PhysicalGradient:
    """Split rows and replicate global loss, count, and gradient sums."""
    if mesh.axis_names != ("data",):
        raise ValueError(
            "Data parallelism requires a one-dimensional data mesh"
        )

    def reduce_gradient(
        parameters: types.Parameters, batch: types.DeviceBatch
    ) -> tuple[jax.Array, jax.Array, types.Parameters]:
        """Prevent positive counts from hiding an invalid replica count."""
        # Mark replica-local differentiation inputs before value_and_grad.
        # Otherwise VMA autodiff inserts its own psum for invariant masters,
        # and the explicit reduction below multiplies gradients by mesh.size.
        # JAX 0.7.2's collective APIs don't carry type annotations.
        local = jax.lax.pvary(  # type: ignore[no-untyped-call]
            parameters, "data"
        )
        loss, count, gradients = gradient(local, batch)
        count = jnp.where(
            jnp.isfinite(count) & (count >= 0), count, jnp.float32(jnp.nan)
        )
        return cast(
            tuple[jax.Array, jax.Array, types.Parameters],
            jax.lax.psum(  # type: ignore[no-untyped-call]
                (loss, count, gradients), "data"
            ),
        )

    # JAX 0.7.2's PartitionSpec constructor has no type annotations.
    replicated = jax.sharding.PartitionSpec()  # type: ignore[no-untyped-call]
    rows = jax.sharding.PartitionSpec("data")  # type: ignore[no-untyped-call]
    return cast(
        PhysicalGradient,
        jax.shard_map(
            reduce_gradient,
            mesh=mesh,
            in_specs=(replicated, rows),
            out_specs=(replicated, replicated, replicated),
        ),
    )


@dataclass(frozen=True)
class StreamingStep:
    """Compile physical gradients and the commit as separate programs.

    The host only selects active slots. Gradients, sums, normalization, and
    the transactional update stay on device, with one commit per logical step.
    This bounds the reverse pass to one physical batch per device. An optional
    data mesh splits global rows while keeping masters and moments replicated.
    """

    gradient: PhysicalGradient
    add: Callable[[types.Parameters, types.Parameters], types.Parameters]
    normalize: Callable[[types.Parameters, jax.Array], types.Parameters]
    transition: Callable[
        [state.State, types.Parameters, jax.Array, jax.Array],
        adamw.CommitResult,
    ]
    inventory: core_parameters.FullParameterInventory
    accumulate: (
        Callable[
            [types.Parameters, types.DeviceBatch, Accumulation], Accumulation
        ]
        | None
    ) = None
    mesh: jax.sharding.Mesh | None = None

    def __call__(
        self,
        full_state: state.State,
        microbatches: types.DeviceBatch,
        active: NDArray[np.bool_] | jax.Array,
    ) -> adamw.CommitResult:
        """Sum active decision gradients, then attempt one donated commit."""
        adamw.validate_full_weight_state_structure(full_state, self.inventory)
        if active.ndim != 1 or np.dtype(active.dtype) != np.dtype(bool):
            raise ValueError("active must be a boolean microbatch vector")
        if not microbatches or any(
            value.ndim < 1 or value.shape[0] != active.shape[0]
            for value in microbatches.values()
        ):
            raise ValueError("microbatches need a common leading axis")
        if self.mesh is not None and any(
            value.ndim < 2 or value.shape[1] % self.mesh.size
            for value in microbatches.values()
        ):
            raise ValueError("Physical rows must be divisible by device count")
        enabled = np.asarray(active)
        total: Accumulation | None = None
        for index in np.flatnonzero(enabled):
            batch = {
                name: value[int(index)] for name, value in microbatches.items()
            }
            if total is None:
                loss, count, gradients = self.gradient(
                    full_state["params"], batch
                )
                total = (
                    loss,
                    count,
                    gradients,
                    jnp.isfinite(count) & (count >= 0),
                )
            elif self.accumulate is not None:
                total = self.accumulate(full_state["params"], batch, total)
            else:
                loss, count, gradients = self.gradient(
                    full_state["params"], batch
                )
                old_loss, old_count, old_gradients, counts_valid = total
                total = (
                    old_loss + loss,
                    old_count + count,
                    self.add(old_gradients, gradients),
                    counts_valid & jnp.isfinite(count) & (count >= 0),
                )
        if total is None:
            nan = jnp.float32(jnp.nan)
            return adamw.CommitResult(
                full_state,
                jnp.asarray(False),
                jnp.int32(adamw.CommitCode.ACCUMULATION_INVALID),
                jnp.float32(0),
                nan,
                nan,
                nan,
            )
        loss, count, gradients, counts_valid = total
        valid = counts_valid & jnp.isfinite(count) & (count > 0)
        denominator = jnp.where(valid, count, jnp.float32(1))
        normalized = self.normalize(gradients, denominator)
        return self.transition(
            full_state, normalized, loss / denominator, valid
        )


def make_streaming_step(
    loss_terms: LossTerms,
    inventory: core_parameters.FullParameterInventory,
    config: adamw.AdamWConfig,
    *,
    fuse_accumulation: bool = False,
    mesh: jax.sharding.Mesh | None = None,
) -> StreamingStep:
    """Build accumulation with optional row parallelism and fused sums.

    Fusing later physical gradients with their previous sums needs device
    memory for both the reverse pass and the carried gradients. Keep it opt-in
    until target-device peak memory and throughput are measured.
    """
    if not inventory.trainable_names:
        raise ValueError("Logical updates need a trainable parameter")
    trainable_names = inventory.trainable_names
    frozen_names = inventory.frozen_names

    def gradient(
        parameters: types.Parameters, batch: types.DeviceBatch
    ) -> tuple[jax.Array, jax.Array, types.Parameters]:
        """Differentiate one physical batch without optimizer state."""
        trainable = {name: parameters[name] for name in trainable_names}
        frozen = {name: parameters[name] for name in frozen_names}

        def objective(
            selected: types.Parameters,
        ) -> tuple[jax.Array, jax.Array]:
            """Join frozen masters only for the forward pass."""
            loss, count = loss_terms({**frozen, **selected}, batch)
            if loss.shape or count.shape:
                raise ValueError("loss and count must be scalars")
            if np.dtype(loss.dtype) != np.dtype(np.float32) or np.dtype(
                count.dtype
            ) != np.dtype(np.float32):
                raise ValueError("loss and count must be float32")
            return loss, count

        (loss, count), gradients = jax.value_and_grad(objective, has_aux=True)(
            trainable
        )
        return loss, count, gradients

    physical_gradient = (
        gradient if mesh is None else _parallel_gradient(gradient, mesh)
    )

    def add(
        accumulated: types.Parameters,
        gradients: types.Parameters,
    ) -> types.Parameters:
        """Keep the full gradient sum on device between physical batches."""
        return cast(
            types.Parameters, jax.tree.map(jnp.add, accumulated, gradients)
        )

    def normalize(
        gradients: types.Parameters, denominator: jax.Array
    ) -> types.Parameters:
        """Divide the complete sum by its decision count exactly once."""
        return cast(
            types.Parameters,
            jax.tree.map(lambda value: value / denominator, gradients),
        )

    def accumulate(
        parameters: types.Parameters,
        batch: types.DeviceBatch,
        previous: Accumulation,
    ) -> Accumulation:
        """Differentiate and add one later slot in the same JIT program."""
        loss, count, gradients = physical_gradient(parameters, batch)
        old_loss, old_count, old_gradients, counts_valid = previous
        return (
            old_loss + loss,
            old_count + count,
            cast(
                types.Parameters,
                jax.tree.map(jnp.add, old_gradients, gradients),
            ),
            counts_valid & jnp.isfinite(count) & (count >= 0),
        )

    return StreamingStep(
        jax.jit(physical_gradient),
        jax.jit(add, donate_argnums=(0,)),
        jax.jit(normalize, donate_argnums=(0,)),
        adamw.make_donated_transaction(inventory, config),
        inventory,
        jax.jit(accumulate, donate_argnums=(2,)) if fuse_accumulation else None,
        mesh,
    )


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
