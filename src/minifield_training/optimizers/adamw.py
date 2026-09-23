"""Transactional full-weight AdamW updates over an FP32 master tree.

This module deliberately consumes an already-normalized logical gradient. Route,
argument, candidate, and physical microbatch normalization belong to the batch
compiler. A transaction either commits every unique master and both Adam moments
once, or returns the input state unchanged with an explicit numeric code.
"""

from collections.abc import Callable, Mapping
import dataclasses
import enum
import functools
import hashlib
import math
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from minifield_training.core import json_io
from minifield_training.core import parameters as core_parameters
from minifield_training.kernels import types
from minifield_training.optimizers import state

_INT32_MAX = np.iinfo(np.int32).max
_IMPLEMENTATION_ID = "minifield.full-weight-adamw/2-portable-f32-normal"


class CommitCode(enum.IntEnum):
    """Result codes that stay representable across a JAX transformation."""

    COMMITTED = 0
    ACCUMULATION_INVALID = 1
    NONFINITE_LOSS = 2
    INVALID_STATE = 3
    NONFINITE_GRADIENT = 4
    STEP_OVERFLOW = 5
    CANDIDATE_INVALID = 6


@dataclasses.dataclass(frozen=True)
class AdamWConfig:
    """Validated scalar settings for one committed logical AdamW update."""

    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    clip_norm: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid numeric settings before a transition is built."""
        values: dict[str, object] = {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "clip_norm": self.clip_norm,
        }
        converted: dict[str, float] = {}
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a non-boolean finite number")
            number = float(value)
            if not math.isfinite(number) or abs(number) > float(
                np.finfo(np.float32).max
            ):
                raise ValueError(f"{name} must be a finite float32 scalar")
            effective = float(np.float32(number))
            if number and effective == 0:
                raise ValueError(f"{name} must be representable in float32")
            if effective and abs(effective) < float(np.finfo(np.float32).tiny):
                raise ValueError(
                    f"{name} must be zero or a normal float32 scalar"
                )
            converted[name] = effective
        if converted["learning_rate"] <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= converted["beta1"] < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= converted["beta2"] < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if converted["epsilon"] <= 0:
            raise ValueError("epsilon must be positive")
        if converted["weight_decay"] < 0:
            raise ValueError("weight_decay must be nonnegative")
        if converted["clip_norm"] <= 0:
            raise ValueError("clip_norm must be positive")
        for name, value in converted.items():
            object.__setattr__(self, name, value)

    @property
    def implementation_identity(self) -> str:
        """Return a canonical digest for checkpoint resume binding."""
        payload = {
            "implementation": _IMPLEMENTATION_ID,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "clip_norm": self.clip_norm,
        }
        return hashlib.sha256(
            json_io.canonical(payload).encode("utf-8")
        ).hexdigest()


class CommitResult(NamedTuple):
    """State and diagnostics for one all-or-nothing logical transaction."""

    state: state.State
    committed: jax.Array
    code: jax.Array
    loss: jax.Array
    gradient_norm: jax.Array
    clipped_gradient_norm: jax.Array
    update_norm: jax.Array


def _inventory_by_name(
    inventory: core_parameters.FullParameterInventory,
) -> dict[str, core_parameters.FullParameterSpec]:
    """Copy immutable inventory specs into a stable name-indexed mapping."""
    result = {spec.name: spec for spec in inventory.specs}
    if tuple(sorted(result)) != inventory.names or len(result) != len(
        inventory.specs
    ):
        raise ValueError("Full-weight inventory contains duplicate names")
    if not result:
        raise ValueError("Full-weight inventory must not be empty")
    if any(spec.master_dtype != "float32" for spec in result.values()):
        raise ValueError("Full-weight inventory must contain FP32 leaves")
    return result


def _trainable_specs(
    inventory: core_parameters.FullParameterInventory,
) -> dict[str, core_parameters.FullParameterSpec]:
    """Return only the leaf specs that receive a gradient and an update."""
    return {
        name: spec
        for name, spec in _inventory_by_name(inventory).items()
        if spec.trainable
    }


def _validate_tree_structure(
    values: Mapping[str, object],
    specs: Mapping[str, core_parameters.FullParameterSpec],
    group: str,
) -> None:
    """Check exact tree membership, shapes, and FP32 dtypes without values."""
    actual = set(values)
    expected = set(specs)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        if any(".lora_" in name for name in extra):
            raise ValueError("Full-weight state cannot contain LoRA tensors")
        raise ValueError(
            f"Full-weight {group} tree mismatch; "
            f"missing={missing}, extra={extra}"
        )
    for name, spec in specs.items():
        value = cast(jax.Array, values[name])
        if tuple(value.shape) != spec.shape:
            raise ValueError(f"Full-weight {group} shape mismatch: {name}")
        try:
            dtype = np.dtype(value.dtype)
        except TypeError as error:
            raise ValueError(
                f"Full-weight {group} dtype is unavailable: {name}"
            ) from error
        if dtype != np.dtype(np.float32):
            raise ValueError(f"Full-weight {group} must be float32: {name}")


def _validate_step_structure(step: object) -> None:
    """Check the scalar int32 step without converting its dynamic value."""
    value = cast(jax.Array, step)
    if tuple(value.shape):
        raise ValueError("Full-weight step must be a scalar")
    try:
        dtype = np.dtype(value.dtype)
    except TypeError as error:
        raise ValueError("Full-weight step dtype is unavailable") from error
    if dtype != np.dtype(np.int32):
        raise ValueError("Full-weight step must be int32")


def validate_gradient_tree(
    gradients: Mapping[str, object],
    inventory: core_parameters.FullParameterInventory,
) -> None:
    """Perform cheap static validation before a traced transaction call.

    A gradient tree covers exactly the trainable leaves: frozen inventory
    leaves never carry a gradient and must not appear here.
    """
    _validate_tree_structure(gradients, _trainable_specs(inventory), "gradient")


def validate_full_weight_state_structure(
    full_state: state.State,
    inventory: core_parameters.FullParameterInventory,
) -> None:
    """Validate state keys, shapes, and dtypes without reading device data."""
    specs = _inventory_by_name(inventory)
    for group in ("params", "m", "v"):
        _validate_tree_structure(full_state[group], specs, group)
    _validate_step_structure(full_state["step"])


def validate_full_weight_state(
    full_state: state.State,
    inventory: core_parameters.FullParameterInventory,
) -> None:
    """Eagerly validate checkpoint/load state including finite master values.

    This intentionally transfers leaves to the host. It belongs at explicit
    initialization, restore, checkpoint, and export boundaries, never the
    model-update hot path.
    """
    validate_full_weight_state_structure(full_state, inventory)
    specs = _inventory_by_name(inventory)
    step = int(np.asarray(full_state["step"]))
    if step < 0 or step > _INT32_MAX:
        raise ValueError("Full-weight step is outside int32 range")
    for name in specs:
        parameter = np.asarray(full_state["params"][name])
        first_moment = np.asarray(full_state["m"][name])
        second_moment = np.asarray(full_state["v"][name])
        if not np.isfinite(parameter).all():
            raise ValueError(f"Full-weight parameter is non-finite: {name}")
        if not np.isfinite(first_moment).all():
            raise ValueError(f"Full-weight first moment is non-finite: {name}")
        if not np.isfinite(second_moment).all() or np.any(second_moment < 0):
            raise ValueError(f"Full-weight second moment is invalid: {name}")


def _replicated_scalar_sharding(
    parameters: types.Parameters,
) -> jax.sharding.Sharding | None:
    """Return replicated scalar sharding from the parameter mesh."""
    first = next(iter(parameters.values()), None)
    sharding = getattr(first, "sharding", None)
    if not isinstance(sharding, jax.sharding.NamedSharding):
        return None
    spec = jax.sharding.PartitionSpec()  # type: ignore[no-untyped-call]
    return jax.sharding.NamedSharding(sharding.mesh, spec)


def _zero_step(parameters: types.Parameters) -> jax.Array:
    """Create step zero with the same committed placement as model state."""
    value = jnp.asarray(0, dtype=jnp.int32)
    sharding = _replicated_scalar_sharding(parameters)
    if sharding is None:
        # Without a named mesh, match the first parameter leaf so later
        # committed states present one stable aval to jitted callers.
        first = next(iter(parameters.values()), None)
        sharding = getattr(first, "sharding", None)
    if sharding is None:
        return value
    return cast(jax.Array, jax.device_put(value, sharding))


def normalize_step_sharding(full_state: state.State) -> state.State:
    """Restore the step with replicated model-mesh sharding."""
    sharding = _replicated_scalar_sharding(full_state["params"])
    if sharding is None:
        return full_state
    step = full_state["step"]
    if getattr(step, "sharding", None) == sharding:
        return full_state
    return {**full_state, "step": jax.device_put(step, sharding)}


def initialize_state(
    parameters: types.Parameters,
    inventory: core_parameters.FullParameterInventory,
) -> state.State:
    """Create a complete zero-moment FP32 state for every unique master leaf.

    Frozen inventory leaves keep permanently zero moments. The shared v2
    checkpoint format requires params, m, and v to carry identical tensor
    membership, so frozen moment slots stay allocated rather than dropped.
    """
    specs = _inventory_by_name(inventory)
    _validate_tree_structure(parameters, specs, "parameter")
    for name in specs:
        if not np.isfinite(np.asarray(parameters[name])).all():
            raise ValueError(f"Full-weight parameter is non-finite: {name}")
    full_state: state.State = {
        "params": {name: parameters[name] for name in specs},
        "m": {name: jnp.zeros_like(parameters[name]) for name in specs},
        "v": {name: jnp.zeros_like(parameters[name]) for name in specs},
        "step": _zero_step(parameters),
    }
    validate_full_weight_state(full_state, inventory)
    return full_state


def _tree_all_finite(values: types.Parameters) -> jax.Array:
    """Return one device boolean without host transfer or Python coercion."""
    checks = [
        jnp.all(jnp.isfinite(value.astype(jnp.float32)))
        for value in values.values()
    ]
    return jnp.all(jnp.stack(checks))


def _moments_valid(full_state: state.State) -> jax.Array:
    """Check all incoming FP32 masters and Adam moments on device."""
    return (
        _tree_all_finite(full_state["params"])
        & _tree_all_finite(full_state["m"])
        & _tree_all_finite(full_state["v"])
        & jnp.all(
            jnp.stack(
                [
                    jnp.all(value.astype(jnp.float32) >= 0)
                    for value in full_state["v"].values()
                ]
            )
        )
    )


def state_values_valid_on_device(full_state: state.State) -> jax.Array:
    """Return one device scalar for valid FP32 leaves and a nonnegative step."""
    return _moments_valid(full_state) & (
        full_state["step"] >= jnp.asarray(0, dtype=jnp.int32)
    )


def _stable_norm(values: types.Parameters) -> jax.Array:
    """Compute an FP32 norm without squaring a large finite leaf first."""
    maximum = jnp.float32(0)
    for value in values.values():
        maximum = jnp.maximum(
            maximum, jnp.max(jnp.abs(value.astype(jnp.float32)))
        )
    safe_maximum = jnp.where(maximum == 0, jnp.float32(1), maximum)
    squared = jnp.float32(0)
    for value in values.values():
        scaled = value.astype(jnp.float32) / safe_maximum
        squared = squared + jnp.sum(scaled * scaled, dtype=jnp.float32)
    return cast(jax.Array, maximum * jnp.sqrt(squared))


def _validate_transition_inputs(
    active_loss: jax.Array, accumulation_valid: jax.Array
) -> None:
    """Check the scalar loss and accumulation flag without values."""
    if tuple(active_loss.shape):
        raise ValueError("active_loss must be a scalar")
    if np.dtype(active_loss.dtype) != np.dtype(np.float32):
        raise ValueError("active_loss must be a real float32 scalar")
    if tuple(accumulation_valid.shape):
        raise ValueError("accumulation_valid must be a scalar")
    if np.dtype(accumulation_valid.dtype) != np.dtype(bool):
        raise ValueError("accumulation_valid must be boolean")


def make_transaction(
    inventory: core_parameters.FullParameterInventory,
    config: AdamWConfig,
) -> Callable[
    [state.State, types.Parameters, jax.Array, jax.Array], CommitResult
]:
    """Build a pure full-weight AdamW commit without parameter donation.

    The returned transition accepts an already-normalized logical gradient
    covering exactly the trainable inventory leaves. Frozen leaves pass through
    bit-identically: their masters and moments are preserved without an update,
    a decay term, or a delta. Its caller must invoke validate_gradient_tree at
    a host boundary for untrusted tree structures. Dynamic finite checks remain
    entirely on device so jax.jit and value_and_grad callers are not broken by
    Python bools.
    """
    specs = _inventory_by_name(inventory)
    trainable = {name: spec for name, spec in specs.items() if spec.trainable}
    decay = {name: spec.decayed for name, spec in specs.items()}
    learning_rate = jnp.float32(config.learning_rate)
    beta1 = jnp.float32(config.beta1)
    beta2 = jnp.float32(config.beta2)
    epsilon = jnp.float32(config.epsilon)
    weight_decay = jnp.float32(config.weight_decay)
    clip_norm = jnp.float32(config.clip_norm)

    def transition(
        full_state: state.State,
        gradients: types.Parameters,
        active_loss: jax.Array,
        accumulation_valid: jax.Array,
    ) -> CommitResult:
        """Commit once or preserve the exact incoming healthy state."""
        for group in ("params", "m", "v"):
            _validate_tree_structure(full_state[group], specs, group)
        _validate_step_structure(full_state["step"])
        _validate_tree_structure(gradients, trainable, "gradient")
        _validate_transition_inputs(active_loss, accumulation_valid)

        loss = active_loss.astype(jnp.float32)
        step = full_state["step"]
        accumulation_ok = accumulation_valid.astype(bool)
        state_ok = _moments_valid(full_state) & (step >= 0)
        loss_ok = jnp.isfinite(loss)
        gradients_ok = _tree_all_finite(gradients)
        step_ok = step < jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        code = jnp.where(
            ~state_ok,
            jnp.int32(CommitCode.INVALID_STATE),
            jnp.where(
                ~accumulation_ok,
                jnp.int32(CommitCode.ACCUMULATION_INVALID),
                jnp.where(
                    ~loss_ok,
                    jnp.int32(CommitCode.NONFINITE_LOSS),
                    jnp.where(
                        ~gradients_ok,
                        jnp.int32(CommitCode.NONFINITE_GRADIENT),
                        jnp.where(
                            ~step_ok,
                            jnp.int32(CommitCode.STEP_OVERFLOW),
                            jnp.int32(CommitCode.COMMITTED),
                        ),
                    ),
                ),
            ),
        )

        def rejected(_: None) -> CommitResult:
            """Return the original state without a false commit claim."""
            nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
            return CommitResult(
                full_state,
                jnp.asarray(False),
                code,
                loss,
                nan,
                nan,
                nan,
            )

        def attempt(_: None) -> CommitResult:
            """Calculate one clipped AdamW candidate from old master state."""
            gradient_norm = _stable_norm(gradients)
            clip_scale = jnp.where(
                gradient_norm == 0,
                jnp.float32(1),
                jnp.minimum(jnp.float32(1), clip_norm / gradient_norm),
            )
            clipped_gradient_norm = gradient_norm * clip_scale
            new_step = step + jnp.asarray(1, dtype=jnp.int32)
            iteration = new_step.astype(jnp.float32)
            m: types.Parameters = {}
            v: types.Parameters = {}
            params: types.Parameters = {}
            delta: types.Parameters = {}
            for name in specs:
                if name not in trainable:
                    # Frozen leaves commit bit-identically: no moment update,
                    # no decay, and no delta entry (zero update contribution).
                    params[name] = full_state["params"][name]
                    m[name] = full_state["m"][name]
                    v[name] = full_state["v"][name]
                    continue
                gradient = gradients[name].astype(jnp.float32) * clip_scale
                first = (
                    beta1 * full_state["m"][name].astype(jnp.float32)
                    + (jnp.float32(1) - beta1) * gradient
                )
                second = (
                    beta2 * full_state["v"][name].astype(jnp.float32)
                    + (jnp.float32(1) - beta2) * gradient * gradient
                )
                first_hat = first / (jnp.float32(1) - beta1**iteration)
                second_hat = second / (jnp.float32(1) - beta2**iteration)
                direction = jnp.where(
                    first_hat == 0,
                    jnp.float32(0),
                    first_hat / (jnp.sqrt(second_hat) + epsilon),
                )
                decay_term = (
                    weight_decay
                    * full_state["params"][name].astype(jnp.float32)
                    if decay[name]
                    else jnp.float32(0)
                )
                updated = full_state["params"][name].astype(
                    jnp.float32
                ) - learning_rate * (direction + decay_term)
                m[name] = first.astype(jnp.float32)
                v[name] = second.astype(jnp.float32)
                params[name] = updated.astype(jnp.float32)
                delta[name] = updated - full_state["params"][name].astype(
                    jnp.float32
                )
            update_norm = _stable_norm(delta)
            candidate: state.State = {
                "params": params,
                "m": m,
                "v": v,
                "step": new_step,
            }
            candidate_ok = (
                jnp.isfinite(gradient_norm)
                & jnp.isfinite(update_norm)
                & _moments_valid(candidate)
            )

            def committed(_: None) -> CommitResult:
                """Publish the candidate state as one committed logical step."""
                return CommitResult(
                    candidate,
                    jnp.asarray(True),
                    jnp.asarray(CommitCode.COMMITTED, dtype=jnp.int32),
                    loss,
                    gradient_norm,
                    clipped_gradient_norm,
                    update_norm,
                )

            def candidate_rejected(_: None) -> CommitResult:
                """Keep the old state if any candidate tensor overflowed."""
                return CommitResult(
                    full_state,
                    jnp.asarray(False),
                    jnp.asarray(CommitCode.CANDIDATE_INVALID, dtype=jnp.int32),
                    loss,
                    gradient_norm,
                    clipped_gradient_norm,
                    update_norm,
                )

            return cast(
                CommitResult,
                jax.lax.cond(candidate_ok, committed, candidate_rejected, None),
            )

        return cast(
            CommitResult,
            jax.lax.cond(
                code == jnp.asarray(CommitCode.COMMITTED, dtype=jnp.int32),
                attempt,
                rejected,
                None,
            ),
        )

    return transition


@functools.cache
def make_donated_transaction(
    inventory: core_parameters.FullParameterInventory,
    config: AdamWConfig,
) -> Callable[
    [state.State, types.Parameters, jax.Array, jax.Array], CommitResult
]:
    """Build a state-donating transaction with identical math.

    Both accepted and rejected paths return CommitResult.state, so a caller
    never retains a donated input buffer as its recovery state.
    """
    return cast(
        Callable[
            [state.State, types.Parameters, jax.Array, jax.Array], CommitResult
        ],
        jax.jit(make_transaction(inventory, config), donate_argnums=(0,)),
    )
