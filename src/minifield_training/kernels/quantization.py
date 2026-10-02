"""Group-128 fake quantizers with FP16 semantic scales and identity STE."""

from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np


class Quantizer(Protocol):
    """A numerical fake quantizer over FP32 matrix masters."""

    @property
    def identity(self) -> str:
        """Return the exact stable numerical algorithm identifier."""

    def effective(self, weight: jax.Array) -> jax.Array:
        """Return decoded FP32 values with identity backward gradient."""

    def validate_shape(self, shape: tuple[int, ...]) -> None:
        """Admit the quantizer's matrix layout before tracing."""


_NF4_LEVELS = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)
_NF4 = jnp.asarray(_NF4_LEVELS, dtype=jnp.float32)


def _nf4_transitions() -> tuple[float, ...]:
    """Find first FP32 values above exact codebook midpoints."""
    levels = tuple(
        float(value) for value in np.asarray(_NF4_LEVELS, dtype=np.float32)
    )
    boundaries = []
    for left, right in zip(levels[:-1], levels[1:], strict=False):
        midpoint = (left + right) * 0.5
        rounded = np.float32(midpoint)
        first_upper = (
            np.nextafter(rounded, np.float32(np.inf))
            if float(rounded) <= midpoint
            else rounded
        )
        boundaries.append(float(first_upper))
    return tuple(boundaries)


_NF4_TRANSITIONS = _nf4_transitions()
_barrier: Callable[[jax.Array], jax.Array] = jax.lax.optimization_barrier


def _groups(weight: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Group complete rows, converting absmax to the stored FP16 value."""
    if weight.ndim != 2 or 0 in weight.shape or weight.shape[1] % 128:
        raise ValueError(
            "Quantized matrix requires rank 2 and K divisible by 128"
        )
    if weight.dtype != jnp.float32:
        raise ValueError("Quantized master must be FP32")
    grouped = weight.reshape(weight.shape[0], weight.shape[1] // 128, 128)
    scales = jnp.max(jnp.abs(grouped), axis=-1, keepdims=True)
    stored_scales = _barrier(scales.astype(jnp.float16))
    return grouped, stored_scales.astype(jnp.float32)


def codes(weight: jax.Array, kind: str) -> tuple[jax.Array, jax.Array]:
    """Return U8 ``[N, K]`` codebook indices and F16 ``[N, K/128]`` scales.

    Ternary codes 0, 1, 2 mean -1, 0, +1; NF4 codes index the NF4 levels.
    These are the runtime's ``minifield.ternary.v1`` and ``minifield.nf4.v1``
    code meanings before bit packing.
    """
    grouped, scales = _groups(weight)
    if kind == "ternary-g128-absmax-f16-v1":
        threshold = jnp.float32(0.5) * scales
        indices = jnp.where(
            scales == 0,
            1,
            jnp.where(
                grouped >= threshold,
                2,
                jnp.where(grouped <= -threshold, 0, 1),
            ),
        )
    elif kind == "nf4-g128-absmax-f16-v1":
        normalized = jnp.where(scales == 0, 0.0, grouped / scales)
        # One fused compare-and-count keeps the compiled program small.
        indices = jnp.sum(
            normalized[..., None] >= jnp.asarray(_NF4_TRANSITIONS, jnp.float32),
            axis=-1,
            dtype=jnp.int32,
        )
    else:
        raise ValueError("Unknown quantizer")
    return (
        indices.reshape(weight.shape).astype(jnp.uint8),
        scales[..., 0].astype(jnp.float16),
    )


def decode(indices: jax.Array, scales: jax.Array, kind: str) -> jax.Array:
    """Turn codes and F16 scales back into FP32 weights."""
    if kind == "ternary-g128-absmax-f16-v1":
        levels = jnp.asarray((-1.0, 0.0, 1.0), jnp.float32)
    elif kind == "nf4-g128-absmax-f16-v1":
        levels = _NF4
    else:
        raise ValueError("Unknown quantizer")
    rows, columns = indices.shape
    values = levels[indices.astype(jnp.int32)].reshape(
        rows, columns // 128, 128
    )
    return (values * scales.astype(jnp.float32)[..., None]).reshape(
        rows, columns
    )


def _decoded(weight: jax.Array, kind: str) -> jax.Array:
    """Decode the reference ternary or NF4 codes with stored-scale semantics."""
    return decode(*codes(weight, kind), kind)


class Group128Quantizer:
    """Reference ternary or NF4 g128 with FP16 scales and STE backward."""

    def __init__(self, kind: str) -> None:
        """Admit only defined numerical identities."""
        if kind not in (
            "ternary-g128-absmax-f16-v1",
            "nf4-g128-absmax-f16-v1",
        ):
            raise ValueError("Unknown group-128 quantizer")
        self._kind = kind

    @property
    def identity(self) -> str:
        """Return the reference rule including scale and codebook semantics."""
        return self._kind

    def effective(self, weight: jax.Array) -> jax.Array:
        """Forward exact decode while leaving the master gradient unchanged."""
        decoded = _decoded(weight, self._kind)
        return jax.lax.stop_gradient(decoded) + (
            weight - jax.lax.stop_gradient(weight)
        )

    def validate_shape(self, shape: tuple[int, ...]) -> None:
        """Require complete 128-column groups in a nonempty matrix."""
        if len(shape) != 2 or 0 in shape or shape[1] % 128:
            raise ValueError("Group-128 quantizer needs a complete matrix")
