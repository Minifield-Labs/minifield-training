"""Shared normalization kernels with explicit cast-order contracts."""

import jax
import jax.numpy as jnp


def rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """Apply FP32-accumulated RMS normalization over the final axis.

    The input is cast to FP32, scaled by the inverse root mean square of its
    final axis plus ``eps``, cast back to ``x.dtype`` and multiplied by
    ``weight`` cast to ``x.dtype`` with ordinary broadcasting. The result
    has the broadcast shape and ``x.dtype``; gradients keep the dtype of
    their respective argument. There is no shape, epsilon, dtype or
    nonfinite validation: NaN inputs propagate and incompatible broadcast
    shapes raise the usual JAX error.
    """
    value = x.astype(jnp.float32)
    value *= jax.lax.rsqrt(
        jnp.mean(value * value, axis=-1, keepdims=True) + eps
    )
    return value.astype(x.dtype) * weight.astype(x.dtype)
