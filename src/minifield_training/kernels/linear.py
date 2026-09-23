"""Dense projection kernels for FP32-master parameters."""

import jax
import jax.numpy as jnp


def full_linear(x: jax.Array, weight: jax.Array) -> jax.Array:
    """Apply one trainable dense tensor stored as an FP32 master."""
    return jnp.matmul(
        x,
        weight.astype(x.dtype).T,
        precision=jax.lax.Precision.HIGHEST,
    )
