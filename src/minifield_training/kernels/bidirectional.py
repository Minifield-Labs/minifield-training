"""Pad-safe bidirectional attention, centered short convolution, and norms."""

import jax
import jax.numpy as jnp


def attention(
    query: jax.Array, key: jax.Array, value: jax.Array, mask: jax.Array
) -> jax.Array:
    """Attend in both directions with GQA and zero fully masked rows.

    Inputs use BTHD layout; mask is BK. XLA accumulates attention logits in
    FP32. A finite sentinel and explicit zero handle empty padded examples.
    """
    result = jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=mask[:, None, None, :].astype(jnp.bool_),
        is_causal=False,
        implementation="xla",
    )
    return jnp.where(jnp.any(mask, axis=-1)[:, None, None, None], result, 0)


def centered_convolution(gated: jax.Array, taps: jax.Array) -> jax.Array:
    """Match PyTorch grouped conv1d with padding=k//2 and right cropping."""
    width = taps.shape[-1]
    padded = jnp.pad(gated, ((0, 0), (width // 2, width // 2), (0, 0)))
    result = jnp.zeros_like(gated)
    for index in range(width):
        result += padded[:, index : index + gated.shape[1]] * taps[:, index]
    return result


def layer_norm(
    value: jax.Array, gain: jax.Array, bias: jax.Array, eps: float = 1e-5
) -> jax.Array:
    """Layer-normalize with FP32 moments and the input activation dtype."""
    full = value.astype(jnp.float32)
    centered = full - jnp.mean(full, axis=-1, keepdims=True)
    result = centered * jax.lax.rsqrt(
        jnp.mean(centered * centered, axis=-1, keepdims=True) + eps
    )
    return (result * gain + bias).astype(value.dtype)


def dropout(value: jax.Array, key: jax.Array, rate: float) -> jax.Array:
    """Apply inverted dropout; rate zero is an identity in inference."""
    if rate == 0:
        return value
    return jnp.where(
        jax.random.bernoulli(key, 1 - rate, value.shape), value / (1 - rate), 0
    )
