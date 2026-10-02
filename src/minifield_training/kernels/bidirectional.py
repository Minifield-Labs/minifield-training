"""Pad-safe bidirectional attention, centered short convolution, and norms."""

import jax
import jax.numpy as jnp


def attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: jax.Array,
    segment_ids: jax.Array | None = None,
) -> jax.Array:
    """Attend in both directions with GQA and zero fully masked rows.

    Inputs use BTHD layout; mask is BK. XLA accumulates attention logits in
    FP32. A finite sentinel and explicit zero handle empty padded examples.
    With packed ``segment_ids`` (BT, 0 for padding), a query sees only active
    keys in its own nonzero segment, and inactive queries return zeros.
    """
    if segment_ids is None:
        valid = mask[:, None, None, :].astype(jnp.bool_)
        present = jnp.any(mask, axis=-1)[:, None, None, None]
    else:
        active = mask.astype(jnp.bool_) & (segment_ids != 0)
        valid = (
            (segment_ids[:, :, None] == segment_ids[:, None, :])
            & active[:, :, None]
            & active[:, None, :]
        )[:, None]
        present = active[:, :, None, None]
    result = jax.nn.dot_product_attention(
        query, key, value, mask=valid, is_causal=False, implementation="xla"
    )
    return jnp.where(present, result, 0)


def centered_convolution(
    gated: jax.Array, taps: jax.Array, segment_ids: jax.Array | None = None
) -> jax.Array:
    """Match PyTorch grouped conv1d with padding=k//2 and right cropping.

    With packed ``segment_ids`` (BT, 0 for padding), a neighbouring tap
    contributes only inside the current token's nonzero segment, exactly as
    zero padding would for an independently encoded sequence.
    """
    width = taps.shape[-1]
    padded = jnp.pad(gated, ((0, 0), (width // 2, width // 2), (0, 0)))
    if segment_ids is not None:
        neighbours = jnp.pad(segment_ids, ((0, 0), (width // 2, width // 2)))
    result = jnp.zeros_like(gated)
    for index in range(width):
        tap = padded[:, index : index + gated.shape[1]] * taps[:, index]
        if segment_ids is not None:
            same = neighbours[:, index : index + gated.shape[1]] == segment_ids
            tap = tap * (same & (segment_ids != 0))[..., None].astype(tap.dtype)
        result += tap
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
