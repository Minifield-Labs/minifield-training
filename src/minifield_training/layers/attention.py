"""Causal attention blocks with QK-norm, RoPE, and residual.

Every variant shares this block layout; only the attention call and any
boundary state consumed or returned differ::

        x
        |
   +----+----+
   |         |
   v         |
rms_norm     |
   |         |
  qkv        |
   |         |
qk_norm      |
   |         |
  rope       |
   |         |
  attn       |
   |         |
out_proj     |
   |         |
   v         |
 (+) <-------+
   |
   v
swiglu_ffn   (pre-norm + gated projection + residual)
   |
   v
  out
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from minifield_training.kernels import attention
from minifield_training.kernels import linear
from minifield_training.kernels import normalization
from minifield_training.kernels import rotary
from minifield_training.layers import feed_forward


class AttentionWeights(NamedTuple):
    """Operator norm, q/k/v/out projections, and per-head q/k norm gains.

    ``key`` and ``value`` may project to fewer heads than ``query``; each
    head count follows from the matrix's leading dimension and ``head_dim``.
    """

    operator_norm: jax.Array
    query: jax.Array
    key: jax.Array
    value: jax.Array
    out: jax.Array
    query_norm: jax.Array
    key_norm: jax.Array


def project_qkv(
    hidden: jax.Array,
    weights: AttentionWeights,
    *,
    head_dim: int,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Project to ``[B, T, H, D]`` and RMS-norm query and key heads."""
    shape = (*hidden.shape[:2], -1, head_dim)
    query = linear.full_linear(hidden, weights.query).reshape(shape)
    key = linear.full_linear(hidden, weights.key).reshape(shape)
    value = linear.full_linear(hidden, weights.value).reshape(shape)
    return (
        normalization.rms_norm(query, weights.query_norm, eps),
        normalization.rms_norm(key, weights.key_norm, eps),
        value,
    )


def _rope_qkv(
    x: jax.Array,
    weights: AttentionWeights,
    positions: jax.Array | None,
    *,
    head_dim: int,
    rope_theta: float,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    hidden = normalization.rms_norm(x, weights.operator_norm, eps)
    query, key, value = project_qkv(hidden, weights, head_dim=head_dim, eps=eps)
    query = rotary.apply_rotary(
        query, rope_theta=rope_theta, head_dim=head_dim, positions=positions
    )
    key = rotary.apply_rotary(
        key, rope_theta=rope_theta, head_dim=head_dim, positions=positions
    )
    return query, key, value


def _attention_residual(
    x: jax.Array,
    attended: jax.Array,
    weights: AttentionWeights,
    ffn: feed_forward.FeedForwardWeights,
    eps: float,
) -> jax.Array:
    hidden = linear.full_linear(attended.reshape(x.shape), weights.out)
    return feed_forward.swiglu_ffn(x + hidden, ffn, eps)


def attention_block(
    x: jax.Array,
    weights: AttentionWeights,
    ffn: feed_forward.FeedForwardWeights,
    attention_mask: jax.Array,
    *,
    head_dim: int,
    rope_theta: float,
    eps: float,
    backend: str,
    positions: jax.Array | None = None,
) -> jax.Array:
    """Run a full-sequence causal attention block.

    ``positions`` overrides the default ``0..T-1`` RoPE positions.
    """
    query, key, value = _rope_qkv(
        x,
        weights,
        positions,
        head_dim=head_dim,
        rope_theta=rope_theta,
        eps=eps,
    )
    attended = attention.causal_attention(
        query, key, value, attention_mask, backend=backend
    )
    return _attention_residual(x, attended, weights, ffn, eps)


def attention_block_prefix(
    x: jax.Array,
    weights: AttentionWeights,
    ffn: feed_forward.FeedForwardWeights,
    attention_mask: jax.Array,
    *,
    head_dim: int,
    rope_theta: float,
    eps: float,
    backend: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the block and return mask-zeroed RoPE'd keys and values.

    The returned keys and values are the layer's shared-context boundary
    state for ``attention_block_suffix``: RoPE is already applied and
    padded columns are zeroed.
    """
    query, key, value = _rope_qkv(
        x, weights, None, head_dim=head_dim, rope_theta=rope_theta, eps=eps
    )
    attended = attention.causal_attention(
        query, key, value, attention_mask, backend=backend
    )
    keep = attention_mask[:, :, None, None].astype(key.dtype)
    return (
        _attention_residual(x, attended, weights, ffn, eps),
        key * keep,
        value * keep,
    )


def attention_block_suffix(
    x: jax.Array,
    weights: AttentionWeights,
    ffn: feed_forward.FeedForwardWeights,
    attention_mask: jax.Array,
    positions: jax.Array,
    prefix_key: jax.Array,
    prefix_value: jax.Array,
    prefix_mask: jax.Array,
    *,
    head_dim: int,
    rope_theta: float,
    eps: float,
    backend: str,
) -> jax.Array:
    """Run a continuation block attending to shared prefix keys/values.

    ``positions`` carries absolute RoPE positions continuing the prefix.
    ``prefix_key``/``prefix_value`` come from ``attention_block_prefix`` or
    an equivalent producer; ``prefix_mask`` marks their valid columns.
    """
    query, key, value = _rope_qkv(
        x,
        weights,
        positions,
        head_dim=head_dim,
        rope_theta=rope_theta,
        eps=eps,
    )
    attended = attention.prefix_causal_attention(
        query,
        prefix_key,
        prefix_value,
        key,
        value,
        prefix_mask,
        attention_mask,
        backend=backend,
    )
    return _attention_residual(x, attended, weights, ffn, eps)


def attention_block_packed(
    x: jax.Array,
    weights: AttentionWeights,
    ffn: feed_forward.FeedForwardWeights,
    attention_mask: jax.Array,
    segment_ids: jax.Array,
    positions: jax.Array,
    *,
    head_dim: int,
    rope_theta: float,
    eps: float,
    backend: str,
) -> jax.Array:
    """Run a packed-batch block confined to same-segment tokens.

    ``positions`` carries per-token RoPE positions that restart at segment
    boundaries; ``segment_ids`` uses 0 for padding.
    """
    query, key, value = _rope_qkv(
        x,
        weights,
        positions,
        head_dim=head_dim,
        rope_theta=rope_theta,
        eps=eps,
    )
    attended = attention.packed_causal_attention(
        query, key, value, attention_mask, segment_ids, backend=backend
    )
    return _attention_residual(x, attended, weights, ffn, eps)


def attention_block_prefill(
    x: jax.Array,
    weights: AttentionWeights,
    ffn: feed_forward.FeedForwardWeights,
    attention_mask: jax.Array,
    *,
    capacity: int,
    head_dim: int,
    rope_theta: float,
    eps: float,
    backend: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the block and return keys/values padded to a fixed KV cache.

    The returned cache holds mask-zeroed RoPE'd keys and values right-padded
    to ``capacity`` columns, in the layout ``attention_block_step`` updates.
    """
    query, key, value = _rope_qkv(
        x, weights, None, head_dim=head_dim, rope_theta=rope_theta, eps=eps
    )
    attended = attention.causal_attention(
        query, key, value, attention_mask, backend=backend
    )
    padding = capacity - key.shape[1]
    if padding < 0:
        raise ValueError("Prompt length exceeds attention cache capacity")
    keep = attention_mask[:, :, None, None].astype(key.dtype)
    pad = ((0, 0), (0, padding), (0, 0), (0, 0))
    return (
        _attention_residual(x, attended, weights, ffn, eps),
        jnp.pad(key * keep, pad),
        jnp.pad(value * keep, pad),
    )


def attention_block_step(
    x: jax.Array,
    weights: AttentionWeights,
    ffn: feed_forward.FeedForwardWeights,
    cache_key: jax.Array,
    cache_value: jax.Array,
    position: jax.Array,
    valid_length: jax.Array,
    *,
    head_dim: int,
    rope_theta: float,
    eps: float,
    backend: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run one decode step and return output plus the updated KV cache.

    ``x`` holds a single token per row. ``position`` is the absolute cache
    slot written this step and doubles as its RoPE position;
    ``valid_length`` is the number of readable columns after the write.
    """
    query, key, value = _rope_qkv(
        x,
        weights,
        position,
        head_dim=head_dim,
        rope_theta=rope_theta,
        eps=eps,
    )
    cache_key = jax.lax.dynamic_update_slice(
        cache_key, key, (0, position, 0, 0)
    )
    cache_value = jax.lax.dynamic_update_slice(
        cache_value, value, (0, position, 0, 0)
    )
    attended = attention.cached_attention(
        query, cache_key, cache_value, valid_length, backend=backend
    )
    return (
        _attention_residual(x, attended, weights, ffn, eps),
        cache_key,
        cache_value,
    )
