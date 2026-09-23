"""Gated depthwise-convolution blocks with pre-norm and residual."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from minifield_training.kernels import convolution
from minifield_training.kernels import linear
from minifield_training.kernels import normalization
from minifield_training.layers import feed_forward


class ConvWeights(NamedTuple):
    """Operator norm gain, fused in-projection, taps, and out-projection.

    ``in_proj`` produces three times the channel count and splits into the
    b gate, c gate, and values. ``conv`` holds ``[channels, taps]`` kernel
    weights; squeezing checkpoint layouts such as ``[channels, 1, taps]``
    is the caller's job.
    """

    operator_norm: jax.Array
    in_proj: jax.Array
    conv: jax.Array
    out: jax.Array


def conv_input_projection(
    hidden: jax.Array, in_proj: jax.Array, attention_mask: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Zero masked columns, project, and split into b/c gates and values."""
    hidden = hidden * attention_mask[..., None].astype(hidden.dtype)
    b_gate, c_gate, values = jnp.split(
        linear.full_linear(hidden, in_proj), 3, axis=-1
    )
    return b_gate, c_gate, values


def _conv_residual(
    x: jax.Array,
    mixed: jax.Array,
    weights: ConvWeights,
    ffn: feed_forward.FeedForwardWeights,
    eps: float,
) -> jax.Array:
    hidden = linear.full_linear(mixed, weights.out)
    return feed_forward.swiglu_ffn(x + hidden, ffn, eps)


def conv_block(
    x: jax.Array,
    weights: ConvWeights,
    ffn: feed_forward.FeedForwardWeights,
    attention_mask: jax.Array,
    *,
    kernel_size: int,
    eps: float,
) -> jax.Array:
    """Run a full-sequence gated convolution block.

    Applies the operator norm, masks and splits the in-projection, runs the
    causal depthwise mix, projects back, and finishes with the residual
    SwiGLU tail.
    """
    hidden = normalization.rms_norm(x, weights.operator_norm, eps)
    b_gate, c_gate, values = conv_input_projection(
        hidden, weights.in_proj, attention_mask
    )
    taps = weights.conv.astype(hidden.dtype)
    mixed = convolution.gated_depthwise_convolution(
        b_gate, c_gate, values, taps, kernel_size=kernel_size
    )
    return _conv_residual(x, mixed, weights, ffn, eps)


def conv_block_prefix(
    x: jax.Array,
    weights: ConvWeights,
    ffn: feed_forward.FeedForwardWeights,
    attention_mask: jax.Array,
    *,
    kernel_size: int,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    """Run the block and return its continuation history.

    The history holds the ``kernel_size - 1`` gated ``b_gate * values``
    columns ending at each row's valid length, zero-filled for short rows.
    """
    hidden = normalization.rms_norm(x, weights.operator_norm, eps)
    b_gate, c_gate, values = conv_input_projection(
        hidden, weights.in_proj, attention_mask
    )
    taps = weights.conv.astype(hidden.dtype)
    mixed = convolution.gated_depthwise_convolution(
        b_gate, c_gate, values, taps, kernel_size=kernel_size
    )
    lengths = jnp.sum(attention_mask, axis=1, dtype=jnp.int32)
    history = convolution.state_tail(b_gate * values, lengths, kernel_size - 1)
    return _conv_residual(x, mixed, weights, ffn, eps), history


def conv_block_suffix(
    x: jax.Array,
    weights: ConvWeights,
    ffn: feed_forward.FeedForwardWeights,
    attention_mask: jax.Array,
    history: jax.Array,
    *,
    kernel_size: int,
    eps: float,
) -> jax.Array:
    """Run a continuation block against a prefix's conv history.

    ``history`` carries the ``kernel_size - 1`` gated ``b_gate * values``
    columns immediately preceding ``x`` so outputs match one uninterrupted
    causal convolution.
    """
    hidden = normalization.rms_norm(x, weights.operator_norm, eps)
    b_gate, c_gate, values = conv_input_projection(
        hidden, weights.in_proj, attention_mask
    )
    taps = weights.conv.astype(hidden.dtype)
    mixed = convolution.gated_depthwise_convolution_with_history(
        b_gate, c_gate, values, taps, history, kernel_size=kernel_size
    )
    return _conv_residual(x, mixed, weights, ffn, eps)


def conv_block_packed(
    x: jax.Array,
    weights: ConvWeights,
    ffn: feed_forward.FeedForwardWeights,
    attention_mask: jax.Array,
    segment_ids: jax.Array,
    *,
    kernel_size: int,
    eps: float,
) -> jax.Array:
    """Run a packed-batch block with segment-isolated taps."""
    hidden = normalization.rms_norm(x, weights.operator_norm, eps)
    b_gate, c_gate, values = conv_input_projection(
        hidden, weights.in_proj, attention_mask
    )
    taps = weights.conv.astype(hidden.dtype)
    mixed = convolution.gated_depthwise_convolution_segmented(
        b_gate, c_gate, values, taps, segment_ids, kernel_size=kernel_size
    )
    return _conv_residual(x, mixed, weights, ffn, eps)


def conv_block_step(
    x: jax.Array,
    weights: ConvWeights,
    ffn: feed_forward.FeedForwardWeights,
    history: jax.Array,
    *,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    """Run one decode step and return the output plus shifted history.

    ``x`` holds a single token per row and is always valid, so the
    in-projection's binary input mask is the identity and skipped here.
    ``history`` holds the ``kernel_size - 1`` gated ``b_gate * values``
    columns preceding the token; the returned history drops the oldest
    column and appends this step's gated input.
    """
    hidden = normalization.rms_norm(x, weights.operator_norm, eps)
    b_gate, c_gate, values = jnp.split(
        linear.full_linear(hidden, weights.in_proj), 3, axis=-1
    )
    taps = weights.conv.astype(hidden.dtype)
    mixed = convolution.gated_depthwise_convolution_with_history(
        b_gate, c_gate, values, taps, history, kernel_size=taps.shape[1]
    )
    new_history = jnp.concatenate((history, b_gate * values), axis=1)[:, 1:, :]
    return _conv_residual(x, mixed, weights, ffn, eps), new_history
