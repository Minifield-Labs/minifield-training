"""SwiGLU feed-forward composition with pre-norm and residual."""

from typing import NamedTuple

import jax

from minifield_training.kernels import linear
from minifield_training.kernels import normalization


class FeedForwardWeights(NamedTuple):
    """Pre-norm gain plus the three SwiGLU projection matrices.

    All arrays are FP32 masters; ``full_linear`` casts each matrix to the
    activation dtype for its projection.
    """

    norm: jax.Array
    gate: jax.Array
    up: jax.Array
    down: jax.Array


def swiglu_ffn(
    x: jax.Array, weights: FeedForwardWeights, eps: float
) -> jax.Array:
    """Return ``x + down(silu(gate(norm(x))) * up(norm(x)))``.

    RMS-normalizes ``x`` over the final axis, applies the gated up and down
    projections, and adds the result to the unnormalized input.
    """
    hidden = normalization.rms_norm(x, weights.norm, eps)
    gate = jax.nn.silu(linear.full_linear(hidden, weights.gate))
    up = linear.full_linear(hidden, weights.up)
    return x + linear.full_linear(gate * up, weights.down)
