"""Small classification projection over one valid sequence state."""

import jax
import jax.numpy as jnp

from minifield_training.kernels import linear


def last_valid_logits(
    hidden: jax.Array, attention_mask: jax.Array, weight: jax.Array
) -> jax.Array:
    """Project only each row's final valid hidden state to class logits."""
    lengths = jnp.sum(attention_mask, axis=1)
    positions = jnp.maximum(lengths - 1, 0)
    selected = hidden[jnp.arange(hidden.shape[0]), positions]
    return linear.full_linear(selected, weight).astype(jnp.float32)
