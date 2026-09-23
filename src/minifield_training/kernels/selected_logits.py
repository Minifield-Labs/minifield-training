"""Memory-bounded selected-token LM-head scoring kernels."""

import jax
import jax.numpy as jnp

from minifield_training.kernels import linear


def selected_hidden_log_probs(
    selected: jax.Array,
    head: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    """Score pre-gathered hidden states through the trainable LM head."""
    logits = linear.full_linear(selected, head).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, target_ids[:, None], axis=-1)[:, 0]


def selected_token_log_probs(
    hidden: jax.Array,
    head: jax.Array,
    positions: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    """Score selected causal targets without a sequence-by-vocabulary grid."""
    selected = hidden[positions[:, 0], positions[:, 1]]
    return selected_hidden_log_probs(selected, head, target_ids)
