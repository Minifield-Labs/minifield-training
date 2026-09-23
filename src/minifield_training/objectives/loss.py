"""Next-token causal loss terms with explicit mask and count semantics."""

import jax
import jax.numpy as jnp


def causal_loss_terms(
    logits: jax.Array,
    input_ids: jax.Array,
    loss_mask: jax.Array,
    attention_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return unaveraged next-token loss and its exact target count.

    Position ``t`` predicts token ``t + 1``: logits, labels, and masks shift
    by one here rather than at the caller. A token contributes only when both
    the caller's ``loss_mask`` and the batch ``attention_mask`` mark it.
    """
    shifted_logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = (loss_mask[:, 1:] * attention_mask[:, 1:]).astype(jnp.float32)
    losses = -jnp.take_along_axis(
        jax.nn.log_softmax(shifted_logits, axis=-1), labels[..., None], axis=-1
    )[..., 0]
    return jnp.sum(losses * mask, dtype=jnp.float32), jnp.sum(
        mask, dtype=jnp.float32
    )


def causal_loss(
    logits: jax.Array,
    input_ids: jax.Array,
    loss_mask: jax.Array,
    attention_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Average next-token loss while preserving zero-count safety."""
    total, count = causal_loss_terms(
        logits, input_ids, loss_mask, attention_mask
    )
    denominator = jnp.where(count > 0, count, jnp.float32(1))
    return total / denominator, count
