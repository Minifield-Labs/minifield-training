"""Hard-label sequence classification with an explicit allowed-class set."""

import jax
import jax.numpy as jnp


def masked_logits(logits: jax.Array, allowed: jax.Array) -> jax.Array:
    """Exclude disallowed classes from training and inference softmaxes."""
    if logits.ndim != 2 or allowed.shape != (logits.shape[1],):
        raise ValueError("Classification logits/allowed shape mismatch")
    return jnp.where(allowed[None, :], logits, -jnp.inf)


def hard_label_terms(
    logits: jax.Array,
    labels: jax.Array,
    valid_rows: jax.Array,
    allowed: jax.Array,
    *,
    safe_class: int,
) -> tuple[jax.Array, jax.Array]:
    """Sum decision NLL and count only valid rows.

    Invalid padded labels never enter a gather, avoiding ``inf * 0`` when a
    caller uses a masked class ID as its padding label. Valid invalid labels
    produce NaN and the shared optimizer rejects that logical update.
    """
    if labels.shape != (logits.shape[0],) or valid_rows.shape != labels.shape:
        raise ValueError("Classification row shape mismatch")
    if not 0 <= safe_class < logits.shape[1]:
        raise ValueError("Safe class is out of range")
    masked = masked_logits(logits, allowed)
    safe_labels = jnp.where(valid_rows, labels, safe_class)
    in_range = (safe_labels >= 0) & (safe_labels < logits.shape[1])
    indexes = jnp.clip(safe_labels, 0, logits.shape[1] - 1)
    log_prob = jax.nn.log_softmax(masked, axis=-1)
    selected = jnp.take_along_axis(log_prob, indexes[:, None], axis=1)[:, 0]
    selected_allowed = allowed[indexes]
    losses = jnp.where(valid_rows, -selected, jnp.float32(0))
    losses = jnp.where(
        jnp.all(~valid_rows | (in_range & selected_allowed)),
        losses,
        jnp.full_like(losses, jnp.nan),
    )
    return jnp.sum(losses, dtype=jnp.float32), jnp.sum(
        valid_rows.astype(jnp.float32), dtype=jnp.float32
    )
