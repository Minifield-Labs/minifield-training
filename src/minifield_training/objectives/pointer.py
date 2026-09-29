"""One masked soft-target cross-entropy for every pointer question type."""

import jax
import jax.numpy as jnp

from minifield_training.kernels import types

# Finite, so fully masked padding questions keep finite log-probabilities.
_MASKED = -1e30


def losses(outputs: types.DeviceBatch, batch: types.DeviceBatch) -> jax.Array:
    """Return ``[requests, questions]`` mean start/end cross-entropy.

    Each pointer takes a softmax over the question's allowed tokens only.
    Targets may be soft; padding questions have zero targets and zero loss.
    """
    allowed = batch["allowed"].astype(bool)
    total = jnp.zeros(allowed.shape[:-1], jnp.float32)
    for end in ("start", "end"):
        log_probs = jax.nn.log_softmax(
            jnp.where(allowed, outputs[end].astype(jnp.float32), _MASKED),
            axis=-1,
        )
        target = batch[end + "_target"].astype(jnp.float32)
        total = total - jnp.sum(
            jnp.where(allowed, target * log_probs, 0), axis=-1
        )
    return total / 2


def terms(
    outputs: types.DeviceBatch, batch: types.DeviceBatch
) -> tuple[jax.Array, jax.Array]:
    """Sum preweighted question losses and weight mass for the step engine.

    Weights come from the batch compiler's type balancing over the complete
    logical update; unsupervised and padding questions carry zero weight.
    """
    weights = batch["field_weight"].astype(jnp.float32)
    return jnp.sum(losses(outputs, batch) * weights), jnp.sum(weights)
