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


def distillation(
    teacher: types.DeviceBatch,
    student: types.DeviceBatch,
    batch: types.DeviceBatch,
    *,
    temperature: float,
) -> jax.Array:
    """Return ``[requests, questions]`` KL(teacher || student) times T².

    Both pointers soften over the question's allowed tokens at the same
    ``temperature``; the mean of start and end is returned. Gradients reach
    only the student.
    """
    allowed = batch["allowed"].astype(bool)
    total = jnp.zeros(allowed.shape[:-1], jnp.float32)

    def log_probs(logits: jax.Array) -> jax.Array:
        scaled = logits.astype(jnp.float32) / temperature
        return jax.nn.log_softmax(jnp.where(allowed, scaled, _MASKED), -1)

    for end in ("start", "end"):
        target = jax.lax.stop_gradient(log_probs(teacher[end]))
        divergence = jnp.exp(target) * (target - log_probs(student[end]))
        total = total + jnp.sum(jnp.where(allowed, divergence, 0), axis=-1)
    return total * (temperature**2) / 2


def distilled_terms(
    dense: types.DeviceBatch,
    quantized: types.DeviceBatch,
    batch: types.DeviceBatch,
    *,
    quantized_weight: float = 1.0,
    distill_weight: float = 1.0,
    temperature: float = 2.0,
) -> tuple[jax.Array, jax.Array]:
    """Train shared weights as a dense parent and a quantized student at once.

    Each question's loss is the dense cross-entropy, plus the quantized
    cross-entropy and the quantized pass's distillation from the dense one,
    weighted as given. Type balancing and the weight mass match ``terms``.
    """
    weights = batch["field_weight"].astype(jnp.float32)
    per_question = (
        losses(dense, batch)
        + quantized_weight * losses(quantized, batch)
        + distill_weight
        * distillation(dense, quantized, batch, temperature=temperature)
    )
    return jnp.sum(per_question * weights), jnp.sum(weights)
