"""Partial-label losses with logical-update type-balanced field weights."""

import jax
import jax.numpy as jnp

from minifield_training.kernels import types


def binary_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Stable Bernoulli cross entropy, including soft targets."""
    return jax.nn.softplus(logits) - targets * logits


def losses(outputs: types.DeviceBatch, batch: types.DeviceBatch) -> jax.Array:
    """Return one loss per row, repeating group CE for candidate rows."""
    logits = outputs["candidate"].astype(jnp.float32)
    owners = batch["field_owner"]
    same = (owners[:, :, None] == owners[:, None, :]) & batch["row_mask"][
        :, None, :
    ]
    scores = jnp.where(same, logits[:, None, :], -1e30)
    log_prob = jax.nn.log_softmax(scores, axis=-1)
    categorical = -jnp.sum(
        jnp.where(same, batch["target"][:, None, :] * log_prob, 0), axis=-1
    )
    binary = binary_loss(outputs["binary"], batch["target"])
    presence = binary_loss(outputs["presence"], batch["target"])
    selectable = (
        batch["selectable"][:, None, :] * batch["token_supervised"][:, :, None]
    )
    token = jnp.sum(
        binary_loss(outputs["tokens"], batch["token_target"]) * selectable,
        axis=-1,
    ) / jnp.maximum(jnp.sum(selectable, axis=-1), 1)
    # 0 extract, 1 choice, 2 noul, 3 score. Padding has zero weight.
    return jnp.where(
        batch["kind"] == 0,
        presence + token,
        jnp.where(batch["kind"] == 2, binary, categorical),
    )


def terms(
    outputs: types.DeviceBatch, batch: types.DeviceBatch
) -> tuple[jax.Array, jax.Array]:
    """Sum preweighted field losses and mass, for the shared step engine.

    Weights are formed over the entire logical update before device slicing:
    one mean per active type, then a mean across active types. Candidate rows
    after the first carry zero weight. Missing labels also carry zero weight.
    """
    weights = batch["field_weight"].astype(jnp.float32)
    return jnp.sum(losses(outputs, batch) * weights), jnp.sum(weights)
