"""LFM2.5 sequence-classifier composition over the shared update engine."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from minifield_training.core import parameters as core_parameters
from minifield_training.engine import step
from minifield_training.kernels import types
from minifield_training.layers import classification as readout
from minifield_training.models.lfm2_5 import model
from minifield_training.objectives import classification as objective
from minifield_training.optimizers import adamw

HEAD_NAME = "classification_head.weight"


def _check_config(cfg: model.Config, allowed: tuple[bool, ...]) -> None:
    """Require a complete finite classification softmax and a tied Base head."""
    if not cfg.tied_embeddings:
        raise ValueError("Classification requires tied pretrained embeddings")
    if len(allowed) < 2 or not any(allowed):
        raise ValueError("Classification needs an allowed action class")


def parameter_inventory(
    cfg: model.Config, allowed: tuple[bool, ...]
) -> core_parameters.FullParameterInventory:
    """Freeze input embeddings and train all remaining trunk and head leaves."""
    _check_config(cfg, allowed)
    shapes = model.expected_shapes(cfg)
    shapes[HEAD_NAME] = (len(allowed), cfg.hidden_size)
    frozen = frozenset({"model.embed_tokens.weight"})
    decay = frozenset(
        name for name, shape in shapes.items() if len(shape) == 2
    ).difference(frozen)
    return core_parameters.build_inventory(
        shapes,
        format_id="minifield.lfm.sequence-classifier/1",
        decayed_names=decay,
        frozen_names=frozen,
    )


def initialize_from_backbone(
    backbone: Mapping[str, jax.Array],
    cfg: model.Config,
    allowed: tuple[bool, ...],
    *,
    head_seed: int,
) -> types.Parameters:
    """Keep every verified pretrained tensor and initialize only the head."""
    _check_config(cfg, allowed)
    if head_seed < 0:
        raise ValueError("Head seed must be nonnegative")
    model.validate_masters(backbone, cfg)
    head = jax.random.normal(
        jax.random.PRNGKey(head_seed),
        (len(allowed), cfg.hidden_size),
        dtype=jnp.float32,
    ) * jnp.float32(0.02)
    return {**backbone, HEAD_NAME: head}


def logits(
    parameters: types.Parameters,
    ids: jax.Array,
    attention_mask: jax.Array,
    cfg: model.Config,
    allowed: tuple[bool, ...],
    *,
    dtype: types.DType = jnp.bfloat16,
    attention_backend: str = "dense",
) -> jax.Array:
    """Score one decision per row without a vocabulary logits tensor."""
    _check_config(cfg, allowed)
    if set(parameters) != set(model.expected_shapes(cfg)) | {HEAD_NAME}:
        raise ValueError("Classification parameter inventory mismatch")
    if attention_mask.shape != ids.shape or ids.ndim != 2:
        raise ValueError("Classification input shape mismatch")
    backbone = {
        name: value for name, value in parameters.items() if name != HEAD_NAME
    }
    hidden = model.hidden_states(
        backbone,
        ids,
        attention_mask,
        cfg,
        dtype=dtype,
        attention_backend=attention_backend,
    )
    return readout.last_valid_logits(
        hidden, attention_mask, parameters[HEAD_NAME]
    )


def predict(
    parameters: types.Parameters,
    ids: jax.Array,
    attention_mask: jax.Array,
    cfg: model.Config,
    allowed: tuple[bool, ...],
    *,
    dtype: types.DType = jnp.bfloat16,
    attention_backend: str = "dense",
) -> jax.Array:
    """Choose only an allowed class for each valid input sequence."""
    values = logits(
        parameters,
        ids,
        attention_mask,
        cfg,
        allowed,
        dtype=dtype,
        attention_backend=attention_backend,
    )
    return jnp.argmax(
        objective.masked_logits(values, jnp.asarray(allowed)), axis=-1
    )


def make_lfm2_5_step(
    cfg: model.Config,
    allowed: tuple[bool, ...],
    inventory: core_parameters.FullParameterInventory,
    optimizer: adamw.AdamWConfig,
    *,
    dtype: types.DType = jnp.bfloat16,
    attention_backend: str = "dense",
) -> step.LogicalStep:
    """Bind last-valid logits and decision-count loss to shared AdamW."""
    _check_config(cfg, allowed)
    expected = parameter_inventory(cfg, allowed)
    if inventory.sha256 != expected.sha256:
        raise ValueError("Classification inventory identity mismatch")
    allowed_array = jnp.asarray(allowed, dtype=jnp.bool_)
    safe_class = allowed.index(True)

    def loss_terms(
        parameters: types.Parameters, batch: types.DeviceBatch
    ) -> tuple[jax.Array, jax.Array]:
        """Return hard-label summed NLL and valid decision count."""
        values = logits(
            parameters,
            batch["input_ids"],
            batch["attention_mask"],
            cfg,
            allowed,
            dtype=dtype,
            attention_backend=attention_backend,
        )
        return objective.hard_label_terms(
            values,
            batch["labels"],
            batch["valid_rows"],
            allowed_array,
            safe_class=safe_class,
        )

    return step.make_step(loss_terms, inventory, optimizer)
