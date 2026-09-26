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
from minifield_training.strategies import quantization

HEAD_NAME = "classification_head.weight"


def _check_config(cfg: model.Config, allowed: tuple[bool, ...]) -> None:
    """Require a complete finite classification softmax and a tied Base head."""
    if not cfg.tied_embeddings:
        raise ValueError("Classification requires tied pretrained embeddings")
    if len(allowed) < 2 or not any(allowed):
        raise ValueError("Classification needs an allowed action class")


def parameter_inventory(
    cfg: model.Config,
    allowed: tuple[bool, ...],
    quantization_strategy: quantization.QuantizationPlan | None = None,
) -> core_parameters.FullParameterInventory:
    """Freeze input embeddings and train all remaining trunk and head leaves."""
    _check_config(cfg, allowed)
    shapes = model.expected_shapes(cfg)
    shapes[HEAD_NAME] = (len(allowed), cfg.hidden_size)
    frozen = frozenset({"model.embed_tokens.weight"})
    decay = frozenset(
        name for name, shape in shapes.items() if len(shape) == 2
    ).difference(frozen)
    dense_inventory = core_parameters.build_inventory(
        shapes,
        format_id="minifield.lfm.sequence-classifier/1",
        decayed_names=decay,
        frozen_names=frozen,
    )
    if quantization_strategy is None:
        return dense_inventory
    projection_set = projection_names(cfg)
    roles = {
        name: (
            "embedding"
            if name == "model.embed_tokens.weight"
            else "head"
            if name == HEAD_NAME
            else "projection"
            if name in projection_set
            else "other"
        )
        for name in shapes
    }
    selected = quantization.select(
        dense_inventory, quantization_strategy, roles
    )
    return core_parameters.build_inventory(
        shapes,
        format_id="minifield.lfm.sequence-classifier/1",
        decayed_names=decay,
        frozen_names=frozen,
        quantization_profile=quantization_strategy.identity,
        quantized_names=selected,
    )


def projection_names(cfg: model.Config) -> frozenset[str]:
    """Name only this model's projection matrices eligible for QAT."""
    return model.projection_names(cfg)


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
    rematerialize_blocks: bool = True,
    inventory: core_parameters.FullParameterInventory | None = None,
    quantization_strategy: quantization.QuantizationPlan | None = None,
) -> jax.Array:
    """Score one decision per row without a vocabulary logits tensor."""
    _check_config(cfg, allowed)
    if set(parameters) != set(model.expected_shapes(cfg)) | {HEAD_NAME}:
        raise ValueError("Classification parameter inventory mismatch")
    if attention_mask.shape != ids.shape or ids.ndim != 2:
        raise ValueError("Classification input shape mismatch")
    if quantization_strategy is not None and inventory is None:
        raise ValueError("Quantized logits require inventory")
    effective = (
        quantization.apply(parameters, inventory, quantization_strategy)
        if inventory is not None
        else parameters
    )
    backbone = {
        name: value for name, value in effective.items() if name != HEAD_NAME
    }
    hidden = model.hidden_states(
        backbone,
        ids,
        attention_mask,
        cfg,
        dtype=dtype,
        attention_backend=attention_backend,
        rematerialize_blocks=rematerialize_blocks,
    )
    return readout.last_valid_logits(
        hidden, attention_mask, effective[HEAD_NAME]
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
    inventory: core_parameters.FullParameterInventory | None = None,
    quantization_strategy: quantization.QuantizationPlan | None = None,
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
        inventory=inventory,
        quantization_strategy=quantization_strategy,
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
    rematerialize_blocks: bool = True,
    quantization_strategy: quantization.QuantizationPlan | None = None,
) -> step.LogicalStep:
    """Bind last-valid logits and decision-count loss to shared AdamW."""
    return step.make_step(
        _loss_terms(
            cfg,
            allowed,
            inventory,
            dtype,
            attention_backend,
            rematerialize_blocks,
            quantization_strategy,
        ),
        inventory,
        optimizer,
    )


def make_lfm2_5_streaming_step(
    cfg: model.Config,
    allowed: tuple[bool, ...],
    inventory: core_parameters.FullParameterInventory,
    optimizer: adamw.AdamWConfig,
    *,
    dtype: types.DType = jnp.bfloat16,
    attention_backend: str = "dense",
    rematerialize_blocks: bool = True,
    fuse_accumulation: bool = False,
    mesh: jax.sharding.Mesh | None = None,
    quantization_strategy: quantization.QuantizationPlan | None = None,
) -> step.StreamingStep:
    """Compile physical gradients, optionally splitting rows over a mesh."""
    return step.make_streaming_step(
        _loss_terms(
            cfg,
            allowed,
            inventory,
            dtype,
            attention_backend,
            rematerialize_blocks,
            quantization_strategy,
        ),
        inventory,
        optimizer,
        fuse_accumulation=fuse_accumulation,
        mesh=mesh,
    )


def _loss_terms(
    cfg: model.Config,
    allowed: tuple[bool, ...],
    inventory: core_parameters.FullParameterInventory,
    dtype: types.DType,
    attention_backend: str,
    rematerialize_blocks: bool,
    quantization_strategy: quantization.QuantizationPlan | None,
) -> step.LossTerms:
    """Bind the exact model inventory to summed hard-label terms."""
    _check_config(cfg, allowed)
    expected = parameter_inventory(cfg, allowed, quantization_strategy)
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
            rematerialize_blocks=rematerialize_blocks,
            inventory=inventory,
            quantization_strategy=quantization_strategy,
        )
        return objective.hard_label_terms(
            values,
            batch["labels"],
            batch["valid_rows"],
            allowed_array,
            safe_class=safe_class,
        )

    return loss_terms
