"""Bind the selected LFM encoder to the MagicBox model and parameter names."""

from collections.abc import Callable
import functools

import jax

from minifield_training.core import parameters as core_parameters
from minifield_training.kernels import quantization as quant_kernels
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import model
from minifield_training.models.magicbox import pointer
from minifield_training.strategies import quantization

QUANTIZERS = {
    "nf4": "nf4-g128-absmax-f16-v1",
    "ternary": "ternary-g128-absmax-f16-v1",
}


def inventory(
    cfg: lfm.Config, fusion: model.Config, *, freeze_embeddings: bool = True
) -> core_parameters.FullParameterInventory:
    """Freeze pretrained token embeddings and train the encoder and heads."""
    if cfg.hidden_size != fusion.encoder_width:
        raise ValueError("Encoder and fusion widths differ")
    return _build_inventory(cfg, model.shapes(fusion), freeze_embeddings)


def pointer_inventory(
    cfg: lfm.Config,
    head: pointer.Config,
    plan: quantization.NamedQuantization | None = None,
) -> core_parameters.FullParameterInventory:
    """Train the encoder and pointer projections; freeze token embeddings.

    With ``plan``, the inventory also declares which encoder projections
    train quantization-aware; embeddings, norms and pointer heads stay FP32.
    """
    if cfg.hidden_size != head.encoder_width:
        raise ValueError("Encoder and pointer widths differ")
    dense = _build_inventory(cfg, pointer.shapes(head), True)
    if plan is None:
        return dense
    projections = encoder_projections(cfg)
    roles = {
        spec.name: (
            "embedding"
            if spec.name == "lfm2.embed_tokens.weight"
            else "head"
            if spec.name.startswith("magicbox.")
            else "projection"
            if spec.name in projections
            else "other"
        )
        for spec in dense.specs
    }
    return _build_inventory(
        cfg,
        pointer.shapes(head),
        True,
        quantization_profile=plan.identity,
        quantized_names=quantization.select(dense, plan, roles),
    )


def encoder_projections(cfg: lfm.Config) -> frozenset[str]:
    """The encoder's attention, convolution and FFN projection matrices."""
    return frozenset(
        name.replace("model.", "lfm2.", 1) for name in lfm.projection_names(cfg)
    )


def quantization_plan(
    cfg: lfm.Config, kind: str
) -> quantization.NamedQuantization:
    """Fake-quantize every encoder projection with one group-128 quantizer."""
    if kind not in QUANTIZERS:
        raise ValueError(f"Unknown quantizer: {kind}")
    return quantization.NamedQuantization(
        quant_kernels.Group128Quantizer(QUANTIZERS[kind]),
        encoder_projections(cfg),
    )


def _build_inventory(
    cfg: lfm.Config,
    head_shapes: dict[str, tuple[int, ...]],
    freeze_embeddings: bool,
    *,
    quantization_profile: str | None = None,
    quantized_names: frozenset[str] = frozenset(),
) -> core_parameters.FullParameterInventory:
    """Decay trainable matrices; optionally freeze the embedding table."""
    shapes = {**encoder.Adapter().expected_shapes(cfg), **head_shapes}
    frozen = (
        frozenset({"lfm2.embed_tokens.weight"})
        if freeze_embeddings
        else frozenset()
    )
    return core_parameters.build_inventory(
        shapes,
        decayed_names=frozenset(
            name
            for name, shape in shapes.items()
            if len(shape) == 2 and name not in frozen
        ),
        frozen_names=frozen,
        format_id="minifield.magicbox.parameters/1",
        source_dtype="float32",
        master_dtype="float32",
        quantization_profile=quantization_profile,
        quantized_names=quantized_names,
    )


def forward(
    parameters: types.Parameters,
    cfg: lfm.Config,
    fusion: model.Config,
    batch: types.DeviceBatch,
    *,
    training: bool = False,
    bf16: bool = True,
) -> types.DeviceBatch:
    """Inject one shared encoder into the architecture."""
    encode = functools.partial(encoder.encode, cfg=cfg, bf16=bf16)

    def apply(
        params: types.Parameters,
        ids: jax.Array,
        mask: jax.Array,
        segment_ids: jax.Array | None,
        positions: jax.Array | None,
    ) -> jax.Array:
        """Bind the encoder's keyword config without copying parameters."""
        return encode(
            params,
            ids=ids,
            mask=mask,
            segment_ids=segment_ids,
            positions=positions,
        )

    return model.forward(parameters, fusion, apply, batch, training=training)


def bind(
    cfg: lfm.Config,
    fusion: model.Config,
    *,
    training: bool = False,
    bf16: bool = True,
) -> Callable[[types.Parameters, types.DeviceBatch], types.DeviceBatch]:
    """Supply the selected model as a neutral two-argument forward callback."""

    def apply(
        params: types.Parameters, batch: types.DeviceBatch
    ) -> types.DeviceBatch:
        return forward(params, cfg, fusion, batch, training=training, bf16=bf16)

    return apply


def bind_pointer(
    cfg: lfm.Config,
    head: pointer.Config,
    *,
    bf16: bool = True,
    attention: encoder.Attention = encoder.DEFAULT_ATTENTION,
) -> Callable[[types.Parameters, types.DeviceBatch], types.DeviceBatch]:
    """Bind the selected encoder to the joint pointer model."""

    def encode(
        params: types.Parameters,
        ids: jax.Array,
        mask: jax.Array,
        segment_ids: jax.Array,
        positions: jax.Array,
    ) -> jax.Array:
        return encoder.encode(
            params,
            cfg,
            ids,
            mask,
            bf16=bf16,
            segment_ids=segment_ids,
            positions=positions,
            attention_options=attention,
        )

    def apply(
        params: types.Parameters, batch: types.DeviceBatch
    ) -> types.DeviceBatch:
        return pointer.forward(params, head, encode, batch)

    return apply
