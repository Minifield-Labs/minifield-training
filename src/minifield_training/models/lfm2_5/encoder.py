"""Pinned LFM2.5 encoder: bidirectional attention and centered short conv."""

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp

from minifield_training.kernels import bidirectional
from minifield_training.kernels import linear
from minifield_training.kernels import normalization
from minifield_training.kernels import rotary
from minifield_training.kernels import types
from minifield_training.layers import attention
from minifield_training.layers import convolution
from minifield_training.layers import feed_forward
from minifield_training.models import contracts
from minifield_training.models.lfm2_5 import model

MAX_SEQUENCE_LENGTH = 8192

SOURCE = contracts.PretrainedSource(
    model_id="LiquidAI/LFM2.5-Encoder-350M",
    revision="b886781f7c6f10ca9b7096e21b83e30a073c2f39",
    config_sha256=(
        "559f88ddcebfb0a7b46ba99a074b17f3a951f278b3010a0f22fdf5d9cc358d11"
    ),
    tokenizer_sha256=(
        "1efc3a6609abf6b63b1f47188d139f3b59973a6a434dffe970a7261a51ed2711"
    ),
    weights_sha256=(
        "cd70e404c3c6c1756b2cf5dc75de2a87788b460e3a6156c1a4ab134b824c2706"
    ),
)


class Adapter:
    """Admit only the published bidirectional masked-LM backbone."""

    source_dtype = "F32"

    def parse_config(self, value: Mapping[str, object]) -> model.Config:
        """Require explicit bidirectional architecture and tied embeddings."""
        if value.get("architectures") != ["Lfm2BidirectionalForMaskedLM"]:
            raise ValueError("Expected the bidirectional LFM2 checkpoint")
        cfg = model.Config.from_dict(value)
        if not cfg.tied_embeddings:
            raise ValueError("Encoder requires tied source embeddings")
        return cfg

    def expected_shapes(self, cfg: model.Config) -> dict[str, tuple[int, ...]]:
        """Use the masked-LM wrapper's lfm2 prefix without a vocabulary head."""
        return {
            name.replace("model.", "lfm2.", 1): shape
            for name, shape in model.expected_shapes(cfg).items()
        }

    def validate_masters(
        self, parameters: Mapping[str, jax.Array], cfg: model.Config
    ) -> None:
        """Validate all admitted tensors as finite FP32 masters."""
        types.validate_parameter_masters(parameters, self.expected_shapes(cfg))


def _block(
    hidden: jax.Array,
    mask: jax.Array,
    params: types.Parameters,
    *,
    cfg: model.Config,
    kind: str,
    segment_ids: jax.Array | None = None,
    positions: jax.Array | None = None,
) -> jax.Array:
    """Compose the source's operator with the family's shared SwiGLU tail."""
    norm = normalization.rms_norm(
        hidden, params["operator_norm.weight"], cfg.norm_eps
    )
    if kind == "conv":
        weights = model.conv_weights(params)
        b_gate, c_gate, values = convolution.conv_input_projection(
            norm, weights.in_proj, mask
        )
        mixed = c_gate * bidirectional.centered_convolution(
            b_gate * values, weights.conv.astype(hidden.dtype), segment_ids
        )
        residual = linear.full_linear(mixed, weights.out)
    else:
        weights_attn = model.attention_weights(params)
        query, key, value = attention.project_qkv(
            norm, weights_attn, head_dim=cfg.head_dim, eps=cfg.norm_eps
        )
        query = rotary.apply_rotary(
            query,
            rope_theta=cfg.rope_theta,
            head_dim=cfg.head_dim,
            positions=positions,
        )
        key = rotary.apply_rotary(
            key,
            rope_theta=cfg.rope_theta,
            head_dim=cfg.head_dim,
            positions=positions,
        )
        mixed = bidirectional.attention(query, key, value, mask, segment_ids)
        residual = linear.full_linear(
            mixed.reshape(hidden.shape), weights_attn.out
        )
    return feed_forward.swiglu_ffn(
        hidden + residual, model.ffn_weights(params), cfg.norm_eps
    )


def _scan_blocks(
    hidden: jax.Array,
    mask: jax.Array,
    params: Mapping[str, jax.Array],
    cfg: model.Config,
    segment_ids: jax.Array | None = None,
    positions: jax.Array | None = None,
) -> jax.Array:
    """Keep one compiled block per operator kind, with distinct layer weights.

    Common tensors travel along the scan's layer axis. Operator-specific
    tensors are stacked separately, so convolution and attention keep their
    original shapes without padding or allocating dummy model parameters.
    Packing is differentiable; checkpoint and optimizer inventories stay flat.
    """
    if not cfg.layer_types:
        return hidden
    layers = [
        types.slice_parameters(params, f"lfm2.layers.{index}.")
        for index in range(len(cfg.layer_types))
    ]
    common_names = set(layers[0]).intersection(*layers)
    common = {
        name: jnp.stack([layer[name] for layer in layers])
        for name in sorted(common_names)
    }
    kinds = tuple(dict.fromkeys(cfg.layer_types))
    operators = {}
    for kind in kinds:
        group = [
            layer
            for layer, layer_kind in zip(layers, cfg.layer_types, strict=True)
            if layer_kind == kind
        ]
        operators[kind] = {
            name: jnp.stack([layer[name] for layer in group])
            for name in sorted(set(group[0]) - common_names)
        }
    counts = dict.fromkeys(kinds, 0)
    offsets = []
    for kind in cfg.layer_types:
        offsets.append(counts[kind])
        counts[kind] += 1
    indices = jnp.asarray([kinds.index(kind) for kind in cfg.layer_types])

    def step(
        activation: jax.Array,
        item: tuple[types.Parameters, jax.Array, jax.Array],
    ) -> tuple[jax.Array, None]:
        """Select the original operator and its weights at this layer."""
        shared, index, offset = item

        def branch(kind: str) -> Callable[[None], jax.Array]:
            """Bind one operator's static layout to the dynamic layer slot."""

            def apply(_: None) -> jax.Array:
                """Apply the unchanged block equations to selected weights."""
                selected = {
                    name: values[offset]
                    for name, values in operators[kind].items()
                }
                return _block(
                    activation,
                    mask,
                    {**shared, **selected},
                    cfg=cfg,
                    kind=kind,
                    segment_ids=segment_ids,
                    positions=positions,
                )

            return apply

        result = jax.lax.switch(
            index, tuple(branch(kind) for kind in kinds), None
        )
        return result, None

    # Scan separates forward/reverse iterations; it doesn't need CSE barriers.
    # JAX 0.7.2 exports checkpoint without a public typing declaration.
    rematerialize = jax.checkpoint  # type: ignore[attr-defined]
    hidden, _ = jax.lax.scan(
        rematerialize(step, prevent_cse=False),
        hidden,
        (common, indices, jnp.asarray(offsets)),
    )
    return hidden


def encode(
    params: Mapping[str, jax.Array],
    cfg: model.Config,
    ids: jax.Array,
    mask: jax.Array,
    *,
    bf16: bool = True,
    segment_ids: jax.Array | None = None,
    positions: jax.Array | None = None,
) -> jax.Array:
    """Encode complete tokens, rematerializing blocks during reverse mode.

    Packed rows pass ``segment_ids`` (0 for padding, matching ``mask``) and
    per-segment ``positions`` together. Each segment then encodes exactly as
    it would alone: attention, convolution taps, and rotary positions stop at
    segment boundaries.
    """
    if (
        ids.ndim != 2
        or mask.shape != ids.shape
        or ids.shape[1] > MAX_SEQUENCE_LENGTH
    ):
        raise ValueError(
            "Encoder input shape or trained context limit violated"
        )
    if (segment_ids is None) != (positions is None) or any(
        value is not None and value.shape != ids.shape
        for value in (segment_ids, positions)
    ):
        raise ValueError("Packed segment IDs and positions must match ids")
    dtype = jnp.bfloat16 if bf16 else jnp.float32
    hidden = params["lfm2.embed_tokens.weight"][ids].astype(dtype)
    hidden = _scan_blocks(hidden, mask, params, cfg, segment_ids, positions)
    return (
        normalization.rms_norm(
            hidden, params["lfm2.embedding_norm.weight"], cfg.norm_eps
        )
        * mask[..., None]
    )
