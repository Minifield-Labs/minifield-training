"""Pinned LFM2.5 encoder: bidirectional attention and centered short conv."""

from collections.abc import Mapping
import functools

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
            b_gate * values, weights.conv.astype(hidden.dtype)
        )
        residual = linear.full_linear(mixed, weights.out)
    else:
        weights_attn = model.attention_weights(params)
        query, key, value = attention.project_qkv(
            norm, weights_attn, head_dim=cfg.head_dim, eps=cfg.norm_eps
        )
        query = rotary.apply_rotary(
            query, rope_theta=cfg.rope_theta, head_dim=cfg.head_dim
        )
        key = rotary.apply_rotary(
            key, rope_theta=cfg.rope_theta, head_dim=cfg.head_dim
        )
        mixed = bidirectional.attention(query, key, value, mask)
        residual = linear.full_linear(
            mixed.reshape(hidden.shape), weights_attn.out
        )
    return feed_forward.swiglu_ffn(
        hidden + residual, model.ffn_weights(params), cfg.norm_eps
    )


def encode(
    params: Mapping[str, jax.Array],
    cfg: model.Config,
    ids: jax.Array,
    mask: jax.Array,
    *,
    bf16: bool = True,
) -> jax.Array:
    """Encode complete tokens, rematerializing blocks during reverse mode."""
    if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[1] > 8192:
        raise ValueError(
            "Encoder input shape or trained context limit violated"
        )
    dtype = jnp.bfloat16 if bf16 else jnp.float32
    hidden = params["lfm2.embed_tokens.weight"][ids].astype(dtype)
    for index, kind in enumerate(cfg.layer_types):
        layer = types.slice_parameters(params, f"lfm2.layers.{index}.")
        block = functools.partial(_block, cfg=cfg, kind=kind)
        # JAX 0.7.2 exports checkpoint without a public typing declaration.
        rematerialize = jax.checkpoint  # type: ignore[attr-defined]
        hidden = rematerialize(block)(hidden, mask, layer)
    return (
        normalization.rms_norm(
            hidden, params["lfm2.embedding_norm.weight"], cfg.norm_eps
        )
        * mask[..., None]
    )
