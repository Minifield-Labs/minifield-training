"""LFM2.5 hybrid decoder: config, checkpoint mapping, and block assembly.

The model maps flat Hugging Face checkpoint names onto shared layer weight
packs, dispatches convolution vs. attention blocks per ``layer_types``, and
owns the per-layer boundary state for shared-context scoring. All block
mathematics lives in ``minifield_training.layers`` and ``kernels``.
"""

from collections.abc import Mapping
import dataclasses
import functools
import math
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from minifield_training.core import parameters as core_parameters
from minifield_training.kernels import linear
from minifield_training.kernels import normalization
from minifield_training.kernels import selected_logits
from minifield_training.kernels import types
from minifield_training.layers import attention as attention_layers
from minifield_training.layers import convolution as convolution_layers
from minifield_training.layers import feed_forward


@dataclasses.dataclass(frozen=True)
class Config:
    """Validated dimensions for an LFM2.5 attention/convolution decoder."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    layer_types: tuple[str, ...]
    conv_kernel: int = 3
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 128000
    tied_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        """Return the dimension shared by attention heads."""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Config":
        """Validate published HF configurations and reject unsupported math."""
        if value.get("model_type") != "lfm2" or value.get("conv_bias", False):
            raise ValueError("Expected an LFM2 model without convolution bias")

        def integer(key: str, default: int | None = None) -> int:
            result = value.get(key, default)
            if (
                not isinstance(result, int)
                or isinstance(result, bool)
                or result <= 0
            ):
                raise ValueError(f"Invalid LFM2 dimension: {key}")
            return result

        def positive(number: object) -> float:
            if isinstance(number, bool) or not isinstance(number, int | float):
                raise ValueError("Expected a finite positive model constant")
            if not math.isfinite(number) or number <= 0:
                raise ValueError("Expected a finite positive model constant")
            return float(number)

        intermediate = integer("intermediate_size")
        if value.get("block_auto_adjust_ff_dim", True):
            intermediate = int(2 * intermediate / 3)
            multiplier = value.get("block_ffn_dim_multiplier", 1.0)
            if multiplier is not None:
                intermediate = int(intermediate * positive(multiplier))
                multiple = integer("block_multiple_of", 256)
                intermediate = multiple * (
                    (intermediate + multiple - 1) // multiple
                )
        layers = value.get("layer_types")
        if (
            not isinstance(layers, list)
            or len(layers) != integer("num_hidden_layers")
            or any(layer not in ("conv", "full_attention") for layer in layers)
        ):
            raise ValueError("Unsupported LFM2 layer sequence")
        rope = value.get("rope_parameters", {})
        if (
            not isinstance(rope, dict)
            or rope.get("rope_type", "default") != "default"
        ):
            raise ValueError("Only unscaled LFM2 RoPE is supported")
        tied = value.get(
            "tie_word_embeddings", value.get("tie_embedding", True)
        )
        if not isinstance(tied, bool):
            raise ValueError("Invalid tied-embedding flag")
        result = cls(
            hidden_size=integer("hidden_size"),
            intermediate_size=intermediate,
            num_attention_heads=integer("num_attention_heads"),
            num_key_value_heads=integer("num_key_value_heads"),
            vocab_size=integer("vocab_size"),
            layer_types=tuple(str(layer) for layer in layers),
            conv_kernel=integer("conv_L_cache", 3),
            norm_eps=positive(value.get("norm_eps", 1e-5)),
            rope_theta=positive(
                rope.get("rope_theta", value.get("rope_theta", 1000000.0))
            ),
            max_position_embeddings=integer("max_position_embeddings", 128000),
            tied_embeddings=tied,
        )
        if (
            result.hidden_size % result.num_attention_heads
            or result.num_attention_heads % result.num_key_value_heads
            or result.head_dim % 2
            or result.intermediate_size <= 0
        ):
            raise ValueError(
                "Invalid LFM2 attention or feed-forward dimensions"
            )
        return result


def expected_shapes(cfg: Config) -> dict[str, tuple[int, ...]]:
    """Return the exact tensor inventory for the supported dense checkpoint."""
    hidden, inner = cfg.hidden_size, cfg.intermediate_size
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (cfg.vocab_size, hidden),
        "model.embedding_norm.weight": (hidden,),
    }
    if not cfg.tied_embeddings:
        shapes["lm_head.weight"] = (cfg.vocab_size, hidden)
    for index, kind in enumerate(cfg.layer_types):
        prefix = f"model.layers.{index}."
        block: dict[str, tuple[int, ...]] = {
            "operator_norm.weight": (hidden,),
            "ffn_norm.weight": (hidden,),
            "feed_forward.w1.weight": (inner, hidden),
            "feed_forward.w3.weight": (inner, hidden),
            "feed_forward.w2.weight": (hidden, inner),
        }
        if kind == "conv":
            block.update(
                {
                    "conv.in_proj.weight": (3 * hidden, hidden),
                    "conv.out_proj.weight": (hidden, hidden),
                    "conv.conv.weight": (hidden, 1, cfg.conv_kernel),
                }
            )
        else:
            key_dim = cfg.num_key_value_heads * cfg.head_dim
            block.update(
                {
                    "self_attn.q_proj.weight": (hidden, hidden),
                    "self_attn.k_proj.weight": (key_dim, hidden),
                    "self_attn.v_proj.weight": (key_dim, hidden),
                    "self_attn.out_proj.weight": (hidden, hidden),
                    "self_attn.q_layernorm.weight": (cfg.head_dim,),
                    "self_attn.k_layernorm.weight": (cfg.head_dim,),
                }
            )
        shapes.update({prefix + key: shape for key, shape in block.items()})
    return shapes


def parameter_inventory(
    cfg: Config,
    *,
    source_dtype: str = "bfloat16",
    master_dtype: str = "float32",
    quantization_profile: str | None = None,
    quantized_names: frozenset[str] = frozenset(),
    frozen_names: frozenset[str] = frozenset(),
) -> core_parameters.FullParameterInventory:
    """Bind metadata with decay for every unfrozen matrix in this family."""
    shapes = expected_shapes(cfg)
    return core_parameters.build_inventory(
        shapes,
        decayed_names=frozenset(
            name
            for name, shape in shapes.items()
            if len(shape) == 2 and name not in frozen_names
        ),
        format_id="minifield.lfm.full-parameters/1",
        source_dtype=source_dtype,
        master_dtype=master_dtype,
        quantization_profile=quantization_profile,
        quantized_names=quantized_names,
        frozen_names=frozen_names,
    )


def validate_parameters(
    parameters: Mapping[str, jax.Array], cfg: Config
) -> None:
    """Check key, shape, and FP32-master dtype against this config."""
    types.validate_parameter_structure(parameters, expected_shapes(cfg))


def validate_masters(parameters: Mapping[str, jax.Array], cfg: Config) -> None:
    """Eagerly reject non-finite masters at load or update boundaries."""
    types.validate_parameter_masters(parameters, expected_shapes(cfg))


class LayerPrefixState(NamedTuple):
    """One layer's differentiable boundary state at a shared context edge.

    ``key`` and ``value`` hold the layer's full context attention cache with
    RoPE already applied to keys and padded columns zeroed. Convolution
    blocks instead retain ``convolution_history``, the ``kernel_size - 1``
    gated ``b_gate * values`` columns ending at each row's valid length.
    """

    key: jax.Array | None
    value: jax.Array | None
    convolution_history: jax.Array | None


class SharedPrefix(NamedTuple):
    """Complete boundary state for one decision's shared context.

    ``hidden`` is the normalized context representation kept so selected
    positions inside the context still resolve, and ``mask`` is the context
    validity mask suffix attention uses to exclude padding keys.
    """

    layers: tuple[LayerPrefixState, ...]
    hidden: jax.Array
    mask: jax.Array


def layer_weights(
    parameters: Mapping[str, jax.Array], index: int
) -> types.Parameters:
    """Return one layer's parameter sub-dict with its prefix stripped."""
    return types.slice_parameters(parameters, f"model.layers.{index}.")


def ffn_weights(
    params: Mapping[str, jax.Array],
) -> feed_forward.FeedForwardWeights:
    """Build the SwiGLU weight pack from one layer's checkpoint tensors."""
    return feed_forward.FeedForwardWeights(
        norm=params["ffn_norm.weight"],
        gate=params["feed_forward.w1.weight"],
        up=params["feed_forward.w3.weight"],
        down=params["feed_forward.w2.weight"],
    )


def conv_weights(
    params: Mapping[str, jax.Array],
) -> convolution_layers.ConvWeights:
    """Build the conv weight pack, squeezing the checkpoint's taps axis."""
    return convolution_layers.ConvWeights(
        operator_norm=params["operator_norm.weight"],
        in_proj=params["conv.in_proj.weight"],
        conv=params["conv.conv.weight"][:, 0, :],
        out=params["conv.out_proj.weight"],
    )


def attention_weights(
    params: Mapping[str, jax.Array],
) -> attention_layers.AttentionWeights:
    """Build the attention weight pack from one layer's checkpoint tensors."""
    return attention_layers.AttentionWeights(
        operator_norm=params["operator_norm.weight"],
        query=params["self_attn.q_proj.weight"],
        key=params["self_attn.k_proj.weight"],
        value=params["self_attn.v_proj.weight"],
        out=params["self_attn.out_proj.weight"],
        query_norm=params["self_attn.q_layernorm.weight"],
        key_norm=params["self_attn.k_layernorm.weight"],
    )


def _block(
    hidden: jax.Array,
    params: types.Parameters,
    attention_mask: jax.Array,
    cfg: Config,
    kind: str,
    attention_backend: str,
) -> jax.Array:
    """Run one full-sequence block, dispatching on the layer type."""
    ffn = ffn_weights(params)
    if kind == "conv":
        return convolution_layers.conv_block(
            hidden,
            conv_weights(params),
            ffn,
            attention_mask,
            kernel_size=cfg.conv_kernel,
            eps=cfg.norm_eps,
        )
    return attention_layers.attention_block(
        hidden,
        attention_weights(params),
        ffn,
        attention_mask,
        head_dim=cfg.head_dim,
        rope_theta=cfg.rope_theta,
        eps=cfg.norm_eps,
        backend=attention_backend,
    )


def _block_prefix(
    hidden: jax.Array,
    params: types.Parameters,
    attention_mask: jax.Array,
    cfg: Config,
    kind: str,
    attention_backend: str,
) -> tuple[jax.Array, LayerPrefixState]:
    """Run one block and retain its shared-context boundary state."""
    ffn = ffn_weights(params)
    if kind == "conv":
        hidden, history = convolution_layers.conv_block_prefix(
            hidden,
            conv_weights(params),
            ffn,
            attention_mask,
            kernel_size=cfg.conv_kernel,
            eps=cfg.norm_eps,
        )
        return hidden, LayerPrefixState(
            key=None, value=None, convolution_history=history
        )
    hidden, key, value = attention_layers.attention_block_prefix(
        hidden,
        attention_weights(params),
        ffn,
        attention_mask,
        head_dim=cfg.head_dim,
        rope_theta=cfg.rope_theta,
        eps=cfg.norm_eps,
        backend=attention_backend,
    )
    return hidden, LayerPrefixState(
        key=key, value=value, convolution_history=None
    )


def _block_suffix(
    hidden: jax.Array,
    params: types.Parameters,
    attention_mask: jax.Array,
    positions: jax.Array,
    prefix_mask: jax.Array,
    state: LayerPrefixState,
    cfg: Config,
    kind: str,
    attention_backend: str,
) -> jax.Array:
    """Run one continuation block conditioned on shared prefix state."""
    ffn = ffn_weights(params)
    if kind == "conv":
        if state.convolution_history is None:
            raise ValueError("Convolution suffix blocks need prefix history")
        return convolution_layers.conv_block_suffix(
            hidden,
            conv_weights(params),
            ffn,
            attention_mask,
            state.convolution_history,
            kernel_size=cfg.conv_kernel,
            eps=cfg.norm_eps,
        )
    if state.key is None or state.value is None:
        raise ValueError("Attention suffix blocks need prefix keys/values")
    return attention_layers.attention_block_suffix(
        hidden,
        attention_weights(params),
        ffn,
        attention_mask,
        positions,
        state.key,
        state.value,
        prefix_mask,
        head_dim=cfg.head_dim,
        rope_theta=cfg.rope_theta,
        eps=cfg.norm_eps,
        backend=attention_backend,
    )


def _block_packed(
    hidden: jax.Array,
    params: types.Parameters,
    attention_mask: jax.Array,
    segment_ids: jax.Array,
    position_ids: jax.Array,
    cfg: Config,
    kind: str,
    attention_backend: str,
) -> jax.Array:
    """Run one packed block honoring segment and position boundaries."""
    ffn = ffn_weights(params)
    if kind == "conv":
        return convolution_layers.conv_block_packed(
            hidden,
            conv_weights(params),
            ffn,
            attention_mask,
            segment_ids,
            kernel_size=cfg.conv_kernel,
            eps=cfg.norm_eps,
        )
    return attention_layers.attention_block_packed(
        hidden,
        attention_weights(params),
        ffn,
        attention_mask,
        segment_ids,
        position_ids,
        head_dim=cfg.head_dim,
        rope_theta=cfg.rope_theta,
        eps=cfg.norm_eps,
        backend=attention_backend,
    )


def _hidden_states(
    parameters: types.Parameters,
    ids: jax.Array,
    attention_mask: jax.Array,
    cfg: Config,
    *,
    dtype: types.DType,
    attention_backend: str,
    rematerialize_blocks: bool,
) -> jax.Array:
    """Decode from one complete FP32 parameter pytree."""
    hidden = parameters["model.embed_tokens.weight"][ids].astype(dtype)
    for index, kind in enumerate(cfg.layer_types):
        block = _block
        if rematerialize_blocks:
            block = jax.checkpoint(  # type: ignore[attr-defined]
                _block, static_argnums=(3, 4, 5)
            )
        hidden = block(
            hidden,
            layer_weights(parameters, index),
            attention_mask,
            cfg,
            kind,
            attention_backend,
        )
    return normalization.rms_norm(
        hidden, parameters["model.embedding_norm.weight"], cfg.norm_eps
    )


def _prefix_hidden_states(
    parameters: types.Parameters,
    ids: jax.Array,
    attention_mask: jax.Array,
    cfg: Config,
    *,
    dtype: types.DType,
    attention_backend: str,
) -> tuple[jax.Array, tuple[LayerPrefixState, ...]]:
    """Decode one shared context and retain each layer's boundary state."""
    hidden = parameters["model.embed_tokens.weight"][ids].astype(dtype)
    states: list[LayerPrefixState] = []
    for index, kind in enumerate(cfg.layer_types):
        rematerialize = jax.checkpoint  # type: ignore[attr-defined]
        hidden, state = rematerialize(_block_prefix, static_argnums=(3, 4, 5))(
            hidden,
            layer_weights(parameters, index),
            attention_mask,
            cfg,
            kind,
            attention_backend,
        )
        states.append(state)
    hidden = normalization.rms_norm(
        hidden, parameters["model.embedding_norm.weight"], cfg.norm_eps
    )
    return hidden, tuple(states)


def _suffix_hidden_states(
    parameters: types.Parameters,
    ids: jax.Array,
    attention_mask: jax.Array,
    positions: jax.Array,
    prefix: SharedPrefix,
    cfg: Config,
    *,
    dtype: types.DType,
    attention_backend: str,
) -> jax.Array:
    """Decode suffix rows conditioned on one shared prefix state."""
    hidden = parameters["model.embed_tokens.weight"][ids].astype(dtype)
    for index, kind in enumerate(cfg.layer_types):
        rematerialize = jax.checkpoint  # type: ignore[attr-defined]
        hidden = rematerialize(_block_suffix, static_argnums=(6, 7, 8))(
            hidden,
            layer_weights(parameters, index),
            attention_mask,
            positions,
            prefix.mask,
            prefix.layers[index],
            cfg,
            kind,
            attention_backend,
        )
    return normalization.rms_norm(
        hidden, parameters["model.embedding_norm.weight"], cfg.norm_eps
    )


def _packed_hidden_states(
    parameters: types.Parameters,
    ids: jax.Array,
    attention_mask: jax.Array,
    segment_ids: jax.Array,
    position_ids: jax.Array,
    cfg: Config,
    *,
    dtype: types.DType,
    attention_backend: str,
) -> jax.Array:
    """Decode a packed batch honoring segment and position boundaries."""
    hidden = parameters["model.embed_tokens.weight"][ids].astype(dtype)
    for index, kind in enumerate(cfg.layer_types):
        rematerialize = jax.checkpoint  # type: ignore[attr-defined]
        hidden = rematerialize(_block_packed, static_argnums=(5, 6, 7))(
            hidden,
            layer_weights(parameters, index),
            attention_mask,
            segment_ids,
            position_ids,
            cfg,
            kind,
            attention_backend,
        )
    return normalization.rms_norm(
        hidden, parameters["model.embedding_norm.weight"], cfg.norm_eps
    )


def hidden_states(
    parameters: types.Parameters,
    ids: jax.Array,
    attention_mask: jax.Array,
    cfg: Config,
    *,
    dtype: types.DType = jnp.bfloat16,
    attention_backend: str = "dense",
    rematerialize_blocks: bool = True,
) -> jax.Array:
    """Decode batches from fully trainable FP32 masters.

    With ``attention_backend="cudnn"`` each row's valid tokens must form a
    contiguous suffix-padded prefix; the dense backend accepts arbitrary
    binary masks. ``rematerialize_blocks=False`` retains block activations
    for backward instead of recomputing them; it only applies to this full
    sequence path.
    """
    validate_parameters(parameters, cfg)
    return _hidden_states(
        parameters,
        ids,
        attention_mask,
        cfg,
        dtype=dtype,
        attention_backend=attention_backend,
        rematerialize_blocks=rematerialize_blocks,
    )


def packed_hidden_states(
    parameters: types.Parameters,
    ids: jax.Array,
    attention_mask: jax.Array,
    segment_ids: jax.Array,
    position_ids: jax.Array,
    cfg: Config,
    *,
    dtype: types.DType = jnp.bfloat16,
    attention_backend: str = "dense",
) -> jax.Array:
    """Decode packed rows with segment-safe attention, RoPE, and conv."""
    validate_parameters(parameters, cfg)
    return _packed_hidden_states(
        parameters,
        ids,
        attention_mask,
        segment_ids,
        position_ids,
        cfg,
        dtype=dtype,
        attention_backend=attention_backend,
    )


@functools.partial(
    jax.jit,
    static_argnames=("cfg", "dtype", "attention_backend"),
)
def prefix_states(
    parameters: types.Parameters,
    ids: jax.Array,
    attention_mask: jax.Array,
    cfg: Config,
    *,
    dtype: types.DType = jnp.bfloat16,
    attention_backend: str = "dense",
) -> SharedPrefix:
    """Decode one shared context and retain its per-layer boundary state.

    The context is a standard suffix-padded causal sequence, so the
    ``cudnn`` backend applies to this call. Retained states stay inside the
    traced objective, so gradients from every conditioned suffix flow back
    through this single forward.
    """
    validate_parameters(parameters, cfg)
    hidden, layers = _prefix_hidden_states(
        parameters,
        ids,
        attention_mask,
        cfg,
        dtype=dtype,
        attention_backend=attention_backend,
    )
    return SharedPrefix(layers=layers, hidden=hidden, mask=attention_mask)


def _suffix_selected_impl(
    parameters: types.Parameters,
    prefix: SharedPrefix,
    suffix_ids: jax.Array,
    suffix_mask: jax.Array,
    suffix_positions: jax.Array,
    context_length: jax.Array,
    cfg: Config,
    positions: jax.Array,
    dtype: types.DType,
    attention_backend: str,
) -> jax.Array:
    """Gather selected hidden states for one shared-context suffix chunk."""
    validate_parameters(parameters, cfg)
    hidden = _suffix_hidden_states(
        parameters,
        suffix_ids,
        suffix_mask,
        suffix_positions,
        prefix,
        cfg,
        dtype=dtype,
        attention_backend=attention_backend,
    )
    on_suffix = positions[:, 1] >= context_length
    suffix_columns = jnp.clip(
        positions[:, 1] - context_length, 0, hidden.shape[1] - 1
    )
    prefix_columns = jnp.clip(positions[:, 1], 0, prefix.hidden.shape[1] - 1)
    return jnp.where(
        on_suffix[:, None],
        hidden[positions[:, 0], suffix_columns],
        prefix.hidden[0, prefix_columns],
    )


def _suffix_scoring_impl(
    parameters: types.Parameters,
    prefix: SharedPrefix,
    suffix_ids: jax.Array,
    suffix_mask: jax.Array,
    suffix_positions: jax.Array,
    context_length: jax.Array,
    cfg: Config,
    positions: jax.Array,
    target_ids: jax.Array,
    dtype: types.DType,
    attention_backend: str,
) -> jax.Array:
    """Recompute each bounded suffix chunk during its reverse pass."""
    selected = _suffix_selected_impl(
        parameters,
        prefix,
        suffix_ids,
        suffix_mask,
        suffix_positions,
        context_length,
        cfg,
        positions,
        dtype,
        attention_backend,
    )
    head = (
        parameters["model.embed_tokens.weight"]
        if cfg.tied_embeddings
        else parameters["lm_head.weight"]
    )
    return selected_logits.selected_hidden_log_probs(selected, head, target_ids)


# Whole-chunk rematerialization does not retain all earlier chunks' layer
# activations for backward. The context boundary states remain shared.
_suffix_selected_checkpoint = jax.checkpoint(  # type: ignore[attr-defined]
    _suffix_selected_impl, static_argnums=(6, 8, 9)
)
_suffix_scoring_checkpoint = jax.checkpoint(  # type: ignore[attr-defined]
    _suffix_scoring_impl, static_argnums=(6, 9, 10)
)


@functools.partial(
    jax.jit,
    static_argnames=("cfg", "dtype", "attention_backend"),
)
def suffix_selected_hidden(
    parameters: types.Parameters,
    prefix: SharedPrefix,
    suffix_ids: jax.Array,
    suffix_mask: jax.Array,
    suffix_positions: jax.Array,
    context_length: jax.Array,
    cfg: Config,
    positions: jax.Array,
    *,
    dtype: types.DType = jnp.bfloat16,
    attention_backend: str = "dense",
) -> jax.Array:
    """Gather selected targets' hidden states on shared-context suffix rows.

    Same layout and boundary semantics as ``suffix_token_log_probs``, but
    stops before the LM-head projection so a caller scanning many slabs pays
    one ``[targets, vocabulary]`` projection instead of one per slab.
    """
    return cast(
        jax.Array,
        _suffix_selected_checkpoint(
            parameters,
            prefix,
            suffix_ids,
            suffix_mask,
            suffix_positions,
            context_length,
            cfg,
            positions,
            dtype,
            attention_backend,
        ),
    )


@functools.partial(
    jax.jit,
    static_argnames=("cfg", "dtype", "attention_backend"),
)
def suffix_token_log_probs(
    parameters: types.Parameters,
    prefix: SharedPrefix,
    suffix_ids: jax.Array,
    suffix_mask: jax.Array,
    suffix_positions: jax.Array,
    context_length: jax.Array,
    cfg: Config,
    positions: jax.Array,
    target_ids: jax.Array,
    *,
    dtype: types.DType = jnp.bfloat16,
    attention_backend: str = "dense",
) -> jax.Array:
    """Score selected targets on suffix rows sharing one context state.

    ``suffix_ids``/``suffix_mask`` are the tokens after the shared context,
    right-padded; ``suffix_positions`` holds their absolute RoPE positions
    and ``context_length`` the shared context's logical length. ``positions``
    indexes targets in absolute context-plus-suffix coordinates; positions
    below ``context_length`` read the retained prefix hidden states.
    """
    return cast(
        jax.Array,
        _suffix_scoring_checkpoint(
            parameters,
            prefix,
            suffix_ids,
            suffix_mask,
            suffix_positions,
            context_length,
            cfg,
            positions,
            target_ids,
            dtype,
            attention_backend,
        ),
    )


def logits(
    hidden: jax.Array,
    parameters: Mapping[str, jax.Array],
    cfg: Config,
) -> jax.Array:
    """Project hidden states through the tied or untied stored LM head."""
    head = (
        parameters["model.embed_tokens.weight"]
        if cfg.tied_embeddings
        else parameters["lm_head.weight"]
    )
    return linear.full_linear(hidden, head).astype(jnp.float32)


def selected_token_log_probs(
    hidden: jax.Array,
    parameters: Mapping[str, jax.Array],
    positions: jax.Array,
    target_ids: jax.Array,
    cfg: Config,
) -> jax.Array:
    """Score selected causal targets through the tied/full LM head."""
    head = (
        parameters["model.embed_tokens.weight"]
        if cfg.tied_embeddings
        else parameters["lm_head.weight"]
    )
    return selected_logits.selected_token_log_probs(
        hidden, head, positions, target_ids
    )


def forward(
    parameters: types.Parameters,
    ids: jax.Array,
    attention_mask: jax.Array,
    cfg: Config,
    *,
    dtype: types.DType = jnp.bfloat16,
    rematerialize_blocks: bool = True,
) -> jax.Array:
    """Return dense diagnostic logits from the complete trainable pytree."""
    hidden = hidden_states(
        parameters,
        ids,
        attention_mask,
        cfg,
        dtype=dtype,
        rematerialize_blocks=rematerialize_blocks,
    )
    return logits(hidden, parameters, cfg)
