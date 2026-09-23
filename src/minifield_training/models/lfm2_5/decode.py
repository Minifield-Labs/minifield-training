"""Fixed-size cached decoding for the LFM2.5 full-weight model.

Prefill uses the next power-of-two prompt bucket while keeping the larger
autoregressive KV capacity in the returned state. Block mathematics lives
in ``minifield_training.layers``; this module owns cache allocation,
validation, and the per-step state transitions.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from minifield_training.kernels import normalization
from minifield_training.kernels import types
from minifield_training.layers import attention as attention_layers
from minifield_training.layers import convolution as convolution_layers
from minifield_training.models.lfm2_5 import model


class DecodeOverflow(ValueError):
    """Raised before a token can be written past a fixed cache capacity."""


class BlockCache(NamedTuple):
    """Per-block cache payload at the explicit block boundary."""

    key: jax.Array | None
    value: jax.Array | None
    convolution_history: jax.Array | None


class DecodeState(NamedTuple):
    """Immutable fixed-capacity decoding state."""

    attention_keys: tuple[jax.Array | None, ...]
    attention_values: tuple[jax.Array | None, ...]
    convolution_history: tuple[jax.Array | None, ...]
    valid_length: int
    capacity: int


def _check_capacity(capacity: int) -> None:
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or capacity <= 0
        or capacity & (capacity - 1)
    ):
        raise ValueError("Cache capacity must be a positive power of two")


def _check_dtype(dtype: types.DType) -> None:
    if np.dtype(dtype) not in {
        np.dtype(jnp.bfloat16),
        np.dtype(jnp.float16),
        np.dtype(jnp.float32),
    }:
        raise ValueError("Cached decoding expects BF16, FP16, or FP32 compute")


def _prompt_capacity(token_count: int, total_capacity: int) -> int:
    """Choose a bounded physical prompt shape independently of KV capacity."""
    if token_count <= 0 or token_count > total_capacity:
        raise ValueError("Prompt length must fit the cache capacity")
    return 1 << (token_count - 1).bit_length()


def _check_model_length(cfg: model.Config, length: int) -> None:
    if length > cfg.max_position_embeddings:
        raise ValueError("Logical length exceeds model max_position_embeddings")


def new_state(
    cfg: model.Config,
    batch_size: int,
    capacity: int,
    *,
    dtype: types.DType = jnp.bfloat16,
) -> DecodeState:
    """Allocate a zeroed fixed-size state for one prompt batch."""
    _check_capacity(capacity)
    _check_dtype(dtype)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    keys: list[jax.Array | None] = []
    values: list[jax.Array | None] = []
    histories: list[jax.Array | None] = []
    for kind in cfg.layer_types:
        if kind == "full_attention":
            keys.append(
                jnp.zeros(
                    (
                        batch_size,
                        capacity,
                        cfg.num_key_value_heads,
                        cfg.head_dim,
                    ),
                    dtype=dtype,
                )
            )
            values.append(
                jnp.zeros(
                    (
                        batch_size,
                        capacity,
                        cfg.num_key_value_heads,
                        cfg.head_dim,
                    ),
                    dtype=dtype,
                )
            )
            histories.append(None)
        else:
            keys.append(None)
            values.append(None)
            histories.append(
                jnp.zeros(
                    (batch_size, cfg.conv_kernel - 1, cfg.hidden_size),
                    dtype=dtype,
                )
            )
    return DecodeState(
        tuple(keys),
        tuple(values),
        tuple(histories),
        valid_length=0,
        capacity=capacity,
    )


def _validate_state(state: DecodeState, cfg: model.Config) -> None:
    _check_capacity(state.capacity)
    if state.valid_length < 0 or state.valid_length > state.capacity:
        raise ValueError("Decode state has an invalid logical position")
    if len(state.attention_keys) != len(cfg.layer_types):
        raise ValueError("Decode state layer count does not match the config")
    for index, kind in enumerate(cfg.layer_types):
        key = state.attention_keys[index]
        value = state.attention_values[index]
        history = state.convolution_history[index]
        if kind == "full_attention":
            if key is None or value is None or history is not None:
                raise ValueError(
                    "Attention cache structure does not match config"
                )
            expected = (
                key.shape[0],
                state.capacity,
                cfg.num_key_value_heads,
                cfg.head_dim,
            )
            if key.shape != expected or value.shape != expected:
                raise ValueError("Attention cache shape does not match config")
        else:
            if key is not None or value is not None or history is None:
                raise ValueError(
                    "Convolution cache structure does not match config"
                )
            expected_history = (
                history.shape[0],
                cfg.conv_kernel - 1,
                cfg.hidden_size,
            )
            if history.shape != expected_history:
                raise ValueError(
                    "Convolution history shape does not match config"
                )


def _cache_batch(state: DecodeState) -> int:
    for cache in (*state.attention_keys, *state.convolution_history):
        if cache is not None:
            return cache.shape[0]
    raise ValueError("Decode state has no cache leaves")


def _prefill_block(
    hidden: jax.Array,
    params: types.Parameters,
    cfg: model.Config,
    kind: str,
    cache: BlockCache,
    valid_length: jax.Array,
    attention_backend: str,
) -> tuple[jax.Array, BlockCache]:
    """Run one prefill block and return its fresh cache payload."""
    positions = jnp.arange(hidden.shape[1], dtype=jnp.int32)
    attention_mask = jnp.broadcast_to(
        (positions < valid_length)[None, :],
        (hidden.shape[0], hidden.shape[1]),
    )
    ffn = model.ffn_weights(params)
    if kind == "conv":
        hidden, history = convolution_layers.conv_block_prefix(
            hidden,
            model.conv_weights(params),
            ffn,
            attention_mask,
            kernel_size=cfg.conv_kernel,
            eps=cfg.norm_eps,
        )
        return hidden, BlockCache(None, None, history)
    if cache.key is None or cache.value is None:
        raise ValueError("Missing attention cache allocation")
    hidden, key, value = attention_layers.attention_block_prefill(
        hidden,
        model.attention_weights(params),
        ffn,
        attention_mask,
        capacity=cache.key.shape[1],
        head_dim=cfg.head_dim,
        rope_theta=cfg.rope_theta,
        eps=cfg.norm_eps,
        backend=attention_backend,
    )
    return hidden, BlockCache(key, value, None)


def _decode_block(
    hidden: jax.Array,
    params: types.Parameters,
    cfg: model.Config,
    kind: str,
    cache: BlockCache,
    position: jax.Array,
    attention_backend: str,
) -> tuple[jax.Array, BlockCache]:
    """Run one single-token block and return its updated cache payload."""
    ffn = model.ffn_weights(params)
    if kind == "conv":
        if cache.convolution_history is None:
            raise ValueError("Missing convolution history")
        hidden, history = convolution_layers.conv_block_step(
            hidden,
            model.conv_weights(params),
            ffn,
            cache.convolution_history,
            eps=cfg.norm_eps,
        )
        return hidden, BlockCache(None, None, history)
    if cache.key is None or cache.value is None:
        raise ValueError("Missing attention KV cache")
    hidden, key, value = attention_layers.attention_block_step(
        hidden,
        model.attention_weights(params),
        ffn,
        cache.key,
        cache.value,
        position,
        position + 1,
        head_dim=cfg.head_dim,
        rope_theta=cfg.rope_theta,
        eps=cfg.norm_eps,
        backend=attention_backend,
    )
    return hidden, BlockCache(key, value, None)


def _prefill_impl(
    parameters: types.Parameters,
    input_ids: jax.Array,
    attention_keys: tuple[jax.Array | None, ...],
    attention_values: tuple[jax.Array | None, ...],
    convolution_history: tuple[jax.Array | None, ...],
    valid_length: jax.Array,
    cfg: model.Config,
    dtype: types.DType,
    capacity: int,
    attention_backend: str,
) -> tuple[
    jax.Array,
    tuple[jax.Array | None, ...],
    tuple[jax.Array | None, ...],
    tuple[jax.Array | None, ...],
]:
    if any(
        cache is not None and cache.shape[1] != capacity
        for cache in attention_keys
    ):
        raise ValueError("Prefill cache shape does not match capacity")
    hidden = parameters["model.embed_tokens.weight"][input_ids].astype(dtype)
    keys = list(attention_keys)
    values = list(attention_values)
    histories = list(convolution_history)
    for index, kind in enumerate(cfg.layer_types):
        hidden, updated = _prefill_block(
            hidden,
            model.layer_weights(parameters, index),
            cfg,
            kind,
            BlockCache(keys[index], values[index], histories[index]),
            valid_length,
            attention_backend,
        )
        keys[index], values[index], histories[index] = updated
    hidden = normalization.rms_norm(
        hidden, parameters["model.embedding_norm.weight"], cfg.norm_eps
    )
    last_hidden = hidden[:, valid_length - 1, :]
    return (
        model.logits(last_hidden, parameters, cfg),
        tuple(keys),
        tuple(values),
        tuple(histories),
    )


_prefill_jit = jax.jit(
    _prefill_impl,
    static_argnames=(
        "cfg",
        "dtype",
        "capacity",
        "attention_backend",
    ),
)


def _decode_one_impl(
    parameters: types.Parameters,
    input_ids: jax.Array,
    attention_keys: tuple[jax.Array | None, ...],
    attention_values: tuple[jax.Array | None, ...],
    convolution_history: tuple[jax.Array | None, ...],
    position: jax.Array,
    cfg: model.Config,
    dtype: types.DType,
    attention_backend: str,
) -> tuple[
    jax.Array,
    tuple[jax.Array | None, ...],
    tuple[jax.Array | None, ...],
    tuple[jax.Array | None, ...],
]:
    hidden = parameters["model.embed_tokens.weight"][input_ids].astype(dtype)
    keys = list(attention_keys)
    values = list(attention_values)
    histories = list(convolution_history)
    for index, kind in enumerate(cfg.layer_types):
        hidden, updated = _decode_block(
            hidden,
            model.layer_weights(parameters, index),
            cfg,
            kind,
            BlockCache(keys[index], values[index], histories[index]),
            position,
            attention_backend,
        )
        keys[index], values[index], histories[index] = updated
    hidden = normalization.rms_norm(
        hidden, parameters["model.embedding_norm.weight"], cfg.norm_eps
    )
    return (
        model.logits(hidden[:, 0, :], parameters, cfg),
        tuple(keys),
        tuple(values),
        tuple(histories),
    )


_decode_one_jit = jax.jit(
    _decode_one_impl,
    static_argnames=("cfg", "dtype", "attention_backend"),
)


def prefill(
    parameters: types.Parameters,
    input_ids: jax.Array,
    state: DecodeState,
    cfg: model.Config,
    *,
    dtype: types.DType = jnp.bfloat16,
    attention_backend: str = "dense",
) -> tuple[jax.Array, DecodeState]:
    """Prefill a prompt and return only logits for its next token."""
    _check_dtype(dtype)
    _validate_state(state, cfg)
    if state.valid_length != 0:
        raise ValueError("Prefill requires a fresh state")
    if input_ids.ndim != 2:
        raise ValueError("Prefill input_ids must have shape [batch, tokens]")
    batch_size, token_count = input_ids.shape
    _check_model_length(cfg, token_count)
    if batch_size != _cache_batch(state):
        raise ValueError("Prompt batch does not match cache batch")
    prompt_capacity = _prompt_capacity(token_count, state.capacity)
    padded = jnp.pad(
        jnp.asarray(input_ids),
        ((0, 0), (0, prompt_capacity - token_count)),
    )
    logits, keys, values, histories = _prefill_jit(
        parameters,
        padded,
        state.attention_keys,
        state.attention_values,
        state.convolution_history,
        jnp.asarray(token_count, dtype=jnp.int32),
        cfg,
        dtype,
        state.capacity,
        attention_backend,
    )
    return logits, DecodeState(
        keys,
        values,
        histories,
        valid_length=token_count,
        capacity=state.capacity,
    )


def decode_one(
    parameters: types.Parameters,
    input_ids: jax.Array,
    state: DecodeState,
    cfg: model.Config,
    *,
    dtype: types.DType = jnp.bfloat16,
    attention_backend: str = "dense",
) -> tuple[jax.Array, DecodeState]:
    """Decode one token with a dynamic device-side cache position."""
    _check_dtype(dtype)
    _validate_state(state, cfg)
    if state.valid_length >= cfg.max_position_embeddings:
        raise ValueError(
            "Decode position exceeds model max_position_embeddings"
        )
    if state.valid_length >= state.capacity:
        raise DecodeOverflow(
            f"Cannot decode at position {state.valid_length}; "
            f"capacity is {state.capacity}"
        )
    if input_ids.ndim == 1:
        input_ids = input_ids[:, None]
    if input_ids.ndim != 2 or input_ids.shape[1] != 1:
        raise ValueError(
            "decode_one input_ids must have shape [batch] or [batch, 1]"
        )
    if input_ids.shape[0] != _cache_batch(state):
        raise ValueError("Decode batch does not match cache batch")
    position = jnp.asarray(state.valid_length, dtype=jnp.int32)
    logits, keys, values, histories = _decode_one_jit(
        parameters,
        input_ids,
        state.attention_keys,
        state.attention_values,
        state.convolution_history,
        position,
        cfg,
        dtype,
        attention_backend,
    )
    return logits, DecodeState(
        keys,
        values,
        histories,
        valid_length=state.valid_length + 1,
        capacity=state.capacity,
    )
