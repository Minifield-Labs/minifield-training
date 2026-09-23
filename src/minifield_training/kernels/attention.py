"""Causal attention kernels for dense and compact LFM2 batches."""

from collections.abc import Callable

import jax
from jax.experimental.pallas.ops.tpu import splash_attention
import jax.numpy as jnp
import numpy as np


def _dense_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    """Score, mask, softmax, and reduce over a broadcastable validity grid."""
    repeats = query.shape[2] // key.shape[2]
    key, value = jnp.repeat(key, repeats, axis=2), jnp.repeat(
        value, repeats, axis=2
    )
    scores = jnp.einsum(
        "bthd,bshd->bhts", query, key, precision=jax.lax.Precision.HIGHEST
    ) * (query.shape[-1] ** -0.5)
    scores = jnp.where(
        valid, scores.astype(jnp.float32), float(np.finfo(np.float32).min)
    )
    return jnp.einsum(
        "bhts,bshd->bthd",
        jax.nn.softmax(scores, axis=-1).astype(query.dtype),
        value,
        precision=jax.lax.Precision.HIGHEST,
    )


def _prefix_key_value(
    query: jax.Array,
    prefix_key: jax.Array,
    prefix_value: jax.Array,
    key: jax.Array,
    value: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Broadcast a shared prefix across the batch and prepend it to K/V."""
    batch = query.shape[0]
    if prefix_key.shape[0] != batch:
        if prefix_key.shape[0] != 1:
            raise ValueError("Shared prefix batch is incompatible")
        prefix_key = jnp.broadcast_to(
            prefix_key, (batch,) + prefix_key.shape[1:]
        )
        prefix_value = jnp.broadcast_to(
            prefix_value, (batch,) + prefix_value.shape[1:]
        )
    return (
        jnp.concatenate((prefix_key, key), axis=1),
        jnp.concatenate((prefix_value, value), axis=1),
    )


def _prefix_valid_mask(
    query: jax.Array,
    prefix_key: jax.Array,
    prefix_mask: jax.Array,
    suffix_mask: jax.Array,
) -> jax.Array:
    """Build the (batch, 1, query, prefix + suffix) prefix validity grid."""
    batch, query_length = query.shape[0], query.shape[1]
    positions = jnp.arange(query_length)
    causal = (positions[:, None] >= positions[None, :])[None, None, :, :]
    suffix_valid = causal & suffix_mask[:, None, None, :].astype(bool)
    prefix_valid = jnp.broadcast_to(
        prefix_mask[:, None, None, :].astype(bool),
        (batch, 1, query_length, prefix_key.shape[1]),
    )
    return jnp.concatenate((prefix_valid, suffix_valid), axis=-1)


def _packed_valid_mask(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array,
    segment_ids: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Build the packed validity grid and the active-token indicator.

    A query may attend to a key only when the key is causal, both tokens share
    one nonzero segment id, and both are mask-active. Inactive queries fall
    back to attending to themselves so every softmax row stays finite.
    """
    if query.shape != key.shape and key.shape != value.shape:
        raise ValueError("Packed attention query/KV shapes are incompatible")
    sequence = query.shape[1]
    positions = jnp.arange(sequence, dtype=jnp.int32)
    causal = (positions[:, None] >= positions[None, :])[None, :, :]
    active = (attention_mask > 0) & (segment_ids != 0)
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    valid = (
        causal
        & same_segment
        & (segment_ids[:, :, None] != 0)
        & active[:, None, :]
        & active[:, :, None]
    )
    self_only = positions[:, None] == positions[None, :]
    valid = valid | ((~active)[:, :, None] & self_only[None, :, :])
    return valid, active


def _padding_segment_ids(active: jax.Array) -> jax.Array:
    """Assign active keys segment 1 and each inactive key a private id.

    Segment ids keep per-row masks inside Splash's static sparsity path:
    queries all run in segment 1, so they see exactly the active keys, and
    every softmax row stays nonempty whenever the sequence starts active.
    """
    positions = jnp.arange(active.shape[-1], dtype=jnp.int32)
    result: jax.Array = jnp.where(
        active, jnp.int32(1), active.shape[-1] + positions
    )
    return result


def _packed_segment_ids(active: jax.Array, segment_ids: jax.Array) -> jax.Array:
    """Keep real segment ids for active tokens, isolate each inactive token."""
    positions = jnp.arange(segment_ids.shape[-1], dtype=jnp.int32)
    return jnp.where(active, segment_ids, segment_ids.max() + positions + 1)


def _round_up(length: int, multiple: int) -> int:
    return -(-length // multiple) * multiple


def _pad_to(x: jax.Array, length: int, axis: int) -> jax.Array:
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, length - x.shape[axis])
    return jnp.pad(x, tuple(pad))


def _splash_mha(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    make_mask: Callable[[int, int], "splash_attention.Mask"],
    query_segment_ids: jax.Array | None,
    key_segment_ids: jax.Array | None,
) -> jax.Array:
    """Run a single-device Splash kernel over batched ``bthd`` operands.

    Splash takes unbatched ``(heads, sequence, dim)`` operands, requires
    sequence lengths divisible by the kernel block size, and supports grouped
    KV heads natively. Sequences are zero-padded to block multiples; padded
    keys carry segment id 0, which no logical segment id uses, so they stay
    invisible to real queries. ``make_mask`` builds the static per-head mask
    over the padded ``(query, key)`` shape. Splash does not scale logits
    itself, so the query is multiplied by ``head_dim ** -0.5`` here.
    """
    block_sizes = splash_attention.BlockSizes
    blocks = block_sizes.get_default()  # type: ignore[no-untyped-call]
    query_length = query.shape[1]
    q_padded = _round_up(query_length, blocks.block_q)
    kv_padded = _round_up(key.shape[1], blocks.block_kv)
    mask = make_mask(q_padded, kv_padded)
    multi_head = splash_attention.MultiHeadMask([mask] * query.shape[2])
    kernel = splash_attention.make_splash_mha_single_device(multi_head)
    query = jnp.transpose(
        _pad_to(query * (query.shape[-1] ** -0.5), q_padded, 1),
        (0, 2, 1, 3),
    )
    key = jnp.transpose(_pad_to(key, kv_padded, 1), (0, 2, 1, 3))
    value = jnp.transpose(_pad_to(value, kv_padded, 1), (0, 2, 1, 3))
    if key_segment_ids is None:
        output = jax.vmap(kernel)(query, key, value)
    else:
        if query_segment_ids is None:
            query_segment_ids = key_segment_ids
        query_segment_ids = _pad_to(query_segment_ids, q_padded, 1)
        key_segment_ids = _pad_to(key_segment_ids, kv_padded, 1)

        def run(
            q: jax.Array,
            k: jax.Array,
            v: jax.Array,
            q_ids: jax.Array,
            kv_ids: jax.Array,
        ) -> jax.Array:
            output: jax.Array = kernel(
                q,
                k,
                v,
                segment_ids=splash_attention.SegmentIds(q=q_ids, kv=kv_ids),
            )
            return output

        output = jax.vmap(run)(
            query, key, value, query_segment_ids, key_segment_ids
        )
    output = jnp.transpose(output, (0, 2, 1, 3))
    return output[:, :query_length]


def dense_causal_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array,
) -> jax.Array:
    """Apply causal attention with a fully materialized score grid.

    Key and value heads are repeated to match the query head count, then each
    query attends to keys that are both causal and mask-active. Scores and
    softmax run in float32 with ``Precision.HIGHEST`` matmuls; probabilities
    are cast back to the query dtype before the value reduction.
    """
    positions = jnp.arange(query.shape[1])
    valid = (positions[:, None] >= positions[None, :])[None, None, :, :]
    valid &= attention_mask[:, None, None, :].astype(bool)
    return _dense_attention(query, key, value, valid)


def dense_prefix_causal_attention(
    query: jax.Array,
    prefix_key: jax.Array,
    prefix_value: jax.Array,
    key: jax.Array,
    value: jax.Array,
    prefix_mask: jax.Array,
    suffix_mask: jax.Array,
) -> jax.Array:
    """Attend suffix queries to a shared prefix and their own causal suffix.

    Every valid prefix key precedes the caller's causal suffix keys, so each
    query sees the same key set in the same order as one uninterrupted
    sequence.
    """
    key, value = _prefix_key_value(query, prefix_key, prefix_value, key, value)
    valid = _prefix_valid_mask(query, prefix_key, prefix_mask, suffix_mask)
    return _dense_attention(query, key, value, valid)


def dense_packed_causal_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array,
    segment_ids: jax.Array,
) -> jax.Array:
    """Apply causal attention confined to same-segment active tokens.

    Inactive query outputs are zeroed so padding never leaks into supervised
    paths.
    """
    valid, active = _packed_valid_mask(
        query, key, value, attention_mask, segment_ids
    )
    output = _dense_attention(query, key, value, valid[:, None, :, :])
    return output * active[:, :, None, None].astype(output.dtype)


def dense_cached_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    valid_length: jax.Array,
) -> jax.Array:
    """Attend one query to the valid prefix of a fixed-size KV cache.

    The query is intentionally non-causal: its cache index is already the
    current absolute position, so causal masking based on query index zero
    would hide every key after slot zero.
    """
    if query.shape[-1] != key.shape[-1] or key.shape != value.shape:
        raise ValueError("Cached attention query/KV shapes are incompatible")
    if query.shape[2] % key.shape[2]:
        raise ValueError("Query heads must be divisible by KV heads")
    positions = jnp.arange(key.shape[1])
    valid = positions[None, None, None, :] < valid_length.reshape(-1, 1, 1, 1)
    return _dense_attention(query, key, value, valid)


def cudnn_causal_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array,
) -> jax.Array:
    """Apply fused cuDNN causal attention to prefix-padded sequences.

    Active counts are summed from ``attention_mask`` and passed as sequence
    lengths, so each row's valid tokens must form a contiguous prefix.
    Causality comes from ``is_causal``, keeping the fused kernel without a
    materialized score mask.
    """
    lengths = jnp.sum(attention_mask, axis=1, dtype=jnp.int32)
    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        query_seq_lengths=lengths,
        key_value_seq_lengths=lengths,
        implementation="cudnn",
    )


def cudnn_prefix_causal_attention(
    query: jax.Array,
    prefix_key: jax.Array,
    prefix_value: jax.Array,
    key: jax.Array,
    value: jax.Array,
    prefix_mask: jax.Array,
    suffix_mask: jax.Array,
) -> jax.Array:
    """Attend suffix queries to a shared prefix and their own causal suffix.

    cuDNN accepts this offset-causal mask without materializing the full
    query-head by query-token by key-token score grid.
    """
    key, value = _prefix_key_value(query, prefix_key, prefix_value, key, value)
    valid = _prefix_valid_mask(query, prefix_key, prefix_mask, suffix_mask)
    return jax.nn.dot_product_attention(
        query, key, value, mask=valid, implementation="cudnn"
    )


def cudnn_packed_causal_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array,
    segment_ids: jax.Array,
) -> jax.Array:
    """Apply causal attention confined to same-segment active tokens.

    Inactive query outputs are zeroed so padding never leaks into supervised
    paths.
    """
    valid, active = _packed_valid_mask(
        query, key, value, attention_mask, segment_ids
    )
    output = jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=valid[:, None, :, :],
        implementation="cudnn",
    )
    return output * active[:, :, None, None].astype(output.dtype)


def cudnn_cached_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    valid_length: jax.Array,
) -> jax.Array:
    """Attend one query to the valid prefix of a fixed-size KV cache.

    The query is intentionally non-causal: its cache index is already the
    current absolute position, so causal masking based on query index zero
    would hide every key after slot zero.
    """
    if query.shape[-1] != key.shape[-1] or key.shape != value.shape:
        raise ValueError("Cached attention query/KV shapes are incompatible")
    if query.shape[2] % key.shape[2]:
        raise ValueError("Query heads must be divisible by KV heads")
    batch_size = query.shape[0]
    query_lengths = jnp.ones((batch_size,), dtype=jnp.int32)
    key_lengths = jnp.full((batch_size,), valid_length, dtype=jnp.int32)
    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        is_causal=False,
        query_seq_lengths=query_lengths,
        key_value_seq_lengths=key_lengths,
        implementation="cudnn",
    )


# ---------------
# Splash Attention is a TPU optimized attention impl like Flash Attention on
# NVIDIA GPUs. It streams smaller chunks of the attention matrices through
# the GPU to calculate the attention scores. GPUs are memory bound so streaming
# smaller chunks means you spend time calculating while you're waiting for
# more matrices to show up.


def splash_causal_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array,
) -> jax.Array:
    """Apply Splash causal attention on TPU to prefix-padded sequences.

    Causality is a static ``CausalMask`` so fully masked blocks are skipped;
    padding folds into key-side segment ids, which give active keys one
    shared segment and each inactive key a private one.
    """
    kv_ids = _padding_segment_ids(attention_mask > 0)
    query_ids = jnp.ones_like(kv_ids)
    return _splash_mha(
        query,
        key,
        value,
        lambda q_len, kv_len: splash_attention.CausalMask((q_len, kv_len)),
        query_ids,
        kv_ids,
    )


def splash_prefix_causal_attention(
    query: jax.Array,
    prefix_key: jax.Array,
    prefix_value: jax.Array,
    key: jax.Array,
    value: jax.Array,
    prefix_mask: jax.Array,
    suffix_mask: jax.Array,
) -> jax.Array:
    """Attend suffix queries to a shared prefix and their own causal suffix.

    A positive ``CausalMask`` offset makes every prefix key visible to every
    query while suffix keys stay causal. Per-row prefix and suffix masks fold
    into segment ids, keeping the kernel mask static.
    """
    key, value = _prefix_key_value(query, prefix_key, prefix_value, key, value)
    prefix_length = key.shape[1] - query.shape[1]
    active = jnp.concatenate((prefix_mask > 0, suffix_mask > 0), axis=1)
    kv_ids = _padding_segment_ids(active)
    query_ids = jnp.ones((query.shape[0], query.shape[1]), dtype=jnp.int32)
    return _splash_mha(
        query,
        key,
        value,
        lambda q_len, kv_len: splash_attention.CausalMask(
            (q_len, kv_len), offset=prefix_length
        ),
        query_ids,
        kv_ids,
    )


def splash_packed_causal_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array,
    segment_ids: jax.Array,
) -> jax.Array:
    """Apply causal attention confined to same-segment active tokens.

    Splash's ``segment_ids`` carry the packed layout directly: active tokens
    keep their real segment id and each inactive token gets a private id, so
    no softmax row is empty. Inactive query outputs are zeroed.
    """
    active = (attention_mask > 0) & (segment_ids != 0)
    ids = _packed_segment_ids(active, segment_ids)
    output = _splash_mha(
        query,
        key,
        value,
        lambda q_len, kv_len: splash_attention.CausalMask((q_len, kv_len)),
        ids,
        ids,
    )
    return output * active[:, :, None, None].astype(output.dtype)


_CAUSAL_BACKENDS = {
    "dense": dense_causal_attention,
    "cudnn": cudnn_causal_attention,
    "splash": splash_causal_attention,
}

_PREFIX_BACKENDS = {
    "dense": dense_prefix_causal_attention,
    "cudnn": cudnn_prefix_causal_attention,
    "splash": splash_prefix_causal_attention,
}

_PACKED_BACKENDS = {
    "dense": dense_packed_causal_attention,
    "cudnn": cudnn_packed_causal_attention,
    "splash": splash_packed_causal_attention,
}

_CACHED_BACKENDS = {
    "dense": dense_cached_attention,
    "cudnn": cudnn_cached_attention,
}


def causal_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array,
    *,
    backend: str,
) -> jax.Array:
    """Apply causal attention on ``dense``, ``cudnn``, or ``splash``."""
    try:
        return _CAUSAL_BACKENDS[backend](query, key, value, attention_mask)
    except KeyError:
        raise ValueError(
            f"Unsupported causal attention backend: {backend}"
        ) from None


def prefix_causal_attention(
    query: jax.Array,
    prefix_key: jax.Array,
    prefix_value: jax.Array,
    key: jax.Array,
    value: jax.Array,
    prefix_mask: jax.Array,
    suffix_mask: jax.Array,
    *,
    backend: str = "dense",
) -> jax.Array:
    """Attend suffix queries to a shared prefix and their own causal suffix."""
    try:
        return _PREFIX_BACKENDS[backend](
            query,
            prefix_key,
            prefix_value,
            key,
            value,
            prefix_mask,
            suffix_mask,
        )
    except KeyError:
        raise ValueError(
            f"Unsupported prefix attention backend: {backend}"
        ) from None


def packed_causal_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array,
    segment_ids: jax.Array,
    *,
    backend: str,
) -> jax.Array:
    """Apply causal attention confined to same-segment active tokens."""
    try:
        return _PACKED_BACKENDS[backend](
            query, key, value, attention_mask, segment_ids
        )
    except KeyError:
        raise ValueError(
            f"Unsupported packed attention backend: {backend}"
        ) from None


def cached_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    valid_length: jax.Array,
    *,
    backend: str = "cudnn",
) -> jax.Array:
    """Attend one query to the valid prefix of a fixed-size KV cache.

    Single-query decode gains nothing from blocked kernels, so the dense path
    is the TPU implementation as well; only ``dense`` and ``cudnn`` are
    accepted.
    """
    try:
        return _CACHED_BACKENDS[backend](query, key, value, valid_length)
    except KeyError:
        raise ValueError(
            f"Unsupported cached attention backend: {backend}"
        ) from None
