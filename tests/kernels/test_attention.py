"""Independent output and backend parity contracts for attention kernels."""

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from minifield_training.kernels import attention

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_BATCH, _LENGTH, _PREFIX, _HEADS, _KV_HEADS, _DIM = 2, 6, 3, 4, 2, 8


@pytest.fixture(autouse=True)
def _cpu_device() -> Iterator[None]:
    """Place each case on CPU and restore the preceding device context."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _arrays(*shapes: tuple[int, ...], seed: int = 0) -> list[jax.Array]:
    """Draw deterministic float32 operands from a seeded generator."""
    generator = np.random.default_rng(seed)
    return [
        jnp.asarray(generator.normal(size=shape), jnp.float32)
        for shape in shapes
    ]


def _suffix_mask() -> jax.Array:
    """Two rows whose active tokens form a contiguous prefix."""
    return jnp.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=jnp.int32)


def _oracle_attention(
    query: FloatArray,
    key: FloatArray,
    value: FloatArray,
    valid: BoolArray,
) -> FloatArray:
    """Evaluate masked scaled-dot-product attention independently in FP64."""
    repeats = query.shape[2] // key.shape[2]
    key = np.repeat(key, repeats, axis=2)
    value = np.repeat(value, repeats, axis=2)
    scores = np.einsum("bthd,bshd->bhts", query, key) * query.shape[-1] ** -0.5
    scores = np.where(valid, scores, np.finfo(np.float64).min)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.sum(
        np.exp(shifted), axis=-1, keepdims=True
    )
    return np.asarray(
        np.einsum("bhts,bshd->bthd", probabilities, value), dtype=np.float64
    )


def _causal_valid(suffix_mask: FloatArray) -> BoolArray:
    """Causal-and-active-key validity over the suffix grid."""
    positions = np.arange(suffix_mask.shape[1])
    causal = positions[:, None] >= positions[None, :]
    return causal[None, None] & (suffix_mask[:, None, None, :] > 0)


def _prefix_valid(
    prefix_mask: FloatArray, suffix_mask: FloatArray
) -> BoolArray:
    """Prefix-mask validity prepended to the causal suffix validity."""
    batch, query_length = suffix_mask.shape
    prefix_length = prefix_mask.shape[1]
    prefix = np.broadcast_to(
        prefix_mask[:, None, None, :] > 0,
        (batch, 1, query_length, prefix_length),
    )
    return np.concatenate((prefix, _causal_valid(suffix_mask)), axis=-1)


def _packed_valid(
    attention_mask: FloatArray, segment_ids: FloatArray
) -> tuple[BoolArray, BoolArray]:
    """Same-segment causal validity with self-fallback for inactive rows."""
    sequence = segment_ids.shape[1]
    positions = np.arange(sequence)
    causal = positions[:, None] >= positions[None, :]
    active = (attention_mask > 0) & (segment_ids != 0)
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    valid = (
        causal[None]
        & same_segment
        & (segment_ids[:, :, None] != 0)
        & active[:, None, :]
        & active[:, :, None]
    )
    self_only = positions[:, None] == positions[None, :]
    return valid | ((~active)[:, :, None] & self_only[None]), active


@pytest.mark.parametrize("compiled", [False, True])
def test_dense_causal_matches_fp64_oracle(compiled: bool) -> None:
    """Dense causal output matches the FP64 masked-softmax formula."""
    query, key, value = _arrays(
        (_BATCH, _LENGTH, _HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
    )
    mask = _suffix_mask()
    function = (
        jax.jit(attention.dense_causal_attention)
        if compiled
        else attention.dense_causal_attention
    )
    output = function(query, key, value, mask)
    expected = _oracle_attention(
        np.asarray(query, np.float64),
        np.asarray(key, np.float64),
        np.asarray(value, np.float64),
        _causal_valid(np.asarray(mask, np.float64)),
    )
    np.testing.assert_allclose(
        np.asarray(output), expected, rtol=2e-6, atol=2e-6
    )
    assert output.shape == query.shape


def test_splash_causal_interpret_matches_dense() -> None:
    """Splash in interpret mode reproduces the dense causal output."""
    query, key, value = _arrays(
        (_BATCH, _LENGTH, _HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
    )
    mask = _suffix_mask()
    expected = attention.dense_causal_attention(query, key, value, mask)
    output = attention.splash_causal_attention(
        query, key, value, mask, interpret=True
    )
    np.testing.assert_allclose(
        np.asarray(output), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_cudnn_causal_xla_matches_dense_on_active_rows() -> None:
    """The fused XLA path reproduces dense rows inside each active prefix.

    Padded query rows carry no contract: ``query_seq_lengths`` leaves them
    unspecified, so only positions below each row's active count compare.
    """
    query, key, value = _arrays(
        (_BATCH, _LENGTH, _HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
    )
    mask = _suffix_mask()
    expected = np.asarray(
        attention.dense_causal_attention(query, key, value, mask)
    )
    output = np.asarray(
        attention.cudnn_causal_attention(
            query, key, value, mask, implementation="xla"
        )
    )
    lengths = np.asarray(mask).sum(axis=1)
    for row, length in enumerate(lengths):
        np.testing.assert_allclose(
            output[row, :length], expected[row, :length], rtol=1e-5, atol=1e-5
        )


def test_cudnn_causal_xla_matches_dense_exactly_when_full() -> None:
    """A fully active mask gives the fused path no unspecified rows."""
    query, key, value = _arrays(
        (_BATCH, _LENGTH, _HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
    )
    mask = jnp.ones((_BATCH, _LENGTH), dtype=jnp.int32)
    expected = attention.dense_causal_attention(query, key, value, mask)
    output = attention.cudnn_causal_attention(
        query, key, value, mask, implementation="xla"
    )
    np.testing.assert_allclose(
        np.asarray(output), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def _prefix_operands() -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Suffix operands plus a per-row shared prefix and both masks."""
    query, prefix_key, prefix_value, key, value = _arrays(
        (_BATCH, _LENGTH, _HEADS, _DIM),
        (_BATCH, _PREFIX, _KV_HEADS, _DIM),
        (_BATCH, _PREFIX, _KV_HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        seed=1,
    )
    prefix_mask = jnp.array([[1, 1, 0], [1, 0, 0]], dtype=jnp.int32)
    return (
        query,
        prefix_key,
        prefix_value,
        key,
        value,
        prefix_mask,
        _suffix_mask(),
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_dense_prefix_matches_fp64_oracle(compiled: bool) -> None:
    """Dense prefix output matches the FP64 oracle over prefix + suffix."""
    query, prefix_key, prefix_value, key, value, prefix_mask, suffix_mask = (
        _prefix_operands()
    )
    function = (
        jax.jit(attention.dense_prefix_causal_attention)
        if compiled
        else attention.dense_prefix_causal_attention
    )
    output = function(
        query, prefix_key, prefix_value, key, value, prefix_mask, suffix_mask
    )
    expected = _oracle_attention(
        np.asarray(query, np.float64),
        np.concatenate(
            (
                np.asarray(prefix_key, np.float64),
                np.asarray(key, np.float64),
            ),
            axis=1,
        ),
        np.concatenate(
            (
                np.asarray(prefix_value, np.float64),
                np.asarray(value, np.float64),
            ),
            axis=1,
        ),
        _prefix_valid(
            np.asarray(prefix_mask, np.float64),
            np.asarray(suffix_mask, np.float64),
        ),
    )
    np.testing.assert_allclose(
        np.asarray(output), expected, rtol=2e-6, atol=2e-6
    )
    assert output.shape == query.shape


def test_splash_prefix_interpret_matches_dense() -> None:
    """Splash in interpret mode reproduces the dense prefix output."""
    query, prefix_key, prefix_value, key, value, prefix_mask, suffix_mask = (
        _prefix_operands()
    )
    expected = attention.dense_prefix_causal_attention(
        query, prefix_key, prefix_value, key, value, prefix_mask, suffix_mask
    )
    output = attention.splash_prefix_causal_attention(
        query,
        prefix_key,
        prefix_value,
        key,
        value,
        prefix_mask,
        suffix_mask,
        interpret=True,
    )
    np.testing.assert_allclose(
        np.asarray(output), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_cudnn_prefix_xla_matches_dense() -> None:
    """The fused XLA path reproduces the dense prefix output."""
    query, prefix_key, prefix_value, key, value, prefix_mask, suffix_mask = (
        _prefix_operands()
    )
    expected = attention.dense_prefix_causal_attention(
        query, prefix_key, prefix_value, key, value, prefix_mask, suffix_mask
    )
    output = attention.cudnn_prefix_causal_attention(
        query,
        prefix_key,
        prefix_value,
        key,
        value,
        prefix_mask,
        suffix_mask,
        implementation="xla",
    )
    np.testing.assert_allclose(
        np.asarray(output), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def _packed_operands() -> (
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Packed operands whose rows mix two segments and inactive tails."""
    query, key, value = _arrays(
        (_BATCH, _LENGTH, _HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        seed=2,
    )
    mask = jnp.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]], dtype=jnp.int32)
    segments = jnp.array(
        [[1, 1, 1, 2, 2, 0], [1, 1, 2, 0, 0, 0]], dtype=jnp.int32
    )
    return query, key, value, mask, segments


@pytest.mark.parametrize("compiled", [False, True])
def test_dense_packed_matches_fp64_oracle(compiled: bool) -> None:
    """Dense packed output matches the FP64 oracle, zeroed off-segment."""
    query, key, value, mask, segments = _packed_operands()
    function = (
        jax.jit(attention.dense_packed_causal_attention)
        if compiled
        else attention.dense_packed_causal_attention
    )
    output = function(query, key, value, mask, segments)
    valid, active = _packed_valid(
        np.asarray(mask, np.float64), np.asarray(segments, np.float64)
    )
    expected = _oracle_attention(
        np.asarray(query, np.float64),
        np.asarray(key, np.float64),
        np.asarray(value, np.float64),
        valid[:, None, :, :],
    )
    expected *= active[:, :, None, None]
    np.testing.assert_allclose(
        np.asarray(output), expected, rtol=2e-6, atol=2e-6
    )
    assert output.shape == query.shape


def test_splash_packed_interpret_matches_dense() -> None:
    """Splash in interpret mode reproduces the dense packed output."""
    query, key, value, mask, segments = _packed_operands()
    expected = attention.dense_packed_causal_attention(
        query, key, value, mask, segments
    )
    output = attention.splash_packed_causal_attention(
        query, key, value, mask, segments, interpret=True
    )
    np.testing.assert_allclose(
        np.asarray(output), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_cudnn_packed_xla_matches_dense() -> None:
    """The fused XLA path reproduces the dense packed output."""
    query, key, value, mask, segments = _packed_operands()
    expected = attention.dense_packed_causal_attention(
        query, key, value, mask, segments
    )
    output = attention.cudnn_packed_causal_attention(
        query, key, value, mask, segments, implementation="xla"
    )
    np.testing.assert_allclose(
        np.asarray(output), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_dense_cached_matches_fp64_oracle() -> None:
    """Dense cached output attends to exactly the valid cache prefix."""
    query, key, value = _arrays(
        (_BATCH, 1, _HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        seed=3,
    )
    valid_length = jnp.array(4, dtype=jnp.int32)
    output = attention.dense_cached_attention(query, key, value, valid_length)
    positions = np.arange(_LENGTH)
    valid = positions[None, None, None, :] < 4
    expected = _oracle_attention(
        np.asarray(query, np.float64),
        np.asarray(key, np.float64),
        np.asarray(value, np.float64),
        np.broadcast_to(valid, (_BATCH, 1, 1, _LENGTH)),
    )
    np.testing.assert_allclose(
        np.asarray(output), expected, rtol=2e-6, atol=2e-6
    )
    assert output.shape == query.shape


def test_cudnn_cached_xla_matches_dense() -> None:
    """The fused XLA path reproduces the dense cached output."""
    query, key, value = _arrays(
        (_BATCH, 1, _HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        seed=3,
    )
    valid_length = jnp.array(4, dtype=jnp.int32)
    expected = attention.dense_cached_attention(query, key, value, valid_length)
    output = attention.cudnn_cached_attention(
        query, key, value, valid_length, implementation="xla"
    )
    np.testing.assert_allclose(
        np.asarray(output), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_dispatchers_route_to_dense_backend() -> None:
    """Each dispatcher forwards to the dense implementation unchanged."""
    query, key, value = _arrays(
        (_BATCH, _LENGTH, _HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
    )
    mask = _suffix_mask()
    np.testing.assert_array_equal(
        np.asarray(
            attention.causal_attention(query, key, value, mask, backend="dense")
        ),
        np.asarray(attention.dense_causal_attention(query, key, value, mask)),
    )
    (
        query,
        prefix_key,
        prefix_value,
        key,
        value,
        prefix_mask,
        suffix_mask,
    ) = _prefix_operands()
    np.testing.assert_array_equal(
        np.asarray(
            attention.prefix_causal_attention(
                query,
                prefix_key,
                prefix_value,
                key,
                value,
                prefix_mask,
                suffix_mask,
                backend="dense",
            )
        ),
        np.asarray(
            attention.dense_prefix_causal_attention(
                query,
                prefix_key,
                prefix_value,
                key,
                value,
                prefix_mask,
                suffix_mask,
            )
        ),
    )
    query, key, value, mask, segments = _packed_operands()
    np.testing.assert_array_equal(
        np.asarray(
            attention.packed_causal_attention(
                query, key, value, mask, segments, backend="dense"
            )
        ),
        np.asarray(
            attention.dense_packed_causal_attention(
                query, key, value, mask, segments
            )
        ),
    )
    query1 = query[:, :1]
    valid_length = jnp.array(4, dtype=jnp.int32)
    np.testing.assert_array_equal(
        np.asarray(
            attention.cached_attention(
                query1, key, value, valid_length, backend="dense"
            )
        ),
        np.asarray(
            attention.dense_cached_attention(query1, key, value, valid_length)
        ),
    )


def test_dispatchers_reject_unknown_backends() -> None:
    """Every dispatcher reports an unsupported backend as ``ValueError``."""
    query, key, value = _arrays(
        (_BATCH, _LENGTH, _HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
        (_BATCH, _LENGTH, _KV_HEADS, _DIM),
    )
    mask = _suffix_mask()
    with pytest.raises(ValueError, match="Unsupported causal"):
        attention.causal_attention(query, key, value, mask, backend="tpu")
    with pytest.raises(ValueError, match="Unsupported prefix"):
        attention.prefix_causal_attention(
            query,
            key[:, :_PREFIX],
            value[:, :_PREFIX],
            key,
            value,
            mask[:, :_PREFIX],
            mask,
            backend="tpu",
        )
    segments = jnp.ones((query.shape[0], query.shape[1]), dtype=jnp.int32)
    with pytest.raises(ValueError, match="Unsupported packed"):
        attention.packed_causal_attention(
            query, key, value, mask, segments, backend="tpu"
        )
    with pytest.raises(ValueError, match="Unsupported cached"):
        attention.cached_attention(
            query[:, :1], key, value, jnp.array(4, jnp.int32), backend="tpu"
        )
