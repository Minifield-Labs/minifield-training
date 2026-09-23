"""Independent block outputs, derivatives and boundary-state tests."""

from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from minifield_training.layers import attention
from minifield_training.layers import convolution
from minifield_training.layers import feed_forward

FloatArray = npt.NDArray[np.float64]


@pytest.fixture(autouse=True)
def _cpu_device() -> Iterator[None]:
    """Keep numerical evidence explicitly on CPU."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _inputs() -> jax.Array:
    """Supply four nonuniform tokens, with two query heads and one KV head."""
    return jnp.array(
        [
            [
                [0.2, -0.6, 0.3, 0.7],
                [0.8, 0.1, -0.5, 0.2],
                [-0.2, 0.9, 0.4, -0.3],
                [0.5, -0.7, 0.6, 0.1],
            ]
        ],
        dtype=jnp.float32,
    )


def _identity_ffn() -> feed_forward.FeedForwardWeights:
    """Keep the tested branch observable; SwiGLU has its own numerical tests."""
    return feed_forward.FeedForwardWeights(
        jnp.ones(4), jnp.zeros((3, 4)), jnp.zeros((3, 4)), jnp.zeros((4, 3))
    )


def _attention_weights() -> attention.AttentionWeights:
    """Make deterministic nonzero projections and unequal per-head gains."""
    rng = np.random.default_rng(29)
    query, key, value, out = (
        jnp.asarray(rng.normal(0, 0.3, shape), dtype=jnp.float32)
        for shape in ((4, 4), (2, 4), (2, 4), (4, 4))
    )
    return attention.AttentionWeights(
        jnp.array([1.0, 0.8, 1.2, 0.9]),
        query,
        key,
        value,
        out,
        jnp.array([0.9, 1.1]),
        jnp.array([1.2, 0.8]),
    )


def _conv_weights() -> convolution.ConvWeights:
    """Make gated channels and asymmetric oldest-to-newest taps."""
    rng = np.random.default_rng(37)
    return convolution.ConvWeights(
        jnp.array([1.0, 0.8, 1.2, 0.9]),
        jnp.asarray(rng.normal(0, 0.4, (12, 4)), dtype=jnp.float32),
        jnp.array(
            [
                [0.2, -0.1, 0.4],
                [0.3, 0.2, -0.5],
                [-0.4, 0.2, 0.1],
                [0.1, -0.3, 0.5],
            ]
        ),
        jnp.asarray(rng.normal(0, 0.4, (4, 4)), dtype=jnp.float32),
    )


def _rms(x: FloatArray, gain: jax.Array) -> FloatArray:
    """Evaluate RMS in FP64, independently of the shared JAX primitive."""
    return np.asarray(
        x
        / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 0.01)
        * np.asarray(gain, dtype=np.float64),
        dtype=np.float64,
    )


def _attention_reference(
    inputs: FloatArray, weights: attention.AttentionWeights
) -> FloatArray:
    """Evaluate causal GQA using token/head loops and 2D rotation matrices."""
    length = inputs.shape[1]
    hidden = _rms(inputs, weights.operator_norm)
    query = _rms(
        (hidden @ np.asarray(weights.query).T).reshape(1, length, 2, 2),
        weights.query_norm,
    )
    key = _rms(
        (hidden @ np.asarray(weights.key).T).reshape(1, length, 1, 2),
        weights.key_norm,
    )
    value = (hidden @ np.asarray(weights.value).T).reshape(1, length, 1, 2)
    for token in range(length):
        rotation = np.array(
            [[np.cos(token), -np.sin(token)], [np.sin(token), np.cos(token)]]
        )
        query[0, token] = query[0, token] @ rotation.T
        key[0, token] = key[0, token] @ rotation.T
    attended = np.zeros_like(query)
    for token in range(length):
        for head in range(2):
            scores = np.array(
                [
                    np.dot(query[0, token, head], key[0, source, 0])
                    / np.sqrt(2)
                    for source in range(token + 1)
                ]
            )
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            attended[0, token, head] = probabilities @ value[0, : token + 1, 0]
    return np.asarray(
        inputs + attended.reshape(inputs.shape) @ np.asarray(weights.out).T,
        dtype=np.float64,
    )


def _conv_reference(
    inputs: FloatArray, weights: convolution.ConvWeights
) -> FloatArray:
    """Use NumPy's signal convolution for each independently gated channel."""
    hidden = _rms(inputs, weights.operator_norm)
    projected = hidden @ np.asarray(weights.in_proj).T
    b_gate, c_gate, values = (
        projected[..., :4],
        projected[..., 4:8],
        projected[..., 8:],
    )
    mixed = np.empty_like(values)
    for row in range(inputs.shape[0]):
        for channel in range(inputs.shape[-1]):
            mixed[row, :, channel] = np.convolve(
                b_gate[row, :, channel] * values[row, :, channel],
                np.asarray(weights.conv)[channel, ::-1],
            )[: inputs.shape[1]]
    return np.asarray(
        inputs + (c_gate * mixed) @ np.asarray(weights.out).T, dtype=np.float64
    )


def _assert_input_derivative(
    function: Callable[[jax.Array], jax.Array],
    reference: Callable[[FloatArray], FloatArray],
) -> None:
    """Compare a nonuniform output probe's derivative to FP64 differences."""
    inputs = _inputs()
    x = np.asarray(inputs, dtype=np.float64)
    probe = np.linspace(-0.7, 1.0, x.size).reshape(x.shape)
    actual = jax.grad(lambda value: (function(value) * probe).sum())(inputs)
    expected = np.zeros_like(x)
    step = 1e-5
    for coordinate in np.ndindex(x.shape):
        upper, lower = x.copy(), x.copy()
        upper[coordinate] += step
        lower[coordinate] -= step
        expected[coordinate] = (
            (reference(upper) - reference(lower)) * probe
        ).sum() / (2 * step)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)


def test_attention_values_and_input_gradient() -> None:
    """Pre-norm, QK gains, RoPE, grouped heads and residual match NumPy."""
    weights = _attention_weights()

    def apply(x: jax.Array) -> jax.Array:
        return attention.attention_block(
            x,
            weights,
            _identity_ffn(),
            jnp.ones((1, 4), jnp.int32),
            head_dim=2,
            rope_theta=10000.0,
            eps=0.01,
            backend="dense",
        )

    expected = _attention_reference(
        np.asarray(_inputs(), dtype=np.float64), weights
    )
    np.testing.assert_allclose(
        jax.jit(apply)(_inputs()), expected, rtol=3e-6, atol=3e-7
    )
    _assert_input_derivative(apply, lambda x: _attention_reference(x, weights))


def test_convolution_values_and_input_gradient() -> None:
    """Gate ordering, tap orientation and residual match signal convolution."""
    weights = _conv_weights()

    def apply(x: jax.Array) -> jax.Array:
        return convolution.conv_block(
            x,
            weights,
            _identity_ffn(),
            jnp.ones((1, 4), jnp.int32),
            kernel_size=3,
            eps=0.01,
        )

    expected = _conv_reference(np.asarray(_inputs(), dtype=np.float64), weights)
    np.testing.assert_allclose(
        jax.jit(apply)(_inputs()), expected, rtol=3e-6, atol=3e-7
    )
    _assert_input_derivative(apply, lambda x: _conv_reference(x, weights))


def test_attention_prefix_suffix_and_cached_steps() -> None:
    """Two-token prefix and incremental cache yield the uninterrupted result."""
    x, weights, ffn = _inputs(), _attention_weights(), _identity_ffn()
    mask = jnp.ones((1, 2), jnp.int32)
    prefix, key, value = attention.attention_block_prefix(
        x[:, :2],
        weights,
        ffn,
        mask,
        head_dim=2,
        rope_theta=10000.0,
        eps=0.01,
        backend="dense",
    )
    suffix = attention.attention_block_suffix(
        x[:, 2:],
        weights,
        ffn,
        mask,
        jnp.array([2, 3]),
        key,
        value,
        mask,
        head_dim=2,
        rope_theta=10000.0,
        eps=0.01,
        backend="dense",
    )
    expected = _attention_reference(np.asarray(x, dtype=np.float64), weights)
    np.testing.assert_allclose(
        jnp.concatenate((prefix, suffix), axis=1),
        expected,
        rtol=3e-6,
        atol=3e-7,
    )
    prefill, cache_key, cache_value = attention.attention_block_prefill(
        x[:, :2],
        weights,
        ffn,
        mask,
        capacity=5,
        head_dim=2,
        rope_theta=10000.0,
        eps=0.01,
        backend="dense",
    )
    np.testing.assert_array_equal(cache_key[:, :2], key)
    np.testing.assert_array_equal(cache_value[:, :2], value)
    np.testing.assert_array_equal(cache_key[:, 2:], 0)
    outputs = [prefill]
    for position in range(2, 4):
        result, cache_key, cache_value = attention.attention_block_step(
            x[:, position : position + 1],
            weights,
            ffn,
            cache_key,
            cache_value,
            jnp.array(position),
            jnp.array(position + 1),
            head_dim=2,
            rope_theta=10000.0,
            eps=0.01,
            backend="dense",
        )
        outputs.append(result)
    np.testing.assert_allclose(
        jnp.concatenate(outputs, axis=1), expected, rtol=3e-6, atol=3e-7
    )
    np.testing.assert_array_equal(cache_value[:, 4:], 0)


def test_convolution_prefix_suffix_and_cached_steps() -> None:
    """Short prefixes zero-fill history; steps append each gated input."""
    x, weights, ffn = _inputs(), _conv_weights(), _identity_ffn()
    prefix, history = convolution.conv_block_prefix(
        x[:, :1],
        weights,
        ffn,
        jnp.ones((1, 1), jnp.int32),
        kernel_size=3,
        eps=0.01,
    )
    np.testing.assert_array_equal(history[:, :1], 0)
    suffix = convolution.conv_block_suffix(
        x[:, 1:],
        weights,
        ffn,
        jnp.ones((1, 3), jnp.int32),
        history,
        kernel_size=3,
        eps=0.01,
    )
    expected = _conv_reference(np.asarray(x, dtype=np.float64), weights)
    np.testing.assert_allclose(
        jnp.concatenate((prefix, suffix), axis=1),
        expected,
        rtol=3e-6,
        atol=3e-7,
    )
    outputs = [prefix]
    for position in range(1, 4):
        result, history = convolution.conv_block_step(
            x[:, position : position + 1],
            weights,
            ffn,
            history,
            eps=0.01,
        )
        outputs.append(result)
    np.testing.assert_allclose(
        jnp.concatenate(outputs, axis=1), expected, rtol=3e-6, atol=3e-7
    )
    hidden = _rms(np.asarray(x, dtype=np.float64), weights.operator_norm)
    gates = np.split(hidden @ np.asarray(weights.in_proj).T, 3, axis=-1)
    np.testing.assert_allclose(
        history, (gates[0] * gates[2])[:, -2:], rtol=3e-6, atol=3e-7
    )


def test_packed_blocks_restart_at_segment_boundaries() -> None:
    """Packed rows restart RoPE and history at each segment boundary."""
    x, ffn = _inputs(), _identity_ffn()
    mask = jnp.ones((1, 4), jnp.int32)
    segments = jnp.array([[1, 1, 2, 2]])
    attn_weights, conv_weights = _attention_weights(), _conv_weights()
    attended = attention.attention_block_packed(
        x,
        attn_weights,
        ffn,
        mask,
        segments,
        jnp.array([[0, 1, 0, 1]]),
        head_dim=2,
        rope_theta=10000.0,
        eps=0.01,
        backend="dense",
    )
    convolved = convolution.conv_block_packed(
        x,
        conv_weights,
        ffn,
        mask,
        segments,
        kernel_size=3,
        eps=0.01,
    )
    independent = [
        np.asarray(x[:, :2], dtype=np.float64),
        np.asarray(x[:, 2:], dtype=np.float64),
    ]
    expected_attention = np.concatenate(
        [_attention_reference(row, attn_weights) for row in independent], axis=1
    )
    expected_conv = np.concatenate(
        [_conv_reference(row, conv_weights) for row in independent], axis=1
    )
    np.testing.assert_allclose(
        attended, expected_attention, rtol=3e-6, atol=3e-7
    )
    np.testing.assert_allclose(convolved, expected_conv, rtol=3e-6, atol=3e-7)


def test_prefix_padding_cannot_enter_returned_state() -> None:
    """Nonzero padded hidden rows contribute neither cached KV nor history."""
    x, ffn = _inputs(), _identity_ffn()
    padded_mask = jnp.array([[1, 0, 0, 0]])
    _, key, value = attention.attention_block_prefix(
        x,
        _attention_weights(),
        ffn,
        padded_mask,
        head_dim=2,
        rope_theta=10000.0,
        eps=0.01,
        backend="dense",
    )
    np.testing.assert_array_equal(key[:, 1:], 0)
    np.testing.assert_array_equal(value[:, 1:], 0)
    weights = _conv_weights()
    _, history = convolution.conv_block_prefix(
        x,
        weights,
        ffn,
        padded_mask,
        kernel_size=3,
        eps=0.01,
    )
    first = _rms(np.asarray(x[:, :1], dtype=np.float64), weights.operator_norm)
    gates = np.split(first @ np.asarray(weights.in_proj).T, 3, axis=-1)
    np.testing.assert_array_equal(history[:, :1], 0)
    np.testing.assert_allclose(
        history[:, 1:], gates[0] * gates[2], rtol=3e-6, atol=3e-7
    )
