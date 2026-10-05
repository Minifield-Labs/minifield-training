"""Independent analytical CPU checks for the encoder's noncausal kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from minifield_training.kernels import attention as causal
from minifield_training.kernels import bidirectional
from minifield_training.kernels import types


@pytest.mark.parametrize("mask_dtype", [jnp.int32, jnp.bool_])
def test_uniform_attention_and_gradient(mask_dtype: types.DType) -> None:
    """Zero queries/keys average valid values with equal input gradients."""
    query = jnp.zeros((1, 3, 2, 2))
    key = jnp.zeros((1, 3, 1, 2))
    values = jnp.asarray([[[[2.0, 4.0]], [[6.0, 8.0]], [[99.0, 99.0]]]])
    mask = jnp.asarray([[1, 1, 0]], dtype=mask_dtype)
    output = bidirectional.attention(query, key, values, mask)
    np.testing.assert_allclose(output, np.broadcast_to([4, 6], (1, 3, 2, 2)))
    gradient = jax.grad(
        lambda value: bidirectional.attention(query, key, value, mask).sum()
    )(values)
    np.testing.assert_allclose(gradient, [[[[3, 3]], [[3, 3]], [[0, 0]]]])
    empty = bidirectional.attention(query, key, values, jnp.zeros_like(mask))
    np.testing.assert_array_equal(empty, np.zeros((1, 3, 2, 2)))


def test_centered_convolution_output_and_gradient() -> None:
    """Asymmetric taps expose direction, padding, and boundary gradients."""
    values = jnp.asarray([[[1.0], [2.0], [3.0], [4.0]]])
    taps = jnp.asarray([[10.0, 100.0, 1000.0]])
    actual = jax.jit(bidirectional.centered_convolution)(values, taps)
    np.testing.assert_array_equal(actual.ravel(), [2100, 3210, 4320, 430])
    gradient = jax.grad(
        lambda x: bidirectional.centered_convolution(x, taps).sum()
    )(values)
    np.testing.assert_array_equal(gradient.ravel(), [110, 1110, 1110, 1100])


def test_packed_attention_averages_within_each_segment() -> None:
    """Zero queries/keys average values only inside their own segment."""
    query = jnp.zeros((1, 6, 2, 2))
    key = jnp.zeros((1, 6, 1, 2))
    values = jnp.asarray(
        [[2.0, 4.0], [6.0, 8.0], [1.0, 1.0], [2.0, 2.0], [6.0, 3.0], [99, 99]]
    ).reshape(1, 6, 1, 2)
    segments = jnp.asarray([[1, 1, 2, 2, 2, 0]])
    output = bidirectional.attention(
        query, key, values, segments != 0, segments
    )
    expected = [[4, 6]] * 2 + [[3, 2]] * 3 + [[0, 0]]
    np.testing.assert_allclose(
        output,
        np.broadcast_to(np.asarray(expected)[None, :, None], (1, 6, 2, 2)),
    )
    gradient = jax.grad(
        lambda value: bidirectional.attention(
            query, key, value, segments != 0, segments
        ).sum()
    )(values)
    # Each of 2 heads spreads 1 over its segment's queries: 2 * n / n = 2.
    np.testing.assert_allclose(
        gradient.reshape(6, 2), [[2, 2]] * 5 + [[0, 0]], rtol=1e-6
    )


def test_packed_convolution_stops_at_segment_boundaries() -> None:
    """Packed [1, 2] and [3, 4] match their separately padded outputs."""
    values = jnp.asarray([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    taps = jnp.asarray([[10.0, 100.0, 1000.0]])
    segments = jnp.asarray([[1, 1, 2, 2, 0]])
    actual = jax.jit(bidirectional.centered_convolution)(values, taps, segments)
    np.testing.assert_array_equal(actual.ravel(), [2100, 210, 4300, 430, 0])
    gradient = jax.grad(
        lambda x: bidirectional.centered_convolution(x, taps, segments).sum()
    )(values)
    np.testing.assert_array_equal(gradient.ravel(), [110, 1100, 110, 1100, 0])


def test_local_window_averages_only_nearby_keys_in_the_segment() -> None:
    """Radius 1 averages each token's same-segment neighbours and itself."""
    query = jnp.zeros((1, 6, 2, 2))
    key = jnp.zeros((1, 6, 1, 2))
    values = jnp.asarray([1.0, 2.0, 4.0, 8.0, 16.0, 99.0]).reshape(1, 6, 1, 1)
    values = jnp.broadcast_to(values, (1, 6, 1, 2))
    segments = jnp.asarray([[1, 1, 1, 2, 2, 0]])
    output = bidirectional.attention(
        query, key, values, segments != 0, segments, window=2
    )
    expected = [1.5, 7 / 3, 3.0, 12.0, 12.0, 0.0]
    np.testing.assert_allclose(output[0, :, 0, 0], expected, rtol=1e-6)


@pytest.mark.parametrize("window", [3, 0, -2])
def test_window_must_be_a_positive_even_width(window: int) -> None:
    """Odd or empty windows have no symmetric radius."""
    operand = jnp.zeros((1, 2, 1, 2))
    with pytest.raises(ValueError, match="even width"):
        bidirectional.attention(
            operand, operand, operand, jnp.ones((1, 2)), window=window
        )


def _random_operands(
    length: int, seed: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array]:
    generator = np.random.default_rng(seed)
    return tuple(  # type: ignore[return-value]
        jnp.asarray(generator.normal(size=shape), jnp.float32)
        for shape in ((2, length, 4, 8), (2, length, 2, 8), (2, length, 2, 8))
    )


@pytest.mark.parametrize("window", [None, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_splash_interpret_matches_dense(
    window: int | None, packed: bool
) -> None:
    """Splash reproduces dense output for padded and packed rows, GQA heads."""
    length = 12
    query, key, value = _random_operands(length)
    segments = jnp.asarray(
        [[1] * 5 + [2] * 4 + [0] * 3, [1] * 8 + [2] * 3 + [0]]
    )
    mask = segments != 0
    segment_ids = segments if packed else None
    expected = bidirectional.attention(
        query, key, value, mask, segment_ids, window=window
    )
    output = jax.jit(
        lambda q, k, v: causal.splash_bidirectional_attention(
            q, k, v, mask, segment_ids, window=window, interpret=True
        )
    )(query, key, value)
    # Padded queries are never read: their keys are masked and the encoder
    # zeroes their outputs, so dense may leave them unspecified.
    active = np.asarray(mask)[:, :, None, None]
    np.testing.assert_allclose(
        np.asarray(output) * active,
        np.asarray(expected) * active,
        rtol=1e-5,
        atol=1e-5,
    )
