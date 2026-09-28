"""Independent analytical CPU checks for the encoder's noncausal kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
