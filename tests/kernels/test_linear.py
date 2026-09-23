"""Independent projection and selected-token scoring contracts on CPU."""

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from minifield_training.kernels import linear
from minifield_training.kernels import selected_logits


@pytest.fixture(autouse=True)
def _cpu_device() -> Iterator[None]:
    """Keep numerical evidence explicitly on CPU."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.mark.parametrize("compiled", [False, True])
def test_projection_output_and_derivatives(compiled: bool) -> None:
    """Batched projection and both derivatives follow matrix calculus."""
    x = np.array([[[0.25, -0.5], [1.5, 0.75]]], dtype=np.float32)
    weight = np.array([[0.5, -0.25], [0.125, 0.75]], dtype=np.float32)
    probe = np.array([[[1.0, -2.0], [0.5, 3.0]]], dtype=np.float32)

    def objective(inputs: jax.Array, matrix: jax.Array) -> jax.Array:
        return (linear.full_linear(inputs, matrix) * probe).sum()

    project = jax.jit(linear.full_linear) if compiled else linear.full_linear
    derivative = jax.grad(objective, argnums=(0, 1))
    if compiled:
        derivative = jax.jit(derivative)
    output = project(jnp.asarray(x), jnp.asarray(weight))
    grad_x, grad_weight = derivative(jnp.asarray(x), jnp.asarray(weight))
    np.testing.assert_array_equal(output, x @ weight.T)
    np.testing.assert_array_equal(grad_x, probe @ weight)
    np.testing.assert_array_equal(
        grad_weight, probe.reshape(-1, 2).T @ x.reshape(-1, 2)
    )
    assert output.dtype == grad_x.dtype == grad_weight.dtype == jnp.float32


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_projection_casts_master_before_multiplication(
    dtype: type[np.generic],
) -> None:
    """Rounded masters cancel exactly before the dot product."""
    inputs = jnp.array([[1024.0, -1024.0]], dtype=dtype)
    weight = jnp.array([[1.0002, 1.0]], dtype=jnp.float32)
    result = linear.full_linear(inputs, weight)
    grad_x, grad_weight = jax.grad(
        lambda x, w: linear.full_linear(x, w).sum(), argnums=(0, 1)
    )(inputs, weight)
    np.testing.assert_array_equal(result, [[0.0]])
    np.testing.assert_array_equal(grad_x, [[1.0, 1.0]])
    np.testing.assert_array_equal(grad_weight, [[1024.0, -1024.0]])
    assert result.dtype == grad_x.dtype == dtype
    assert grad_weight.dtype == jnp.float32


def test_selected_scoring_output_and_derivatives() -> None:
    """Log-likelihood and derivatives match a NumPy softmax calculation."""
    hidden = np.array([[0.2, -0.4], [0.7, 0.3]], dtype=np.float32)
    head = np.array([[0.5, -0.2], [-0.1, 0.6], [0.8, 0.3]], np.float32)
    targets = jnp.array([2, 0], dtype=jnp.int32)
    logits = hidden.astype(np.float64) @ head.astype(np.float64).T
    probability = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    expected = np.log(probability[[0, 1], [2, 0]])
    slope = -probability
    slope[[0, 1], [2, 0]] += 1
    score = selected_logits.selected_hidden_log_probs
    output = jax.jit(score)(jnp.asarray(hidden), jnp.asarray(head), targets)
    grad_x, grad_head = jax.grad(
        lambda x, w: score(x, w, targets).sum(), argnums=(0, 1)
    )(jnp.asarray(hidden), jnp.asarray(head))
    np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(grad_x, slope @ head, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        grad_head, slope.T @ hidden, rtol=2e-6, atol=2e-7
    )
    assert output.dtype == jnp.float32


def test_selected_positions_keep_only_requested_rows() -> None:
    """Only requested rows contribute scores and receive hidden gradients."""
    hidden = np.arange(12, dtype=np.float32).reshape(2, 3, 2) / 10
    head = np.array([[0.5, -0.2], [-0.1, 0.6], [0.8, 0.3]], np.float32)
    positions = jnp.array([[1, 2], [0, 1]], dtype=jnp.int32)
    targets = jnp.array([2, 0], dtype=jnp.int32)
    selected = np.array([hidden[1, 2], hidden[0, 1]], dtype=np.float64)
    logits = selected @ head.T
    probability = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    expected_scores = np.log(probability[[0, 1], [2, 0]])
    slopes = -probability
    slopes[[0, 1], [2, 0]] += 1
    expected_gradient = np.zeros_like(hidden)
    expected_gradient[1, 2], expected_gradient[0, 1] = slopes @ head
    result = selected_logits.selected_token_log_probs(
        jnp.asarray(hidden), jnp.asarray(head), positions, targets
    )
    gradient = jax.grad(
        lambda x: selected_logits.selected_token_log_probs(
            x, jnp.asarray(head), positions, targets
        ).sum()
    )(jnp.asarray(hidden))
    np.testing.assert_allclose(result, expected_scores, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        gradient, expected_gradient, rtol=2e-6, atol=2e-7
    )
