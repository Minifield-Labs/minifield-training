"""SwiGLU values and derivatives against an independent FP64 calculation."""

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from minifield_training.layers import feed_forward

FloatArray = npt.NDArray[np.float64]


@pytest.fixture(autouse=True)
def _cpu_device() -> Iterator[None]:
    """Keep numerical evidence explicitly on CPU."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _reference(x: FloatArray, weights: list[FloatArray]) -> FloatArray:
    """Evaluate RMS pre-norm and sigmoid-gated projections in NumPy FP64."""
    norm, gate, up, down = weights
    hidden = x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 0.01) * norm
    gate_value = hidden @ gate.T
    activation = gate_value / (1 + np.exp(-gate_value)) * (hidden @ up.T)
    return np.asarray(x + activation @ down.T, dtype=np.float64)


@pytest.mark.parametrize("compiled", [False, True])
def test_swiglu_output_and_all_argument_gradients(compiled: bool) -> None:
    """Every input and master derivative matches FP64 central differences."""
    rng = np.random.default_rng(12)
    arrays = [
        rng.normal(0, 0.3, shape)
        for shape in ((1, 2, 2), (2,), (3, 2), (3, 2), (2, 3))
    ]
    arrays[1] += 1
    x, *weights = [value.astype(np.float64) for value in arrays]
    inputs = jnp.asarray(x, dtype=jnp.float32)
    pack = feed_forward.FeedForwardWeights(
        *(jnp.asarray(value, dtype=jnp.float32) for value in weights)
    )

    def objective(
        value: jax.Array, params: feed_forward.FeedForwardWeights
    ) -> tuple[jax.Array, jax.Array]:
        output = feed_forward.swiglu_ffn(value, params, eps=0.01)
        return jnp.square(output).sum(), output

    evaluate = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)
    if compiled:
        evaluate = jax.jit(evaluate)
    (_, output), (grad_x, grad_weights) = evaluate(inputs, pack)
    np.testing.assert_allclose(
        output, _reference(x, weights), rtol=3e-6, atol=2e-7
    )
    step = 1e-5
    for index, actual in enumerate([grad_x, *grad_weights]):
        expected = np.zeros_like(arrays[index])
        for coordinate in np.ndindex(expected.shape):
            perturbations = [value.copy() for value in arrays]
            perturbations[index][coordinate] += step
            upper = np.square(
                _reference(perturbations[0], perturbations[1:])
            ).sum()
            perturbations[index][coordinate] -= 2 * step
            lower = np.square(
                _reference(perturbations[0], perturbations[1:])
            ).sum()
            expected[coordinate] = (upper - lower) / (2 * step)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
        assert actual.dtype == jnp.float32
    assert output.shape == inputs.shape
