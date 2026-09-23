"""Independent output, cast, gradient and broadcast contracts for rms_norm."""

from collections.abc import Iterator
import math

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from minifield_training.kernels import normalization

FloatArray = npt.NDArray[np.float64]


@pytest.fixture(autouse=True)
def _cpu_device() -> Iterator[None]:
    """Place each case on CPU and restore the preceding device context."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _reference_output(
    x: FloatArray, weight: FloatArray, eps: float
) -> FloatArray:
    """Evaluate the same normalization contract independently in FP64."""
    scale = (np.mean(x * x, axis=-1, keepdims=True) + eps) ** -0.5
    return np.asarray(x * scale * weight, dtype=np.float64)


def _reference_input_grad(
    x: FloatArray, weight: FloatArray, eps: float
) -> FloatArray:
    """Differentiate the output sum against the input in FP64.

    For each final-axis row with inverse RMS ``r``, the derivative is
    ``weight * r - x * r**3 * sum(weight * x) / width``.
    """
    scale = (np.mean(x * x, axis=-1, keepdims=True) + eps) ** -0.5
    dot = np.sum(weight * x, axis=-1, keepdims=True)
    return np.asarray(
        weight * scale - x * scale**3 * dot / x.shape[-1], dtype=np.float64
    )


def _reference_weight_grad(
    x: FloatArray, weight: FloatArray, eps: float
) -> FloatArray:
    """Sum normalized inputs over the broadcast leading axes in FP64."""
    scale = (np.mean(x * x, axis=-1, keepdims=True) + eps) ** -0.5
    normalized = x * scale
    leading = tuple(range(normalized.ndim - weight.ndim))
    return np.asarray(normalized.sum(axis=leading), dtype=np.float64)


def _forward_and_grads(
    x: jax.Array, weight: jax.Array, eps: float, compiled: bool
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return output and output-sum gradients, eager or under ``jax.jit``."""

    def objective(
        inputs: jax.Array, weights: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        output = normalization.rms_norm(inputs, weights, eps)
        return output.sum(), output

    function = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)
    if compiled:
        function = jax.jit(function)
    output: jax.Array
    grad_x: jax.Array
    grad_weight: jax.Array
    (_, output), (grad_x, grad_weight) = function(x, weight)
    return output, grad_x, grad_weight


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_binary_exact_fixture_output_and_gradients(
    dtype: type[np.generic], compiled: bool
) -> None:
    """x=[3,4], weight=[2,-3], eps=3.5 is binary-exact in each dtype."""
    x = jnp.array([3.0, 4.0], dtype)
    weight = jnp.array([2.0, -3.0], dtype)
    output, grad_x, grad_weight = _forward_and_grads(x, weight, 3.5, compiled)
    np.testing.assert_array_equal(np.asarray(output), np.array([1.5, -3.0]))
    np.testing.assert_array_equal(
        np.asarray(grad_x), np.array([0.640625, -0.5625])
    )
    np.testing.assert_array_equal(
        np.asarray(grad_weight), np.array([0.75, 1.0])
    )
    assert output.shape == x.shape == (2,)
    assert output.dtype == grad_x.dtype == x.dtype == dtype
    assert grad_weight.dtype == weight.dtype


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_mixed_low_precision_input_with_fp32_weight(
    dtype: type[np.generic], compiled: bool
) -> None:
    """Low-precision input with FP32 weight keeps each argument's grad dtype."""
    x = jnp.array([3.0, 4.0], dtype)
    weight = jnp.array([2.0, -3.0], jnp.float32)
    output, grad_x, grad_weight = _forward_and_grads(
        x, weight, 3.5, compiled=compiled
    )
    np.testing.assert_array_equal(np.asarray(output), np.array([1.5, -3.0]))
    np.testing.assert_array_equal(
        np.asarray(grad_x), np.array([0.640625, -0.5625])
    )
    np.testing.assert_array_equal(
        np.asarray(grad_weight), np.array([0.75, 1.0])
    )
    assert output.dtype == grad_x.dtype == dtype
    assert grad_weight.dtype == jnp.float32


def test_rank_two_batch_matches_fp64_oracle() -> None:
    """A rank-2 batch with channel weights matches the FP64 formula."""
    x64 = np.array(
        [[0.375, -1.25, 2.5, -0.875], [1.125, 0.625, -0.5, 3.0]],
        dtype=np.float64,
    )
    weight64 = np.array([0.5, -0.25, 1.25, 2.0], dtype=np.float64)
    x = jnp.asarray(x64.astype(np.float32))
    weight = jnp.asarray(weight64.astype(np.float32))
    output, grad_x, grad_weight = _forward_and_grads(
        x, weight, 0.25, compiled=False
    )
    np.testing.assert_allclose(
        np.asarray(output),
        _reference_output(x64, weight64, 0.25),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(grad_x),
        _reference_input_grad(x64, weight64, 0.25),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(grad_weight),
        _reference_weight_grad(x64, weight64, 0.25),
        rtol=2e-6,
        atol=2e-6,
    )
    assert output.shape == x.shape == grad_x.shape
    assert grad_weight.shape == weight.shape


def test_rank_three_batch_matches_fp64_oracle() -> None:
    """A rank-3 batch reduces the weight gradient across leading axes."""
    x64 = np.arange(24, dtype=np.float64).reshape(2, 3, 4) * 0.125 - np.array(
        [1.5, 0.625, -0.25, 0.875]
    )
    weight64 = np.array([-1.25, 0.5, 2.0, -0.75], dtype=np.float64)
    x = jnp.asarray(x64.astype(np.float32))
    weight = jnp.asarray(weight64.astype(np.float32))
    output, grad_x, grad_weight = _forward_and_grads(
        x, weight, 0.5, compiled=True
    )
    np.testing.assert_allclose(
        np.asarray(output),
        _reference_output(x64, weight64, 0.5),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(grad_x),
        _reference_input_grad(x64, weight64, 0.5),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(grad_weight),
        _reference_weight_grad(x64, weight64, 0.5),
        rtol=2e-6,
        atol=2e-6,
    )
    assert output.shape == x.shape == (2, 3, 4)
    assert grad_weight.shape == (4,)


def test_zero_row_output_and_gradients() -> None:
    """Zero row: zero output, weight/sqrt(eps) input grad, zero weight grad."""
    x = jnp.zeros((3,), jnp.float32)
    weight = jnp.array([2.0, -3.0, 0.5], jnp.float32)
    output, grad_x, grad_weight = _forward_and_grads(
        x, weight, 0.5, compiled=False
    )
    np.testing.assert_array_equal(np.asarray(output), np.zeros(3))
    np.testing.assert_allclose(
        np.asarray(grad_x),
        np.asarray(weight) / math.sqrt(0.5),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(np.asarray(grad_weight), np.zeros(3))


@pytest.mark.parametrize("compiled", [False, True])
def test_fp32_accumulation_survives_fp16_square_overflow(
    compiled: bool,
) -> None:
    """x=[256,256] FP16, eps=0 returns [1,1]; FP16 squaring would overflow."""
    x = jnp.array([256.0, 256.0], jnp.float16)
    weight = jnp.ones((2,), jnp.float16)
    function = (
        jax.jit(normalization.rms_norm) if compiled else normalization.rms_norm
    )
    output = function(x, weight, 0.0)
    np.testing.assert_array_equal(
        np.asarray(output), np.ones(2, dtype=np.float16)
    )
    assert output.dtype == jnp.float16


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("dtype", "weight_value", "expected"),
    [
        (jnp.float16, 1.3, np.array([0.75, 1.5])),
        (jnp.bfloat16, 1.75, np.array([1.015625, 2.03125])),
    ],
    ids=["float16", "bfloat16"],
)
def test_normalized_values_cast_before_weight_multiply(
    dtype: type[np.generic],
    weight_value: float,
    expected: FloatArray,
    compiled: bool,
) -> None:
    """Normalized FP32 values round to ``x.dtype`` before weight multiply."""
    x = jnp.array([1.0, 2.0], dtype)
    weight = jnp.array([weight_value, weight_value], jnp.float32)
    function = (
        jax.jit(normalization.rms_norm) if compiled else normalization.rms_norm
    )
    output = function(x, weight, 0.5)
    np.testing.assert_array_equal(np.asarray(output), expected)
    assert output.dtype == dtype


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (jnp.float16, np.array([0.974609375, 1.2998046875])),
        (jnp.bfloat16, np.array([0.97265625, 1.296875])),
    ],
    ids=["float16", "bfloat16"],
)
def test_weight_casts_to_input_dtype_before_multiply(
    dtype: type[np.generic], expected: FloatArray, compiled: bool
) -> None:
    """The FP32 weight rounds to ``x.dtype`` before the final multiply."""
    x = jnp.array([3.0, 4.0], dtype)
    weight = jnp.array([1.3, 1.3], jnp.float32)
    function = (
        jax.jit(normalization.rms_norm) if compiled else normalization.rms_norm
    )
    output = function(x, weight, 3.5)
    np.testing.assert_array_equal(np.asarray(output), expected)
    assert output.dtype == dtype


def test_scalar_weight_broadcasts_and_reduces_its_gradient() -> None:
    """A scalar weight broadcasts over every element and sums its gradient."""
    x64 = np.arange(24, dtype=np.float64).reshape(2, 3, 4) * 0.125 - 1.5
    weight64 = np.asarray(2.0, dtype=np.float64)
    x = jnp.asarray(x64.astype(np.float32))
    weight = jnp.asarray(weight64.astype(np.float32))
    output, grad_x, grad_weight = _forward_and_grads(
        x, weight, 0.5, compiled=False
    )
    np.testing.assert_allclose(
        np.asarray(output),
        _reference_output(x64, weight64, 0.5),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(grad_x),
        _reference_input_grad(x64, weight64, 0.5),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(grad_weight),
        _reference_weight_grad(x64, weight64, 0.5),
        rtol=2e-6,
        atol=2e-6,
    )
    assert output.shape == x.shape == (2, 3, 4)
    assert grad_weight.shape == ()


@pytest.mark.parametrize("compiled", [False, True])
def test_weight_broadcast_expands_output_and_reduces_input_gradient(
    compiled: bool,
) -> None:
    """Expanded weight axes repeat outputs and sum into input gradients."""
    x = jnp.array([3.0, 4.0], jnp.float32)
    weight = jnp.array([[2.0, -3.0], [-1.0, 4.0]], jnp.float32)
    output, grad_x, grad_weight = _forward_and_grads(x, weight, 3.5, compiled)
    np.testing.assert_array_equal(
        np.asarray(output), np.array([[1.5, -3.0], [-0.75, 4.0]])
    )
    np.testing.assert_array_equal(
        np.asarray(grad_x), np.array([0.0859375, 0.03125])
    )
    np.testing.assert_array_equal(
        np.asarray(grad_weight), np.array([[0.75, 1.0], [0.75, 1.0]])
    )
    assert output.shape == (2, 2)
    assert grad_x.shape == x.shape == (2,)
    assert grad_weight.shape == weight.shape == (2, 2)
    assert output.dtype == grad_x.dtype == grad_weight.dtype == jnp.float32


def test_incompatible_weight_broadcast_raises_jax_error() -> None:
    """Ordinary JAX broadcasting failure is preserved, not sanitized."""
    with pytest.raises(TypeError, match="broadcasting"):
        normalization.rms_norm(
            jnp.ones((2,), jnp.float32),
            jnp.ones((3,), jnp.float32),
            1.0,
        )


def test_nan_input_propagates_to_the_whole_row() -> None:
    """A NaN element poisons the row's inverse RMS and output."""
    output = normalization.rms_norm(
        jnp.array([np.nan, 1.0], jnp.float32),
        jnp.ones((2,), jnp.float32),
        0.5,
    )
    assert bool(jnp.isnan(output).all())
