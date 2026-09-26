"""Independent group-128 output and STE checks on CPU."""

import jax
import jax.numpy as jnp
import numpy as np

from minifield_training.kernels import quantization


def test_ternary_stored_scale_boundaries_and_ste() -> None:
    """FP16 scale is semantic; half ties quantize away from zero."""
    weight = np.zeros((1, 128), dtype=np.float32)
    weight[0, :7] = [2, -2, 1, -1, 0.99, -0.99, 0]
    master = jnp.asarray(weight)
    quantizer = quantization.Group128Quantizer("ternary-g128-absmax-f16-v1")
    expected = np.zeros_like(weight)
    expected[0, :7] = [2, -2, 2, -2, 0, 0, 0]
    for function in (quantizer.effective, jax.jit(quantizer.effective)):
        np.testing.assert_array_equal(function(master), expected)
        np.testing.assert_array_equal(
            jax.grad(lambda value, fn=function: jnp.sum(fn(value)))(master),
            np.ones_like(weight),
        )


def test_nf4_codebook_and_zero_group() -> None:
    """NF4 nearest levels and all-zero handling are distinct from INT4."""
    weight = np.zeros((2, 128), dtype=np.float32)
    weight[0, :4] = [2.0, -2.0, 0.0, 0.16]
    effective = jax.jit(
        quantization.Group128Quantizer("nf4-g128-absmax-f16-v1").effective
    )(jnp.asarray(weight))
    expected = np.zeros_like(weight)
    expected[0, :4] = [2.0, -2.0, 0.0, 2 * 0.07958029955625534]
    np.testing.assert_array_equal(effective, expected)


def test_fp16_underflow_group_decodes_zero() -> None:
    """An underflowed stored scale doesn't create a NaN forward value."""
    tiny = jnp.full((1, 128), 1e-9, dtype=jnp.float32)
    for kind in ("ternary-g128-absmax-f16-v1", "nf4-g128-absmax-f16-v1"):
        effective = quantization.Group128Quantizer(kind).effective
        for function in (effective, jax.jit(effective)):
            output = function(tiny)
            np.testing.assert_array_equal(output, np.zeros((1, 128)))
            np.testing.assert_array_equal(
                jax.grad(lambda value, fn=function: jnp.sum(fn(value)))(tiny),
                np.ones((1, 128)),
            )


def test_ternary_nonrepresentable_scale_and_nextafter_ties() -> None:
    """Stored FP16 scale determines the exact half-away threshold."""
    weight = np.zeros((1, 128), dtype=np.float32)
    weight[0, :7] = [
        1.0003,
        0.5,
        np.nextafter(np.float32(0.5), np.float32(0)),
        np.nextafter(np.float32(0.5), np.float32(1)),
        -0.5,
        np.nextafter(np.float32(-0.5), np.float32(0)),
        np.nextafter(np.float32(-0.5), np.float32(-1)),
    ]
    expected = np.zeros_like(weight)
    expected[0, :7] = [1, 1, 0, 1, -1, 0, -1]
    effective = quantization.Group128Quantizer(
        "ternary-g128-absmax-f16-v1"
    ).effective
    for function in (effective, jax.jit(effective)):
        np.testing.assert_array_equal(function(jnp.asarray(weight)), expected)


def test_nf4_nextafter_midpoints_independent_oracle() -> None:
    """Every FP32 midpoint neighbor selects the mathematically nearer level."""
    levels = np.asarray(
        (
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ),
        dtype=np.float32,
    )
    values = [np.float32(-1), np.float32(1)]
    for left, right in zip(levels[:-1], levels[1:], strict=False):
        midpoint = np.float32((np.float64(left) + np.float64(right)) / 2)
        values.extend(
            (
                np.nextafter(midpoint, np.float32(-np.inf)),
                midpoint,
                np.nextafter(midpoint, np.float32(np.inf)),
            )
        )
    weight = np.zeros((1, 128), dtype=np.float32)
    weight[0, : len(values)] = values
    expected = np.zeros_like(weight)
    for index, value in enumerate(values):
        distances = np.abs(levels.astype(np.float64) - np.float64(value))
        expected[0, index] = levels[int(np.argmin(distances))]
    effective = quantization.Group128Quantizer(
        "nf4-g128-absmax-f16-v1"
    ).effective
    for function in (effective, jax.jit(effective)):
        master = jnp.asarray(weight)
        np.testing.assert_array_equal(function(master), expected)
        np.testing.assert_array_equal(
            jax.grad(lambda value, fn=function: jnp.sum(fn(value)))(master),
            np.ones_like(weight),
        )
