"""Analytical masked soft-target pointer cross-entropy."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from minifield_training.objectives import pointer


def _batch() -> dict[str, jax.Array]:
    """Question 0: uniform over 4 allowed. 1: soft over 2. 2: padding."""
    allowed = np.zeros((1, 3, 6), np.int32)
    allowed[0, 0, :4] = 1
    allowed[0, 1, 4:] = 1
    start = np.zeros((1, 3, 6), np.float32)
    start[0, 0, 2] = 1
    start[0, 1, 4:] = (0.25, 0.75)
    return {
        "allowed": jnp.asarray(allowed),
        "start_target": jnp.asarray(start),
        "end_target": jnp.asarray(start),
        "field_weight": jnp.asarray([[2.0, 1.0, 0.0]]),
    }


def _outputs() -> dict[str, jax.Array]:
    """Masked positions carry huge logits that must not leak into softmaxes."""
    logits = np.zeros((1, 3, 6), np.float32)
    logits[0, 0, 4:] = 1e4
    logits[0, 1, 4:] = (0.0, math.log(3.0))
    logits[0, 1, :4] = 1e4
    return {"start": jnp.asarray(logits), "end": jnp.asarray(logits)}


def test_losses_match_hand_derived_values() -> None:
    """Uniform question costs log 4; the soft pair costs its cross-entropy."""
    losses = np.asarray(pointer.losses(_outputs(), _batch()))
    # Softmax over (0, log 3) is (1/4, 3/4).
    soft = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    np.testing.assert_allclose(
        losses, [[math.log(4), soft, 0.0]], rtol=1e-6, atol=1e-7
    )
    total, mass = pointer.terms(_outputs(), _batch())
    np.testing.assert_allclose(total, 2 * math.log(4) + soft, rtol=1e-6)
    np.testing.assert_allclose(mass, 3.0)


def test_gradient_is_finite_and_zero_outside_allowed_tokens() -> None:
    """Padding questions and masked tokens receive exactly zero gradient."""
    batch = _batch()
    gradient = jax.grad(
        lambda logits: pointer.terms({"start": logits, "end": logits}, batch)[0]
    )(_outputs()["start"])
    values = np.asarray(gradient)
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values[0, 0, 4:], 0)
    np.testing.assert_array_equal(values[0, 1, :4], 0)
    np.testing.assert_array_equal(values[0, 2], 0)
    # d/dz of weight * CE is weight * (softmax - target), summed over ends.
    np.testing.assert_allclose(
        values[0, 0, :4], 2 * (np.full(4, 0.25) - [0, 0, 1, 0]), rtol=1e-6
    )


def test_distillation_matches_hand_derived_kl_and_ignores_masked_tokens() -> (
    None
):
    """KL between (1/4, 3/4) and uniform over the soft pair, times T²."""
    batch = _batch()
    teacher = _outputs()
    flat = {name: jnp.zeros_like(value) for name, value in teacher.items()}
    divergence = np.asarray(
        pointer.distillation(teacher, flat, batch, temperature=1.0)
    )
    pair = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
    np.testing.assert_allclose(divergence, [[0.0, pair, 0.0]], atol=1e-6)
    hot = pointer.distillation(teacher, flat, batch, temperature=2.0)
    assert float(hot[0, 1]) > 0
    same = pointer.distillation(teacher, teacher, batch, temperature=2.0)
    np.testing.assert_allclose(same, 0, atol=1e-6)


def test_distillation_trains_only_the_student() -> None:
    """The dense pass gets cross-entropy gradient only, never from the KL."""
    batch = _batch()
    student = {"start": jnp.zeros((1, 3, 6)), "end": jnp.zeros((1, 3, 6))}

    def kl_only(teacher_logits: jax.Array) -> jax.Array:
        teacher = {"start": teacher_logits, "end": teacher_logits}
        return (
            pointer.distilled_terms(
                teacher, student, batch, quantized_weight=0.0
            )[0]
            - pointer.terms(teacher, batch)[0]
        )

    gradient = jax.grad(kl_only)(_outputs()["start"])
    np.testing.assert_allclose(gradient, 0, atol=1e-6)


def test_distilled_terms_reduce_to_dense_terms_without_the_student() -> None:
    """Zero student weights leave exactly the dense objective."""
    batch, outputs = _batch(), _outputs()
    total, mass = pointer.distilled_terms(
        outputs, outputs, batch, quantized_weight=0.0, distill_weight=0.0
    )
    expected = pointer.terms(outputs, batch)
    np.testing.assert_allclose(total, expected[0], rtol=1e-6)
    np.testing.assert_allclose(mass, expected[1])
