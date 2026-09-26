"""Independent hard-label and allowed-class loss checks."""

import jax
import jax.numpy as jnp
import numpy as np

from minifield_training.objectives import classification


def test_excluded_class_and_padded_label_have_finite_terms() -> None:
    """PAD never enters softmax or the inactive-row gather."""
    values = jnp.asarray([[0.0, 1.0, 20.0], [2.0, 0.0, 40.0]])
    allowed = jnp.asarray([True, True, False])
    total, count = classification.hard_label_terms(
        values,
        jnp.asarray([1, 2]),
        jnp.asarray([True, False]),
        allowed,
        safe_class=0,
    )
    expected = np.log(np.exp(0.0) + np.exp(1.0)) - 1.0
    np.testing.assert_allclose(total, expected, rtol=1e-6)
    assert float(count) == 1.0
    gradient = jax.grad(
        lambda logits: classification.hard_label_terms(
            logits,
            jnp.asarray([1, 2]),
            jnp.asarray([True, False]),
            allowed,
            safe_class=0,
        )[0]
    )(values)
    assert np.isfinite(np.asarray(gradient)).all()
    assert float(gradient[0, 2]) == 0.0
    assert float(gradient[1, 2]) == 0.0


def test_valid_disallowed_label_is_rejected_numerically() -> None:
    """Dynamic malformed labels cause a nonfinite rejected update."""
    total, count = classification.hard_label_terms(
        jnp.zeros((1, 3)),
        jnp.asarray([2]),
        jnp.asarray([True]),
        jnp.asarray([True, True, False]),
        safe_class=0,
    )
    assert np.isnan(float(total))
    assert float(count) == 1.0
