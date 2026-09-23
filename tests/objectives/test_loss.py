"""Independent contracts for the causal loss reduction."""

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from minifield_training.objectives import loss


def _reference_terms(
    logits: npt.NDArray[np.generic],
    input_ids: npt.NDArray[np.generic],
    loss_mask: npt.NDArray[np.generic],
    attention_mask: npt.NDArray[np.generic],
) -> tuple[float, float]:
    """Independent oracle: logsumexp minus label logit, masked, summed."""
    shifted = np.asarray(logits[:, :-1, :], dtype=np.float64)
    labels = np.asarray(input_ids)[:, 1:]
    mask = (
        np.asarray(loss_mask)[:, 1:] * np.asarray(attention_mask)[:, 1:]
    ).astype(np.float64)
    log_total = np.log(
        np.exp(shifted - shifted.max(axis=-1, keepdims=True)).sum(axis=-1)
    ) + shifted.max(axis=-1)
    chosen = np.take_along_axis(shifted, labels[..., None], axis=-1)[..., 0]
    losses = log_total - chosen
    return float(np.sum(losses * mask)), float(np.sum(mask))


def test_terms_match_independent_oracle() -> None:
    """Compare masked terms against a float64 logsumexp oracle."""
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((2, 5, 11)).astype(np.float32)
    ids = rng.integers(0, 11, (2, 5)).astype(np.int32)
    loss_mask = np.array([[1, 1, 1, 0, 0], [1, 0, 1, 1, 0]], dtype=np.int32)
    attention_mask = np.array(
        [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=np.int32
    )
    expected_total, expected_count = _reference_terms(
        logits, ids, loss_mask, attention_mask
    )
    total, count = loss.causal_loss_terms(
        jnp.asarray(logits),
        jnp.asarray(ids),
        jnp.asarray(loss_mask),
        jnp.asarray(attention_mask),
    )
    assert total.dtype == jnp.float32
    assert count.dtype == jnp.float32
    np.testing.assert_allclose(float(total), expected_total, rtol=1e-5)
    assert float(count) == expected_count == 4.0


def test_mask_excludes_padding_and_unsupervised_positions() -> None:
    """Count only positions marked by both masks."""
    logits = np.zeros((1, 4, 3), dtype=np.float32)
    logits[0, :, 0] = 10.0
    ids = np.array([[0, 0, 0, 0]], dtype=np.int32)
    attention_mask = np.ones((1, 4), dtype=np.int32)
    loss_mask = np.array([[1, 1, 0, 1]], dtype=np.int32)
    total, count = loss.causal_loss_terms(
        jnp.asarray(logits),
        jnp.asarray(ids),
        jnp.asarray(loss_mask),
        jnp.asarray(attention_mask),
    )
    expected = 2.0 * -np.log(np.exp(10.0) / (np.exp(10.0) + 2.0))
    np.testing.assert_allclose(float(total), expected, rtol=1e-4, atol=1e-6)
    assert float(count) == 2.0


def test_attention_mask_still_bounds_supervision() -> None:
    """Drop loss-marked positions that fall on padding."""
    logits = np.zeros((1, 3, 2), dtype=np.float32)
    ids = np.array([[0, 1, 1]], dtype=np.int32)
    loss_mask = np.ones((1, 3), dtype=np.int32)
    attention_mask = np.array([[1, 1, 0]], dtype=np.int32)
    total, count = loss.causal_loss_terms(
        jnp.asarray(logits),
        jnp.asarray(ids),
        jnp.asarray(loss_mask),
        jnp.asarray(attention_mask),
    )
    np.testing.assert_allclose(float(total), float(np.log(2.0)), rtol=1e-6)
    assert float(count) == 1.0


def test_zero_count_average_is_zero_not_nan() -> None:
    """Return zero loss and zero count for an empty mask."""
    logits = np.zeros((1, 3, 2), dtype=np.float32)
    ids = np.zeros((1, 3), dtype=np.int32)
    zeros = np.zeros((1, 3), dtype=np.int32)
    average, count = loss.causal_loss(
        jnp.asarray(logits),
        jnp.asarray(ids),
        jnp.asarray(zeros),
        jnp.asarray(zeros),
    )
    assert float(count) == 0.0
    assert float(average) == 0.0


def test_average_divides_by_count() -> None:
    """Divide the unaveraged total by the exact target count."""
    logits = np.zeros((1, 3, 2), dtype=np.float32)
    ids = np.array([[0, 0, 0]], dtype=np.int32)
    loss_mask = np.ones((1, 3), dtype=np.int32)
    attention_mask = np.ones((1, 3), dtype=np.int32)
    average, count = loss.causal_loss(
        jnp.asarray(logits),
        jnp.asarray(ids),
        jnp.asarray(loss_mask),
        jnp.asarray(attention_mask),
    )
    np.testing.assert_allclose(float(average), float(np.log(2.0)), rtol=1e-6)
    assert float(count) == 2.0
