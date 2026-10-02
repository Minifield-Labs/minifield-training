"""Warm-up and cosine decay factors over committed updates."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from minifield_training.optimizers import schedule


def test_warmup_rises_linearly_then_cosine_decays_to_the_floor() -> None:
    """The first update moves, the peak lands after warm-up, the end holds."""
    plan = schedule.WarmupCosine(4, 12, final_fraction=0.1)
    factors = [float(plan.factor(jnp.int32(step))) for step in range(14)]
    np.testing.assert_allclose(factors[:4], [0.25, 0.5, 0.75, 1.0])
    middle = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 4 / 8))
    np.testing.assert_allclose(factors[8], middle, rtol=1e-6)
    np.testing.assert_allclose(factors[12:], [0.1, 0.1], rtol=1e-6)
    assert all(a >= b for a, b in zip(factors[4:], factors[5:], strict=False))


def test_no_warmup_starts_at_the_peak() -> None:
    """Zero warm-up updates begins decay from the full rate."""
    plan = schedule.WarmupCosine(0, 10, final_fraction=0.0)
    assert float(plan.factor(jnp.int32(0))) == 1.0
    assert float(plan.factor(jnp.int32(10))) == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    ("warmup", "total", "final"),
    ((-1, 10, 0.1), (10, 10, 0.1), (0, 0, 0.1), (1, 10, 1.5)),
)
def test_invalid_schedules_are_rejected(
    warmup: int, total: int, final: float
) -> None:
    """Warm-up must end before the last update and the floor is a fraction."""
    with pytest.raises(ValueError):
        schedule.WarmupCosine(warmup, total, final_fraction=final)
