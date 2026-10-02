"""Optax AdamW commit against an independent float64 calculation."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from minifield_training.core import parameters as core_parameters
from minifield_training.optimizers import adamw
from minifield_training.optimizers import optax_adamw
from minifield_training.optimizers import state

_CONFIG = adamw.AdamWConfig(
    learning_rate=0.1, beta1=0.9, beta2=0.95, weight_decay=0.2, clip_norm=0.5
)


def _inventory() -> core_parameters.FullParameterInventory:
    """A decayed matrix, an undecayed vector, and a frozen vector."""
    return core_parameters.build_inventory(
        {"matrix": (2, 2), "vector": (3,), "frozen": (2,)},
        format_id="optax-test/1",
        decayed_names=frozenset({"matrix"}),
        frozen_names=frozenset({"frozen"}),
    )


def _state() -> state.State:
    return adamw.initialize_state(
        {
            "matrix": jnp.asarray([[0.5, -1.0], [2.0, 0.25]]),
            "vector": jnp.asarray([1.0, -0.5, 0.75]),
            "frozen": jnp.asarray([3.0, 4.0]),
        },
        _inventory(),
    )


def _gradients(scale: float = 1.0) -> dict[str, jax.Array]:
    return {
        "matrix": jnp.asarray([[0.3, -0.2], [0.1, 0.4]]) * scale,
        "vector": jnp.asarray([-0.6, 0.2, 0.5]) * scale,
    }


def _oracle(
    steps: list[dict[str, npt.NDArray[np.float32]]],
) -> dict[str, dict[str, npt.NDArray[np.float64]]]:
    """Clip by the global norm, then decoupled AdamW, in float64."""
    params = {
        name: np.asarray(value, np.float64)
        for name, value in _state()["params"].items()
    }
    m = {name: np.zeros_like(params[name]) for name in ("matrix", "vector")}
    v = {name: np.zeros_like(params[name]) for name in ("matrix", "vector")}
    for iteration, gradients in enumerate(steps, start=1):
        norm = np.sqrt(
            sum(np.sum(g.astype(np.float64) ** 2) for g in gradients.values())
        )
        scale = min(1.0, _CONFIG.clip_norm / norm)
        for name, gradient in gradients.items():
            g = gradient.astype(np.float64) * scale
            m[name] = _CONFIG.beta1 * m[name] + (1 - _CONFIG.beta1) * g
            v[name] = _CONFIG.beta2 * v[name] + (1 - _CONFIG.beta2) * g * g
            direction = (m[name] / (1 - _CONFIG.beta1**iteration)) / (
                np.sqrt(v[name] / (1 - _CONFIG.beta2**iteration))
                + _CONFIG.epsilon
            )
            decay = (
                _CONFIG.weight_decay * params[name] if name == "matrix" else 0
            )
            params[name] = params[name] - _CONFIG.learning_rate * (
                direction + decay
            )
    return {"params": params, "m": m, "v": v}


def test_two_commits_match_clipped_decoupled_adamw() -> None:
    """Clipping, bias correction, and decay masks match float64 math."""
    transition = jax.jit(optax_adamw.make_transaction(_inventory(), _CONFIG))
    current = _state()
    steps = [_gradients(), _gradients(0.3)]
    for gradients in steps:
        result = transition(
            current, gradients, jnp.float32(1.5), jnp.asarray(True)
        )
        assert bool(result.committed)
        assert int(result.code) == adamw.CommitCode.COMMITTED
        current = result.state
    expected = _oracle(
        [{k: np.asarray(v) for k, v in g.items()} for g in steps]
    )
    assert int(current["step"]) == 2
    for group in ("params", "m", "v"):
        for name, value in expected[group].items():
            np.testing.assert_allclose(
                current[group][name], value, rtol=2e-6, atol=1e-7
            )
    np.testing.assert_array_equal(current["params"]["frozen"], [3.0, 4.0])
    np.testing.assert_array_equal(current["m"]["frozen"], [0.0, 0.0])
    # The first gradient's norm is sqrt(0.95) > 0.5, so it was clipped. The
    # second is 0.3 of that, about 0.29, so it passes through unclipped.
    norm = 0.3 * np.sqrt(0.95)
    np.testing.assert_allclose(result.gradient_norm, norm, rtol=1e-6)
    np.testing.assert_allclose(result.clipped_gradient_norm, norm, rtol=1e-6)


@pytest.mark.parametrize(
    ("gradient", "loss", "valid", "code"),
    [
        (float("nan"), 1.0, True, adamw.CommitCode.NONFINITE_GRADIENT),
        (0.1, float("inf"), True, adamw.CommitCode.NONFINITE_LOSS),
        (0.1, 1.0, False, adamw.CommitCode.ACCUMULATION_INVALID),
    ],
)
def test_rejection_keeps_the_exact_incoming_state(
    gradient: float, loss: float, valid: bool, code: int
) -> None:
    """A non-finite step writes nothing and names the reason."""
    incoming = _state()
    expected = jax.tree.map(lambda value: np.asarray(value).copy(), incoming)
    gradients = _gradients()
    gradients["vector"] = gradients["vector"].at[0].set(gradient)
    result = optax_adamw.make_transaction(_inventory(), _CONFIG)(
        incoming, gradients, jnp.float32(loss), jnp.asarray(valid)
    )
    assert not bool(result.committed)
    assert int(result.code) == code
    for actual, saved in zip(
        jax.tree.leaves(result.state), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual, saved)


def test_identity_is_separate_from_the_transactional_commit() -> None:
    """Checkpoints from the 2 transactions can't resume into each other."""
    identity = optax_adamw.implementation_identity(_CONFIG)
    assert identity == optax_adamw.implementation_identity(_CONFIG)
    assert identity != _CONFIG.implementation_identity
    assert identity != optax_adamw.implementation_identity(
        adamw.AdamWConfig(learning_rate=0.2)
    )
