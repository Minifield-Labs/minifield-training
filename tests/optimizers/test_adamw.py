"""Independent contracts for the AdamW update transaction."""

import math

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from minifield_training.core import parameters as core_parameters
from minifield_training.optimizers import adamw
from minifield_training.optimizers import state


def _inventory() -> core_parameters.FullParameterInventory:
    """One decayed matrix, one plain vector, and two frozen leaves."""
    return core_parameters.build_inventory(
        {
            "w.matrix": (2, 3),
            "w.vector": (3,),
            "w.frozen_matrix": (2, 2),
            "w.frozen_vector": (4,),
        },
        format_id="test.inventory/1",
        frozen_names=frozenset({"w.frozen_matrix", "w.frozen_vector"}),
    )


def _parameters() -> dict[str, jnp.ndarray]:
    return {
        "w.matrix": jnp.asarray(
            [[0.5, -0.25, 0.125], [1.0, -0.5, 0.75]], dtype=jnp.float32
        ),
        "w.vector": jnp.asarray([0.25, -0.5, 1.5], dtype=jnp.float32),
        "w.frozen_matrix": jnp.asarray(
            [[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32
        ),
        "w.frozen_vector": jnp.asarray([5.0, 6.0, 7.0, 8.0], dtype=jnp.float32),
    }


def _gradients() -> dict[str, jnp.ndarray]:
    return {
        "w.matrix": jnp.asarray(
            [[0.1, 0.2, -0.1], [0.05, -0.2, 0.3]], dtype=jnp.float32
        ),
        "w.vector": jnp.asarray([-0.4, 0.1, 0.2], dtype=jnp.float32),
    }


def _make_state() -> state.State:
    full_state = adamw.initialize_state(_parameters(), _inventory())
    moments = {
        "w.matrix": jnp.asarray(
            [[0.01, 0.0, -0.01], [0.02, 0.0, 0.01]], dtype=jnp.float32
        ),
        "w.vector": jnp.asarray([0.05, -0.05, 0.0], dtype=jnp.float32),
        "w.frozen_matrix": jnp.zeros((2, 2), dtype=jnp.float32),
        "w.frozen_vector": jnp.zeros((4,), dtype=jnp.float32),
    }
    variances = {
        "w.matrix": jnp.asarray(
            [[0.001, 0.002, 0.001], [0.0, 0.004, 0.002]], dtype=jnp.float32
        ),
        "w.vector": jnp.asarray([0.01, 0.0, 0.03], dtype=jnp.float32),
        "w.frozen_matrix": jnp.zeros((2, 2), dtype=jnp.float32),
        "w.frozen_vector": jnp.zeros((4,), dtype=jnp.float32),
    }
    return {**full_state, "m": moments, "v": variances}


def _oracle_update(
    parameter: npt.NDArray[np.float64],
    gradient: npt.NDArray[np.float64],
    first: npt.NDArray[np.float64],
    second: npt.NDArray[np.float64],
    clip_scale: float,
    config: adamw.AdamWConfig,
    iteration: int,
    *,
    decayed: bool,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Independent float64 AdamW oracle for one leaf."""
    g = np.asarray(gradient, dtype=np.float64) * clip_scale
    m = (
        config.beta1 * np.asarray(first, dtype=np.float64)
        + (1 - config.beta1) * g
    )
    v = (
        config.beta2 * np.asarray(second, dtype=np.float64)
        + (1 - config.beta2) * g * g
    )
    m_hat = m / (1 - config.beta1**iteration)
    v_hat = v / (1 - config.beta2**iteration)
    direction = np.where(
        m_hat == 0, 0.0, m_hat / (np.sqrt(v_hat) + config.epsilon)
    )
    decay_term = (
        config.weight_decay * np.asarray(parameter, dtype=np.float64)
        if decayed
        else 0.0
    )
    updated = np.asarray(parameter, dtype=np.float64) - config.learning_rate * (
        direction + decay_term
    )
    return updated, m, v


def _run(
    full_state: state.State,
    gradients: dict[str, jnp.ndarray],
    *,
    config: adamw.AdamWConfig | None = None,
    active_loss: jnp.ndarray | None = None,
    accumulation_valid: jnp.ndarray | None = None,
) -> adamw.CommitResult:
    transition = adamw.make_transaction(
        _inventory(), config or adamw.AdamWConfig(learning_rate=0.1)
    )
    return transition(
        full_state,
        gradients,
        active_loss if active_loss is not None else jnp.float32(1.0),
        (
            accumulation_valid
            if accumulation_valid is not None
            else jnp.asarray(True)
        ),
    )


def test_initialize_state_zeroes_moments_and_step() -> None:
    """Create zero moments, preserved masters, and an int32 step zero."""
    full_state = adamw.initialize_state(_parameters(), _inventory())
    assert int(full_state["step"]) == 0
    assert full_state["step"].dtype == jnp.int32
    for name, parameter in _parameters().items():
        np.testing.assert_array_equal(full_state["params"][name], parameter)
        assert not np.any(np.asarray(full_state["m"][name]))
        assert not np.any(np.asarray(full_state["v"][name]))


def test_commit_matches_independent_oracle() -> None:
    """Commit one update equal to a float64 AdamW oracle."""
    config = adamw.AdamWConfig(learning_rate=0.1, clip_norm=10.0)
    full_state = _make_state()
    result = _run(full_state, _gradients(), config=config)
    assert bool(result.committed)
    assert int(result.code) == adamw.CommitCode.COMMITTED
    assert int(result.state["step"]) == 1

    gradient_norm = math.sqrt(
        sum(
            float(np.sum(np.asarray(g, dtype=np.float64) ** 2))
            for g in _gradients().values()
        )
    )
    clip_scale = min(1.0, config.clip_norm / gradient_norm)
    np.testing.assert_allclose(
        float(result.gradient_norm), gradient_norm, rtol=1e-5
    )
    np.testing.assert_allclose(
        float(result.clipped_gradient_norm),
        gradient_norm * clip_scale,
        rtol=1e-5,
    )

    expected_delta: list[npt.NDArray[np.float64]] = []
    for name in ("w.matrix", "w.vector"):
        decayed = name == "w.matrix"
        updated, m, v = _oracle_update(
            np.asarray(full_state["params"][name]),
            np.asarray(_gradients()[name]),
            np.asarray(full_state["m"][name]),
            np.asarray(full_state["v"][name]),
            clip_scale,
            config,
            iteration=1,
            decayed=decayed,
        )
        np.testing.assert_allclose(
            np.asarray(result.state["params"][name]), updated, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(result.state["m"][name]), m, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(result.state["v"][name]), v, rtol=1e-5
        )
        expected_delta.append(
            updated - np.asarray(full_state["params"][name], dtype=np.float64)
        )

    update_norm = math.sqrt(
        sum(float(np.sum(delta**2)) for delta in expected_delta)
    )
    np.testing.assert_allclose(
        float(result.update_norm), update_norm, rtol=1e-5
    )


def test_frozen_leaves_commit_bit_identically() -> None:
    """Keep frozen masters and moments untouched with no delta."""
    full_state = _make_state()
    result = _run(full_state, _gradients())
    assert bool(result.committed)
    for name in ("w.frozen_matrix", "w.frozen_vector"):
        np.testing.assert_array_equal(
            np.asarray(result.state["params"][name]),
            np.asarray(full_state["params"][name]),
        )
        np.testing.assert_array_equal(
            np.asarray(result.state["m"][name]),
            np.asarray(full_state["m"][name]),
        )
        np.testing.assert_array_equal(
            np.asarray(result.state["v"][name]),
            np.asarray(full_state["v"][name]),
        )


def test_clipping_scales_the_logical_gradient() -> None:
    """Clip a large gradient to the declared norm before moments update."""
    config = adamw.AdamWConfig(learning_rate=0.1, clip_norm=0.05)
    gradients = {
        name: jnp.asarray(np.asarray(value) * 100.0)
        for name, value in _gradients().items()
    }
    result = _run(_make_state(), gradients, config=config)
    assert bool(result.committed)
    np.testing.assert_allclose(
        float(result.clipped_gradient_norm), config.clip_norm, rtol=1e-5
    )


def _assert_state_unchanged(
    result: adamw.CommitResult, full_state: state.State
) -> None:
    assert not bool(result.committed)
    for group in ("params", "m", "v"):
        for name, value in full_state[group].items():
            np.testing.assert_array_equal(
                np.asarray(result.state[group][name]), np.asarray(value)
            )
    np.testing.assert_array_equal(
        np.asarray(result.state["step"]), np.asarray(full_state["step"])
    )


def test_nonfinite_gradient_aborts_without_writes() -> None:
    """Reject a NaN gradient before any leaf is written."""
    full_state = _make_state()
    gradients = dict(_gradients())
    gradients["w.vector"] = jnp.asarray([jnp.nan, 0.0, 0.0])
    result = _run(full_state, gradients)
    assert int(result.code) == adamw.CommitCode.NONFINITE_GRADIENT
    _assert_state_unchanged(result, full_state)


def test_nonfinite_loss_aborts_without_writes() -> None:
    """Reject a NaN loss before any leaf is written."""
    full_state = _make_state()
    result = _run(full_state, _gradients(), active_loss=jnp.asarray(jnp.nan))
    assert int(result.code) == adamw.CommitCode.NONFINITE_LOSS
    _assert_state_unchanged(result, full_state)


def test_invalid_accumulation_aborts_without_writes() -> None:
    """Reject an invalid accumulation flag before any leaf is written."""
    full_state = _make_state()
    result = _run(
        full_state, _gradients(), accumulation_valid=jnp.asarray(False)
    )
    assert int(result.code) == adamw.CommitCode.ACCUMULATION_INVALID
    _assert_state_unchanged(result, full_state)


def test_invalid_state_aborts_without_writes() -> None:
    """Reject non-finite incoming moments before any leaf is written."""
    full_state = _make_state()
    broken = dict(full_state["m"])
    broken["w.vector"] = jnp.asarray([jnp.inf, 0.0, 0.0])
    result = _run({**full_state, "m": broken}, _gradients())
    assert int(result.code) == adamw.CommitCode.INVALID_STATE


def test_negative_variance_aborts() -> None:
    """Reject a negative second moment as invalid incoming state."""
    full_state = _make_state()
    broken = dict(full_state["v"])
    broken["w.vector"] = jnp.asarray([-1.0, 0.0, 0.0])
    result = _run({**full_state, "v": broken}, _gradients())
    assert int(result.code) == adamw.CommitCode.INVALID_STATE


def test_step_overflow_aborts() -> None:
    """Reject a commit that would exceed the int32 step range."""
    full_state = _make_state()
    result = _run(
        {**full_state, "step": jnp.asarray(2**31 - 1, dtype=jnp.int32)},
        _gradients(),
    )
    assert int(result.code) == adamw.CommitCode.STEP_OVERFLOW


def test_overflowing_candidate_preserves_state() -> None:
    """Keep the old state when the candidate update is non-finite."""
    config = adamw.AdamWConfig(
        learning_rate=1.0, weight_decay=4.0, clip_norm=1e30
    )
    full_state = _make_state()
    params = dict(full_state["params"])
    params["w.matrix"] = jnp.full((2, 3), 1e38, dtype=jnp.float32)
    gradients = {
        "w.matrix": jnp.zeros((2, 3), dtype=jnp.float32),
        "w.vector": jnp.zeros((3,), dtype=jnp.float32),
    }
    result = _run({**full_state, "params": params}, gradients, config=config)
    assert int(result.code) == adamw.CommitCode.CANDIDATE_INVALID
    np.testing.assert_array_equal(
        np.asarray(result.state["params"]["w.matrix"]),
        np.asarray(params["w.matrix"]),
    )


def test_zero_gradient_commits_with_decay_only() -> None:
    """Apply only the decay term when moments and gradients are zero."""
    config = adamw.AdamWConfig(learning_rate=0.1, weight_decay=0.5)
    full_state = adamw.initialize_state(_parameters(), _inventory())
    gradients = {
        "w.matrix": jnp.zeros((2, 3), dtype=jnp.float32),
        "w.vector": jnp.zeros((3,), dtype=jnp.float32),
    }
    result = _run(full_state, gradients, config=config)
    assert bool(result.committed)
    np.testing.assert_allclose(
        np.asarray(result.state["params"]["w.matrix"]),
        np.asarray(full_state["params"]["w.matrix"])
        * (1 - config.learning_rate * config.weight_decay),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(result.state["params"]["w.vector"]),
        np.asarray(full_state["params"]["w.vector"]),
    )


def test_gradient_tree_covers_exactly_trainable_leaves() -> None:
    """Accept trainable-only gradients and reject membership drift."""
    inventory = _inventory()
    adamw.validate_gradient_tree(_gradients(), inventory)
    with pytest.raises(ValueError, match="tree mismatch"):
        adamw.validate_gradient_tree(
            {"w.matrix": _gradients()["w.matrix"]}, inventory
        )
    with pytest.raises(ValueError, match="tree mismatch"):
        adamw.validate_gradient_tree(
            {**_gradients(), "w.frozen_vector": jnp.zeros((4,))}, inventory
        )


def test_state_structure_rejects_drift() -> None:
    """Reject missing, extra, misshaped, and non-FP32 state leaves."""
    inventory = _inventory()
    full_state = _make_state()
    adamw.validate_full_weight_state_structure(full_state, inventory)
    params = dict(full_state["params"])
    del params["w.vector"]
    with pytest.raises(ValueError, match="tree mismatch"):
        adamw.validate_full_weight_state_structure(
            {**full_state, "params": params}, inventory
        )
    wrong_shape = dict(full_state["params"])
    wrong_shape["w.vector"] = jnp.zeros((4,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        adamw.validate_full_weight_state_structure(
            {**full_state, "params": wrong_shape}, inventory
        )
    wrong_dtype = dict(full_state["params"])
    wrong_dtype["w.vector"] = jnp.zeros((3,), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="float32"):
        adamw.validate_full_weight_state_structure(
            {**full_state, "params": wrong_dtype}, inventory
        )


def test_full_state_rejects_nonfinite_leaves() -> None:
    """Eagerly reject non-finite masters or moments at boundaries."""
    inventory = _inventory()
    full_state = _make_state()
    adamw.validate_full_weight_state(full_state, inventory)
    params = dict(full_state["params"])
    params["w.vector"] = jnp.asarray([jnp.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        adamw.validate_full_weight_state(
            {**full_state, "params": params}, inventory
        )


def test_config_rejects_invalid_scalars() -> None:
    """Reject invalid optimizer settings before a transition is built."""
    with pytest.raises(ValueError, match="positive"):
        adamw.AdamWConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        adamw.AdamWConfig(learning_rate=0.1, beta1=1.0)
    with pytest.raises(ValueError, match="positive"):
        adamw.AdamWConfig(learning_rate=0.1, epsilon=0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        adamw.AdamWConfig(learning_rate=0.1, weight_decay=-0.1)
    with pytest.raises(ValueError, match="positive"):
        adamw.AdamWConfig(learning_rate=0.1, clip_norm=0.0)
    with pytest.raises(ValueError, match="finite"):
        adamw.AdamWConfig(learning_rate=float("nan"))
    with pytest.raises(ValueError, match="non-boolean"):
        adamw.AdamWConfig(learning_rate=True)


def test_implementation_identity_is_stable() -> None:
    """Digest the implementation tag and scalar settings deterministically."""
    first = adamw.AdamWConfig(learning_rate=0.1)
    second = adamw.AdamWConfig(learning_rate=0.1)
    other = adamw.AdamWConfig(learning_rate=0.2)
    assert first.implementation_identity == second.implementation_identity
    assert first.implementation_identity != other.implementation_identity
    assert len(first.implementation_identity) == 64


def test_donated_transaction_matches_plain_commit() -> None:
    """Produce identical results with and without state donation."""
    config = adamw.AdamWConfig(learning_rate=0.1)
    plain = _run(_make_state(), _gradients(), config=config)
    donated_transition = adamw.make_donated_transaction(_inventory(), config)
    donated = donated_transition(
        _make_state(), _gradients(), jnp.float32(1.0), jnp.asarray(True)
    )
    assert bool(donated.committed) == bool(plain.committed)
    for name in _parameters():
        np.testing.assert_array_equal(
            np.asarray(donated.state["params"][name]),
            np.asarray(plain.state["params"][name]),
        )
