"""Independent checks for token-weighted logical updates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from minifield_training.core import parameters
from minifield_training.engine import step
from minifield_training.optimizers import adamw
from minifield_training.optimizers import state


def _inventory() -> parameters.FullParameterInventory:
    """Return one trainable scalar and one frozen scalar."""
    return parameters.build_inventory(
        {"weight": (), "frozen": ()},
        format_id="synthetic/1",
        decayed_names=frozenset(),
        frozen_names=frozenset({"frozen"}),
    )


def _terms(
    params: dict[str, jax.Array], batch: dict[str, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """Weighted squared error with an explicit supervised count."""
    residual = params["weight"] * batch["x"] + params["frozen"] - batch["y"]
    mask = batch["mask"]
    return (
        jnp.sum(mask * residual * residual, dtype=jnp.float32),
        jnp.sum(mask, dtype=jnp.float32),
    )


def _setup() -> (
    tuple[parameters.FullParameterInventory, adamw.AdamWConfig, state.State]
):
    """Return a healthy zero-moment state and unclipped Adam settings."""
    inventory = _inventory()
    initial = adamw.initialize_state(
        {"weight": jnp.float32(2), "frozen": jnp.float32(1)}, inventory
    )
    return inventory, adamw.AdamWConfig(0.1, clip_norm=100), initial


def _batch() -> dict[str, jax.Array]:
    """Return 2 slots with unequal counts of 1 and 3."""
    return {
        "x": jnp.asarray([[1, 0, 0], [2, 3, 4]], dtype=jnp.float32),
        "y": jnp.zeros((2, 3), dtype=jnp.float32),
        "mask": jnp.asarray([[1, 0, 0], [1, 1, 1]], dtype=jnp.float32),
    }


def _same_state(left: object, right: object) -> None:
    """Check all incoming leaves, including frozen moments and step."""
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_token_weighting_matches_analytical_gradient_and_large_batch() -> None:
    """Logical update uses total target count, never slot means."""
    inventory, config, initial = _setup()
    update = step.make_step(_terms, inventory, config)
    micro = update(initial, _batch(), jnp.asarray([True, True]))
    large = {name: value.reshape((1, 6)) for name, value in _batch().items()}
    combined = update(initial, large, jnp.asarray([True]))
    # d/dw sum((w*x+1)^2)/4 = 2*(3+10+21+36)/4 = 35.
    # The first Adam moment is (1-beta1)*gradient = 3.5.
    assert bool(micro.committed)
    assert int(micro.state["step"]) == 1
    np.testing.assert_allclose(micro.state["m"]["weight"], 3.5, rtol=1e-6)
    np.testing.assert_allclose(micro.loss, 41.0, rtol=1e-6)
    for key in ("params", "m", "v"):
        for name in initial[key]:
            np.testing.assert_allclose(
                micro.state[key][name], combined.state[key][name], rtol=1e-6
            )
    for actual, expected in (
        (micro.state["params"], initial["params"]),
        (micro.state["m"], initial["m"]),
        (micro.state["v"], initial["v"]),
    ):
        np.testing.assert_array_equal(actual["frozen"], expected["frozen"])
    jitted = jax.jit(update)(initial, _batch(), jnp.asarray([True, True]))
    np.testing.assert_allclose(
        jitted.state["params"]["weight"],
        micro.state["params"]["weight"],
        rtol=1e-6,
    )


@pytest.mark.parametrize("bad_count", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_total_count_rejects_exact_state(bad_count: float) -> None:
    """Zero and invalid target counts cannot commit."""
    inventory, config, initial = _setup()

    def bad_terms(
        params: dict[str, jax.Array], batch: dict[str, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Keep loss differentiable while supplying an invalid count."""
        return params["weight"] * batch["x"][0], jnp.float32(bad_count)

    result = step.make_step(bad_terms, inventory, config)(
        initial, {"x": jnp.ones((1, 1))}, jnp.asarray([True])
    )
    assert not bool(result.committed)
    assert int(result.code) == adamw.CommitCode.ACCUMULATION_INVALID
    _same_state(result.state, initial)


@pytest.mark.parametrize("bad_kind", ["loss", "gradient"])
def test_nonfinite_objective_rejects_exact_state(bad_kind: str) -> None:
    """A bad active loss or gradient preserves all healthy input leaves."""
    inventory, config, initial = _setup()

    def bad_terms(
        params: dict[str, jax.Array], batch: dict[str, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Select a nonfinite primal or derivative independently."""
        weight = params["weight"]
        if bad_kind == "loss":
            return weight * batch["x"][0] + jnp.float32(jnp.nan), jnp.float32(1)
        return jnp.sqrt(weight - weight), jnp.float32(1)

    result = step.make_step(bad_terms, inventory, config)(
        initial, {"x": jnp.ones((1, 1))}, jnp.asarray([True])
    )
    assert not bool(result.committed)
    _same_state(result.state, initial)


def test_inactive_poison_is_skipped_and_active_poison_rejects() -> None:
    """A masked slot never evaluates invalid data inside the objective."""
    inventory, config, initial = _setup()
    update = step.make_step(_terms, inventory, config)
    batch = _batch()
    batch["x"] = batch["x"].at[1, :].set(jnp.nan)
    safe = update(initial, batch, jnp.asarray([True, False]))
    poisoned = update(initial, batch, jnp.asarray([True, True]))
    assert bool(safe.committed)
    np.testing.assert_allclose(safe.loss, 9.0)
    assert not bool(poisoned.committed)
    _same_state(poisoned.state, initial)


def test_bad_batch_structure_rejected() -> None:
    """Physical slots and active vector must line up before tracing."""
    inventory, config, initial = _setup()
    update = step.make_step(_terms, inventory, config)
    with pytest.raises(ValueError, match="common leading axis"):
        update(initial, {"x": jnp.ones((2, 1))}, jnp.asarray([True]))


@pytest.mark.parametrize("fuse_accumulation", [False, True])
def test_streaming_matches_scanned_update_and_skips_inactive(
    fuse_accumulation: bool,
) -> None:
    """Both streaming programs preserve count-weighted AdamW semantics."""
    inventory, config, initial = _setup()
    physical = _batch()
    scanned = step.make_step(_terms, inventory, config)(
        initial, physical, jnp.asarray([True, True])
    )
    scanned_values = jax.tree.map(
        lambda value: np.asarray(value).copy(), scanned
    )
    _, _, fresh_initial = _setup()
    streamed = step.make_streaming_step(
        _terms, inventory, config, fuse_accumulation=fuse_accumulation
    )(fresh_initial, physical, jnp.asarray([True, True]))
    assert bool(streamed.committed)
    np.testing.assert_allclose(streamed.loss, scanned_values.loss, rtol=1e-6)
    for group in ("params", "m", "v"):
        for name in initial[group]:
            np.testing.assert_allclose(
                streamed.state[group][name],
                scanned_values.state[group][name],
                rtol=1e-6,
            )

    _, _, second_initial = _setup()
    physical["x"] = physical["x"].at[1].set(jnp.nan)
    skipped = step.make_streaming_step(
        _terms, inventory, config, fuse_accumulation=fuse_accumulation
    )(second_initial, physical, jnp.asarray([True, False]))
    assert bool(skipped.committed)
    np.testing.assert_allclose(skipped.loss, 9.0)


def test_fused_accumulation_reuses_donated_state_across_updates() -> None:
    """The fused sum agrees with separate addition for consecutive commits."""
    inventory, config, baseline_state = _setup()
    _, _, fused_state = _setup()
    baseline = step.make_streaming_step(_terms, inventory, config)
    fused = step.make_streaming_step(
        _terms, inventory, config, fuse_accumulation=True
    )
    assert fused.accumulate is not None
    for _ in range(2):
        baseline_result = baseline(
            baseline_state, _batch(), np.asarray([True, True])
        )
        fused_result = fused(fused_state, _batch(), np.asarray([True, True]))
        assert bool(baseline_result.committed)
        assert bool(fused_result.committed)
        np.testing.assert_allclose(
            fused_result.loss, baseline_result.loss, rtol=1e-6
        )
        for group in ("params", "m", "v"):
            for name in baseline_result.state[group]:
                np.testing.assert_allclose(
                    fused_result.state[group][name],
                    baseline_result.state[group][name],
                    rtol=1e-6,
                    atol=1e-7,
                )
        baseline_state = baseline_result.state
        fused_state = fused_result.state
    assert int(fused_state["step"]) == 2


@pytest.mark.parametrize("bad_count", [-1.0, float("nan"), float("inf")])
def test_fused_accumulation_rejects_invalid_later_count(
    bad_count: float,
) -> None:
    """One invalid physical count rejects even when the total stays positive."""
    inventory, config, initial = _setup()
    snapshot = jax.tree.map(lambda value: np.asarray(value).copy(), initial)

    def counted_terms(
        params: dict[str, jax.Array], batch: dict[str, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Provide a differentiable loss and an independent target count."""
        return params["weight"] * batch["x"][0], batch["count"][0]

    result = step.make_streaming_step(
        counted_terms, inventory, config, fuse_accumulation=True
    )(
        initial,
        {
            "x": jnp.ones((2, 1), dtype=jnp.float32),
            "count": jnp.asarray([[3.0], [bad_count]], dtype=jnp.float32),
        },
        np.asarray([True, True]),
    )
    assert not bool(result.committed)
    assert int(result.code) == adamw.CommitCode.ACCUMULATION_INVALID
    _same_state(result.state, snapshot)


def test_fused_accumulation_rejects_invalid_later_gradient() -> None:
    """A later NaN derivative rejects a finite summed loss and count."""
    inventory, config, initial = _setup()
    snapshot = jax.tree.map(lambda value: np.asarray(value).copy(), initial)

    def bad_gradient_terms(
        params: dict[str, jax.Array], batch: dict[str, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Choose a finite primal with a poisoned derivative in slot 2."""
        weight = params["weight"]
        feature = batch["x"][0]
        loss = jax.lax.cond(
            feature < 0,
            lambda _: jnp.sqrt(weight - weight),
            lambda _: weight * feature,
            None,
        )
        return loss, jnp.float32(1)

    result = step.make_streaming_step(
        bad_gradient_terms, inventory, config, fuse_accumulation=True
    )(
        initial,
        {"x": jnp.asarray([[1.0], [-1.0]], dtype=jnp.float32)},
        np.asarray([True, True]),
    )
    assert not bool(result.committed)
    assert int(result.code) == adamw.CommitCode.NONFINITE_GRADIENT
    np.testing.assert_allclose(result.loss, 1.0)
    _same_state(result.state, snapshot)


def test_streaming_rejects_invalid_count_without_committing() -> None:
    """A bad physical count leaves every donated input leaf recoverable."""
    inventory, config, initial = _setup()
    snapshot = jax.tree.map(lambda value: np.asarray(value).copy(), initial)

    def bad_terms(
        params: dict[str, jax.Array], batch: dict[str, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Expose a finite gradient paired with an invalid count."""
        return params["weight"] * batch["x"][0], jnp.float32(-1)

    result = step.make_streaming_step(bad_terms, inventory, config)(
        initial, {"x": jnp.ones((1, 1))}, jnp.asarray([True])
    )
    assert not bool(result.committed)
    assert int(result.code) == adamw.CommitCode.ACCUMULATION_INVALID
    _same_state(result.state, snapshot)
