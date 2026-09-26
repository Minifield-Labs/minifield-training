"""Deterministic physical SFT update fixtures."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from minifield_training.batching import contracts
from minifield_training.batching import dense
from minifield_training.batching import sft as sft_batching
from minifield_training.core import parameters
from minifield_training.datasets.tokenization import TokenizedExample
from minifield_training.engine import step
from minifield_training.optimizers import adamw


def _example(name: str, ids: tuple[int, ...]) -> TokenizedExample:
    """Build valid next-token supervision with the first token unscored."""
    return TokenizedExample(
        name, "g", "train", ids, (0,) + (1,) * (len(ids) - 1), "tok", "tpl"
    )


def test_partial_slots_are_inert_and_cursor_stable() -> None:
    """Every real example appears once in a stable seeded order."""
    examples = [_example(str(index), (1, index + 2, 3)) for index in range(5)]
    strategy = dense.DenseBatchStrategy(
        contracts.BatchShape(2, 2, 5, 0, 10), sft_batching.TokenTargets()
    )
    updates = list(strategy.iter_updates(examples, seed=7))
    again = list(strategy.iter_updates(examples, seed=7))
    assert len(updates) == 2
    names = [name for update in updates for name in update.example_ids]
    assert len(names) == len(set(names)) == 5
    assert names == [name for update in again for name in update.example_ids]
    resumed = list(strategy.iter_updates(examples, start_update=1, seed=7))
    assert resumed[0].example_ids == updates[1].example_ids
    last = updates[1]
    assert np.asarray(last.active).tolist() == [True, False]
    assert int(np.asarray(last.microbatches["loss_mask"]).sum()) == 2
    assert int(np.asarray(last.microbatches["attention_mask"]).sum()) == 3
    assert np.asarray(last.microbatches["input_ids"]).shape == (2, 2, 5)
    assert np.asarray(last.microbatches["loss_mask"])[1].sum() == 0


def test_invalid_ids_and_position_zero_target_rejected() -> None:
    """Bad token bounds and unscorable labels fail before JAX execution."""
    invalid = _example("bad", (1, 10))
    with pytest.raises(ValueError, match="Invalid input sequence"):
        list(
            dense.DenseBatchStrategy(
                contracts.BatchShape(1, 1, 2, 0, 10),
                sft_batching.TokenTargets(),
            ).iter_updates([invalid], seed=0)
        )
    early = TokenizedExample(
        "early", "g", "train", (1, 2), (1, 0), "tok", "tpl"
    )
    with pytest.raises(ValueError, match="Invalid tokenized"):
        list(
            dense.DenseBatchStrategy(
                contracts.BatchShape(1, 1, 2, 0, 10),
                sft_batching.TokenTargets(),
            ).iter_updates([early], seed=0)
        )


def test_padded_update_matches_real_subset() -> None:
    """Inactive microbatches and padded rows leave the optimizer unchanged."""
    example = _example("one", (1, 2, 3))
    physical = next(
        dense.DenseBatchStrategy(
            contracts.BatchShape(2, 2, 4, 0, 5), sft_batching.TokenTargets()
        ).iter_updates([example], seed=0)
    )
    inventory = parameters.build_inventory(
        {"weight": ()},
        format_id="synthetic/1",
        decayed_names=frozenset(),
    )
    state = adamw.initialize_state({"weight": jnp.float32(1)}, inventory)

    def terms(
        params: dict[str, jax.Array], batch: dict[str, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Measure a differentiable token-weighted synthetic objective."""
        mask = batch["loss_mask"].astype(jnp.float32)
        values = params["weight"] * batch["input_ids"].astype(jnp.float32)
        return jnp.sum(mask * values * values), jnp.sum(mask)

    update = step.make_step(terms, inventory, adamw.AdamWConfig(0.1))
    padded = update(state, physical.microbatches, jnp.asarray(physical.active))
    compact = {
        key: value[:1, :1] for key, value in physical.microbatches.items()
    }
    subset = update(state, compact, jnp.asarray([True]))
    assert bool(padded.committed)
    np.testing.assert_allclose(padded.loss, subset.loss, rtol=1e-6)
    np.testing.assert_allclose(
        padded.state["params"]["weight"],
        subset.state["params"]["weight"],
        rtol=1e-6,
    )
