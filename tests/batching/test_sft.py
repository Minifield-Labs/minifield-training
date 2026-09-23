"""Deterministic physical SFT update fixtures."""

from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from minifield_training.batching.sft import iter_updates
from minifield_training.core import parameters
from minifield_training.datasets.tokenization import TokenizedExample
from minifield_training.engine import step
from minifield_training.optimizers import adamw


class _Settings(TypedDict):
    """Typed keyword arguments for a fixed physical update shape."""

    microbatches: int
    rows_per_microbatch: int
    sequence_length: int
    pad_token_id: int
    vocab_size: int
    seed: int


def _example(name: str, ids: tuple[int, ...]) -> TokenizedExample:
    """Build valid next-token supervision with the first token unscored."""
    return TokenizedExample(
        name, "g", "train", ids, (0,) + (1,) * (len(ids) - 1), "tok", "tpl"
    )


def test_partial_slots_are_inert_and_cursor_stable() -> None:
    """Every real example appears once in a stable seeded order."""
    examples = [_example(str(index), (1, index + 2, 3)) for index in range(5)]
    settings: _Settings = dict(
        microbatches=2,
        rows_per_microbatch=2,
        sequence_length=5,
        pad_token_id=0,
        vocab_size=10,
        seed=7,
    )
    updates = list(iter_updates(examples, **settings))
    again = list(iter_updates(examples, **settings))
    assert len(updates) == 2
    names = [name for update in updates for name in update.example_ids]
    assert len(names) == len(set(names)) == 5
    assert names == [name for update in again for name in update.example_ids]
    resumed = list(iter_updates(examples, start_update=1, **settings))
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
    with pytest.raises(ValueError, match="invalid tokenized"):
        list(
            iter_updates(
                [invalid],
                microbatches=1,
                rows_per_microbatch=1,
                sequence_length=2,
                pad_token_id=0,
                vocab_size=10,
                seed=0,
            )
        )
    early = TokenizedExample(
        "early", "g", "train", (1, 2), (1, 0), "tok", "tpl"
    )
    with pytest.raises(ValueError, match="invalid tokenized"):
        list(
            iter_updates(
                [early],
                microbatches=1,
                rows_per_microbatch=1,
                sequence_length=2,
                pad_token_id=0,
                vocab_size=10,
                seed=0,
            )
        )


def test_padded_update_matches_real_subset() -> None:
    """Inactive microbatches and padded rows leave the optimizer unchanged."""
    example = _example("one", (1, 2, 3))
    physical = next(
        iter_updates(
            [example],
            microbatches=2,
            rows_per_microbatch=2,
            sequence_length=4,
            pad_token_id=0,
            vocab_size=5,
            seed=0,
        )
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
    padded = update(state, physical.microbatches, physical.active)
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
