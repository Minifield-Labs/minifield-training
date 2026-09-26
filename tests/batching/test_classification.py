"""Fixed-shape labeled sequence admission and cursor checks."""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import pytest

from minifield_training.batching import classification as batching
from minifield_training.batching import contracts
from minifield_training.batching import dense
from minifield_training.datasets.labeled import LabeledSequence


def test_padded_decision_batches_preserve_records() -> None:
    """Exact observations and labels survive physical padding."""
    examples = [
        LabeledSequence("a", "episode-1", (3, 4, 5), 1),
        LabeledSequence("b", "episode-1", (6, 7), 0),
        LabeledSequence("c", "episode-2", (8,), 1),
    ]
    batches = list(
        dense.DenseBatchStrategy(
            contracts.BatchShape(2, 2, 4, 0, 10),
            batching.ClassTargets((True, True, False), 2),
        ).iter_updates(examples, seed=4)
    )
    assert len(batches) == 1
    batch = batches[0]
    assert set(batch.example_ids) == {"a", "b", "c"}
    assert int(np.asarray(batch.microbatches["valid_rows"]).sum()) == 3
    assert int(np.asarray(batch.active).sum()) == 2
    assert int(np.asarray(batch.microbatches["labels"] == 2).sum()) == 1
    expected = {example.id: example.input_ids for example in examples}
    physical = np.asarray(batch.microbatches["input_ids"]).reshape(4, 4)
    for slot, row_id in enumerate(batch.example_ids):
        ids = expected[row_id]
        assert tuple(physical[slot, : len(ids)]) == ids


def test_active_flags_stay_on_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch construction doesn't place the host slot-selection vector."""
    asarray = jnp.asarray

    def reject_active_conversion(value: NDArray[np.generic]) -> jax.Array:
        """Catch an active-vector conversion while allowing model inputs."""
        if value.shape == (2,) and value.dtype == np.dtype(np.bool_):
            raise AssertionError("active flags were placed on device")
        return asarray(value)

    monkeypatch.setattr(jnp, "asarray", reject_active_conversion)
    batch = next(
        dense.DenseBatchStrategy(
            contracts.BatchShape(2, 1, 2, 0, 3),
            batching.ClassTargets((True,), 0),
        ).iter_updates([LabeledSequence("a", "ep", (1,), 0)], seed=0)
    )
    assert isinstance(batch.active, np.ndarray)
    assert batch.active.dtype == np.dtype(np.bool_)
    assert batch.active.tolist() == [True, False]


def test_rejects_masked_class_and_overlength() -> None:
    """Host admission catches bad decisions and silent truncation."""

    def admit(example: LabeledSequence) -> None:
        """Exercise the public batch boundary with fixed tiny settings."""
        list(
            dense.DenseBatchStrategy(
                contracts.BatchShape(1, 1, 2, 0, 9),
                batching.ClassTargets((True, True, False), 2),
            ).iter_updates([example], seed=0)
        )

    with pytest.raises(ValueError, match="Invalid labeled sequence"):
        admit(LabeledSequence("pad", "ep", (1,), 2))
    with pytest.raises(ValueError, match="Invalid input sequence"):
        admit(LabeledSequence("long", "ep", (1, 2, 3), 1))
