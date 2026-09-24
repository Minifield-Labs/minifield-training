"""Fixed-shape labeled sequence admission and cursor checks."""

import numpy as np
import pytest

from minifield_training.batching import classification
from minifield_training.datasets.labeled import LabeledSequence


def test_padded_decision_batches_preserve_records() -> None:
    """Exact observations and labels survive physical padding."""
    examples = [
        LabeledSequence("a", "episode-1", (3, 4, 5), 1),
        LabeledSequence("b", "episode-1", (6, 7), 0),
        LabeledSequence("c", "episode-2", (8,), 1),
    ]
    batches = list(
        classification.iter_updates(
            examples,
            microbatches=2,
            rows_per_microbatch=2,
            sequence_length=4,
            pad_token_id=0,
            vocab_size=10,
            allowed_classes=(True, True, False),
            padding_label=2,
            seed=4,
        )
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


def test_rejects_masked_class_and_overlength() -> None:
    """Host admission catches bad decisions and silent truncation."""

    def admit(example: LabeledSequence) -> None:
        """Exercise the public batch boundary with fixed tiny settings."""
        list(
            classification.iter_updates(
                [example],
                microbatches=1,
                rows_per_microbatch=1,
                sequence_length=2,
                pad_token_id=0,
                vocab_size=9,
                allowed_classes=(True, True, False),
                padding_label=2,
                seed=0,
            )
        )

    with pytest.raises(ValueError, match="Invalid labeled sequence"):
        admit(LabeledSequence("pad", "ep", (1,), 2))
    with pytest.raises(ValueError, match="Invalid labeled sequence"):
        admit(LabeledSequence("long", "ep", (1, 2, 3), 1))
