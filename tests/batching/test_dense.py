"""Both supervision modes obey the same physical batch contract."""

import numpy as np

from minifield_training.batching import classification
from minifield_training.batching import contracts
from minifield_training.batching import dense
from minifield_training.batching import sft
from minifield_training.datasets.labeled import LabeledSequence
from minifield_training.datasets.tokenization import TokenizedExample


def test_target_encoders_share_observations_order_and_resume() -> None:
    """Changing supervision preserves the shared observation and cursor path."""
    shape = contracts.BatchShape(2, 1, 4, 0, 10)
    tokens = [
        TokenizedExample(
            str(i), "game", "train", (1, i + 2), (0, 1), "tok", "tpl"
        )
        for i in range(3)
    ]
    labels = [
        LabeledSequence(row.id, "game", row.input_ids, 1) for row in tokens
    ]
    token_strategy: contracts.BatchStrategy[TokenizedExample] = (
        dense.DenseBatchStrategy(shape, sft.TokenTargets())
    )
    class_strategy: contracts.BatchStrategy[LabeledSequence] = (
        dense.DenseBatchStrategy(
            shape, classification.ClassTargets((True, True, False), 2)
        )
    )
    token_updates = list(
        token_strategy.iter_updates(tokens, seed=0, shuffle=False)
    )
    class_updates = list(
        class_strategy.iter_updates(labels, seed=0, shuffle=False)
    )
    assert [batch.example_ids for batch in token_updates] == [
        ("0", "1"),
        ("2",),
    ]
    for token_batch, class_batch in zip(
        token_updates, class_updates, strict=True
    ):
        assert (
            type(token_batch) is type(class_batch) is contracts.PhysicalUpdate
        )
        assert token_batch.example_ids == class_batch.example_ids
        assert isinstance(token_batch.active, np.ndarray)
        np.testing.assert_array_equal(token_batch.active, class_batch.active)
        for key in ("input_ids", "attention_mask"):
            np.testing.assert_array_equal(
                token_batch.microbatches[key], class_batch.microbatches[key]
            )
    resumed = next(
        class_strategy.iter_updates(
            labels, seed=0, shuffle=False, start_update=1
        )
    )
    assert resumed.example_ids == ("2",)
    assert resumed.active.tolist() == [True, False]
    np.testing.assert_array_equal(
        resumed.microbatches["input_ids"], [[[1, 4, 0, 0]], [[0, 0, 0, 0]]]
    )
    np.testing.assert_array_equal(resumed.microbatches["labels"], [[1], [2]])
    np.testing.assert_array_equal(
        resumed.microbatches["valid_rows"], [[True], [False]]
    )
    np.testing.assert_array_equal(
        token_updates[-1].microbatches["loss_mask"],
        [[[0, 1, 0, 0]], [[0, 0, 0, 0]]],
    )
