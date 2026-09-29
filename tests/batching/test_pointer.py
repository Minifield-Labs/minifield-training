"""Joint sequence layout, allowed tokens, and pointer targets."""

from collections.abc import Sequence

import numpy as np
import pytest

from minifield_training.batching import pointer as batching
from minifield_training.datasets import fields
from minifield_training.datasets import pointer


def _record(identifier: str = "r") -> pointer.Record:
    source = fields.Encoding(
        (1, 5, 6, 7),
        ((0, 0), (0, 3), (4, 7), (7, 7)),
        (True, False, False, False),
    )
    return pointer.Record(
        identifier,
        "Ada Lee",
        source,
        (
            pointer.Question(
                "name",
                fields.Kind.EXTRACT,
                (1, 9),
                (pointer.Option("absent", (1, 8)),),
                (0.0,),
                True,
                (1, 3),
            ),
            pointer.Question(
                "tier",
                fields.Kind.CHOICE,
                (1, 10, 10),
                (pointer.Option("a", (1, 11)), pointer.Option("b", (1, 12))),
                (0.25, 0.75),
                True,
            ),
            pointer.Question(
                "open",
                fields.Kind.BINARY,
                (1, 13),
                (
                    pointer.Option("false", (1, 14)),
                    pointer.Option("true", (1, 15)),
                ),
                (0.0, 0.0),
                False,
            ),
        ),
    )


def _weights(kinds: Sequence[int]) -> Sequence[float]:
    return [float(kind + 1) for kind in kinds]


def test_layout_places_queries_options_then_source() -> None:
    """Hand-derived positions for a 3-question request."""
    ids, placed = batching.layout(_record())
    assert ids == [1, 9, 1, 8, 1, 10, 10, 1, 11, 1, 12] + [
        1,
        13,
        1,
        14,
        1,
        15,
        1,
        5,
        6,
        7,
    ]
    assert placed.queries == (0, 4, 11)
    assert placed.options == ((2,), (7, 9), (13, 15))
    assert placed.source_start == 17


def test_targets_masks_and_weights() -> None:
    """Extraction may point at selectable source tokens or its absent mark."""
    shape = batching.Shape(2, 1, 24, 4, 16, 0)
    update = batching.build([_record()], shape, weighting=_weights)
    arrays = {
        key: np.asarray(value) for key, value in update.microbatches.items()
    }
    assert update.active.tolist() == [True, False]
    np.testing.assert_array_equal(arrays["query_index"][0, 0], [0, 4, 11, 0])
    np.testing.assert_array_equal(arrays["kind"][0, 0], [0, 1, 2, 0])
    # Source tokens 1 and 2 are selectable; the BOS and empty token aren't.
    np.testing.assert_array_equal(
        np.flatnonzero(arrays["allowed"][0, 0, 0]), [2, 18, 19]
    )
    np.testing.assert_array_equal(
        np.flatnonzero(arrays["allowed"][0, 0, 1]), [7, 9]
    )
    np.testing.assert_array_equal(
        np.flatnonzero(arrays["allowed"][0, 0, 2]), [13, 15]
    )
    assert not arrays["allowed"][0, 0, 3].any()
    # "Ada Lee" spans source tokens 1..2: start at 18, end at 19.
    np.testing.assert_array_equal(
        np.flatnonzero(arrays["start_target"][0, 0, 0]), [18]
    )
    np.testing.assert_array_equal(
        np.flatnonzero(arrays["end_target"][0, 0, 0]), [19]
    )
    assert arrays["start_target"][0, 0, 1, 7] == 0.25
    assert arrays["end_target"][0, 0, 1, 9] == 0.75
    assert not arrays["start_target"][0, 0, 2].any()
    # Supervised extract (kind 0) and choice (kind 1) get the injected weights.
    np.testing.assert_array_equal(arrays["field_weight"][0, 0], [1, 2, 0, 0])
    np.testing.assert_array_equal(
        arrays["input_mask"][0, 0], [1] * 21 + [0] * 3
    )
    assert not arrays["input_mask"][1].any()


def test_overflow_raises_instead_of_truncating() -> None:
    """Too many tokens or questions fails before any write is used."""
    for shape in (
        batching.Shape(1, 1, 20, 4, 16, 0),
        batching.Shape(1, 1, 24, 2, 16, 0),
    ):
        with pytest.raises(ValueError, match="exceeds configured"):
            batching.build([_record()], shape, weighting=_weights)


def test_strategy_visits_each_record_once() -> None:
    """Seeded order covers every record and pads the final update."""
    strategy = batching.PointerBatchStrategy(
        batching.Shape(1, 2, 24, 4, 16, 0), _weights
    )
    records = [_record(str(index)) for index in range(3)]
    updates = list(strategy.iter_updates(records, seed=5))
    assert strategy.update_count(records) == len(updates) == 2
    seen = [identity for update in updates for identity in update.example_ids]
    assert sorted(seen) == ["0", "1", "2"]
    resumed = list(strategy.iter_updates(records, seed=5, start_update=1))
    assert resumed[0].example_ids == updates[1].example_ids
