"""Hand-checked placements and round trips for sequence packing."""

import numpy as np
import pytest

from minifield_training.batching import packing

_SEQUENCES = ((11, 12, 13), (21, 22, 23, 24, 25), (31, 32), (41, 42, 43, 44))


def test_first_fit_decreasing_places_whole_sequences() -> None:
    """Lengths 5, 4, 3, 2 fill two rows of 8 in a hand-derived layout."""
    packed = packing.pack(_SEQUENCES, 2, 8, pad_token_id=0)
    assert packed.placements == (
        packing.Placement(0, 5, 3),
        packing.Placement(0, 0, 5),
        packing.Placement(1, 4, 2),
        packing.Placement(1, 0, 4),
    )
    np.testing.assert_array_equal(
        packed.input_ids,
        [[21, 22, 23, 24, 25, 11, 12, 13], [41, 42, 43, 44, 31, 32, 0, 0]],
    )
    np.testing.assert_array_equal(
        packed.segment_ids, [[2, 2, 2, 2, 2, 1, 1, 1], [4, 4, 4, 4, 3, 3, 0, 0]]
    )
    np.testing.assert_array_equal(
        packed.positions, [[0, 1, 2, 3, 4, 0, 1, 2], [0, 1, 2, 3, 0, 1, 0, 0]]
    )
    assert packing.rows_required([len(item) for item in _SEQUENCES], 8) == 2


def test_gather_index_recovers_each_sequence() -> None:
    """Flattened packed tokens gather back into padded input-order rows."""
    packed = packing.pack(_SEQUENCES, 3, 8, pad_token_id=99)
    index, mask = packing.gather_index(packed.placements, 8, 6)
    gathered = packed.input_ids.reshape(-1)[index]
    for item, sequence in enumerate(_SEQUENCES):
        np.testing.assert_array_equal(gathered[item, : len(sequence)], sequence)
        np.testing.assert_array_equal(
            mask[item], [1] * len(sequence) + [0] * (6 - len(sequence))
        )
    np.testing.assert_array_equal(index[mask == 0], 0)
    # The third row stays empty padding because two rows suffice.
    np.testing.assert_array_equal(packed.segment_ids[2], 0)
    np.testing.assert_array_equal(packed.input_ids[2], 99)


def test_equal_lengths_keep_input_order() -> None:
    """Ties place earlier sequences first, so packing is reproducible."""
    packed = packing.pack(((1, 1), (2, 2), (3, 3)), 2, 4, pad_token_id=0)
    assert [placement.row for placement in packed.placements] == [0, 0, 1]
    assert [placement.start for placement in packed.placements] == [0, 2, 0]


@pytest.mark.parametrize(
    ("sequences", "rows", "message"),
    [
        (((1,) * 9,), 4, "1 to 8 tokens"),
        (((),), 4, "1 to 8 tokens"),
        (((1,) * 5, (2,) * 5, (3,) * 5), 2, "more than 2 packed rows"),
    ],
)
def test_admission_rejects_instead_of_truncating(
    sequences: tuple[tuple[int, ...], ...], rows: int, message: str
) -> None:
    """Overlong, empty, and over-capacity inputs fail before any write."""
    with pytest.raises(ValueError, match=message):
        packing.pack(sequences, rows, 8, pad_token_id=0)


def test_gather_view_must_fit_each_sequence() -> None:
    """A narrower gather view would silently drop tokens, so it raises."""
    packed = packing.pack(_SEQUENCES, 2, 8, pad_token_id=0)
    with pytest.raises(ValueError, match="wider than the gather view"):
        packing.gather_index(packed.placements, 8, 4)


def test_exact_fill_and_one_token_sequences() -> None:
    """Rows that end on a boundary and 1-token sequences pack without gaps."""
    packed = packing.pack(
        ((1, 2, 3), (4,), (5, 6), (7, 8)), 2, 4, pad_token_id=0
    )
    np.testing.assert_array_equal(
        packed.input_ids, [[1, 2, 3, 4], [5, 6, 7, 8]]
    )
    np.testing.assert_array_equal(
        packed.segment_ids, [[1, 1, 1, 2], [3, 3, 4, 4]]
    )
    np.testing.assert_array_equal(
        packed.positions, [[0, 1, 2, 0], [0, 1, 0, 1]]
    )
    assert packing.rows_required([3, 1, 2, 2], 4) == 2


def test_plan_rows_first_fit_closes_full_rows() -> None:
    """Hand-traced first-fit: rows close when tokens or questions run out."""
    sizes = [(6, 1), (5, 1), (4, 1), (3, 1), (2, 1), (7, 1)]
    # 6 opens row A; 5 opens B; 4 fills A (closes); 3 and 2 fill B (closes);
    # 7 opens C, which the end of input flushes.
    assert packing.plan_rows(sizes, (10, 3)) == [[0, 2], [1, 3, 4], [5]]
    # With 2 question slots, a row closes after 2 items regardless of tokens.
    assert packing.plan_rows([(1, 1)] * 3, (100, 2)) == [[0, 1], [2]]


def test_plan_rows_bounds_open_rows() -> None:
    """Opening a third row closes the fullest of the 2 open ones first."""
    sizes = [(9, 1), (8, 1), (7, 1), (2, 1)]
    assert packing.plan_rows(sizes, (10, 5), open_limit=2) == [
        [0],
        [1, 3],
        [2],
    ]


def test_plan_rows_rejects_items_that_cant_fit_alone() -> None:
    """An oversized item or a bad setting fails before any planning."""
    with pytest.raises(ValueError, match="Item 1 can't fit"):
        packing.plan_rows([(3, 1), (11, 1)], (10, 3))
    with pytest.raises(ValueError, match="Invalid open-row limit"):
        packing.plan_rows([(3, 1)], (10, 3), open_limit=0)


def test_plan_updates_places_every_item_once_per_seed() -> None:
    """Each seed gives one reproducible, complete, grouped plan."""
    rng = np.random.default_rng(0)
    sizes = [
        (int(rng.integers(1, 9)), int(rng.integers(1, 3))) for _ in range(50)
    ]
    plan = packing.plan_updates(sizes, (16, 4), 3, seed=7)
    assert plan == packing.plan_updates(sizes, (16, 4), 3, seed=7)
    assert plan != packing.plan_updates(sizes, (16, 4), 3, seed=8)
    placed = [index for update in plan for row in update for index in row]
    assert sorted(placed) == list(range(50))
    assert all(1 <= len(update) <= 3 for update in plan)
    assert all(len(update) == 3 for update in plan[:-1])
    for row in (row for update in plan for row in update):
        assert sum(sizes[index][0] for index in row) <= 16
        assert sum(sizes[index][1] for index in row) <= 4


def test_thin_keeps_each_source_at_its_weight_reproducibly() -> None:
    """Unlisted sources stay whole; a weighted one keeps about its share."""
    labels = ["ner"] * 4000 + ["roles"] * 500
    kept = packing.thin(labels, {"ner": 0.25}, seed=3)
    assert kept == packing.thin(labels, {"ner": 0.25}, seed=3)
    assert kept != packing.thin(labels, {"ner": 0.25}, seed=4)
    assert kept == sorted(kept)
    assert [index for index in kept if labels[index] == "roles"] == list(
        range(4000, 4500)
    )
    ner = sum(labels[index] == "ner" for index in kept)
    # Binomial(4000, 0.25): mean 1000, SD about 27.
    assert 900 < ner < 1100


def test_thin_rejects_weights_outside_zero_to_one() -> None:
    """Upweighting would repeat records within an epoch."""
    for weight in (0.0, 1.5):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            packing.thin(["a"], {"a": weight}, seed=0)
