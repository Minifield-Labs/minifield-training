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
