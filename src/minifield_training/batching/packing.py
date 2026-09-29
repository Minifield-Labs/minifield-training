"""Deterministic packing of short token sequences into fixed-length rows."""

from collections.abc import Sequence
import dataclasses

import numpy as np
from numpy.typing import NDArray


@dataclasses.dataclass(frozen=True)
class Placement:
    """Where one sequence landed: packed row, first column, and token count."""

    row: int
    start: int
    length: int


@dataclasses.dataclass(frozen=True)
class PackedRows:
    """Host arrays for segment-isolated encoding of several sequences per row.

    ``segment_ids`` is 0 for padding and ``index + 1`` for the sequence at
    ``index`` of the packer's input. ``positions`` restart at 0 for every
    sequence. ``placements`` follows input order.
    """

    input_ids: NDArray[np.int32]
    segment_ids: NDArray[np.int32]
    positions: NDArray[np.int32]
    placements: tuple[Placement, ...]


def _first_fit_decreasing(
    lengths: Sequence[int], length: int, rows: int | None
) -> tuple[list[tuple[int, int]], int]:
    """Place the longest sequences first; ties keep input order."""
    if length < 1:
        raise ValueError("Packed row length must be positive")
    if any(size < 1 or size > length for size in lengths):
        raise ValueError(
            f"Every packed sequence needs 1 to {length} tokens; got "
            f"{min(lengths)} to {max(lengths)}"
        )
    used: list[int] = []
    starts: list[tuple[int, int]] = [(0, 0)] * len(lengths)
    for index in sorted(range(len(lengths)), key=lambda item: -lengths[item]):
        size = lengths[index]
        row = next(
            (
                candidate
                for candidate, filled in enumerate(used)
                if filled + size <= length
            ),
            len(used),
        )
        if row == len(used):
            if rows is not None and row == rows:
                raise ValueError(
                    f"Sequences need more than {rows} packed rows of "
                    f"{length} tokens"
                )
            used.append(0)
        starts[index] = (row, used[row])
        used[row] += size
    return starts, len(used)


def rows_required(lengths: Sequence[int], length: int) -> int:
    """Return how many rows of ``length`` tokens ``pack`` would fill."""
    return _first_fit_decreasing(lengths, length, None)[1]


def pack(
    sequences: Sequence[Sequence[int]],
    rows: int,
    length: int,
    *,
    pad_token_id: int,
) -> PackedRows:
    """Pack whole sequences into ``rows`` fixed rows without truncation.

    Sequences never split across rows. Overflow raises instead of dropping
    tokens, so a caller's configured capacity is an admission limit.
    """
    if rows < 1:
        raise ValueError("Packed row count must be positive")
    lengths = [len(sequence) for sequence in sequences]
    starts, _ = _first_fit_decreasing(lengths, length, rows)
    input_ids = np.full((rows, length), pad_token_id, dtype=np.int32)
    segment_ids = np.zeros((rows, length), dtype=np.int32)
    positions = np.zeros((rows, length), dtype=np.int32)
    placements = []
    for index, (sequence, (row, start)) in enumerate(
        zip(sequences, starts, strict=True)
    ):
        stop = start + len(sequence)
        input_ids[row, start:stop] = sequence
        segment_ids[row, start:stop] = index + 1
        positions[row, start:stop] = np.arange(len(sequence))
        placements.append(Placement(row, start, len(sequence)))
    return PackedRows(input_ids, segment_ids, positions, tuple(placements))


def gather_index(
    placements: Sequence[Placement], row_length: int, width: int
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Map each sequence back to a ``[sequences, width]`` view of its tokens.

    Indices address the flattened ``[rows * row_length]`` packed axis. Columns
    past a sequence's length point at index 0 and carry mask 0.
    """
    if any(placement.length > width for placement in placements):
        raise ValueError("A packed sequence is wider than the gather view")
    columns = np.arange(width, dtype=np.int32)
    index = np.zeros((len(placements), width), dtype=np.int32)
    mask = np.zeros((len(placements), width), dtype=np.int32)
    for item, placement in enumerate(placements):
        valid = columns < placement.length
        index[item] = np.where(
            valid, placement.row * row_length + placement.start + columns, 0
        )
        mask[item] = valid
    return index, mask
