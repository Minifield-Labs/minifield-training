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


def plan_rows(
    sizes: Sequence[Sequence[int]],
    capacity: Sequence[int],
    *,
    open_limit: int = 64,
    close_below: float = 0.05,
) -> list[list[int]]:
    """Group items into rows by online first-fit, keeping the given order.

    ``sizes[i]`` and ``capacity`` share dimensions, such as tokens and
    questions. Each item joins the first open row with room in every
    dimension, or opens a new row. A row closes once any dimension's free
    space falls below ``close_below`` of its capacity. Opening a row beyond
    ``open_limit`` closes the fullest open row by the first dimension, which
    bounds how far an item can land behind its position. Returns item indices
    per row: closed rows in closing order, then still-open rows. The result
    depends only on the inputs.
    """
    if open_limit < 1 or not 0 <= close_below < 1 or not capacity:
        raise ValueError("Invalid open-row limit, threshold, or capacity")
    rows: list[list[int]] = []
    open_rows: list[tuple[list[int], list[int]]] = []
    for index, size in enumerate(sizes):
        if len(size) != len(capacity) or any(
            not 0 <= need <= limit
            for need, limit in zip(size, capacity, strict=True)
        ):
            raise ValueError(f"Item {index} can't fit an empty row")
        position = next(
            (
                slot
                for slot, (used, _) in enumerate(open_rows)
                if all(
                    have + need <= limit
                    for have, need, limit in zip(
                        used, size, capacity, strict=True
                    )
                )
            ),
            None,
        )
        if position is None:
            if len(open_rows) == open_limit:
                fullest = max(
                    range(len(open_rows)),
                    key=lambda slot: open_rows[slot][0][0],
                )
                rows.append(open_rows.pop(fullest)[1])
            open_rows.append(([0] * len(capacity), []))
            position = len(open_rows) - 1
        used, members = open_rows[position]
        members.append(index)
        for dimension, need in enumerate(size):
            used[dimension] += need
        if any(
            limit - have < close_below * limit
            for have, limit in zip(used, capacity, strict=True)
        ):
            rows.append(open_rows.pop(position)[1])
    rows.extend(members for _, members in open_rows)
    return rows


def plan_updates(
    sizes: Sequence[Sequence[int]],
    capacity: Sequence[int],
    rows_per_update: int,
    *,
    seed: int,
    open_limit: int = 64,
    close_below: float = 0.05,
) -> list[list[list[int]]]:
    """Shuffle items by ``seed``, pack them into rows, and group the rows.

    Returns updates, each a list of at most ``rows_per_update`` rows of
    original item indices. Every item appears exactly once. Rows follow the
    shuffled order instead of sorting by size, so updates stay mixed.
    """
    if rows_per_update < 1:
        raise ValueError("Updates need at least one row")
    order = np.random.default_rng(seed).permutation(len(sizes))
    rows = [
        [int(order[index]) for index in row]
        for row in plan_rows(
            [sizes[int(index)] for index in order],
            capacity,
            open_limit=open_limit,
            close_below=close_below,
        )
    ]
    return [
        rows[start : start + rows_per_update]
        for start in range(0, len(rows), rows_per_update)
    ]
