"""Joint question-and-source sequences with pointer targets per question.

Each row holds one or more whole requests back to back. A request's tokens
share one segment ID and restart positions at 0, so a segment-aware encoder
reads it exactly as if it were alone. Every question may point only at tokens
of its own request.
"""

from collections.abc import Callable, Iterator, Sequence
import dataclasses
import math

import numpy as np

from minifield_training.batching import contracts
from minifield_training.batching import schema_fields
from minifield_training.datasets import fields
from minifield_training.datasets import pointer


@dataclasses.dataclass(frozen=True)
class Shape:
    """Fixed row, token, and per-row question capacity for one update."""

    microbatches: int
    rows: int
    sequence_tokens: int
    questions: int
    vocab_size: int
    pad_token_id: int

    @property
    def capacity(self) -> int:
        """Number of real or padded rows in a logical update."""
        return self.microbatches * self.rows

    def __post_init__(self) -> None:
        """Require positive dimensions and an admitted padding token."""
        if (
            min(
                self.microbatches,
                self.rows,
                self.sequence_tokens,
                self.questions,
                self.vocab_size,
            )
            < 1
            or not 0 <= self.pad_token_id < self.vocab_size
        ):
            raise ValueError("Invalid pointer batch dimensions or padding")


@dataclasses.dataclass(frozen=True)
class Layout:
    """Where one request's queries, option markers, and source landed."""

    queries: tuple[int, ...]
    options: tuple[tuple[int, ...], ...]
    source_start: int


def layout(record: pointer.Record) -> tuple[list[int], Layout]:
    """Join the prefix, each question with its options, then the source."""
    ids: list[int] = list(record.prefix)
    queries: list[int] = []
    options: list[tuple[int, ...]] = []
    for question in record.questions:
        queries.append(len(ids))
        ids.extend(question.query)
        markers = []
        for option in question.options:
            markers.append(len(ids))
            ids.extend(option.ids)
        options.append(tuple(markers))
    source_start = len(ids)
    ids.extend(record.source.ids)
    return ids, Layout(tuple(queries), tuple(options), source_start)


def _allocate(shape: Shape) -> contracts.HostBatch:
    """Keep the row axis first inside each physical batch for sharding."""
    base = (shape.microbatches, shape.rows)
    tokens = (*base, shape.sequence_tokens)
    grid = (*base, shape.questions, shape.sequence_tokens)
    return {
        "input_ids": np.full(tokens, shape.pad_token_id, np.int32),
        "input_mask": np.zeros(tokens, np.int32),
        "segment_ids": np.zeros(tokens, np.int32),
        "positions": np.zeros(tokens, np.int32),
        "query_index": np.zeros((*base, shape.questions), np.int32),
        "kind": np.zeros((*base, shape.questions), np.int32),
        "field_weight": np.zeros((*base, shape.questions), np.float32),
        "allowed": np.zeros(grid, np.int32),
        "start_target": np.zeros(grid, np.float32),
        "end_target": np.zeros(grid, np.float32),
    }


def _write_question(
    arrays: contracts.HostBatch,
    target: tuple[int, int, int],
    question: pointer.Question,
    markers: list[int],
    source: tuple[int, fields.Encoding],
) -> None:
    """Allow a question's own options and, for extraction, its source."""
    start, encoding = source
    arrays["kind"][target] = question.kind
    arrays["field_weight"][target] = float(question.supervised)
    arrays["allowed"][*target, markers] = 1
    for name in ("start_target", "end_target"):
        arrays[name][*target, markers] = question.targets
    if question.kind != fields.Kind.EXTRACT:
        return
    arrays["allowed"][*target, start + np.flatnonzero(encoding.selectable)] = 1
    if question.span is not None:
        present = 1 - question.targets[0]
        arrays["start_target"][*target, start + question.span[0]] = present
        arrays["end_target"][*target, start + question.span[1] - 1] = present


def _write_row(
    arrays: contracts.HostBatch,
    slot: tuple[int, int],
    records: Sequence[pointer.Record],
    shape: Shape,
) -> None:
    """Write whole requests back to back, one segment per request."""
    offset, question = 0, 0
    for segment, record in enumerate(records, start=1):
        ids, placed = layout(record)
        if offset + len(ids) > shape.sequence_tokens or question + len(
            record.questions
        ) > (shape.questions):
            raise ValueError(
                f"Row with record {record.id} exceeds configured batch shape"
            )
        span = slice(offset, offset + len(ids))
        arrays["input_ids"][*slot, span] = ids
        arrays["input_mask"][*slot, span] = 1
        arrays["segment_ids"][*slot, span] = segment
        arrays["positions"][*slot, span] = np.arange(len(ids))
        for index, item in enumerate(record.questions):
            target = (*slot, question + index)
            arrays["query_index"][target] = offset + placed.queries[index]
            _write_question(
                arrays,
                target,
                item,
                [offset + marker for marker in placed.options[index]],
                (offset + placed.source_start, record.source),
            )
        offset += len(ids)
        question += len(record.questions)


def build_rows(
    rows: Sequence[Sequence[pointer.Record]],
    shape: Shape,
    *,
    weighting: Callable[[Sequence[int]], Sequence[float]],
    allow_unsupervised: bool = False,
) -> contracts.PhysicalUpdate:
    """Pack rows of whole requests; weight fields across the whole update.

    ``example_ids`` lists every record in row order. Unused rows and later
    unused microbatches are inert padding.
    """
    records = [record for row in rows for record in row]
    if not rows or not all(rows) or len(rows) > shape.capacity:
        raise ValueError("Logical update has an invalid row count")
    if len({record.id for record in records}) != len(records):
        raise ValueError("Duplicate example ID")
    for record in records:
        pointer.validate(record, vocab_size=shape.vocab_size)
    arrays = _allocate(shape)
    for index, row in enumerate(rows):
        _write_row(arrays, divmod(index, shape.rows), row, shape)
    return schema_fields.weighted_update(
        arrays,
        weighting,
        microbatches=shape.microbatches,
        slots=shape.rows,
        filled=len(rows),
        example_ids=tuple(record.id for record in records),
        allow_unsupervised=allow_unsupervised,
    )


def build(
    records: Sequence[pointer.Record],
    shape: Shape,
    *,
    weighting: Callable[[Sequence[int]], Sequence[float]],
    allow_unsupervised: bool = False,
) -> contracts.PhysicalUpdate:
    """Place one request per row."""
    return build_rows(
        [[record] for record in records],
        shape,
        weighting=weighting,
        allow_unsupervised=allow_unsupervised,
    )


@dataclasses.dataclass(frozen=True)
class PointerBatchStrategy:
    """Pad whole requests, one per row, to one fixed shape; never truncate."""

    shape: Shape
    weighting: Callable[[Sequence[int]], Sequence[float]]

    def update_count(self, examples: Sequence[pointer.Record]) -> int:
        """Count complete and padded updates independently of order."""
        return math.ceil(len(examples) / self.shape.capacity)

    def pack(
        self,
        examples: Sequence[pointer.Record],
        *,
        seed: int,
        update: int,
        allow_unsupervised: bool = False,
    ) -> contracts.PhysicalUpdate:
        """Pack one logical update; layout depends only on each record."""
        del seed, update
        return build(
            examples,
            self.shape,
            weighting=self.weighting,
            allow_unsupervised=allow_unsupervised,
        )

    def pack_rows(
        self, rows: Sequence[Sequence[pointer.Record]], update: int
    ) -> contracts.PhysicalUpdate:
        """Pack planned rows of several requests each, for a planned stream."""
        del update
        return build_rows(rows, self.shape, weighting=self.weighting)

    def iter_updates(
        self,
        examples: Sequence[pointer.Record],
        *,
        seed: int,
        start_update: int = 0,
        shuffle: bool = True,
    ) -> Iterator[contracts.PhysicalUpdate]:
        """Visit each record once in a reproducible seeded order."""
        if seed < 0 or start_update < 0:
            raise ValueError("Invalid batch seed or cursor")
        order = (
            np.random.default_rng(seed).permutation(len(examples))
            if shuffle
            else np.arange(len(examples))
        )
        capacity = self.shape.capacity
        for update in range(start_update, self.update_count(examples)):
            yield self.pack(
                [
                    examples[int(index)]
                    for index in order[
                        update * capacity : (update + 1) * capacity
                    ]
                ],
                seed=seed,
                update=update,
            )
