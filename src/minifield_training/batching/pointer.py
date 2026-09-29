"""Joint question-and-source sequences with pointer targets per question."""

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
    """Fixed request, token, and question capacity for one logical update."""

    microbatches: int
    requests: int
    sequence_tokens: int
    questions: int
    vocab_size: int
    pad_token_id: int

    @property
    def capacity(self) -> int:
        """Number of real or padded requests in a logical update."""
        return self.microbatches * self.requests

    def __post_init__(self) -> None:
        """Require positive dimensions and an admitted padding token."""
        if (
            min(
                self.microbatches,
                self.requests,
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
    """Concatenate each question with its options, then the source."""
    ids: list[int] = []
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
    """Keep the request axis first inside each physical batch for sharding."""
    base = (shape.microbatches, shape.requests)
    grid = (*base, shape.questions, shape.sequence_tokens)
    return {
        "input_ids": np.full(
            (*base, shape.sequence_tokens), shape.pad_token_id, np.int32
        ),
        "input_mask": np.zeros((*base, shape.sequence_tokens), np.int32),
        "query_index": np.zeros((*base, shape.questions), np.int32),
        "kind": np.zeros((*base, shape.questions), np.int32),
        "field_weight": np.zeros((*base, shape.questions), np.float32),
        "allowed": np.zeros(grid, np.int32),
        "start_target": np.zeros(grid, np.float32),
        "end_target": np.zeros(grid, np.float32),
    }


def _write(
    arrays: contracts.HostBatch,
    slot: tuple[int, int],
    record: pointer.Record,
    shape: Shape,
) -> None:
    """Write one request; every question may point at its allowed tokens."""
    ids, placed = layout(record)
    if len(ids) > shape.sequence_tokens or len(record.questions) > (
        shape.questions
    ):
        raise ValueError(f"Record {record.id} exceeds configured batch shape")
    arrays["input_ids"][*slot, : len(ids)] = ids
    arrays["input_mask"][*slot, : len(ids)] = 1
    selectable = placed.source_start + np.flatnonzero(record.source.selectable)
    for index, question in enumerate(record.questions):
        target = (*slot, index)
        markers = list(placed.options[index])
        arrays["query_index"][target] = placed.queries[index]
        arrays["kind"][target] = question.kind
        arrays["field_weight"][target] = float(question.supervised)
        arrays["allowed"][*target, markers] = 1
        for name in ("start_target", "end_target"):
            arrays[name][*target, markers] = question.targets
        if question.kind != fields.Kind.EXTRACT:
            continue
        arrays["allowed"][*target, selectable] = 1
        if question.span is not None:
            start, end = question.span
            present = 1 - question.targets[0]
            arrays["start_target"][
                *target, placed.source_start + start
            ] = present
            arrays["end_target"][
                *target, placed.source_start + end - 1
            ] = present


def build(
    records: Sequence[pointer.Record],
    shape: Shape,
    *,
    weighting: Callable[[Sequence[int]], Sequence[float]],
    allow_unsupervised: bool = False,
) -> contracts.PhysicalUpdate:
    """Apply caller-owned field weights across the whole logical update."""
    if not records or len(records) > shape.capacity:
        raise ValueError("Logical update has invalid request count")
    if len({record.id for record in records}) != len(records):
        raise ValueError("Duplicate example ID")
    for record in records:
        pointer.validate(record, vocab_size=shape.vocab_size)
    arrays = _allocate(shape)
    for index, record in enumerate(records):
        _write(arrays, divmod(index, shape.requests), record, shape)
    return schema_fields.weighted_update(
        arrays,
        weighting,
        microbatches=shape.microbatches,
        requests=shape.requests,
        example_ids=tuple(record.id for record in records),
        allow_unsupervised=allow_unsupervised,
    )


@dataclasses.dataclass(frozen=True)
class PointerBatchStrategy:
    """Pad whole requests to one fixed joint shape; never truncate."""

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
