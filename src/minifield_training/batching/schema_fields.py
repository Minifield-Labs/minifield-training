"""Request-aligned schema batches with explicit token and weighting policies."""

from collections.abc import Callable, Iterator, Sequence
import dataclasses
import hashlib
import math

import numpy as np

from minifield_training.batching import contracts
from minifield_training.batching import packing
from minifield_training.datasets import fields
from minifield_training.kernels import types


@dataclasses.dataclass(frozen=True)
class Shape:
    """Static TPU buckets; overflow is explicit and never truncates labels."""

    microbatches: int
    requests: int
    source_tokens: int
    schema_tokens: int
    schema_rows: int
    vocab_size: int
    pad_token_id: int
    schema_sequences: int | None = None

    @property
    def capacity(self) -> int:
        """Number of real or padded requests in a logical update."""
        return self.microbatches * self.requests

    @property
    def packed_sequences(self) -> int:
        """Encoder rows of ``schema_tokens`` that hold one request's rows.

        The default of one encoder row per schema row always fits. A smaller
        explicit count packs several short schema rows into each encoder row.
        """
        return (
            self.schema_rows
            if self.schema_sequences is None
            else self.schema_sequences
        )

    def __post_init__(self) -> None:
        """Require explicit dimensions, vocabulary, and padding admission."""
        if (
            min(
                self.microbatches,
                self.requests,
                self.source_tokens,
                self.schema_tokens,
                self.schema_rows,
                self.vocab_size,
                self.packed_sequences,
            )
            < 1
            or not 0 <= self.pad_token_id < self.vocab_size
        ):
            raise ValueError("Invalid schema batch dimensions or token policy")


def _allocate(shape: Shape) -> contracts.HostBatch:
    """Keep request axis first inside each physical batch for row sharding."""
    base = (shape.microbatches, shape.requests)
    row = (*base, shape.schema_rows)
    source = (*base, shape.source_tokens)
    result: contracts.HostBatch = {}
    for name in ("source_ids", "source_mask", "selectable"):
        result[name] = np.zeros(source, dtype=np.int32)
    for name in ("schema_ids", "schema_mask"):
        result[name] = np.zeros((*row, shape.schema_tokens), dtype=np.int32)
    for name in ("source_ids", "schema_ids"):
        result[name].fill(shape.pad_token_id)
    for name in ("field_owner", "kind", "row_mask", "token_supervised"):
        result[name] = np.zeros(row, dtype=np.int32)
    result["row_seed"] = np.zeros(row, dtype=np.uint32)
    for name in ("target", "field_weight"):
        result[name] = np.zeros(row, dtype=np.float32)
    result["token_target"] = np.zeros(
        (*row, shape.source_tokens), dtype=np.float32
    )
    # Packed encoder inputs: short schema rows share fixed encoder rows.
    packed = (*base, shape.packed_sequences, shape.schema_tokens)
    result["packed_schema_ids"] = np.full(
        packed, shape.pad_token_id, dtype=np.int32
    )
    for name in ("packed_schema_segments", "packed_schema_positions"):
        result[name] = np.zeros(packed, dtype=np.int32)
    result["schema_token_index"] = np.zeros(
        (*row, shape.schema_tokens), dtype=np.int32
    )
    return result


def _write(
    arrays: contracts.HostBatch,
    slot: tuple[int, int],
    record: fields.Record,
    shape: Shape,
    seed: int,
    update: int,
) -> None:
    """Write a whole request, keeping every candidate in its field group."""
    length = len(record.source.ids)
    rows = sum(len(field.rows) for field in record.fields)
    if (
        length > shape.source_tokens
        or rows > shape.schema_rows
        or any(
            len(ids) > shape.schema_tokens
            for field in record.fields
            for ids in field.rows
        )
    ):
        raise ValueError(f"Record {record.id} exceeds configured batch shape")
    arrays["source_ids"][*slot, :length] = record.source.ids
    arrays["source_mask"][*slot, :length] = 1
    arrays["selectable"][*slot, :length] = record.source.selectable
    row = 0
    for owner, field in enumerate(record.fields):
        for index, ids in enumerate(field.rows):
            target_slot = (*slot, row)
            arrays["schema_ids"][*target_slot, : len(ids)] = ids
            arrays["schema_mask"][*target_slot, : len(ids)] = 1
            arrays["field_owner"][target_slot] = owner
            arrays["kind"][target_slot] = field.kind
            arrays["row_mask"][target_slot] = 1
            arrays["target"][target_slot] = field.targets[index]
            arrays["field_weight"][target_slot] = float(
                field.supervised and index == 0
            )
            arrays["token_supervised"][target_slot] = int(
                field.token_supervised
            )
            if field.span is not None:
                arrays["token_target"][
                    *target_slot, field.span[0] : field.span[1]
                ] = 1
            identity = f"{seed}:{update}:{record.id}:{field.key}:{index}"
            arrays["row_seed"][target_slot] = int.from_bytes(
                hashlib.sha256(identity.encode()).digest()[:4], "little"
            )
            row += 1
    try:
        packed = packing.pack(
            [ids for field in record.fields for ids in field.rows],
            shape.packed_sequences,
            shape.schema_tokens,
            pad_token_id=shape.pad_token_id,
        )
    except ValueError as error:
        raise ValueError(
            f"Record {record.id} exceeds configured batch shape"
        ) from error
    arrays["packed_schema_ids"][slot] = packed.input_ids
    arrays["packed_schema_segments"][slot] = packed.segment_ids
    arrays["packed_schema_positions"][slot] = packed.positions
    token_index, _ = packing.gather_index(
        packed.placements, shape.schema_tokens, shape.schema_tokens
    )
    arrays["schema_token_index"][*slot, :rows] = token_index


def build(
    records: Sequence[fields.Record],
    shape: Shape,
    *,
    seed: int,
    update: int,
    weighting: Callable[[Sequence[int]], Sequence[float]],
    allow_unsupervised: bool = False,
) -> contracts.PhysicalUpdate:
    """Apply caller-owned logical field weights before physical slicing."""
    if not records or len(records) > shape.capacity:
        raise ValueError("Logical update has invalid request count")
    if len({record.id for record in records}) != len(records):
        raise ValueError("Duplicate example ID")
    for record in records:
        fields.validate(record, vocab_size=shape.vocab_size)
    arrays = _allocate(shape)
    for index, record in enumerate(records):
        _write(
            arrays, divmod(index, shape.requests), record, shape, seed, update
        )
    return weighted_update(
        arrays,
        weighting,
        microbatches=shape.microbatches,
        slots=shape.requests,
        filled=len(records),
        example_ids=tuple(record.id for record in records),
        allow_unsupervised=allow_unsupervised,
    )


def weighted_update(
    arrays: contracts.HostBatch,
    weighting: Callable[[Sequence[int]], Sequence[float]],
    *,
    microbatches: int,
    slots: int,
    filled: int,
    example_ids: tuple[str, ...],
    allow_unsupervised: bool,
) -> contracts.PhysicalUpdate:
    """Weight labeled fields over the whole update and mark active slots.

    ``arrays`` holds ``field_weight`` (1 for labeled fields) and ``kind``.
    The caller's weighting runs once, before physical microbatch slicing.
    ``filled`` of the ``microbatches * slots`` physical slots hold data, in
    microbatch-major order; later microbatches are inactive.
    """
    labeled = arrays["field_weight"].astype(bool)
    weights = np.asarray(
        weighting(arrays["kind"][labeled].tolist()), dtype=np.float32
    )
    if (
        weights.shape != (int(labeled.sum()),)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
    ):
        raise ValueError("Invalid field weighting result")
    arrays["field_weight"][labeled] = weights
    if not np.any(weights) and not allow_unsupervised:
        raise ValueError("no_supervision")
    # Host arrays are accepted by the engine and transferred one microbatch
    # at a time; retaining the complete dataset on accelerators is unnecessary.
    batch: types.DeviceBatch = arrays  # type: ignore[assignment]
    active = np.arange(microbatches) < math.ceil(filled / slots)
    return contracts.PhysicalUpdate(batch, active, example_ids)


def bucket(
    records: Sequence[fields.Record],
    maximum: Shape,
    *,
    min_tokens: int = 1,
    min_rows: int = 1,
) -> Shape:
    """Choose power-of-two observation buckets within caller admission caps."""
    if not records or min(min_tokens, min_rows) < 1:
        raise ValueError("Bucket requires records and positive minimums")

    def axis(required: int, limit: int, minimum: int) -> int:
        if not 1 <= required <= limit:
            raise ValueError(
                f"Record requires {required} positions; limit is {limit}"
            )
        return min(limit, max(minimum, 1 << (required - 1).bit_length()))

    schema_tokens = axis(
        max(
            len(row)
            for record in records
            for field in record.fields
            for row in field.rows
        ),
        maximum.schema_tokens,
        min_tokens,
    )
    sequences = None
    if maximum.schema_sequences is not None:
        lengths = [
            [len(row) for field in record.fields for row in field.rows]
            for record in records
        ]

        def required(tokens: int) -> int:
            return max(packing.rows_required(row, tokens) for row in lengths)

        # Row length and packed sequence count trade off: widen packed rows
        # until the request fits the sequence cap, within the token cap.
        while (
            required(schema_tokens) > maximum.schema_sequences
            and schema_tokens < maximum.schema_tokens
        ):
            schema_tokens = min(maximum.schema_tokens, schema_tokens * 2)
        sequences = axis(required(schema_tokens), maximum.schema_sequences, 1)
    return dataclasses.replace(
        maximum,
        source_tokens=axis(
            max(len(record.source.ids) for record in records),
            maximum.source_tokens,
            min_tokens,
        ),
        schema_tokens=schema_tokens,
        schema_sequences=sequences,
        schema_rows=axis(
            max(
                sum(len(field.rows) for field in record.fields)
                for record in records
            ),
            maximum.schema_rows,
            min_rows,
        ),
    )


@dataclasses.dataclass(frozen=True)
class SchemaBatchStrategy:
    """Pack whole requests using injected objective weights and token policy."""

    shape: Shape
    weighting: Callable[[Sequence[int]], Sequence[float]]
    min_tokens: int = 1
    min_rows: int = 1
    fixed_shape: bool = False

    def update_count(self, examples: Sequence[fields.Record]) -> int:
        """Count complete and padded updates independently of record order."""
        return math.ceil(len(examples) / self.shape.capacity)

    def pack(
        self,
        examples: Sequence[fields.Record],
        *,
        seed: int,
        update: int,
        allow_unsupervised: bool = False,
    ) -> contracts.PhysicalUpdate:
        """Pack a complete update with fixed or bucketed observation axes."""
        return build(
            examples,
            self.shape
            if self.fixed_shape
            else bucket(
                examples,
                self.shape,
                min_tokens=self.min_tokens,
                min_rows=self.min_rows,
            ),
            seed=seed,
            update=update,
            weighting=self.weighting,
            allow_unsupervised=allow_unsupervised,
        )

    def iter_updates(
        self,
        examples: Sequence[fields.Record],
        *,
        seed: int,
        start_update: int = 0,
        shuffle: bool = True,
    ) -> Iterator[contracts.PhysicalUpdate]:
        """Implement the shared batch strategy contract for schema records."""
        if seed < 0 or start_update < 0:
            raise ValueError("Invalid batch seed or cursor")
        if len({record.id for record in examples}) != len(examples):
            raise ValueError("Duplicate example ID")
        order = (
            np.random.default_rng(seed).permutation(len(examples))
            if shuffle
            else np.arange(len(examples))
        )
        for update in range(start_update, self.update_count(examples)):
            selected = order[
                update
                * self.shape.capacity : (update + 1)
                * self.shape.capacity
            ]
            yield self.pack(
                [examples[int(index)] for index in selected],
                seed=seed,
                update=update,
            )
