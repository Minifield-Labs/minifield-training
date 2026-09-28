"""Request-aligned MagicBox batches with replay keys and type-balanced mass."""

from collections.abc import Sequence
import dataclasses
import hashlib
import math

import numpy as np

from minifield_training.batching import contracts
from minifield_training.datasets import magicbox
from minifield_training.kernels import types


@dataclasses.dataclass(frozen=True)
class Shape:
    """Static TPU buckets; overflow is explicit and never truncates labels."""

    microbatches: int = 4
    requests: int = 8
    source_tokens: int = 1024
    schema_tokens: int = 512
    schema_rows: int = 64

    @property
    def capacity(self) -> int:
        """Number of real or padded requests in a logical update."""
        return self.microbatches * self.requests

    def __post_init__(self) -> None:
        """Require positive static axes within the encoder context limit."""
        if (
            min(dataclasses.astuple(self)) < 1
            or max(self.source_tokens, self.schema_tokens) > 8192
        ):
            raise ValueError("Invalid MagicBox batch dimensions")


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
    for name in ("field_owner", "kind", "row_mask", "token_supervised"):
        result[name] = np.zeros(row, dtype=np.int32)
    result["row_seed"] = np.zeros(row, dtype=np.uint32)
    for name in ("target", "field_weight"):
        result[name] = np.zeros(row, dtype=np.float32)
    result["token_target"] = np.zeros(
        (*row, shape.source_tokens), dtype=np.float32
    )
    return result


def _write(
    arrays: contracts.HostBatch,
    slot: tuple[int, int],
    record: magicbox.Record,
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
        raise ValueError(f"Record {record.id} exceeds configured batch bucket")
    if any(
        not 0 <= token < 65536
        for ids in (
            record.source.ids,
            *(ids for field in record.fields for ids in field.rows),
        )
        for token in ids
    ):
        raise ValueError("Token outside pretrained vocabulary")
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


def build(
    records: Sequence[magicbox.Record],
    shape: Shape,
    *,
    seed: int,
    update: int,
    type_weights: tuple[float, float, float, float] = (1, 1, 1, 1),
    allow_unsupervised: bool = False,
) -> contracts.PhysicalUpdate:
    """Normalize over all devices and microbatches before slicing arrays."""
    if not records or len(records) > shape.capacity:
        raise ValueError("Logical update has invalid request count")
    arrays = _allocate(shape)
    for index, record in enumerate(records):
        _write(
            arrays, divmod(index, shape.requests), record, shape, seed, update
        )
    counts = [
        float(np.sum(arrays["field_weight"] * (arrays["kind"] == kind)))
        for kind in range(4)
    ]
    if any(not math.isfinite(weight) or weight < 0 for weight in type_weights):
        raise ValueError("Invalid type weights")
    active_types = sum(
        weight
        for count, weight in zip(counts, type_weights, strict=True)
        if count > 0
    )
    if not active_types and not allow_unsupervised:
        raise ValueError("no_supervision")
    for kind, count in enumerate(counts):
        if count and active_types:
            selected = arrays["kind"] == kind
            arrays["field_weight"][selected] *= type_weights[kind] / (
                count * active_types
            )
    # Host arrays are accepted by the engine and transferred one microbatch
    # at a time; retaining the complete dataset on accelerators is unnecessary.
    batch: types.DeviceBatch = arrays  # type: ignore[assignment]
    active = np.arange(shape.microbatches) < math.ceil(
        len(records) / shape.requests
    )
    return contracts.PhysicalUpdate(
        batch, active, tuple(record.id for record in records)
    )
