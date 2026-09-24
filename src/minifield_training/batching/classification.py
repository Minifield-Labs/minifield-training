"""Fixed-shape physical batches for one-label sequence decisions."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from minifield_training.datasets.labeled import LabeledSequence
from minifield_training.kernels.types import DeviceBatch


@dataclass(frozen=True)
class PhysicalUpdate:
    """One logical update and its real source-record IDs."""

    microbatches: DeviceBatch
    active: NDArray[np.bool_]
    example_ids: tuple[str, ...]


def iter_updates(
    examples: Sequence[LabeledSequence],
    *,
    microbatches: int,
    rows_per_microbatch: int,
    sequence_length: int,
    pad_token_id: int,
    vocab_size: int,
    allowed_classes: tuple[bool, ...],
    padding_label: int,
    seed: int,
    start_update: int = 0,
) -> Iterator[PhysicalUpdate]:
    """Shuffle once per seed and yield finite, right-padded decision batches.

    ``start_update`` skips complete updates in the same deterministic order.
    The caller changes ``seed`` per epoch and stores the global cursor.
    """
    if (
        min(microbatches, rows_per_microbatch, sequence_length, vocab_size) < 1
        or not 0 <= pad_token_id < vocab_size
        or not allowed_classes
        or not any(allowed_classes)
        or not 0 <= padding_label < len(allowed_classes)
        or seed < 0
        or start_update < 0
    ):
        raise ValueError("Invalid classification batch configuration")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("Duplicate decision record ID")
    for example in examples:
        if (
            not example.id
            or not example.group_id
            or not 1 <= len(example.input_ids) <= sequence_length
            or not isinstance(example.label, int)
            or isinstance(example.label, bool)
            or not 0 <= example.label < len(allowed_classes)
            or not allowed_classes[example.label]
            or any(
                not isinstance(token, int)
                or isinstance(token, bool)
                or not 0 <= token < vocab_size
                for token in example.input_ids
            )
        ):
            raise ValueError(f"Invalid labeled sequence: {example.id}")
    order = np.random.default_rng(seed).permutation(len(examples))
    capacity = microbatches * rows_per_microbatch
    for update_index in range(start_update, math.ceil(len(order) / capacity)):
        chosen = order[update_index * capacity : (update_index + 1) * capacity]
        ids = np.full(
            (microbatches, rows_per_microbatch, sequence_length),
            pad_token_id,
            dtype=np.int32,
        )
        attention = np.zeros_like(ids)
        labels = np.full(
            (microbatches, rows_per_microbatch),
            padding_label,
            dtype=np.int32,
        )
        valid_rows = np.zeros(labels.shape, dtype=np.bool_)
        active = np.zeros(microbatches, dtype=np.bool_)
        names: list[str] = []
        for slot, source_index in enumerate(chosen):
            example = examples[int(source_index)]
            microbatch, row = divmod(slot, rows_per_microbatch)
            length = len(example.input_ids)
            ids[microbatch, row, :length] = example.input_ids
            attention[microbatch, row, :length] = 1
            labels[microbatch, row] = example.label
            valid_rows[microbatch, row] = True
            active[microbatch] = True
            names.append(example.id)
        yield PhysicalUpdate(
            {
                "input_ids": jnp.asarray(ids),
                "attention_mask": jnp.asarray(attention),
                "labels": jnp.asarray(labels),
                "valid_rows": jnp.asarray(valid_rows),
            },
            active,
            tuple(names),
        )
