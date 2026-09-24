"""One dense batching strategy with interchangeable target encoders."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math

import jax.numpy as jnp
import numpy as np

from minifield_training.batching import contracts


@dataclass(frozen=True)
class DenseBatchStrategy[RecordT: contracts.SequenceRecord]:
    """Own dense admission, ordering, padding, and device transfer."""

    shape: contracts.BatchShape
    targets: contracts.TargetEncoder[RecordT]

    def update_count(self, examples: Sequence[RecordT]) -> int:
        """Count complete and partial dense updates in an epoch."""
        return math.ceil(len(examples) / self.shape.capacity)

    def _validate(self, examples: Sequence[RecordT]) -> None:
        """Check observation identity and token bounds once per epoch."""
        if len({example.id for example in examples}) != len(examples):
            raise ValueError("Duplicate example ID")
        for example in examples:
            if (
                not example.id
                or not 1 <= len(example.input_ids) <= self.shape.sequence_length
                or any(
                    not isinstance(token, int)
                    or isinstance(token, bool)
                    or not 0 <= token < self.shape.vocab_size
                    for token in example.input_ids
                )
            ):
                raise ValueError(f"Invalid input sequence: {example.id}")
            self.targets.validate(example)

    def iter_updates(
        self,
        examples: Sequence[RecordT],
        *,
        seed: int,
        start_update: int = 0,
        shuffle: bool = True,
    ) -> Iterator[contracts.PhysicalUpdate]:
        """Pad one record per row; resume only at complete update boundaries."""
        if seed < 0 or start_update < 0:
            raise ValueError("Invalid batch seed or cursor")
        self._validate(examples)
        order = (
            np.random.default_rng(seed).permutation(len(examples))
            if shuffle
            else np.arange(len(examples))
        )
        for index in range(start_update, self.update_count(examples)):
            selected = order[
                index * self.shape.capacity : (index + 1) * self.shape.capacity
            ]
            yield self._pack([examples[int(item)] for item in selected])

    def _pack(self, examples: Sequence[RecordT]) -> contracts.PhysicalUpdate:
        """Share observation padding while delegating only target semantics."""
        shape = self.shape
        ids = np.full(
            (
                shape.microbatches,
                shape.rows_per_microbatch,
                shape.sequence_length,
            ),
            shape.pad_token_id,
            dtype=np.int32,
        )
        attention = np.zeros_like(ids)
        active = np.zeros(shape.microbatches, dtype=np.bool_)
        targets = self.targets.allocate(shape)
        if {"input_ids", "attention_mask"}.intersection(targets):
            raise ValueError("Target encoder cannot replace observation arrays")
        for index, example in enumerate(examples):
            slot = divmod(index, shape.rows_per_microbatch)
            length = len(example.input_ids)
            ids[slot][:length] = example.input_ids
            attention[slot][:length] = 1
            active[slot[0]] = True
            self.targets.write(targets, slot, example)
        arrays = {"input_ids": ids, "attention_mask": attention, **targets}
        return contracts.PhysicalUpdate(
            {name: jnp.asarray(value) for name, value in arrays.items()},
            active,
            tuple(example.id for example in examples),
        )
