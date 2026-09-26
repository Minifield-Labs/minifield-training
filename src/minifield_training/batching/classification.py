"""One-label decision supervision for the shared batching strategy."""

from dataclasses import dataclass

import numpy as np

from minifield_training.batching import contracts
from minifield_training.datasets.labeled import LabeledSequence


@dataclass(frozen=True)
class ClassTargets:
    """Admit playable labels and mask padded decision rows."""

    allowed_classes: tuple[bool, ...]
    padding_label: int

    def __post_init__(self) -> None:
        """Require at least one allowed class and a valid padding ID."""
        if (
            not self.allowed_classes
            or not any(self.allowed_classes)
            or not 0 <= self.padding_label < len(self.allowed_classes)
        ):
            raise ValueError("Invalid classification target configuration")

    def validate(self, example: LabeledSequence) -> None:
        """Reject malformed or masked labels before the objective runs."""
        if (
            not example.group_id
            or not isinstance(example.label, int)
            or isinstance(example.label, bool)
            or not 0 <= example.label < len(self.allowed_classes)
            or not self.allowed_classes[example.label]
        ):
            raise ValueError(f"Invalid labeled sequence: {example.id}")

    def allocate(self, shape: contracts.BatchShape) -> contracts.HostBatch:
        """Supply finite padding labels behind a false valid-row mask."""
        dimensions = (shape.microbatches, shape.rows_per_microbatch)
        return {
            "labels": np.full(dimensions, self.padding_label, dtype=np.int32),
            "valid_rows": np.zeros(dimensions, dtype=np.bool_),
        }

    def write(
        self,
        targets: contracts.HostBatch,
        slot: tuple[int, int],
        example: LabeledSequence,
    ) -> None:
        """Store one admitted class ID and mark its row real."""
        targets["labels"][slot] = example.label
        targets["valid_rows"][slot] = True
