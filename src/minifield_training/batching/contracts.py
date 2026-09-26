"""Shared records and structural interfaces for batch construction."""

from abc import abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from minifield_training.kernels.types import DeviceBatch

type HostBatch = dict[str, NDArray[np.generic]]


class SequenceRecord(Protocol):
    """Observation fields shared by token and decision supervision."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable identity of a real observation."""

    @property
    @abstractmethod
    def input_ids(self) -> tuple[int, ...]:
        """Complete, unpadded observation tokens."""


@dataclass(frozen=True)
class BatchShape:
    """Fixed physical dimensions and admitted token vocabulary."""

    microbatches: int
    rows_per_microbatch: int
    sequence_length: int
    pad_token_id: int
    vocab_size: int

    def __post_init__(self) -> None:
        """Reject invalid dimensions before any examples are consumed."""
        if (
            min(
                self.microbatches,
                self.rows_per_microbatch,
                self.sequence_length,
                self.vocab_size,
            )
            < 1
            or not 0 <= self.pad_token_id < self.vocab_size
        ):
            raise ValueError("Invalid batch shape or vocabulary")

    @property
    def capacity(self) -> int:
        """Number of physical rows in one logical update."""
        return self.microbatches * self.rows_per_microbatch


@dataclass(frozen=True)
class PhysicalUpdate:
    """One logical update, host slot flags, and real record identities."""

    microbatches: DeviceBatch
    active: NDArray[np.bool_]
    example_ids: tuple[str, ...]


class TargetEncoder[RecordT](Protocol):
    """Task-specific admission and supervision for a dense batch."""

    @abstractmethod
    def validate(self, example: RecordT) -> None:
        """Reject malformed supervision before device transfer."""

    @abstractmethod
    def allocate(self, shape: BatchShape) -> HostBatch:
        """Allocate finite inert targets for a complete physical update."""

    @abstractmethod
    def write(
        self, targets: HostBatch, slot: tuple[int, int], example: RecordT
    ) -> None:
        """Write one real row's targets into the supplied host arrays."""


class BatchStrategy[RecordT](Protocol):
    """Interchangeable construction of resumable logical updates."""

    @property
    @abstractmethod
    def shape(self) -> BatchShape:
        """Physical dimensions owned by this strategy."""

    @abstractmethod
    def update_count(self, examples: Sequence[RecordT]) -> int:
        """Number of logical updates per epoch, independent of shuffle seed."""

    @abstractmethod
    def iter_updates(
        self,
        examples: Sequence[RecordT],
        *,
        seed: int,
        start_update: int = 0,
        shuffle: bool = True,
    ) -> Iterator[PhysicalUpdate]:
        """Visit each real record once in a reproducible epoch order."""


class BatchSource(Protocol):
    """A replayable stream starting at a global update cursor."""

    @abstractmethod
    def __call__(
        self, start_update: int, deadline: float | None, /
    ) -> Iterator[PhysicalUpdate]:
        """Yield unread updates and release resources when closed."""
