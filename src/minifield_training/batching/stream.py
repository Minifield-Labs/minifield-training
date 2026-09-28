"""Bounded record compilation with deterministic epoch and update cursors."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import time

from minifield_training.batching import contracts

type ChunkReader[RawT] = Callable[[int, int], Sequence[RawT]]


@dataclass(frozen=True)
class EpochStream[RawT, RecordT]:
    """Replay fixed-capacity updates over caller-owned ordered epoch views.

    ``read_epoch(epoch)`` returns a reader for the half-open row interval
    ``[start, stop)``. It must preserve that epoch's order on every replay.
    Only selected rows are compiled; ``pack`` receives their global update
    index so task-specific randomness can reproduce an interrupted run.
    """

    record_count: int
    capacity: int
    epochs: int
    read_epoch: Callable[[int], ChunkReader[RawT]]
    compile_record: Callable[[RawT], RecordT]
    pack: Callable[[Sequence[RecordT], int], contracts.PhysicalUpdate]

    def __post_init__(self) -> None:
        """Reject counts that cannot define exact update boundaries."""
        values = (self.record_count, self.capacity, self.epochs)
        if (
            any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in values
            )
            or self.record_count < 0
            or min(self.capacity, self.epochs) < 1
        ):
            raise ValueError("Invalid stream record count, capacity, or epochs")

    @property
    def updates_per_epoch(self) -> int:
        """Include a final partial chunk without dropping real records."""
        return (self.record_count + self.capacity - 1) // self.capacity

    def __call__(
        self, start_update: int, deadline: float | None = None, /
    ) -> Iterator[contracts.PhysicalUpdate]:
        """Read only unread chunks, checking the deadline between updates."""
        if (
            not isinstance(start_update, int)
            or isinstance(start_update, bool)
            or start_update < 0
        ):
            raise ValueError("Invalid update cursor")
        if not self.updates_per_epoch:
            return
        epoch, offset = divmod(start_update, self.updates_per_epoch)
        for epoch_index in range(epoch, self.epochs):
            read = self.read_epoch(epoch_index)
            for index in range(offset, self.updates_per_epoch):
                if deadline is not None and time.monotonic() >= deadline:
                    return
                start = index * self.capacity
                stop = min(start + self.capacity, self.record_count)
                rows = read(start, stop)
                if len(rows) != stop - start:
                    raise ValueError(
                        "Epoch reader returned an incorrect row count"
                    )
                records = [self.compile_record(row) for row in rows]
                update = epoch_index * self.updates_per_epoch + index
                yield self.pack(records, update)
            offset = 0
