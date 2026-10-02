"""Bounded record compilation with deterministic epoch and update cursors."""

from collections.abc import Callable, Iterator, Sequence
import dataclasses
from dataclasses import dataclass
import queue
import threading
import time

from minifield_training.batching import contracts

type ChunkReader[RawT] = Callable[[int, int], Sequence[RawT]]
# One epoch's updates, each a list of rows of record indices.
type EpochPlan = Sequence[Sequence[Sequence[int]]]


class Prefetch[T]:
    """Produce up to ``depth`` items ahead of the consumer on a thread.

    Items arrive in source order, and a source exception is raised on the
    consuming thread. ``close()`` stops the producer, which then closes the
    source on its own thread, so a generator is never driven from 2 threads.
    Tokenizers and NumPy release the GIL, so host preparation overlaps
    accelerator work.
    """

    def __init__(self, source: Iterator[T], depth: int) -> None:
        if depth < 1:
            raise ValueError("Prefetch depth must be positive")
        self._source = source
        self._items: queue.Queue[tuple[str, object]] = queue.Queue(depth)
        self._stop = threading.Event()
        self._finished = False
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _put(self, item: tuple[str, object]) -> bool:
        """Offer one item until it's queued or the consumer has closed."""
        while not self._stop.is_set():
            try:
                self._items.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        """Run the source to completion, a stop request, or an error."""
        try:
            for item in self._source:
                if not self._put(("item", item)):
                    return
            self._put(("done", None))
        # Any source failure, including KeyboardInterrupt, belongs to the
        # consumer, which re-raises it on the training thread.
        except BaseException as error:  # pylint: disable=broad-exception-caught
            self._put(("error", error))
        finally:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()

    def __iter__(self) -> "Prefetch[T]":
        return self

    def __next__(self) -> T:
        if self._finished:
            raise StopIteration
        kind, value = self._items.get()
        if kind == "item":
            return value  # type: ignore[return-value]
        self._finished = True
        if kind == "error":
            assert isinstance(value, BaseException)
            raise value
        raise StopIteration

    def close(self) -> None:
        """Stop producing and wait for the source to close."""
        self._stop.set()
        self._thread.join()


def _prefetched[T](updates: Iterator[T], depth: int) -> Iterator[T]:
    """Return the iterator itself, or a prefetcher when depth is positive."""
    return Prefetch(updates, depth) if depth else updates


@dataclass(frozen=True)
class EpochStream[RawT, RecordT]:
    """Replay fixed-capacity updates over caller-owned ordered epoch views.

    ``read_epoch(epoch)`` returns a reader for the half-open row interval
    ``[start, stop)``. It must preserve that epoch's order on every replay.
    Only selected rows are compiled; ``pack`` receives their global update
    index so task-specific randomness can reproduce an interrupted run.
    ``prefetch`` prepares that many updates ahead on a background thread.
    """

    record_count: int
    capacity: int
    epochs: int
    read_epoch: Callable[[int], ChunkReader[RawT]]
    compile_record: Callable[[RawT], RecordT]
    pack: Callable[[Sequence[RecordT], int], contracts.PhysicalUpdate]
    prefetch: int = 0

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
            or self.prefetch < 0
        ):
            raise ValueError("Invalid stream record count, capacity, or epochs")

    @property
    def updates_per_epoch(self) -> int:
        """Include a final partial chunk without dropping real records."""
        return (self.record_count + self.capacity - 1) // self.capacity

    @property
    def total_updates(self) -> int:
        """Count every update across all epochs."""
        return self.epochs * self.updates_per_epoch

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
        return _prefetched(self._updates(start_update, deadline), self.prefetch)

    def _updates(
        self, start_update: int, deadline: float | None
    ) -> Iterator[contracts.PhysicalUpdate]:
        """Yield the updates from ``start_update`` in replay order."""
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


@dataclass(frozen=True)
class PlannedStream[RawT, RecordT]:
    """Replay a deterministic per-epoch plan of updates made of packed rows.

    ``plan(epoch)`` returns that epoch's updates; each update is a list of
    rows and each row a list of record indices. It must return the same plan
    for an epoch on every call, so the global update cursor resumes exactly.
    ``read(index)`` returns one raw record, and only planned records are
    compiled, when their update comes up. ``pack(rows, update)`` receives the
    compiled records grouped by row. The number of records per update varies;
    the rows per update don't. ``prefetch`` prepares that many updates ahead
    on a background thread.
    """

    epochs: int
    plan: Callable[[int], EpochPlan]
    read: Callable[[int], RawT]
    compile_record: Callable[[RawT], RecordT]
    pack: Callable[[Sequence[Sequence[RecordT]], int], contracts.PhysicalUpdate]
    prefetch: int = 0
    _plans: dict[int, EpochPlan] = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Reject an empty run or a negative prefetch depth."""
        if self.epochs < 1 or self.prefetch < 0:
            raise ValueError("Invalid planned stream epochs or prefetch")

    def epoch_plan(self, epoch: int) -> EpochPlan:
        """Build an epoch's plan once and reuse it."""
        if epoch not in self._plans:
            self._plans[epoch] = self.plan(epoch)
        return self._plans[epoch]

    @property
    def total_updates(self) -> int:
        """Count every update across all epochs."""
        return sum(len(self.epoch_plan(epoch)) for epoch in range(self.epochs))

    def __call__(
        self, start_update: int, deadline: float | None = None, /
    ) -> Iterator[contracts.PhysicalUpdate]:
        """Start at a global update index, checking the deadline between."""
        if (
            not isinstance(start_update, int)
            or isinstance(start_update, bool)
            or start_update < 0
        ):
            raise ValueError("Invalid update cursor")
        return _prefetched(self._updates(start_update, deadline), self.prefetch)

    def _updates(
        self, start_update: int, deadline: float | None
    ) -> Iterator[contracts.PhysicalUpdate]:
        """Yield planned updates from ``start_update`` in replay order."""
        first = 0
        for epoch in range(self.epochs):
            updates = self.epoch_plan(epoch)
            for offset in range(max(start_update - first, 0), len(updates)):
                if deadline is not None and time.monotonic() >= deadline:
                    return
                rows = [
                    [self.compile_record(self.read(index)) for index in row]
                    for row in updates[offset]
                ]
                yield self.pack(rows, first + offset)
            first += len(updates)
