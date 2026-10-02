"""Independent epoch order, partial chunks, resume, and deadline evidence."""

from collections.abc import Sequence
import dataclasses
import time

import numpy as np
import pytest

from minifield_training.batching import contracts
from minifield_training.batching import stream

_ORDERS = ((4, 0, 3, 1, 2), (1, 3, 2, 4, 0), (2, 4, 0, 3, 1))
_EXPECTED = [
    ("row-4", "row-0"),
    ("row-3", "row-1"),
    ("row-2",),
    ("row-1", "row-3"),
    ("row-2", "row-4"),
    ("row-0",),
    ("row-2", "row-4"),
    ("row-0", "row-3"),
    ("row-1",),
]
_READS = [
    (0, 0, 2),
    (0, 2, 4),
    (0, 4, 5),
    (1, 0, 2),
    (1, 2, 4),
    (1, 4, 5),
    (2, 0, 2),
    (2, 2, 4),
    (2, 4, 5),
]


@dataclasses.dataclass
class _Fixture:
    """Record caller operations over fixed independent epoch orders."""

    epochs: list[int] = dataclasses.field(default_factory=list)
    reads: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    compiled: list[str] = dataclasses.field(default_factory=list)
    packed: list[tuple[int, tuple[str, ...]]] = dataclasses.field(
        default_factory=list
    )

    def read_epoch(self, epoch: int) -> stream.ChunkReader[int]:
        """Open one ordered view while keeping chunk access observable."""
        self.epochs.append(epoch)

        def read(start: int, stop: int) -> Sequence[int]:
            self.reads.append((epoch, start, stop))
            return _ORDERS[epoch][start:stop]

        return read

    def compile_record(self, raw: int) -> str:
        """Attach a distinct representation to selected raw values only."""
        value = f"row-{raw}"
        self.compiled.append(value)
        return value

    def pack(
        self, records: Sequence[str], update: int
    ) -> contracts.PhysicalUpdate:
        """Expose the global update identity without altering record order."""
        self.packed.append((update, tuple(records)))
        return contracts.PhysicalUpdate({}, np.asarray([True]), tuple(records))

    def source(self, record_count: int = 5) -> stream.EpochStream[int, str]:
        """Compose an in-memory source with the real shared cursor owner."""
        return stream.EpochStream(
            record_count=record_count,
            capacity=2,
            epochs=3,
            read_epoch=self.read_epoch,
            compile_record=self.compile_record,
            pack=self.pack,
        )


@pytest.mark.parametrize("cursor", (0, 1, 2, 3, 4, 8, 9, 10))
def test_partial_chunks_and_resume_preserve_global_order(cursor: int) -> None:
    """Boundary and in-epoch resumes compile only their exact unread suffix."""
    fixture = _Fixture()
    source = fixture.source()
    assert source.updates_per_epoch == 3
    batches = list(source(cursor))
    expected = _EXPECTED[cursor:]
    assert [batch.example_ids for batch in batches] == expected
    assert fixture.packed == list(enumerate(_EXPECTED))[cursor:]
    assert fixture.reads == _READS[cursor:]
    assert fixture.compiled == [row for chunk in expected for row in chunk]
    assert fixture.epochs == list(range(cursor // 3, 3))


def test_deadline_stops_before_reading_or_compiling_the_next_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stopped stream can resume from the next unconsumed global update."""
    fixture = _Fixture()
    times = iter((9.0, 10.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    batches = list(fixture.source()(0, 10.0))
    assert [batch.example_ids for batch in batches] == _EXPECTED[:1]
    assert fixture.reads == _READS[:1]
    assert fixture.compiled == ["row-4", "row-0"]
    monkeypatch.setattr(time, "monotonic", lambda: 9.0)
    batches.extend(fixture.source()(1, 10.0))
    assert [batch.example_ids for batch in batches] == _EXPECTED
    assert fixture.packed == list(enumerate(_EXPECTED))
    assert fixture.reads == _READS


def test_expired_deadline_reads_no_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The absolute cutoff is inclusive and leaves all rows uncompiled."""
    fixture = _Fixture()
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    assert not list(fixture.source()(4, 10.0))
    assert not fixture.reads
    assert not fixture.compiled
    assert not fixture.packed


def test_empty_source_never_opens_an_epoch() -> None:
    """An empty immutable split has zero updates and no cursor division."""
    fixture = _Fixture()
    source = fixture.source(record_count=0)
    assert source.updates_per_epoch == 0
    assert not list(source(0))
    assert not fixture.epochs


@pytest.mark.parametrize(
    ("record_count", "capacity", "epochs"),
    ((-1, 2, 3), (5, 0, 3), (5, 2, 0), (5, True, 3)),
)
def test_invalid_dimensions_are_rejected(
    record_count: int, capacity: int, epochs: int
) -> None:
    """Invalid dimensions cannot alter exact update and epoch arithmetic."""
    with pytest.raises(ValueError, match="Invalid stream"):
        dataclasses.replace(
            _Fixture().source(),
            record_count=record_count,
            capacity=capacity,
            epochs=epochs,
        )


@pytest.mark.parametrize("cursor", (-1, True))
def test_invalid_cursor_is_rejected(cursor: int) -> None:
    """Invalid cursors fail before asking the caller to prepare data."""
    fixture = _Fixture()
    with pytest.raises(ValueError, match="Invalid update cursor"):
        list(fixture.source()(cursor))
    assert not fixture.epochs


@pytest.mark.parametrize("row_count", (1, 3))
def test_reader_cannot_silently_drop_or_repeat_records(row_count: int) -> None:
    """A reader must return the requested count before compilation starts."""
    fixture = _Fixture()
    source = dataclasses.replace(
        fixture.source(),
        read_epoch=lambda _epoch: lambda _start, _stop: [0] * row_count,
    )
    with pytest.raises(ValueError, match="incorrect row count"):
        next(source(0))
    assert not fixture.compiled
    assert not fixture.packed
