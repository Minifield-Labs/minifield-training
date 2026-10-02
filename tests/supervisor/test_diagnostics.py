"""Host-memory evidence remains available without accelerator dependencies."""

import json
from pathlib import Path

import pytest

from minifield_training.supervisor import diagnostics


def test_memory_snapshot(tmp_path: Path) -> None:
    """Parse independent Linux fixtures and tolerate missing proc files."""
    assert not diagnostics.memory_snapshot(tmp_path)
    (tmp_path / "self").mkdir()
    (tmp_path / "self/status").write_text(
        "Name:\tpython\nVmRSS:\t123 kB\nVmHWM:\t456 kB\n"
    )
    (tmp_path / "meminfo").write_text("MemAvailable: 789 kB\n")
    assert diagnostics.memory_snapshot(tmp_path) == {
        "rss_kib": 123,
        "peak_rss_kib": 456,
        "available_kib": 789,
    }


def test_monitor_flushes_on_failure(tmp_path: Path) -> None:
    """A failed stage still leaves start and exit samples on disk."""
    with (
        pytest.raises(RuntimeError, match="stage failed"),
        diagnostics.monitor(tmp_path, interval=60),
    ):
        raise RuntimeError("stage failed")
    samples = [
        json.loads(line)
        for line in next(tmp_path.glob("memory-*.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert len(samples) == 2
    assert all(sample["event"] == "host_memory" for sample in samples)
    assert samples[1]["elapsed_seconds"] >= samples[0]["elapsed_seconds"]
    with (
        pytest.raises(ValueError, match="positive"),
        diagnostics.monitor(tmp_path, interval=0),
    ):
        pass
