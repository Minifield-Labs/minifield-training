"""Periodic Linux host-memory evidence without importing an accelerator."""

from collections.abc import Iterator
import contextlib
import json
import os
from pathlib import Path
import threading
import time


def memory_snapshot(proc: Path = Path("/proc")) -> dict[str, int]:
    """Read process RSS/peak and host availability in KiB when exposed."""
    result = {}
    for path, fields in (
        (proc / "self/status", {"VmRSS": "rss_kib", "VmHWM": "peak_rss_kib"}),
        (proc / "meminfo", {"MemAvailable": "available_kib"}),
    ):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            name, separator, value = line.partition(":")
            if separator and name in fields:
                result[fields[name]] = int(value.split()[0])
    return result


@contextlib.contextmanager
def monitor(directory: Path, *, interval: float = 10) -> Iterator[None]:
    """Persist and print host-memory samples until the caller exits.

    Samples flush immediately so earlier evidence survives SIGKILL. Linux
    supplies RSS, peak RSS, and host availability; other hosts emit elapsed
    time and PID. This reports host memory, independently of device memory.
    """
    if interval <= 0:
        raise ValueError("Diagnostic interval must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stopped = threading.Event()
    path = directory / f"memory-{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as log:

        def sample() -> None:
            """Flush one sample to disk and the notebook's live output."""
            event = {
                "event": "host_memory",
                "pid": os.getpid(),
                "elapsed_seconds": round(time.monotonic() - started, 1),
                **memory_snapshot(),
            }
            line = json.dumps(event)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)

        def watch() -> None:
            """Wait interruptibly between samples."""
            while not stopped.wait(interval):
                sample()

        sample()
        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join()
            sample()
