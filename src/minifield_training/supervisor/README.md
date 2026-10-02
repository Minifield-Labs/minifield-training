# Worker process and transport supervision

`diagnostics.memory_snapshot` reads Linux process RSS, peak RSS, and available
host memory in KiB. `diagnostics.monitor` prints and flushes periodic JSONL
samples to a caller-selected directory, including on context exit. Other hosts
emit elapsed time and PID without fabricated memory readings.

The TPU notebook consumes this accelerator-free monitor around expensive stages.
Tests in `tests/supervisor/test_diagnostics.py` cover parsing, persistence, and
cleanup. Process scheduling and remote delivery remain unimplemented.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
