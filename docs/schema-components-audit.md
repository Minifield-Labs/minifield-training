# Schema-component ownership audit, September 28, 2026

Scope: the MagicBox implementation introduced in `eb2f895`, with notebook
updates through `658d749`. An independent reviewer audited ownership and
semantic duplication before changes and reviewed the refactor afterwards.

## Findings and corrections

| Finding | Resulting owner |
| --- | --- |
| Product templates, wire keys, and BOS admission mixed with records | Neutral `datasets.fields`; product `examples/magicbox/data.py` |
| Model vocabulary/context hardcoded in batching | Explicit batch token policy; context bound in encoder adapter |
| Objective field normalization inside packing | `objectives.schema_fields.balance_types`, injected into batching |
| Repeated cursor/chunk/deadline loops in examples | `batching.stream.EpochStream`, used by both examples |
| Concrete model binding in a branded strategy | Example composition; neutral `strategies.schema_fields` |
| Metric aggregation and inference packing in examples | `evaluation.schema_fields`, with injected model/record access |
| Product response formatting inside shared decoding | Example data adapter; neutral `evaluation.field_decode` |
| Bundle writer and tensor restore under strategies | Shared `checkpoints.bundle` and `artifacts.files` |
| Duplicate, weaker checkpoint selection | Shared `checkpoints.discovery`, used by both examples |
| Notebook contains base64 training source | Direct GitHub clone and exact commit checkout |

The existing import gate permitted these misplaced responsibilities. The clone
gate compares conservative AST fingerprints in package source and cannot prove
semantic reuse or detect all example-owned duplication. No gate was weakened.

## Compatibility and independent evidence

The reviewer compared the previous batcher with the new one for 3 uneven
records, shape `(2, 2, 16, 64, 8)`, seed `17`, and update `13`. All 13 arrays,
active flags, and record IDs were bit-identical, including weights and replay
seeds. The records were the existing synthetic fixture plus a choice-only copy
and an extraction/binary copy with distinct IDs.

Parameter names, parameter inventory format, dataset templates and checksums,
bundle format, source-identity shape keys, and each example's shuffle order are
preserved. Newly explicit vocabulary/padding derive from the same encoder and
existing pad ID. The caller's encoder identity already binds vocabulary.

Independent tests use BOS `7`, pad `10`, vocabulary `11`, direct neutral records,
and an unrelated four-scalar forward. They cover analytical loss and gradients
across microbatch layouts, arbitrary injected weights, a 100,000-token vocabulary,
an 8,193-position neutral batch, partial supervision, and finite optimization.
Evaluation uses a separate zero-logit model with soft labels and unequal counts.
Replay tests freeze epoch-boundary ordering and partial-tail behavior. Artifact
tests cover complete/foreign checkpoints, malformed inventories, and tampering.

The second review caught an encoder limit regression (RoPE metadata allowed
128,000 while the encoder admits 8,192) and contradictory neutral extraction
labels. The encoder now exposes its own limit; neutral admission checks span,
presence, and task consistency. Dedicated boundary/regression tests cover both.

Review also caught unnecessary supervised loss computation during inference and
repeated bundle inspection. Prediction now skips the loss path when requested;
bundle loading admits metadata after one inspection, then independently verifies
the saved weight digest during tensor loading. Regression tests cover both paths.

Git checkout tests execute the notebook against temporary repositories, covering
pinned revisions, fetch, reruns, and preservation of local changes. These checks
use no remote training or model weights. TPU execution and throughput remain
unmeasured.

## Completed checks

`UV_CACHE_DIR=.uv-cache uv run --no-sync python scripts/check_quality.py` passed:
architecture, duplication, documentation, Pyink, Ruff, Pylint, strict Mypy,
513 tests, distribution builds, and isolated wheel installation/host imports.
The tests ran on macOS with Python 3.12.11; all 513 passed in 168.63 seconds.

`XLA_FLAGS=--xla_force_host_platform_device_count=8 .venv/bin/python -m
examples.magicbox.parallel_smoke` passed with 8 simulated CPU devices and FP32.
Maximum absolute gradient difference was `3.5762786865234375e-7`, loss was
`0.8969940543174744`, and supervised mass was `1.0`.

The final notebook pins published commit
`588a9e278ed2592e342fe8e56446e005e4842841`. Its checkout cell cloned that revision
from GitHub in a temporary directory, verified the shared modules, and reran with
a clean worktree. All code cells compiled; 18 notebook tests passed again after
the pin changed. Dependency installation and TPU training weren't run by this
checkout check.
