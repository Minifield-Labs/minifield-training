# Repository instructions

## Purpose

The Minifield Python/JAX training worker, packaged as an independent library
with executable quality gates.

Read [README.md](README.md), [architecture](docs/architecture.md), and
[procedure](docs/procedure.md) before changing implementation. Read the
destination module's README too.

## Ownership and dependency direction

- Backend owns authorization, durable jobs, queues, leases, canonical run state,
  and the artifact registry. Platform owns product UI. Training consumes
  versioned immutable inputs and reports computation progress and artifacts.
- `core` owns neutral CPU metadata and types, using only the standard library.
  Shared array aliases belong in `kernels/types.py`; optimizer state belongs in
  `optimizers/state.py`.
- `kernels` owns numerical operations. `layers` owns reusable composition.
  `models` owns architecture, parameter mapping, and cache state. Common code
  mustn't depend on a model family.
- `objectives` owns supervision and reduction mathematics. `datasets` owns
  records, admission, splits, tokenization, and verified caches. `batching` owns
  packing and iteration. Separate public context from private supervision.
- Batching strategies implement `batching.contracts.BatchStrategy` and return
  its single `PhysicalUpdate` type. Dense target encoders share the common
  iterator. Engines consume protocols, with concrete selection at composition.
- `optimizers` owns update transactions. `engine` owns shared lifecycle.
  `strategies` composes public interfaces; it mustn't grow copied model blocks,
  checkpoint writers, or runner loops.
- Pretrained release pins and family-specific admission belong under that
  model family. Shared loaders require an explicit source and model adapter.
- `artifacts` owns CPU inspection and file integrity. `checkpoints` owns training
  state persistence. `execution` owns wire normalization and worker entrypoints.
  `supervisor` owns process isolation, transport, local GPU slots, and delivery.
- The executable dependency policy and architecture document define allowed
  edges. A new edge needs an explicit architecture decision and tests. Don't hide
  a violation through local or dynamic imports or a widened allowlist.

## Quality and duplicate control

Use Python 3.12 and repository-pinned uv 0.11.30. Preserve Google Python style,
module imports, Pyink, Ruff, Google Pylint, and strict Mypy for source, scripts,
and tests. Explain narrow native-library suppressions locally.

Run these commands from this repository:

```sh
uv sync --locked
uv run --no-sync python scripts/check_quality.py
```

Run meaningful component tests during development, then the complete gate,
including its build and isolated-wheel checks. Architecture, duplicate, and
documentation checks supplement lint, typing, and tests. Never disable a rule
or expand a baseline to pass a change. Exceptions need narrow scope, a concrete
semantic reason, and review.

Search for equivalent existing behavior before adding a helper. If 2 consumers
share a contract, make them call one owner. If semantics differ, name and test
the difference. Structural duplication checks can't establish semantic
equivalence. Don't merge independent numerical oracles into production code,
split functions to evade limits, or rename copies to evade detection.

Test public behavior using independent expected values, frozen synthetic
fixtures, analytical results, or an independent numerical oracle. Expected
values computed by the implementation under test aren't independent evidence.

Numerical code needs declared output, gradient, state, precision, shape, and
device coverage. CPU tests don't qualify CUDA behavior or performance. Unrun
checks remain explicitly unrun. Never fabricate measurements or widen
tolerances to excuse a refactor.

## Documentation and completion

Update the module README with implementation. Keep contracts, examples,
commands, and capability claims synchronized. Reserved directories don't
implement features.

Keep credentials, customer data, weights, generated datasets, caches, and run
output outside Git. Commit only small synthetic fixtures. Don't publish a
remote, deploy, or start paid/remote jobs without authorization.

Preserve unrelated changes and other contributors' work. Use conventional
commits. Implementation is complete only when required checks pass and finished
changes are committed on this repository's `main`. Verify `main` contains the
commit. Never force-push, discard unrelated work, or rewrite shared history.

If a required dependency, compatibility decision, permission, integration, or
qualification is blocked, report exact evidence, checks run, branch and commit,
and the remaining step.

Write concise documentation. Use contractions naturally. Never use em dashes.
