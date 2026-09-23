# Development and acceptance procedure

Owner: repository maintainers. No training strategy or deployed worker is
qualified here yet.

## Local development

1. Read `AGENTS.md`, architecture and the destination module README. Confirm
   file ownership and preserve unrelated changes.
2. Install the pinned Python 3.12/uv environment with `uv sync --locked`.
   Dependency additions require intentional `pyproject.toml` and lockfile
   changes.
3. Run focused tests, then `uv run --no-sync python scripts/check_quality.py`.
   It includes configured lint, type, structural, test, package build and
   isolated-wheel checks. Review all output and the final diff.
4. Commit accepted local work with a conventional commit and integrate it into
   this repository's `main`. Verify the commit is present before claiming local
   implementation complete.

The quality command is the authority for checks it actually runs. A passing
gate establishes only those checks. It doesn't establish numerical
correctness, a usable training run or a deployed integration.

## Component acceptance

A component is ready for local acceptance when its public contract works,
meaningful tests pass, import direction is allowed, consumers use the shared
owner and documentation matches code. It must build standalone and leave no
duplicate destination fallback for the same contract.

For an installed public entrypoint, test the built wheel from a temporary
directory and isolated environment without checkout paths on `PYTHONPATH`. Use
bounded synthetic fixtures. If no public runtime entrypoint exists, say so and
verify only package surfaces introduced by the task.

Declare numerical qualification's shapes, precision, modes, target device,
output/gradient/state bounds and consumer matrix before running it. Record exact
code, fixture identities, lockfile, device/software, commands, results,
exclusions and limitations.

CPU-tested code can remain available for development while GPU qualification
is pending if its task explicitly allows that acceptance level. Don't select or
advertise it as a supported runtime path until required evidence exists.
Preserve independent oracles and report unrun checks plainly.

## Future worker execution boundary

This is the intended first-worker acceptance path. It describes responsibilities
to build, not commands already implemented here.

1. Backend authorizes and creates a durable versioned job with immutable inputs.
   Platform submits and displays jobs through Backend.
2. Supervisor claims an attempt, stages verified inputs, reserves a local GPU
   slot and launches an isolated worker. Backend remains the authority for
   queue, lease and canonical run state.
3. Execution normalizes the assignment once and validates model/objective/data
   capabilities before expensive device initialization.
4. Dataset preparation verifies identity, admission, duplicate groups, splits,
   serialization, tokenization and caches. Batching defines deterministic
   packing and iteration independently from objective mathematics.
5. Strategy composition selects model, objective, optimizer, evaluator and
   artifact policy. The shared engine owns lifecycle and cadence. Models use
   common layers and kernels within qualified contracts.
6. Checkpoints capture the committed update, numerical state, identity,
   data/RNG cursor and restore semantics. Artifact inspection verifies bytes and
   manifests again at persistence/publication boundaries.
7. Supervisor relays progress and publishes validated immutable artifacts.
   Backend owns durable records and final status. Runtime consumers independently
   validate exported versioned formats.

Cancellation, nonfinite-update rejection, lease loss, crash recovery, replay,
upload retry, resumed-next-update equivalence and final identity all require
evidence. Shared helpers don't remove trust-boundary verification.

## First complete strategy and release acceptance

Run a bounded synthetic fixture through the installed public path with
deterministic inputs, meaningful metrics, valid checkpoint/final artifacts and
no import-path bypasses. Exercise cancellation and restore through the same lifecycle.

Qualify representative CUDA modes and shapes against independent results while
preserving complete logical work and reduction. Measure performance with matched
inputs and declared thresholds. Report compilation, warm execution,
synchronization, memory and actual device evidence when relevant.

A worker replacement also requires a real product-submitted job through deployed
Backend and Platform, cancellation, recovery under a new attempt, durable
metrics/artifacts and final bundle loading by the intended consumer. Local tests
and historical reports don't establish those outcomes. Deployment and paid or
remote execution require authorization for that work.

## Reporting

State what works, the contract or behavior changed, checks actually run, commit
on `main`, and remaining device/product evidence. Keep local completion,
numerical qualification and release acceptance distinct.

If checks, permissions, compatibility or integration are blocked, give exact
evidence, branch and commit holding the work, and the smallest remaining action.
Never change external capability claims merely because a new module landed.
