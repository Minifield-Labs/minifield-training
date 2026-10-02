# Parameter update transactions

Optimizer creation, partitioning, gradient accumulation, clipping, finite-update policy, and application of state updates. Frozen and tied leaves, precision, transaction ordering, and loss-scale behavior need explicit contracts. Keep adapters independent of concrete models and strategies so update corrections reach each training route through one implementation.

Status: full-weight AdamW implemented. `state.py` owns the neutral `State`
tree (`params`, `m`, `v`, `step`) and the `Step` callable alias. `adamw.py`
owns the all-or-nothing commit transaction: clipped logical gradients,
Adam moments with bias correction, caller-selected weight decay, frozen-leaf
passthrough, device-side finite checks, and `CommitCode` rejection
diagnostics. Both take inventory metadata from `core.parameters`; nothing
here knows a model family.
Second moments use one per-leaf device reduction for finite and nonnegative
values in both incoming-state and candidate checks.
Eager state validation reduces each shape on device and transfers three
booleans per leaf. It preserves named rejection errors without materializing
NumPy copies of every master and moment tensor. The shape-local validation
kernel is cached across initialization, resume, and checkpoint boundaries.

Build metadata with `core.parameters.build_inventory(..., decayed_names=...)`.
AdamW applies decay exactly where the inventory's `decayed` flag is true;
trainable matrices may be excluded and vectors may be included. Frozen masters
and moments pass through unchanged. The builder rejects frozen/decayed overlap.

Tests: `uv run --no-sync pytest tests/optimizers/test_adamw.py` covers independent
AdamW results, explicit matrix/vector decay selection, frozen state, clipping,
rejected transactions, state validation and donated updates on CPU. CUDA
qualification remains unrun.

The executable dependency policy is [architecture.toml](../../../architecture.toml).

## Optional optax commit

`optax_adamw.make_transaction(inventory, config)` is an opt-in alternative with
the same calling convention, `CommitResult`, and `params`/`m`/`v`/`step` state
layout. It applies `optax.clip_by_global_norm` then `optax.adamw` (decay masked
by the inventory) to the trainable leaves, and skips the whole update when the
loss, count, or gradients aren't finite or the step would overflow. It doesn't
recheck incoming masters and moments or every candidate value inside the step;
the runner validates state at startup, restore, and each checkpoint. On the
pinned TPU compiler those in-step checks were most of the commit's compile
memory. `update_norm` is NaN. `optax_adamw.implementation_identity(config)`
binds checkpoints to this transaction and the optax version, so checkpoints
never resume across the two commits. The default transactional `adamw` commit
is unchanged. Tests compare 2 commits with an independent float64 clipped
AdamW, check every rejection code, and run the 8-device data-parallel cases.
