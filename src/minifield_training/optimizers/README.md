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

Build metadata with `core.parameters.build_inventory(..., decayed_names=...)`.
AdamW applies decay exactly where the inventory's `decayed` flag is true;
trainable matrices may be excluded and vectors may be included. Frozen masters
and moments pass through unchanged. The builder rejects frozen/decayed overlap.

Tests: `uv run --no-sync pytest tests/optimizers/test_adamw.py` covers independent
AdamW results, explicit matrix/vector decay selection, frozen state, clipping,
rejected transactions, state validation and donated updates on CPU. CUDA
qualification remains unrun.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
