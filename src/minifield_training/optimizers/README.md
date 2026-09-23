# Parameter update transactions

Optimizer creation, partitioning, gradient accumulation, clipping, finite-update policy, and application of state updates. Frozen and tied leaves, precision, transaction ordering, and loss-scale behavior need explicit contracts. Keep adapters independent of concrete models and strategies so update corrections reach each training route through one implementation.

Status: full-weight AdamW implemented. `state.py` owns the neutral `State`
tree (`params`, `m`, `v`, `step`) and the `Step` callable alias. `adamw.py`
owns the all-or-nothing commit transaction: clipped logical gradients,
Adam moments with bias correction, explicit matrix-only decay, frozen-leaf
passthrough, device-side finite checks, and `CommitCode` rejection
diagnostics. Both take inventory metadata from `core.parameters`; nothing
here knows a model family.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
