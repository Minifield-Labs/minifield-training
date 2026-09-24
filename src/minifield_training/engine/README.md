# Shared logical update

`step.make_step(loss_terms, inventory, adamw_config)` builds one pure update.
The callback receives the full FP32 parameter dictionary and one physical batch,
then returns scalar FP32 **summed** loss and supervised-target count. The engine
differentiates only `inventory.trainable_names`, scans active microbatches,
divides loss and gradients once by the total target count, and calls the shared
AdamW transaction once. Frozen masters and moments pass through unchanged.

```python
from minifield_training.engine import step

update = step.make_step(loss_terms, inventory, adamw_config)
result = update(optimizer_state, microbatches, active)
```

Every `microbatches` value has shape `[M, ...]` with the same fixed `M`; `active`
is a boolean `[M]` vector. The callback defines the remaining physical-batch
shape. Inactive slots skip the callback and contribute finite zeros. Active
nonfinite loss or gradient, negative/nonfinite count, or zero total count rejects
the entire update. `result.code` and `result.committed` are AdamW diagnostics;
rejection returns the exact healthy incoming state. Malformed trees, scalar
contracts, and leading dimensions raise `ValueError` at tracing time.

The update is compatible with `jax.jit`. FP32 masters, moments, accumulated
loss, gradients and count are retained. CPU synthetic tests cover token-weighted
equivalence, an analytical gradient, frozen state, invalid inputs, and eager/JIT
agreement. CUDA and mixed-device performance remain unqualified. Checkpoints,
epoch scheduling and packed-batch construction aren't part of `step`.

`step.make_streaming_step` keeps the same loss/count and AdamW contract for a
single-device run, but compiles one physical gradient, device-side addition,
normalization, and donated commit as separate programs. The host selects active
microbatches and never reads gradient values. This bounds the compiled reverse
pass to one physical batch instead of embedding it inside a full-model scan.
The classifier runner calls this form directly; other logical steps retain the
scanned JIT path.
The donated optimizer consumes its input buffers, including on a rejected
commit. Continue from the returned `CommitResult.state` in either case.

The executable dependency policy is [architecture.toml](../../../architecture.toml).

`classification_run.run` is the bounded single-device host lifecycle for
hard-label sequence updates. It JIT-compiles a scanned logical update or calls
the already compiled stages of a streaming update,
passes fixed `[M, B, T]` batches from host records, checkpoints the complete
state after committed updates, and resumes deterministic epoch shuffles from
the saved global next-batch cursor. `max_steps` and/or `max_seconds` bound each
invocation. An optional callback runs at checkpoint boundaries for product
gameplay; its metrics and a report callback are caller-owned. A requested TPU
must be the sole visible JAX device or startup fails clearly. The caller
supplies an explicit persistent checkpoint path. CPU tests cover save/restore,
cursor advance, callback boundaries, and TPU absence. TPU compilation,
throughput, and full-model gameplay remain unverified here.
