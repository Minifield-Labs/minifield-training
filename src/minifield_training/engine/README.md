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
The training runner calls this form directly when supplied a streaming step;
other logical steps retain the scanned JIT path.
The donated optimizer consumes its input buffers, including on a rejected
commit. Continue from the returned `CommitResult.state` in either case.

The training runner reports the first update's wall time separately because
it can include compilation. `warm_updates_per_second` divides later committed
updates by their summed update-call time; it excludes batch construction,
checkpoints, gameplay and the first update. `last_update_seconds` is the most
recent update-call time. Pass `annotate_steps=True` to label every update with
its global `train` step number in a JAX trace, including resumed updates.
The runner never starts or exports a trace. Callers choose the capture window;
the Polyomino example restricts it to a short run and exports after the final
checkpoint. Profiling output stays in the caller's configured directory.

`step.make_streaming_step(..., fuse_accumulation=True)` combines each later
physical gradient with the existing sum in one donated JIT program. The
default keeps gradient and addition separate. The fused path has CPU numerical
coverage, but its v5e memory peak and speed remain unmeasured.

The executable dependency policy is [architecture.toml](../../../architecture.toml).

`training_run.run` is the bounded single-device host lifecycle for
caller-supplied supervision. It JIT-compiles a scanned logical update or calls
the already compiled stages of a streaming update,
consumes physical updates through batching protocols, checkpoints the complete
state after committed updates, and resumes deterministic epoch shuffles from
the saved global next-batch cursor. `max_steps` and/or `max_seconds` bound each
invocation. An optional callback runs at checkpoint boundaries for product
gameplay; its metrics and a report callback are caller-owned. A requested TPU
must be the sole visible JAX device or startup fails clearly. The caller
supplies an explicit persistent checkpoint path. CPU tests cover save/restore,
cursor advance, callback boundaries, and TPU absence. TPU compilation,
throughput, and full-model gameplay remain unverified here.

For finite records, supply `examples` and a `batch_strategy` implementing
`batching.contracts.BatchStrategy[RecordT]`. The runner owns epoch seeds and the
global update cursor; the strategy owns physical shape and supervision.
`RunConfig` contains only replay seed, cadence and run bounds. SFT and
classification use the same runner without a task-specific import.

For a replayable stream, pass `examples=None` and a `batch_source` implementing
`batching.contracts.BatchSource`. Its `__call__(next_batch, deadline)` receives
the global update index and an absolute `time.monotonic()` deadline or `None`.
The callback returns an iterator of
`batching.contracts.PhysicalUpdate` values starting at that global logical
update index. It must produce the same unread updates after restore, keep
record identities unique, and provide enough batches to reach the run's
step or time bound. It must stop waiting for input at the deadline. The
runner saves committed work before closing the iterator on a normal stop, and
raises if the source ends early. Finite `examples` keep their seeded epoch
behavior; passing both input modes is an error. The checkpoint cursor
identifies the next unread update. Its data/source IDs must identify the
source and deterministic settings as well as any stored inputs.
