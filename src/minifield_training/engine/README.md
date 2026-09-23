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
agreement. CUDA and mixed-device performance remain unqualified. Host loops,
checkpoints, scheduling and packed-batch construction aren't part of this module.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
