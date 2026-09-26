# Causal SFT composition

`sft.make_lfm2_5_step(cfg, inventory, optimizer, dtype=jnp.bfloat16)` binds the
public LFM2.5 dense forward call and `objectives.loss.causal_loss_terms` to the
shared logical update. The batch dictionary contains `input_ids`,
`attention_mask`, and `loss_mask`, each shaped `[M, B, T]`. Position `t` predicts
token `t + 1`; a target counts only when both masks mark that target position.
Each physical batch returns its summed NLL and exact target count, so unequal
microbatch counts receive correct weight.

```python
from minifield_training.strategies import sft

update = sft.make_lfm2_5_step(cfg, inventory, adamw_config)
result = update(optimizer_state, microbatches, active)
```

`active` is boolean `[M]`. Fixed shapes allow `jax.jit`; inactive slots aren't
evaluated. Parameters and Adam moments stay FP32 while model computation uses
the selected dtype. A small deterministic CPU fixture verifies real model
learning with FP32 computation and BF16 compute with FP32 masters. CUDA,
throughput, packed examples and chunked selected-token scoring aren't qualified.
Callers must admit token IDs and masks before entering traced updates.

## Sequence classification

`classification.initialize_from_backbone` validates an exact tied-embedding
LFM2.5 backbone and adds only `classification_head.weight`. Its inventory
freezes `model.embed_tokens.weight`, keeps the remaining trunk trainable, and
updates the small head. `make_lfm2_5_step` sends hard-label loss sums and
decision counts through `engine.step.make_step`; `predict` excludes disallowed
classes. Callers choose the class mask, compute dtype, and attention backend.
For 8 action outputs with class 7 reserved, pass
`(True, True, True, True, True, True, True, False)`.

The tiny CPU fixture verifies finite shared-engine updates, a changed head,
bit-identical frozen embeddings, and safe masked padding labels. It doesn't
qualify the full pretrained model or TPU speed.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
