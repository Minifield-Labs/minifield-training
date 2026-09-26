# Training composition

`pretrained.load_verified(directory, source, model)` verifies config, tokenizer
and weight hashes, then loads the caller's exact tensor inventory as FP32
masters. It requires a `models.contracts.PretrainedSource` and a
`PretrainedModel[ConfigT]` adapter. Release pins, architecture parsing, tensor
mapping and source dtype belong to the model family. The shared loader has no
default family or release. It returns the parsed config and admitted backbone;
task head initialization remains a separate composition step.

## Causal SFT

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

`make_lfm2_5_streaming_step(..., rematerialize_blocks=False)` lets a training
run compare ordinary block autodiff with the default checkpointed block
autodiff. The flag applies to the complete-sequence model path; prefix,
suffix, and packed paths keep their existing checkpoint policy. Both choices
retain the same forward mathematics. CPU conv and attention fixtures show
bit-identical outputs, FP32 gradient agreement within 1e-6 relative error per
leaf, and BF16 gradient error bounded by 2.5% per leaf in that fixture from
floating-point evaluation order. The plain path retains more activations and
may exceed device memory. TPU memory, gradient drift, and speed require a
measured run before selecting it by default.

`make_lfm2_5_streaming_step(..., fuse_accumulation=True)` also exposes the
shared engine's fused gradient sum. It removes separate add programs after
the first active microbatch. This path is opt-in until v5e memory and speed
are measured.

`make_lfm2_5_streaming_step(..., mesh=mesh)` delegates data parallelism to the
shared engine. The `data` mesh splits global physical rows while parameters
and optimizer state stay replicated. CPU tests cover 8-device gradient
agreement on a tiny conv/attention classifier, including fully padded replicas.
Full Base-model TPU memory and throughput still require hardware evidence.

The executable dependency policy is [architecture.toml](../../../architecture.toml).

## Optional QAT composition

`quantization.QuantizationPlan` is the injected selection and numerical
contract; `NamedQuantization` combines exact names with a numerical quantizer.
Classification assigns semantic roles from model-owned projection
names and rejects embeddings, output heads, norms, depthwise taps, frozen
leaves, and incompatible shapes before JIT. Resolved names and quantizer
identity enter the inventory hash, so exact resume rejects a changed plan.
The dense default preserves existing identity.

Classification training, prediction, and SFT training pass the same effective
weights into model forward. FP32 masters and Adam moments stay unquantized.
This first recipe applies full fake quantization from update 1. A ramp schedule
and TPU qualification need separate evidence.
