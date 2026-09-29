# Model families

Family configurations, checkpoint parameter mapping, block-stack assembly,
and cache adapters. Families compose `layers` and `kernels`; checkpoint
names and model configs never cross into them. A model module is mostly a
mapping from saved parameter names to shared weight packs plus the code that
decides which layer goes where.

`contracts.PretrainedModel[ConfigT]` defines config parsing, expected tensor
shapes, source dtype, and master validation. Each family implements that
interface beside its model code. `contracts.PretrainedSource` records immutable
file identities; the shared loader verifies them before admitting tensors.

`lfm2_5.pretrained` owns `Adapter` and the pinned `BASE` release:
`LiquidAI/LFM2.5-230M-Base` revision
`9d2be5519834990d30996f878b6771cccbd24f2c`. Obtain `config.json`,
`tokenizer.json`, and `model.safetensors` from that revision into one directory,
then explicitly compose the loader:

```python
from minifield_training.models.lfm2_5 import pretrained as lfm_source
from minifield_training.strategies import pretrained

cfg, parameters = pretrained.load_verified(
    directory, lfm_source.BASE, lfm_source.Adapter()
)
```

The pinned SHA-256 values verify each file
before the 132 BF16 backbone tensors become FP32 masters. The source is covered
by Liquid AI's LFM Open License v1.0. This repository doesn't redistribute its
weights or tokenizer. Forward numerical parity against the released weights is
still unverified here.

`lfm2_5.model.projection_names(cfg)` lists the exact attention, feed-forward,
and convolution projection matrices in this model's expected inventory.
Quantization strategies use it to select candidates before JIT; embeddings,
output heads, norms, and depthwise taps aren't candidates.

## MagicBox and bidirectional LFM

`lfm2_5.encoder` owns the pinned `LiquidAI/LFM2.5-Encoder-350M` release
`b886781f7c6f10ca9b7096e21b83e30a073c2f39`. Its adapter admits the 148 FP32
`lfm2.*` tensors, totaling 354,483,968 parameters. It composes centered
short convolutions, bidirectional pad-masked GQA, QK RMSNorm, RoPE, and the
family's SwiGLU blocks. A layer scan compiles one block per operator kind,
selecting each layer's original weights in order. Common tensors and each
operator's distinct tensors are stacked separately without dummy weights.
Every block rematerializes in reverse mode. Flat FP32 masters, checkpoint
names, and optimizer state retain their existing format. It doesn't
construct or load an unused vocabulary head. `encoder.MAX_SEQUENCE_LENGTH`
owns its admitted 8,192-token limit independently of the source RoPE metadata.
Passing `segment_ids` and per-segment `positions` together encodes packed rows;
each segment matches its separately padded encoding in FP32 and BF16 CPU tests.

`magicbox.model` accepts an injected shared encoder callable. Source tokens
are encoded once per request; independent schema rows read the resulting
source memory through 2 pre-norm fusion blocks. Defaults are width 256,
4 heads, FFN multiplier 2, dropout 0.1, and extraction match width 128.
One scalar candidate head serves both choice and score. Separate binary and
presence heads and token-membership logits complete the four output types.
Schema rows are encoded packed: `model.encode_schema` runs the encoder once
over every request's packed rows, and the forward gathers each row back with
`schema_token_index` before fusion. The encoder callable takes
`(params, ids, mask, segment_ids, positions)`; source encoding passes `None`
for both. Fusion rows run in chunks without detaching either encoder path.

`magicbox.pointer` is the current MagicBox training model. One encoder pass
reads each request's questions, options, and source together. Every question
answers by pointing: start and end query/key projections score every token of
its request, and the objective masks them to the question's allowed tokens.
There are no per-type heads, fusion blocks, dropout, or per-row loops.
`pointer.forward` returns FP32 `[rows, questions, tokens]` start and end
logits. A CPU test compares them with an independent NumPy computation.
A row may pack several requests: the batch's `segment_ids` and `positions`
reach the encoder, and each question's allowed tokens stay inside its own
request. A CPU test checks that packed requests match separate rows in
losses, allowed logits, and gradients.

`magicbox.cache.SchemaCache` caches pre-projection schema token outputs for
inference. Call `get` before JIT tracing, supplying tokenizer, template, and
precision revisions in `context`. It binds packed schema inputs and immutable
encoder leaf identities, rejects replaced weights, and isn't serializable.
Pass the validated result as `schema_hidden` to the model. Training rejects
that argument. Fusion-only updates don't invalidate pre-projection caches.

CPU checks cover published tensor shapes, noncausal behavior, FP32/BF16
padding, shared gradients, permutation, 2/3/17/65 candidates, row chunking,
cache rejection, and packed versus one-row-per-sequence outputs and every
master gradient. Repeated and interleaved operator layouts also compare
outputs and every master gradient with the former unrolled schedule in FP32
and BF16. That comparison disables CPU excess precision to enforce the
declared BF16 rounding in both graphs. See
[the training guide](../../../docs/magicbox.md).
