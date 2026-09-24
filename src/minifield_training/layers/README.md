# Reusable model layers

These blocks accept weight packs and arrays. Model adapters own checkpoint
names, configuration, block order and per-layer cache storage. Import the
concrete `attention`, `convolution` or `feed_forward` module directly.

## Shapes and precision

Inputs and outputs use `[batch, tokens, channels]`. Projection matrices use
`[out_channels, in_channels]` and are FP32 masters; each projection casts its
matrix to the activation dtype and requests the highest JAX dot precision.
RMS normalization accumulates in FP32 and returns the activation dtype.
Bias, dropout and parameter initialization aren't part of these contracts.
Callers validate compatible dimensions, binary masks and cache bounds.

`classification.last_valid_logits` gathers the final valid state of each
right-padded sequence and calls the existing FP32-master linear kernel once
for `[batch, classes]` logits. It never constructs vocabulary-wide logits.

`feed_forward.swiglu_ffn(x, weights, eps)` applies pre-RMS normalization,
`SiLU(gate) * up`, a down projection and the original-input residual.
`FeedForwardWeights` contains `norm`, `gate`, `up` and `down`.
Both attention and convolution blocks finish with this same SwiGLU block.

## Attention

`AttentionWeights` supplies operator RMS gain, query/key/value/output
projections and query/key RMS gains. Every attention block has operator
pre-norm, per-head QK RMS normalization, unscaled full-head RoPE with
split-half rotation, causal attention, output projection and a residual.
The even `head_dim` divides the query width; query head count must be an
integer multiple of KV head count. QK gains broadcast over the head dimension.
The projected query width equals the input channel count. These are concrete
mathematical assumptions; an adapter requiring different math needs a distinct
contract.

- `attention_block` processes a full sequence. RoPE positions default to
  `0..tokens-1`; explicit scalar, `[tokens]` or `[batch, tokens]` positions
  override them. The `[batch, tokens]` mask identifies valid keys.
- `attention_block_prefix` also returns rotated keys and values in
  `[batch, tokens, kv_heads, head_dim]`, with padding zeroed.
  `attention_block_suffix` consumes these arrays and their prefix mask;
  suffix positions continue the prefix's absolute positions. A one-row
  prefix can be shared across suffix rows.
- `attention_block_packed` isolates causal attention by nonzero segment ID.
  Positions must restart at each segment boundary; segment ID 0 is padding.
- `attention_block_prefill` pads rotated KV state to the requested capacity.
  `attention_block_step` accepts one token per row, writes the scalar
  `position` slot and reads `valid_length` columns, including that write.
  Callers ensure `0 <= position < capacity` and a contiguous valid prefix.

Each call passes its explicit backend to the shared attention kernels.
Padding masks govern the attention branch; the block's residual and feed-forward
path can still produce nonzero outputs at padded query positions.

## Convolution

`ConvWeights` supplies operator RMS gain, a `[3 * channels, channels]`
in-projection, `[channels, kernel_size]` taps and an output projection.
The in-projection splits into b gate, c gate and values. Causal depthwise
convolution mixes `b_gate * values`, then multiplies by `c_gate`, projects and
adds the original input. Taps run oldest to newest; the final tap acts on the
current token. Masks zero projected inputs, while residual outputs remain.

`conv_block` starts with zero history. `conv_block_prefix` also returns the
last `kernel_size - 1` valid gated inputs as
`[batch, kernel_size - 1, channels]`, left-zero-filled for short prefixes.
Masks must describe contiguous right-padded sequences for this history.
`conv_block_suffix` consumes that state, including a one-row shared history.
`conv_block_step` consumes one valid token, drops the oldest history column
and appends its gated input. `conv_block_packed` prevents taps from crossing
nonzero segment IDs; segment ID 0 is padding.

## CPU evidence

`tests/layers/test_feed_forward.py` compares FP32 eager/JIT outputs and all
input/weight gradients to NumPy FP64 values and central differences.
`tests/layers/test_blocks.py` checks small FP32 dense-attention and convolution
outputs against independent NumPy calculations, input gradients against
central differences, packed segment isolation, padding exclusion from state,
and prefix/suffix and token-cache results against uninterrupted calculations.
Fixtures use 4 channels, 2 query heads, 1 KV head and 3 convolution taps.
Output bounds are `rtol=3e-6, atol=3e-7`; block input-gradient bounds are
`rtol=3e-5, atol=3e-6`. Feed-forward bounds are declared in its tests.

These tests qualify the covered CPU shapes and FP32 operations. Mixed-precision
whole blocks, all block-weight gradients, accelerator backends, larger shapes
and performance still need separate evidence. The shared primitive tests cover
additional dtype and backend modes described in the kernel README.
