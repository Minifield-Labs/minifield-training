# Numerical primitives

Reference and optimized numerical operations, numerical array aliases, backend selection, and the parity requirements that connect them. Kernels own normalization, attention, recurrence, and projections when their actual contracts justify sharing. Each optimized path needs independently established outputs, gradients, shapes, precision, and device evidence.

Status: normalization, attention, projection and selected-token scoring have
direct CPU tests. Layer tests cover FP32 rotation and convolution composition;
each contract below describes its evidence. Splash runs on CPU through Pallas
interpret mode and cuDNN wrappers through the `xla` implementation, so CUDA/TPU
qualification and performance evidence are still pending.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.

## Implemented contracts

### `normalization.rms_norm`

`rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array` applies
FP32-accumulated RMS normalization over the final axis. It casts `x` to FP32,
scales it by `jax.lax.rsqrt(mean(x * x, axis=-1) + eps)`, casts the normalized
values back to `x.dtype`, and multiplies by `weight` cast to `x.dtype` under
ordinary broadcasting. The result has the broadcast shape and `x.dtype`;
gradients keep the dtype of their respective argument. There is no shape,
epsilon, dtype or nonfinite validation: NaN inputs propagate and incompatible
broadcast shapes raise the usual JAX error.

```python
import jax.numpy as jnp

from minifield_training.kernels import normalization

x = jnp.array([3.0, 4.0], dtype=jnp.float32)
weight = jnp.array([2.0, -3.0], dtype=jnp.float32)
normalization.rms_norm(x, weight, eps=3.5).tolist()  # [1.5, -3.0]
```

Consumers: attention, convolution and feed-forward layers, plus final model
normalization. They all call this owner for the same RMS contract.

Evidence: `tests/kernels/test_normalization.py` covers eager and `jax.jit`
outputs, input/weight gradients, FP32/FP16/BF16 and mixed dtypes, ranks 1–3,
scalar/channel/broadcast weights, cast-order detectors, a zero row, FP16
overflow, broadcasting failure and NaN propagation.

### `attention`

Causal attention over `bthd` operands in four shapes, each with a `dense`
reference path (fully materialized score grid, FP32 softmax, GQA head
repetition) and optimized backends:

- `dense_causal_attention` / `cudnn_causal_attention` /
  `splash_causal_attention` for prefix-padded rows.
- `dense_prefix_causal_attention` / `cudnn_prefix_causal_attention` /
  `splash_prefix_causal_attention` for a shared prefix plus causal suffix.
- `dense_packed_causal_attention` / `cudnn_packed_causal_attention` /
  `splash_packed_causal_attention` for same-segment packed sequences.
- `dense_cached_attention` / `cudnn_cached_attention` for one query against a
  fixed-size KV cache prefix.

The `cudnn_*` functions take `implementation: Literal["cudnn", "xla"]`,
defaulting to `cudnn`; `xla` runs the same fused
`jax.nn.dot_product_attention` path without cuDNN. The `splash_*` functions
take `interpret: bool`, defaulting to `False`; `True` selects Pallas
interpret mode, the only supported path off TPU. `causal_attention`,
`prefix_causal_attention`, `packed_causal_attention` and `cached_attention`
dispatch on a `backend` name and raise `ValueError` for unknown ones.

```python
import jax.numpy as jnp

from minifield_training.kernels import attention

query = jnp.ones((1, 4, 2, 8))
key = jnp.ones((1, 4, 2, 8))
value = jnp.ones((1, 4, 2, 8))
mask = jnp.ones((1, 4), jnp.int32)
attention.causal_attention(query, key, value, mask, backend="dense")
```


### Projection and selected-token scoring

`linear.full_linear(x, weight)` projects `[..., in_channels]` activations
through `[out_channels, in_channels]` FP32 masters. It casts the master to
`x.dtype` before multiplication, requests `Precision.HIGHEST`, and returns
`[..., out_channels]` in the activation dtype. There is no bias.
All shared layer projections, model output heads and selected-token scoring
call this owner.

`selected_logits.selected_hidden_log_probs(selected, head, target_ids)`
projects `[selected_count, channels]`, casts logits to FP32, applies log-softmax
and gathers one target per row. `selected_token_log_probs` first gathers
`[batch_index, token_index]` positions from a hidden sequence. Both return
FP32 `[selected_count]`, materializing only the selected-row vocabulary grid.
Model scoring is their current consumer. Callers validate target and position
bounds.

`tests/kernels/test_linear.py` independently checks projection outputs and
input/master gradients, eager/JIT FP32, FP16/BF16 master-cast order and gradient
dtypes, FP32 scoring values/gradients, and selected-row gathering. Whole
mixed-precision scoring and accelerator projection paths remain unqualified.

### Rotation and convolution

`rotary.apply_rotary` accepts `[batch, tokens, heads, head_dim]`, rotates the
full even head dimension in split halves, and uses unscaled frequencies
`rope_theta ** (-2*i/head_dim)`. Positions may be scalar, `[tokens]` or
`[batch, tokens]`; omission uses token indices. Angles are FP32; sine and cosine
cast to activation dtype before multiplication. Attention layers consume it.

`convolution.gated_depthwise_convolution` returns
`c_gate * causal_depthwise(b_gate * values)` over `[batch, tokens, channels]`
with `[channels, kernel_size]` taps ordered oldest to newest. The segmented
variant excludes taps from different or zero segment IDs. The history variant
prepends `[batch, kernel_size - 1, channels]` gated inputs, allowing a one-row
history to broadcast across rows. `state_tail` extracts the last requested
number of valid columns, left-zero-filling short rows. Convolution layers own
the projection and residual around these kernels.

Direct layer tests cover their composition on small FP32 CPU inputs, including
independent outputs/input gradients, packed boundaries, prefix continuation
and token-cache state. See the [layer contracts](../layers/README.md) for the
exact evidence and remaining coverage.

`types.py` owns numerical array aliases and parameter-structure validation;
CPU metadata remains in `core`. Attention kernels are consumed by the shared
attention layer and keep their own independent backend tests in
`tests/kernels/test_attention.py`. CPU cuDNN-wrapper and Splash-interpret
checks don't establish CUDA or TPU qualification.
