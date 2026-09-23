# Numerical primitives

Reference and optimized numerical operations, numerical array aliases, backend selection, and the parity requirements that connect them. Kernels own normalization, attention, recurrence, and projections when their actual contracts justify sharing. Each optimized path needs independently established outputs, gradients, shapes, precision, and device evidence.

Status: first contract implemented. `normalization.rms_norm` is CPU-tested;
CUDA qualification is still pending and no model, performance or
checkpoint-resume claim is made.

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

Consumers: only the focused test file today. Layers and models that need RMS
normalization must call this owner instead of reimplementing the contract.

Evidence: `tests/kernels/test_normalization.py` covers eager and `jax.jit`
outputs, input/weight gradients, FP32/FP16/BF16 and mixed dtypes, ranks 1–3,
scalar/channel/broadcast weights, cast-order detectors, a zero row, FP16
overflow, broadcasting failure and NaN propagation.
