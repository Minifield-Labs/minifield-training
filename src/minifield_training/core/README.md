# Neutral metadata and host contracts

Identifiers, immutable descriptors, error vocabulary, capability metadata, and
versioned values that need only the Python standard library. Keep numerical
array types in `kernels/types.py` and model execution contracts in
`models/contracts.py`. A host inspecting an input or artifact must never
initialize a numerical backend.

Status: JSON/digest helpers, parameter records and inventory construction are
implemented; the rest of this module's planned surface remains unimplemented.
The dependency policy is [architecture.toml](../../../architecture.toml).

## `minifield_training.core.json_io`

Package initializers carry no re-exports, so import the concrete module:

```python
from minifield_training.core import json_io

json_io.canonical({"b": 1, "a": [True, None]})  # '{"a":[true,null],"b":1}'
```

`canonical(value: object) -> str` returns `json.dumps` output with
`ensure_ascii=False`, `sort_keys=True`, `separators=(",", ":")` and
`allow_nan=False`. String keys sort recursively by Python string order. Numeric
keys sort numerically before conversion to JSON strings, so `{10: "b", 2: "a"}`
encodes as `'{"2":"a","10":"b"}'`. Floats keep Python's spelling, tuples encode
as arrays, and other non-string keys follow standard-library handling.
Unsupported values and incomparable mixed keys raise
`TypeError`; circular containers and NaN or infinite numbers raise `ValueError`.
There is no added newline, encoding step or Unicode normalization, and lone
surrogates are preserved. A stricter wire contract (UTF-16 key ordering, a safe
integer domain, scalar validation) would belong to a separate module.

`digest_file(path: pathlib.Path) -> str` streams binary reads of 8 MiB into
SHA-256 and returns the lowercase hex digest. It applies no admission, symlink
or expected-hash policy; those belong to callers.

Tests: `uv run --no-sync pytest tests/core/test_json_io.py` with literal
expected bytes in [test_json_io.py](../../../tests/core/test_json_io.py).
Parameter inventory identity and optimizer configuration identity use canonical
JSON. CPU artifact inspection remains unimplemented.

## `minifield_training.core.parameters`

Build an immutable inventory from unique stored parameter shapes and an explicit
weight-decay policy:

```python
from minifield_training.core import parameters

inventory = parameters.build_inventory(
    {"embedding": (16, 8), "projection": (4, 8)},
    format_id="example.parameters/1",
    decayed_names=frozenset({"projection"}),
)
inventory.names  # ("embedding", "projection")
inventory.trainable_parameter_count  # 160
```

`build_inventory` sorts parameter names, counts scalars and computes a SHA-256
digest from canonical JSON. Identity includes the caller's format ID, shapes,
dtypes, trainability, decay membership and quantization metadata. Source dtypes
are `bfloat16`, `float16` or `float32`; master dtype must be `float32`.

The required `decayed_names` set selects weight decay independently of tensor
rank or name. An empty set disables decay. Callers choose their policy, including
whether embeddings or vectors receive decay. `frozen_names` excludes leaves
from gradients and updates. Unknown decay/frozen/quantized names are rejected,
as is overlap between frozen and decayed names. A nonempty quantization profile
is required when quantized names are supplied.

`FullParameterSpec` is a frozen record for one supplied parameter row.
`FullParameterInventory` is a frozen record holding the spec tuple plus
caller-supplied totals, dtypes, quantization profile and digest. Its views
preserve the supplied spec order and multiplicity: `names` returns every row,
`trainable_names`/`frozen_names` filter on `trainable`, and
`trainable_parameter_count` sums `math.prod(spec.shape)` over trainable rows.

Direct record construction performs no validation, sorting, deduplication,
hashing or dtype enforcement. Use `build_inventory` for those supported
construction checks. Tied aliases must already be resolved to unique stored
rows before calling it. The model adapter supplies shapes and policy; the AdamW
transaction consumes the resulting trainability and decay flags. Checkpoint
inspection remains unimplemented.

Tests: `uv run --no-sync pytest tests/core/` covers record views and immutability,
explicit decay selection, rejected memberships, FP32 masters and digest changes.

`quantization.QuantizationStrategy` is the host-safe selection protocol. It
resolves exact names against an inventory and caller-supplied semantic roles
before numerical tracing. Existing dense identity is unchanged.
