# Neutral metadata and host contracts

Identifiers, immutable descriptors, error vocabulary, capability metadata, and
versioned values that need only the Python standard library. Keep numerical
array types in `kernels/types.py` and model execution contracts in
`models/contracts.py`. A host inspecting an input or artifact must never
initialize a numerical backend.

Status: JSON/digest helpers and parameter metadata records are implemented;
the rest of this module's planned surface remains unimplemented. The dependency
policy is [architecture.toml](../../../architecture.toml).

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
The intended next consumers are neutral parameter inventory identity and CPU
artifact inspection; neither is implemented yet.

## `minifield_training.core.parameters`

Immutable records describing a caller-supplied parameter inventory:

```python
from minifield_training.core import parameters

spec = parameters.FullParameterSpec(
    name="decoder.weight",
    shape=(4, 8),
    source_dtype="bfloat16",
    master_dtype="float32",
    trainable=True,
    decayed=True,
    quantized=False,
)
inventory = parameters.FullParameterInventory(
    specs=(spec,),
    parameter_count=32,
    source_dtype="bfloat16",
    master_dtype="float32",
    quantization_profile=None,
    sha256="caller-supplied digest",
)
inventory.names  # ("decoder.weight",)
inventory.trainable_parameter_count  # 32
```

`FullParameterSpec` is a frozen record for one supplied parameter row.
`FullParameterInventory` is a frozen record holding the spec tuple plus
caller-supplied totals, dtypes, quantization profile and digest. Its views
preserve the supplied spec order and multiplicity: `names` returns every row,
`trainable_names`/`frozen_names` filter on `trainable`, and
`trainable_parameter_count` sums `math.prod(spec.shape)` over trainable rows.

The records never sort, validate, deduplicate, compute a digest, enforce FP32,
infer decay or resolve tied aliases. Tied leaves must already appear as their
unique stored rows before construction. Inventory membership, ordering, dtype
and digest policies belong to the constructing builder and to consumers such
as the optimizer transaction and checkpoint inspection, none of which are
implemented here yet.

Tests: `uv run --no-sync pytest tests/core/test_parameters.py` with literal
fixtures in [test_parameters.py](../../../tests/core/test_parameters.py).
