# Training state persistence

Saving and restoring parameters, optimizer state, progress, RNG state, and compatibility metadata through artifact primitives. Distinguish exact continuation from a warm start. Preserve or explicitly translate parameter structure and stored identities. A new package name or import edit must never silently turn an existing checkpoint into an incompatible one.

`tensors.load_masters` admits an immutable safetensors file only when its
SHA-256, exact key set, shapes, and declared source dtype agree. BF16 values are
converted bit-exactly into FP32 masters; nonfinite values fail admission. The
reader supports BF16, FP16, and FP32 and rejects gaps, overlaps, malformed
headers, and trailing bytes. Callers supply model-specific expected shapes.

`training_state.save` writes a new directory atomically with FP32 parameters,
Adam moments, the int32 step, and a JSON manifest. `load` verifies the file
hash, inventory, optimizer configuration, run, data, and source identities and
returns the next unread batch cursor. A warm start loads pretrained tensors and
initializes fresh moments; a resume loads all saved tensors and the cursor.
The writer makes each host tensor contiguous before safetensors serialization,
including strided accelerator transfers such as convolution weights, while
preserving scalar shapes.
`load_warm_start_masters` can explicitly admit a full-state tensor file named
`model.safetensors` when a prior checkpoint used that filename. It verifies
the same manifest hash, exact tensor inventory, and cursor identities before
returning only FP32 masters. Ordinary `load` still reads
`state.safetensors` for exact continuation.
Checkpoint paths must be on persistent storage when used in Colab. The caller
chooses storage and never overwrites an existing checkpoint directory.

The optional `storage` extra provides safetensors for checkpoint writing and
reading. Numerical JAX remains supplied by the `numerical` extra. Base package
imports stay dependency-free.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.

`inference_output.OutputStrategy` is separate from resumable full-state
checkpoints. `DenseEffectiveOutput` writes already-effective FP32 tensors to
one immutable safetensors weight asset with `format=pt`, source lineage,
inventory identity, and optional quantization lineage. It validates exact
keys, shapes, dtype, and finiteness, then makes strided values contiguous.
The runtime dense weight path can read this format. This asset isn't a complete
runtime bundle; config/tokenizer packaging and admission remain separate.
It doesn't reduce storage size. Mixed dense embeddings plus packed projections
need a future runtime per-tensor precision contract.

`bundle.save` atomically packages dense FP32 weights, explicitly named assets,
and caller-supplied metadata into an immutable directory. The caller owns the
format and model configuration. The writer adds a `files` SHA-256 map to
canonical `config.json`; `model.safetensors` keeps the existing dense output's
source and inventory metadata. Assets are individual relative file paths,
including nested paths such as `tokenizer/tokenizer.json`.
`bundle.inspect(directory, expected_files=...)` verifies the exact declared
file inventory and every checksum through `artifacts.files`.
`bundle.load_parameters` additionally restores the caller's exact FP32 tensor
shapes. When the inventory depends on bundle configuration, use
`bundle.load(directory, expected_files=..., inventory=adapter)`: it inspects
once, passes verified metadata to the caller's format/source adapter, then
loads the adapter's expected tensor shapes and returns metadata and parameters.
Each side asset is hashed once; the tensor loader preserves its independent
weight checksum verification. Callers validate their own metadata format and
source policy before constructing model configuration. Independent classifier
and regressor toy metadata, tampering, invalid shapes, atomic failure, and
adapter ordering/checksum passes are covered in `tests/checkpoints/test_bundle.py`.

`discovery.latest_checkpoint` selects the numerically newest complete
`step-00000000` directory for explicit run, data, and source identities. It
requires exactly `manifest.json` and `state.safetensors`, ignores symlinks and
partial directories, and requires a plain integer cursor matching the step
name. Set `reject_mismatched=True` to reject complete directories from another
identity instead of filtering them out. Discovery does structural selection;
`training_state.load` still verifies persisted tensors and full compatibility.
`tests/checkpoints/test_discovery.py` covers numerical ordering, incomplete
and foreign checkpoints, and cursor/path ambiguity without model imports.
