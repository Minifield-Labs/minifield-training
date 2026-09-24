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
including strided accelerator transfers such as convolution weights.
Checkpoint paths must be on persistent storage when used in Colab. The caller
chooses storage and never overwrites an existing checkpoint directory.

The optional `storage` extra provides safetensors for checkpoint writing and
reading. Numerical JAX remains supplied by the `numerical` extra. Base package
imports stay dependency-free.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
