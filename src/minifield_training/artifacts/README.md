# Host-safe artifact inspection

Versioned manifests, file hashing, payload verification, serialization metadata, and CPU-only artifact inspection. This layer serves worker admission, dataset caches, and checkpoint inspection. Preserve exact wire and hash contracts as formats evolve; serializers that happen to produce similar JSON may still have different byte identities and compatibility requirements.

`files.FileEntry` declares a relative path, SHA-256, and optional exact byte
size. `files.verify(root, entries, expected_paths=...)` admits unique canonical
paths, rejects symlink files and directories, verifies bytes, and returns the
contained local paths. The optional expected set checks manifest membership;
unlisted disk files don't become declared assets. Dataset manifests and
inference bundles share this CPU-only contract.

`files.relative_path` validates destination names before writing, and
`files.contained_file` resolves an existing regular file without following
artifact symlinks. Paths use slash separators and cannot contain parent
traversal, redundant components, absolute roots, or Windows drive prefixes.

For example, verify a 3-byte payload without loading an accelerator library:

```python
from minifield_training.artifacts import files

paths = files.verify(root, [files.FileEntry("part.bin", sha256, 3)])
```

`tests/artifacts/test_files.py` covers independent known SHA-256 bytes,
duplicates, inventory and size mismatches, tampering, and symlink/traversal
rejection. File checks don't prevent a concurrent process from replacing
artifacts after admission; callers own immutable storage.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
