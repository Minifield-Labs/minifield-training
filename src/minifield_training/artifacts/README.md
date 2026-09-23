# Host-safe artifact inspection

Versioned manifests, file hashing, payload verification, serialization metadata, and CPU-only artifact inspection. This layer serves worker admission, dataset caches, and checkpoint inspection. Preserve exact wire and hash contracts as formats evolve; serializers that happen to produce similar JSON may still have different byte identities and compatibility requirements.

Status: reserved. No implementation yet. Add a docstring-only `__init__.py`
with the first real module; this directory currently contributes no executable
behavior.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
