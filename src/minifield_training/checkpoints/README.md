# Training state persistence

Saving and restoring parameters, optimizer state, progress, RNG state, and compatibility metadata through artifact primitives. Distinguish exact continuation from a warm start. Preserve or explicitly translate parameter structure and stored identities. A new package name or import edit must never silently turn an existing checkpoint into an incompatible one.

Status: reserved. No implementation yet. Add a docstring-only `__init__.py`
with the first real module; this directory currently contributes no executable
behavior.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
