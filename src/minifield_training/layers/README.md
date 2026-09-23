# Reusable model layers

Composition of kernels into reusable parameterized layers, with explicit parameter layout, state transitions, masks, and precision. Shared prefill and decode behavior belongs here when the semantics match. Model families should call these owners so an accepted kernel or layer improvement reaches every declared consumer.

Status: reserved. No implementation yet. Add a docstring-only `__init__.py`
with the first real module; this directory currently contributes no executable
behavior.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
