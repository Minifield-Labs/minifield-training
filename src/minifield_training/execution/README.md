# Wire adaptation and worker entrypoints

Normalize versioned job envelopes, inspect immutable inputs, validate admission, and expose worker commands without loading numerical backends during host inspection. Add exact reviewed lazy dispatch edges only when a real strategy is available. Backend remains the authority for durable jobs, authorization, leases, canonical run state, and artifact registration.

Status: reserved. No implementation yet. Add a docstring-only `__init__.py`
with the first real module; this directory currently contributes no executable
behavior.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
