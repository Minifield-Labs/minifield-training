# Data admission and reusable datasets

Record parsing, schema validation, split assignment, tokenizer adapters, and verified preprocessing caches. Separate public model context from private labels and supervision. Assign holdouts before repetition or augmentation, and verify actual cache payloads. This is a host-safe layer; device arrays and batch execution belong elsewhere.

Status: reserved. No implementation yet. Add a docstring-only `__init__.py`
with the first real module; this directory currently contributes no executable
behavior.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
