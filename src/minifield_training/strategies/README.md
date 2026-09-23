# Thin composition of training capabilities

Select supported models, objectives, optimizer policies, datasets, and shared engine behavior. Declare actual capability requirements and reject unsupported combinations before allocating a model. Strategies may depend on concrete model families. Keep numerical blocks, dataset transformations, optimizer transactions, and checkpoint implementations in their shared owners rather than copying them into runners.

Status: reserved. No implementation yet. Add a docstring-only `__init__.py`
with the first real module; this directory currently contributes no executable
behavior.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
