# Shared training lifecycle

Initialization, forward/backward execution, evaluation cadence, update boundaries, checkpoint scheduling, progress, and interruption handling for supported execution plans. Consume public model and objective contracts. Strategies provide composition and policy; they must not grow parallel copies of this lifecycle. Introduce abstractions only after concrete consumers establish their required behavior.

Status: reserved. No implementation yet. Add a docstring-only `__init__.py`
with the first real module; this directory currently contributes no executable
behavior.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
