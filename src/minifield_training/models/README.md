# Model architecture and execution contracts

Architecture-specific parameter trees, block composition, cache state, configuration, and external weight mapping. models/contracts.py is the reserved shared contract surface. Individual model families must stay independent. Strategies select concrete families through public interfaces; generic engine and objective code must never know a family name.

Status: reserved. No implementation yet. Add a docstring-only `__init__.py`
with the first real module; this directory currently contributes no executable
behavior.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
