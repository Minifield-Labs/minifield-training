# Batch construction and iteration

Packing, sequence masks, microbatch construction, iteration, and conversion into device-ready inputs. Record order, segment boundaries, padding, positions, and loss masks must agree with the dataset contract. Reuse one implementation where SFT and calibration share semantics, while keeping their differing sampling policies explicit and independently tested.

Status: reserved. No implementation yet. Add a docstring-only `__init__.py`
with the first real module; this directory currently contributes no executable
behavior.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
