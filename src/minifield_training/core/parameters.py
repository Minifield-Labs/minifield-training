"""Immutable parameter metadata records for host-safe inventory contracts."""

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class FullParameterSpec:
    """Caller-supplied metadata for one stored parameter row."""

    name: str
    shape: tuple[int, ...]
    source_dtype: str
    master_dtype: str
    trainable: bool
    decayed: bool
    quantized: bool


@dataclasses.dataclass(frozen=True)
class FullParameterInventory:
    """Caller-supplied inventory metadata with derived membership views."""

    specs: tuple[FullParameterSpec, ...]
    parameter_count: int
    source_dtype: str
    master_dtype: str
    quantization_profile: str | None
    sha256: str

    @property
    def names(self) -> tuple[str, ...]:
        """Return supplied leaf names in order, including duplicates."""
        return tuple(spec.name for spec in self.specs)

    @property
    def trainable_names(self) -> tuple[str, ...]:
        """Return names of rows marked trainable, in supplied order."""
        return tuple(spec.name for spec in self.specs if spec.trainable)

    @property
    def frozen_names(self) -> tuple[str, ...]:
        """Return names of rows marked frozen, in supplied order."""
        return tuple(spec.name for spec in self.specs if not spec.trainable)

    @property
    def trainable_parameter_count(self) -> int:
        """Return the scalar count over leaves that receive updates."""
        return sum(
            math.prod(spec.shape) for spec in self.specs if spec.trainable
        )
