"""Host-safe selection contracts for optional fake quantization."""

from collections.abc import Mapping
from typing import Protocol

from minifield_training.core import parameters


class QuantizationStrategy(Protocol):
    """Select exact matrix names from caller-supplied semantic roles."""

    @property
    def identity(self) -> str:
        """Return a stable quantizer and selection policy identifier."""

    def select(
        self,
        inventory: parameters.FullParameterInventory,
        roles: Mapping[str, str],
    ) -> frozenset[str]:
        """Resolve exact eligible names before tracing a training step."""
