"""Model-neutral metadata and pretrained admission interfaces."""

from abc import abstractmethod
from collections.abc import Mapping
import dataclasses
from typing import Protocol

import jax


@dataclasses.dataclass(frozen=True)
class PretrainedSource:
    """Immutable source identities for one external pretrained release."""

    model_id: str
    revision: str
    config_sha256: str
    tokenizer_sha256: str
    weights_sha256: str


class PretrainedModel[ConfigT](Protocol):
    """Family-owned config parsing, tensor mapping, and master validation."""

    @property
    @abstractmethod
    def source_dtype(self) -> str:
        """Safetensors dtype admitted for this model representation."""

    @abstractmethod
    def parse_config(self, value: Mapping[str, object]) -> ConfigT:
        """Parse and validate this family's published configuration."""

    @abstractmethod
    def expected_shapes(self, cfg: ConfigT) -> Mapping[str, tuple[int, ...]]:
        """Map the configuration to the exact source tensor inventory."""

    @abstractmethod
    def validate_masters(
        self, parameters: Mapping[str, jax.Array], cfg: ConfigT
    ) -> None:
        """Apply family-specific validation to admitted FP32 masters."""
