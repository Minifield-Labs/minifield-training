"""Pinned LFM2.5 releases and their pretrained model adapter."""

from collections.abc import Mapping

import jax

from minifield_training.models import contracts
from minifield_training.models.lfm2_5 import model

BASE = contracts.PretrainedSource(
    model_id="LiquidAI/LFM2.5-230M-Base",
    revision="9d2be5519834990d30996f878b6771cccbd24f2c",
    config_sha256=(
        "f7d0bcc454b7a30fa471b1e7b9e359e" "11fb25b56f5b4ffd59bb18248e3c2ea3d"
    ),
    tokenizer_sha256=(
        "df1d8d5ec5d091b460562ffd545e4a5e" "91d17d4a0db7ebe733be34ed374377bd"
    ),
    weights_sha256=(
        "e91eb22c0aeae0bcbea8ade56f5cfe3c" "f91bca0c34e859adacae8f4445416fe6"
    ),
)


class Adapter:
    """Admit the supported LFM2.5 BF16 backbone through the shared loader."""

    @property
    def source_dtype(self) -> str:
        """The pinned release stores BF16 backbone tensors."""
        return "BF16"

    def parse_config(self, value: Mapping[str, object]) -> model.Config:
        """Reject unsupported LFM configuration and layer variants."""
        return model.Config.from_dict(value)

    def expected_shapes(
        self, cfg: model.Config
    ) -> Mapping[str, tuple[int, ...]]:
        """Expose this family's checkpoint parameter mapping."""
        return model.expected_shapes(cfg)

    def validate_masters(
        self, parameters: Mapping[str, jax.Array], cfg: model.Config
    ) -> None:
        """Verify all admitted family parameters are finite FP32 masters."""
        model.validate_masters(parameters, cfg)
