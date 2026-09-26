"""Verified pretrained LFM2.5 backbone admission for strategy composition."""

import dataclasses
import json
from pathlib import Path
from typing import cast

import jax

from minifield_training.checkpoints import tensors
from minifield_training.core import json_io
from minifield_training.models.lfm2_5 import model


@dataclasses.dataclass(frozen=True)
class Source:
    """Immutable source identities for one external pretrained release."""

    model_id: str
    revision: str
    config_sha256: str
    tokenizer_sha256: str
    weights_sha256: str


BASE = Source(
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


def load_verified(
    directory: Path, source: Source = BASE
) -> tuple[model.Config, dict[str, jax.Array]]:
    """Load a complete pinned backbone from an already downloaded directory.

    This is a warm start, never a training-state resume. The separate action
    head is initialized by its consuming strategy after all source checks pass.
    """
    config_path = directory / "config.json"
    tokenizer_path = directory / "tokenizer.json"
    weights_path = directory / "model.safetensors"
    if json_io.digest_file(config_path) != source.config_sha256:
        raise ValueError("Pretrained config SHA-256 mismatch")
    if json_io.digest_file(tokenizer_path) != source.tokenizer_sha256:
        raise ValueError("Pretrained tokenizer SHA-256 mismatch")
    config_value: object = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_value, dict):
        raise ValueError("Pretrained config must be an object")
    cfg = model.Config.from_dict(cast(dict[str, object], config_value))
    parameters = tensors.load_masters(
        weights_path,
        model.expected_shapes(cfg),
        source_dtype="BF16",
        sha256=source.weights_sha256,
    )
    model.validate_masters(parameters, cfg)
    return cfg, parameters
