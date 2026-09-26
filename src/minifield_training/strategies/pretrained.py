"""Verified pretrained admission through caller-supplied model adapters."""

import json
from pathlib import Path
from typing import cast

import jax

from minifield_training.checkpoints import tensors
from minifield_training.core import json_io
from minifield_training.models import contracts


def load_verified[ConfigT](
    directory: Path,
    source: contracts.PretrainedSource,
    model: contracts.PretrainedModel[ConfigT],
) -> tuple[ConfigT, dict[str, jax.Array]]:
    """Verify source files, then admit the model's exact backbone inventory.

    This warm start requires an explicit source and model adapter. It does not
    initialize task heads or restore a training-state checkpoint.
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
    cfg = model.parse_config(cast(dict[str, object], config_value))
    parameters = tensors.load_masters(
        weights_path,
        model.expected_shapes(cfg),
        source_dtype=model.source_dtype,
        sha256=source.weights_sha256,
    )
    model.validate_masters(parameters, cfg)
    return cfg, parameters
