"""Versioned dense MagicBox inference bundles independent of optimizer state."""

import dataclasses
import json
from pathlib import Path

from examples.magicbox import composition as magicbox
from minifield_training.checkpoints import bundle as bundle_io
from minifield_training.core import json_io
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import model


def fusion_config(value: object) -> model.Config:
    """Reconstruct exact saved fusion settings without accepting extra keys."""
    fields = json_io.object_map(value)
    if set(fields) != {
        field.name for field in dataclasses.fields(model.Config)
    }:
        raise ValueError("Unknown or missing fusion configuration")
    return model.Config(
        encoder_width=int(str(fields["encoder_width"])),
        width=int(str(fields["width"])),
        layers=int(str(fields["layers"])),
        heads=int(str(fields["heads"])),
        ffn_multiplier=int(str(fields["ffn_multiplier"])),
        match_width=int(str(fields["match_width"])),
        dropout=float(str(fields["dropout"])),
        row_chunk=int(str(fields["row_chunk"])),
    )


_EXPECTED_FILES = frozenset(
    {
        "model.safetensors",
        "encoder.json",
        "tokenizer/tokenizer.json",
        "tokenizer/contract.json",
    }
)


def save(
    directory: Path,
    parameters: types.Parameters,
    cfg: lfm.Config,
    fusion: model.Config,
    *,
    encoder_config: Path,
    tokenizer: Path,
    step: int,
) -> None:
    """Supply product metadata to the shared immutable bundle writer."""
    if json_io.digest_file(encoder_config) != encoder.SOURCE.config_sha256:
        raise ValueError("Encoder config must match the pinned source")
    bundle_io.save(
        directory,
        parameters,
        magicbox.inventory(cfg, fusion),
        metadata={
            "format": "minifield.magicbox.model/1",
            "fusion": dataclasses.asdict(fusion),
            "step": step,
            "source": dataclasses.asdict(encoder.SOURCE),
            "decode": {"presence_threshold": 0.5, "confidence": None},
        },
        assets={
            "encoder.json": encoder_config,
            "tokenizer/tokenizer.json": tokenizer / "tokenizer.json",
            "tokenizer/contract.json": tokenizer / "contract.json",
        },
        source_model=encoder.SOURCE.model_id,
        source_revision=encoder.SOURCE.revision,
    )


def _configuration(
    directory: Path, metadata: dict[str, object]
) -> tuple[lfm.Config, model.Config]:
    """Interpret verified assets using the product's metadata contract."""
    if metadata.get("format") != "minifield.magicbox.model/1" or metadata.get(
        "source"
    ) != dataclasses.asdict(encoder.SOURCE):
        raise ValueError("Unknown MagicBox bundle")
    files = json_io.object_map(metadata["files"])
    if files["encoder.json"] != encoder.SOURCE.config_sha256:
        raise ValueError("Bundle encoder configuration changed")
    cfg = encoder.Adapter().parse_config(
        json_io.object_map(json.loads((directory / "encoder.json").read_text()))
    )
    fusion = fusion_config(metadata["fusion"])
    return cfg, fusion


def load(
    directory: Path,
) -> tuple[lfm.Config, model.Config, types.Parameters, dict[str, object]]:
    """Admit model metadata during one shared bundle inspection and restore."""
    metadata, parameters = bundle_io.load(
        directory,
        expected_files=_EXPECTED_FILES,
        inventory=lambda metadata: magicbox.inventory(
            *_configuration(directory, metadata)
        ),
    )
    cfg, fusion = _configuration(directory, metadata)
    return cfg, fusion, parameters, json_io.object_map(metadata["decode"])
