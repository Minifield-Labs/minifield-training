"""Versioned dense MagicBox inference bundles independent of optimizer state."""

import dataclasses
import json
from pathlib import Path

from examples.magicbox import composition as magicbox
from examples.magicbox import data
from minifield_training.checkpoints import bundle as bundle_io
from minifield_training.core import json_io
from minifield_training.core import parameters as core_parameters
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import model
from minifield_training.models.magicbox import pointer

POINTER_FORMAT = "minifield.magicbox.model/3"


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
    _save(
        directory,
        parameters,
        magicbox.inventory(cfg, fusion),
        {
            "format": "minifield.magicbox.model/2",
            "fusion": dataclasses.asdict(fusion),
        },
        encoder_config=encoder_config,
        tokenizer=tokenizer,
        step=step,
    )


def save_pointer(
    directory: Path,
    parameters: types.Parameters,
    cfg: lfm.Config,
    head: pointer.Config,
    *,
    encoder_config: Path,
    tokenizer: Path,
    step: int,
) -> None:
    """Write a joint pointer bundle; its template versions the input layout."""
    _save(
        directory,
        parameters,
        magicbox.pointer_inventory(cfg, head),
        {
            "format": POINTER_FORMAT,
            "head": dataclasses.asdict(head),
            "template": data.POINTER_TEMPLATE,
        },
        encoder_config=encoder_config,
        tokenizer=tokenizer,
        step=step,
    )


def _save(
    directory: Path,
    parameters: types.Parameters,
    inventory: core_parameters.FullParameterInventory,
    model_metadata: dict[str, object],
    *,
    encoder_config: Path,
    tokenizer: Path,
    step: int,
) -> None:
    """Share the pinned encoder, tokenizer, and decode metadata."""
    if json_io.digest_file(encoder_config) != encoder.SOURCE.config_sha256:
        raise ValueError("Encoder config must match the pinned source")
    bundle_io.save(
        directory,
        parameters,
        inventory,
        metadata={
            **model_metadata,
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


def _encoder_config(
    directory: Path, metadata: dict[str, object], formats: tuple[str, ...]
) -> lfm.Config:
    """Admit a known format and the pinned encoder configuration."""
    if metadata.get("format") not in formats or metadata.get(
        "source"
    ) != dataclasses.asdict(encoder.SOURCE):
        raise ValueError("Unknown MagicBox bundle")
    files = json_io.object_map(metadata["files"])
    if files["encoder.json"] != encoder.SOURCE.config_sha256:
        raise ValueError("Bundle encoder configuration changed")
    return encoder.Adapter().parse_config(
        json_io.object_map(json.loads((directory / "encoder.json").read_text()))
    )


def _configuration(
    directory: Path, metadata: dict[str, object]
) -> tuple[lfm.Config, model.Config]:
    """Interpret verified assets using the product's metadata contract."""
    cfg = _encoder_config(
        directory,
        metadata,
        ("minifield.magicbox.model/1", "minifield.magicbox.model/2"),
    )
    return cfg, fusion_config(metadata["fusion"])


def _pointer_configuration(
    directory: Path, metadata: dict[str, object]
) -> tuple[lfm.Config, pointer.Config]:
    """Admit exactly the saved pointer widths and input template."""
    cfg = _encoder_config(directory, metadata, (POINTER_FORMAT,))
    head = json_io.object_map(metadata["head"])
    if metadata.get("template") != data.POINTER_TEMPLATE or set(head) != {
        field.name for field in dataclasses.fields(pointer.Config)
    }:
        raise ValueError("Unknown pointer template or head configuration")
    return cfg, pointer.Config(
        encoder_width=int(str(head["encoder_width"])),
        pointer_width=int(str(head["pointer_width"])),
    )


def load(
    directory: Path,
) -> tuple[lfm.Config, model.Config, types.Parameters, dict[str, object]]:
    """Admit model metadata during one shared bundle inspection and restore."""
    metadata, parameters = bundle_io.load(
        directory,
        expected_files=_EXPECTED_FILES,
        inventory=lambda metadata: magicbox.inventory(
            *_configuration(directory, metadata),
            freeze_embeddings=metadata["format"]
            == "minifield.magicbox.model/2",
        ),
    )
    cfg, fusion = _configuration(directory, metadata)
    return cfg, fusion, parameters, json_io.object_map(metadata["decode"])


def load_pointer(
    directory: Path,
) -> tuple[lfm.Config, pointer.Config, types.Parameters, dict[str, object]]:
    """Restore a joint pointer bundle and its decode settings."""
    metadata, parameters = bundle_io.load(
        directory,
        expected_files=_EXPECTED_FILES,
        inventory=lambda metadata: magicbox.pointer_inventory(
            *_pointer_configuration(directory, metadata)
        ),
    )
    cfg, head = _pointer_configuration(directory, metadata)
    return cfg, head, parameters, json_io.object_map(metadata["decode"])
