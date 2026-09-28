"""Versioned dense MagicBox inference bundles independent of optimizer state."""

import dataclasses
import json
import os
from pathlib import Path
import shutil
import tempfile

from minifield_training.checkpoints import inference_output
from minifield_training.checkpoints import tensors
from minifield_training.core import json_io
from minifield_training.datasets import magicbox as data
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import model
from minifield_training.strategies import magicbox


def fusion_config(value: object) -> model.Config:
    """Reconstruct exact saved fusion settings without accepting extra keys."""
    fields = data.object_map(value)
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
    """Atomically publish a complete inference bundle."""
    if directory.exists():
        raise FileExistsError(directory)
    if json_io.digest_file(encoder_config) != encoder.SOURCE.config_sha256:
        raise ValueError("Encoder config must match the pinned source")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".magicbox-", dir=directory.parent
    ) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        inference_output.DenseEffectiveOutput().write(
            staged / "model.safetensors",
            parameters,
            magicbox.inventory(cfg, fusion),
            source_model=encoder.SOURCE.model_id,
            source_revision=encoder.SOURCE.revision,
        )
        shutil.copyfile(encoder_config, staged / "encoder.json")
        shutil.copytree(tokenizer, staged / "tokenizer")
        files = {
            name: json_io.digest_file(staged / name)
            for name in (
                "model.safetensors",
                "encoder.json",
                "tokenizer/tokenizer.json",
                "tokenizer/contract.json",
            )
        }
        metadata = {
            "format": "minifield.magicbox.model/1",
            "fusion": dataclasses.asdict(fusion),
            "step": step,
            "files": files,
            "source": dataclasses.asdict(encoder.SOURCE),
            "decode": {"presence_threshold": 0.5, "confidence": None},
        }
        (staged / "config.json").write_text(json_io.canonical(metadata))
        os.rename(staged, directory)


def load(
    directory: Path,
) -> tuple[lfm.Config, model.Config, types.Parameters, dict[str, object]]:
    """Verify every asset and restore one shared encoder with all task heads."""
    metadata = data.object_map(
        json.loads((directory / "config.json").read_text())
    )
    if metadata.get("format") != "minifield.magicbox.model/1" or metadata.get(
        "source"
    ) != dataclasses.asdict(encoder.SOURCE):
        raise ValueError("Unknown MagicBox bundle")
    files = data.object_map(metadata["files"])
    expected = {
        "model.safetensors",
        "encoder.json",
        "tokenizer/tokenizer.json",
        "tokenizer/contract.json",
    }
    if set(files) != expected:
        raise ValueError("Incomplete model bundle")
    for name, digest in files.items():
        if json_io.digest_file(directory / name) != digest:
            raise ValueError(f"Changed bundle asset: {name}")
    if files["encoder.json"] != encoder.SOURCE.config_sha256:
        raise ValueError("Bundle encoder configuration changed")
    cfg = encoder.Adapter().parse_config(
        data.object_map(json.loads((directory / "encoder.json").read_text()))
    )
    fusion = fusion_config(metadata["fusion"])
    shapes = {
        spec.name: spec.shape for spec in magicbox.inventory(cfg, fusion).specs
    }
    parameters = tensors.load_masters(
        directory / "model.safetensors",
        shapes,
        source_dtype="F32",
        sha256=str(files["model.safetensors"]),
    )
    return cfg, fusion, parameters, data.object_map(metadata["decode"])
