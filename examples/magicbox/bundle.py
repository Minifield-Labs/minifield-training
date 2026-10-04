"""Versioned dense MagicBox inference bundles independent of optimizer state."""

import dataclasses
import json
from pathlib import Path

from examples.magicbox import composition as magicbox
from examples.magicbox import data
from minifield_training.checkpoints import bundle as bundle_io
from minifield_training.checkpoints import inference_output
from minifield_training.core import json_io
from minifield_training.core import parameters as core_parameters
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import model
from minifield_training.models.magicbox import pointer

POINTER_FORMAT = "minifield.magicbox.model/3"
# Device bundles: trimmed vocabulary, FP32 or runtime-packed projections.
DEVICE_FORMAT = "minifield.magicbox.model/4"


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
    template: str = data.POINTER_TEMPLATE,
) -> None:
    """Write a joint pointer bundle; its template versions the input layout."""
    _save(
        directory,
        parameters,
        magicbox.pointer_inventory(cfg, head),
        {
            "format": POINTER_FORMAT,
            "head": dataclasses.asdict(head),
            "template": template,
        },
        encoder_config=encoder_config,
        tokenizer=tokenizer,
        step=step,
    )


def save_device(
    directory: Path,
    parameters: types.Parameters,
    cfg: lfm.Config,
    head: pointer.Config,
    *,
    vocabulary: tuple[int, ...],
    quantizer: str | None,
    encoder_config: Path,
    tokenizer: Path,
    step: int,
    template: str = data.POINTER_TEMPLATE,
) -> None:
    """Write a device bundle from trimmed-vocabulary masters.

    ``vocabulary`` lists the kept original token IDs; ``tokenizer`` holds the
    matching trimmed tokenizer. With ``quantizer`` (a ``composition``
    quantizer name), encoder projections are stored as the runtime's packed
    codes and scales; otherwise every tensor is FP32.
    """
    trimmed = dataclasses.replace(cfg, vocab_size=len(vocabulary))
    plan = (
        None
        if quantizer is None
        else magicbox.quantization_plan(trimmed, quantizer)
    )
    inventory = magicbox.pointer_inventory(trimmed, head, plan)
    _save(
        directory,
        parameters,
        inventory,
        {
            "format": DEVICE_FORMAT,
            "head": dataclasses.asdict(head),
            "template": template,
            "vocabulary": {
                "size": len(vocabulary),
                "trimmed_from": tokenizer_contract_source(tokenizer),
            },
            "weights": "fp32" if quantizer is None else quantizer,
        },
        encoder_config=encoder_config,
        tokenizer=tokenizer,
        step=step,
        output=(
            None if plan is None else inference_output.PackedGroup128Output()
        ),
    )


def tokenizer_contract_source(tokenizer: Path) -> str:
    """The pinned tokenizer digest a trimmed tokenizer was cut from."""
    contract = json_io.object_map(
        json.loads((tokenizer / "contract.json").read_text())
    )
    return str(contract["trimmed_from"])


def _save(
    directory: Path,
    parameters: types.Parameters,
    inventory: core_parameters.FullParameterInventory,
    model_metadata: dict[str, object],
    *,
    encoder_config: Path,
    tokenizer: Path,
    step: int,
    output: inference_output.OutputStrategy | None = None,
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
        output=output,
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
    directory: Path,
    metadata: dict[str, object],
    templates: tuple[str, ...] = (data.POINTER_TEMPLATE,),
) -> tuple[lfm.Config, pointer.Config]:
    """Admit exactly the saved pointer widths and input template."""
    cfg = _encoder_config(directory, metadata, (POINTER_FORMAT, DEVICE_FORMAT))
    if metadata["format"] == DEVICE_FORMAT:
        vocabulary = json_io.object_map(metadata["vocabulary"])
        cfg = dataclasses.replace(cfg, vocab_size=int(str(vocabulary["size"])))
    head = json_io.object_map(metadata["head"])
    if metadata.get("template") not in templates or set(head) != {
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
    templates: tuple[str, ...] = (data.POINTER_TEMPLATE,),
) -> tuple[lfm.Config, pointer.Config, types.Parameters, dict[str, object]]:
    """Restore a joint pointer bundle and its decode settings.

    ``templates`` lists the input layouts the caller can compile requests
    for; a bundle trained on another layout is refused.

    Device bundles return the trimmed vocabulary size in the config and, when
    packed, the decoded FP32 weights the quantized model runs.
    """
    # config.json carries no checksum of its own; the load below verifies
    # every file it declares before any tensor is read.
    declared = json_io.object_map(
        json.loads((directory / "config.json").read_text(encoding="utf-8"))
    )
    weights = declared.get("weights", "fp32")
    if weights != "fp32" and weights not in magicbox.QUANTIZERS:
        raise ValueError("Unknown bundle weight storage")

    def inventory(
        loaded: dict[str, object],
    ) -> core_parameters.FullParameterInventory:
        cfg, head = _pointer_configuration(directory, loaded, templates)
        plan = (
            None
            if weights == "fp32"
            else magicbox.quantization_plan(cfg, str(weights))
        )
        return magicbox.pointer_inventory(cfg, head, plan)

    metadata, parameters = bundle_io.load(
        directory,
        expected_files=_EXPECTED_FILES,
        inventory=inventory,
        packed=weights != "fp32",
    )
    cfg, head = _pointer_configuration(directory, metadata, templates)
    return cfg, head, parameters, json_io.object_map(metadata["decode"])
