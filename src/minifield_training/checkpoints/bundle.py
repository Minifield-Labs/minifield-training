"""Atomic dense inference bundles with caller-owned metadata and assets."""

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import cast

from minifield_training.artifacts import files
from minifield_training.checkpoints import inference_output
from minifield_training.checkpoints import tensors
from minifield_training.core import json_io
from minifield_training.core import parameters as core_parameters
from minifield_training.kernels import types


def save(
    directory: Path,
    parameters: types.Parameters,
    inventory: core_parameters.FullParameterInventory,
    *,
    metadata: Mapping[str, object],
    assets: Mapping[str, Path],
    source_model: str,
    source_revision: str,
) -> None:
    """Publish weights, assets, and canonical ``config.json`` together.

    The caller owns metadata format/version and relative asset names. The
    writer adds only ``files``, mapping each asset and ``model.safetensors``
    to its SHA-256. Source and inventory identities remain in the safetensors
    header written by ``DenseEffectiveOutput``. Existing bundles are immutable.
    """
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(directory)
    if "files" in metadata:
        raise ValueError("Bundle file checksums are owned by the writer")
    for name, source in assets.items():
        files.relative_path(name)
        if name in {"config.json", "model.safetensors"}:
            raise ValueError(f"Reserved bundle asset: {name}")
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Invalid bundle source asset: {source}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{directory.name}-", dir=directory.parent
    ) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        inference_output.DenseEffectiveOutput().write(
            staged / "model.safetensors",
            parameters,
            inventory,
            source_model=source_model,
            source_revision=source_revision,
        )
        for name, source in assets.items():
            destination = staged / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        checksums = {
            name: json_io.digest_file(staged / name)
            for name in ("model.safetensors", *assets)
        }
        (staged / "config.json").write_text(
            json_io.canonical({**metadata, "files": checksums}),
            encoding="utf-8",
        )
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(directory)
        os.rename(staged, directory)


def inspect(
    directory: Path, *, expected_files: frozenset[str]
) -> dict[str, object]:
    """Verify the exact declared asset set and return caller-owned metadata."""
    metadata = json_io.object_map(
        json.loads(
            files.contained_file(directory, "config.json").read_text(
                encoding="utf-8"
            )
        )
    )
    checksums = json_io.object_map(metadata.get("files"))
    if any(not isinstance(digest, str) for digest in checksums.values()):
        raise ValueError("Invalid bundle checksums")
    files.verify(
        directory,
        (
            files.FileEntry(name, cast(str, digest))
            for name, digest in checksums.items()
        ),
        expected_paths=expected_files,
    )
    return metadata


def load_parameters(
    directory: Path,
    inventory: core_parameters.FullParameterInventory,
    *,
    expected_files: frozenset[str],
) -> types.Parameters:
    """Restore exact dense FP32 weights after verifying all bundle assets."""
    _, parameters = load(
        directory,
        expected_files=expected_files,
        inventory=lambda _: inventory,
    )
    return parameters


def load(
    directory: Path,
    *,
    expected_files: frozenset[str],
    inventory: Callable[
        [dict[str, object]], core_parameters.FullParameterInventory
    ],
) -> tuple[dict[str, object], types.Parameters]:
    """Inspect once, admit caller metadata, then restore exact dense tensors.

    ``inventory`` receives metadata only after all declared files pass byte
    verification. It validates the caller's model format and source policy,
    then supplies expected tensor shapes. The tensor loader independently
    rechecks the inspected weight digest before decoding. This keeps model
    configuration admission between file inspection and tensor allocation
    without repeating every asset's checksum pass.
    """
    metadata = inspect(directory, expected_files=expected_files)
    checksums = cast(dict[str, str], metadata["files"])
    if "model.safetensors" not in checksums:
        raise ValueError("Bundle is missing model.safetensors")
    weight_sha256 = checksums["model.safetensors"]
    expected_inventory = inventory(metadata)
    parameters = tensors.load_masters(
        directory / "model.safetensors",
        {spec.name: spec.shape for spec in expected_inventory.specs},
        source_dtype="F32",
        sha256=weight_sha256,
    )
    return metadata, parameters
