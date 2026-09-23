"""Versioned, content-verified local storage for tokenized examples."""

from collections.abc import Iterable, Iterator
from dataclasses import asdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from minifield_training.core.json_io import canonical
from minifield_training.core.json_io import digest_file
from minifield_training.datasets.conversations import _reject_constant
from minifield_training.datasets.conversations import _unique_pairs
from minifield_training.datasets.preparation import assign_split
from minifield_training.datasets.tokenization import TokenizedExample

_VERSION = 1


@dataclass(frozen=True)
class PreparationSettings:
    """Settings that determine a reusable tokenized dataset."""

    mode: str
    seed: str
    validation_fraction: float
    max_tokens: int
    overlength: str


def _safe_path(path: Path) -> None:
    """Reject relative paths and symlink traversal for local artifacts."""
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("artifact paths must be absolute and symlink-free")


def _identity(
    source: Path,
    tokenizer_asset: Path,
    template_asset: Path,
    settings: PreparationSettings,
) -> dict[str, object]:
    """Compute identity from actual input bytes and explicit settings."""
    for path in (source, tokenizer_asset, template_asset):
        _safe_path(path)
        if not path.is_file():
            raise ValueError("missing input asset")
    if (
        settings.mode not in {"all", "turn"}
        or not settings.seed
        or not 0 <= settings.validation_fraction < 1
        or settings.max_tokens < 2
        or settings.overlength not in {"error", "drop"}
    ):
        raise ValueError("invalid preparation settings")
    return {
        "source_sha256": digest_file(source),
        "tokenizer_sha256": digest_file(tokenizer_asset),
        "template_sha256": digest_file(template_asset),
        "settings": asdict(settings),
    }


def _validate(
    example: TokenizedExample,
    identity: dict[str, object],
    settings: PreparationSettings,
) -> None:
    """Check identity, shape, masks, and split before persistence or reuse."""
    ids = example.input_ids
    mask = example.loss_mask
    if (
        not example.id
        or not example.source_group
        or example.split
        != assign_split(
            example.source_group,
            seed=settings.seed,
            validation_fraction=settings.validation_fraction,
        )
        or example.tokenizer_id != identity["tokenizer_sha256"]
        or example.template_id != identity["template_sha256"]
        or not 2 <= len(ids) <= settings.max_tokens
        or len(ids) != len(mask)
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in ids
        )
        or any(item not in (0, 1) or isinstance(item, bool) for item in mask)
        or mask[0] != 0
        or not any(mask[1:])
    ):
        raise ValueError("invalid prepared example")


def _decode(line: str) -> TokenizedExample:
    """Read one strict versioned row without leaking payload in errors."""
    value = json.loads(
        line, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant
    )
    if not isinstance(value, dict) or set(value) != {
        "id",
        "source_group",
        "split",
        "input_ids",
        "loss_mask",
        "tokenizer_id",
        "template_id",
    }:
        raise ValueError("invalid prepared row schema")
    for key in ("id", "source_group", "split", "tokenizer_id", "template_id"):
        if not isinstance(value[key], str):
            raise ValueError("invalid prepared row identity")
    for key in ("input_ids", "loss_mask"):
        if not isinstance(value[key], list):
            raise ValueError("invalid prepared row array")
    return TokenizedExample(
        value["id"],
        value["source_group"],
        value["split"],
        tuple(value["input_ids"]),
        tuple(value["loss_mask"]),
        value["tokenizer_id"],
        value["template_id"],
    )


def _temp_path(directory: Path) -> Path:
    """Create a private staging file inside the artifact directory."""
    descriptor, name = tempfile.mkstemp(dir=directory, prefix=".staging-")
    os.close(descriptor)
    return Path(name)


def save_prepared(
    directory: Path,
    examples: Iterable[TokenizedExample],
    *,
    source: Path,
    tokenizer_asset: Path,
    template_asset: Path,
    settings: PreparationSettings,
) -> Path:
    """Publish a manifest after validating a content-named payload."""
    _safe_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("manifest symlinks are unsupported")
    identity = _identity(source, tokenizer_asset, template_asset, settings)
    data_stage = _temp_path(directory)
    try:
        count = 0
        names: set[str] = set()
        with data_stage.open("w", encoding="utf-8", newline="\n") as output:
            for example in examples:
                _validate(example, identity, settings)
                if example.id in names:
                    raise ValueError("duplicate prepared example id")
                names.add(example.id)
                output.write(canonical(asdict(example)) + "\n")
                count += 1
            output.flush()
            os.fsync(output.fileno())
        if count == 0:
            raise ValueError("prepared artifact must contain examples")
        checksum = digest_file(data_stage)
        payload_name = f"{checksum}.jsonl"
        os.replace(data_stage, directory / payload_name)
        manifest = {
            "version": _VERSION,
            "count": count,
            "checksum": checksum,
            "payload": payload_name,
            "identity": identity,
        }
        manifest_stage = _temp_path(directory)
        try:
            manifest_stage.write_text(
                canonical(manifest) + "\n", encoding="utf-8"
            )
            os.replace(manifest_stage, manifest_path)
        finally:
            manifest_stage.unlink(missing_ok=True)
        return manifest_path
    finally:
        data_stage.unlink(missing_ok=True)


def iter_prepared(
    directory: Path,
    *,
    source: Path,
    tokenizer_asset: Path,
    template_asset: Path,
    settings: PreparationSettings,
) -> Iterator[TokenizedExample]:
    """Verify manifest and every payload row before yielding any example."""
    _safe_path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("manifest symlinks are unsupported")
    try:
        manifest: Any = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError) as error:
        raise ValueError("invalid prepared manifest") from error
    identity = _identity(source, tokenizer_asset, template_asset, settings)
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {"version", "count", "checksum", "payload", "identity"}
        or manifest["version"] != _VERSION
        or isinstance(manifest["version"], bool)
        or manifest["identity"] != identity
        or not isinstance(manifest["checksum"], str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest["checksum"]) is None
        or manifest["payload"] != manifest["checksum"] + ".jsonl"
        or not isinstance(manifest["count"], int)
        or isinstance(manifest["count"], bool)
        or manifest["count"] < 1
    ):
        raise ValueError("stale or invalid prepared manifest")
    payload = directory / manifest["payload"]
    if payload.is_symlink() or not payload.is_file():
        raise ValueError("unsafe or missing prepared payload")
    if digest_file(payload) != manifest["checksum"]:
        raise ValueError("prepared payload checksum mismatch")
    names: set[str] = set()
    count = 0
    with payload.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                example = _decode(line)
                _validate(example, identity, settings)
            except (ValueError, TypeError) as error:
                raise ValueError("invalid prepared payload row") from error
            if example.id in names:
                raise ValueError("duplicate prepared example id")
            names.add(example.id)
            count += 1
    if count != manifest["count"]:
        raise ValueError("prepared payload count mismatch")
    with payload.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield _decode(line)
