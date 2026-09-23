"""Prepared artifact integrity and stale identity checks."""

from dataclasses import replace
import json
from pathlib import Path
from typing import cast

import pytest

from minifield_training.core.json_io import canonical
from minifield_training.core.json_io import digest_file
from minifield_training.datasets.prepared import PreparationSettings
from minifield_training.datasets.prepared import iter_prepared
from minifield_training.datasets.prepared import save_prepared
from minifield_training.datasets.tokenization import TokenizedExample


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, PreparationSettings]:
    """Write actual synthetic source and tokenizer/template asset bytes."""
    source = tmp_path / "source.jsonl"
    tokenizer = tmp_path / "tokenizer.json"
    template = tmp_path / "template.jinja"
    source.write_text('{"id":"r"}\n', encoding="utf-8")
    tokenizer.write_text('{"vocab":{"x":1}}', encoding="utf-8")
    template.write_text("{{ messages }}", encoding="utf-8")
    return (
        source,
        tokenizer,
        template,
        PreparationSettings(
            mode="all",
            seed="seed",
            validation_fraction=0.0,
            max_tokens=8,
            overlength="error",
        ),
    )


def _example(tokenizer: Path, template: Path) -> TokenizedExample:
    """Bind example identity to the real asset files."""
    return TokenizedExample(
        "example",
        "group",
        "train",
        (1, 2, 3),
        (0, 1, 1),
        digest_file(tokenizer),
        digest_file(template),
    )


def test_roundtrip_and_stale_asset_rejected(tmp_path: Path) -> None:
    """Cache replay depends on source, tokenizer, template, and settings."""
    source, tokenizer, template, settings = _inputs(tmp_path)
    artifact = tmp_path / "prepared"
    expected = _example(tokenizer, template)
    save_prepared(
        artifact,
        [expected],
        source=source,
        tokenizer_asset=tokenizer,
        template_asset=template,
        settings=settings,
    )
    assert list(
        iter_prepared(
            artifact,
            source=source,
            tokenizer_asset=tokenizer,
            template_asset=template,
            settings=settings,
        )
    ) == [expected]
    template.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        list(
            iter_prepared(
                artifact,
                source=source,
                tokenizer_asset=tokenizer,
                template_asset=template,
                settings=settings,
            )
        )
    with pytest.raises(ValueError, match="invalid prepared example"):
        save_prepared(
            artifact,
            [expected],
            source=source,
            tokenizer_asset=tokenizer,
            template_asset=template,
            settings=settings,
        )


def test_corrupt_payload_and_rehashed_bad_mask_rejected(tmp_path: Path) -> None:
    """Checksums and row validation both protect prepared examples."""
    source, tokenizer, template, settings = _inputs(tmp_path)
    artifact = tmp_path / "prepared"
    manifest_path = save_prepared(
        artifact,
        [_example(tokenizer, template)],
        source=source,
        tokenizer_asset=tokenizer,
        template_asset=template,
        settings=settings,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = artifact / manifest["payload"]
    original = payload.read_text(encoding="utf-8")
    payload.write_text(
        original.replace('"loss_mask":[0,1,1]', '"loss_mask":[1,0,0]'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum"):
        list(
            iter_prepared(
                artifact,
                source=source,
                tokenizer_asset=tokenizer,
                template_asset=template,
                settings=settings,
            )
        )
    new_hash = digest_file(payload)
    new_path = artifact / f"{new_hash}.jsonl"
    payload.rename(new_path)
    manifest["checksum"] = new_hash
    manifest["payload"] = new_path.name
    manifest_path.write_text(canonical(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid prepared payload row"):
        list(
            iter_prepared(
                artifact,
                source=source,
                tokenizer_asset=tokenizer,
                template_asset=template,
                settings=settings,
            )
        )


def test_wrong_settings_and_unsafe_paths(tmp_path: Path) -> None:
    """A cache cannot be reused for changed preparation or a symlink path."""
    source, tokenizer, template, settings = _inputs(tmp_path)
    artifact = tmp_path / "prepared"
    save_prepared(
        artifact,
        [_example(tokenizer, template)],
        source=source,
        tokenizer_asset=tokenizer,
        template_asset=template,
        settings=settings,
    )
    with pytest.raises(ValueError, match="stale"):
        list(
            iter_prepared(
                artifact,
                source=source,
                tokenizer_asset=tokenizer,
                template_asset=template,
                settings=replace(settings, mode="turn"),
            )
        )
    alias = tmp_path / "alias"
    alias.symlink_to(artifact, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink-free"):
        list(
            iter_prepared(
                alias,
                source=source,
                tokenizer_asset=tokenizer,
                template_asset=template,
                settings=settings,
            )
        )


def test_float_manifest_version_and_mask_are_rejected(tmp_path: Path) -> None:
    """JSON numeric equality can't admit a float schema version or mask."""
    source, tokenizer, template, settings = _inputs(tmp_path)
    artifact = tmp_path / "prepared"
    expected = _example(tokenizer, template)
    bad_mask = replace(
        expected, loss_mask=cast(tuple[int, ...], (0.0, 1.0, 1.0))
    )
    with pytest.raises(ValueError, match="invalid prepared example"):
        save_prepared(
            artifact,
            [bad_mask],
            source=source,
            tokenizer_asset=tokenizer,
            template_asset=template,
            settings=settings,
        )
    manifest_path = save_prepared(
        artifact,
        [expected],
        source=source,
        tokenizer_asset=tokenizer,
        template_asset=template,
        settings=settings,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = 1.0
    manifest_path.write_text(canonical(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid prepared manifest"):
        list(
            iter_prepared(
                artifact,
                source=source,
                tokenizer_asset=tokenizer,
                template_asset=template,
                settings=settings,
            )
        )
