"""Checkpoint discovery shares numeric ordering and complete cursor matching."""

import json
from pathlib import Path

import pytest

from minifield_training.checkpoints import discovery


def _checkpoint(root: Path, step: int, *, run_id: str = "run") -> Path:
    """Write independent structural fixtures without a numerical backend."""
    directory = root / f"step-{step:08d}"
    directory.mkdir(parents=True)
    (directory / "state.safetensors").write_bytes(b"loader verifies this later")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "cursor": {
                    "run_id": run_id,
                    "data_sha256": "data",
                    "source_id": "source",
                    "next_batch": step,
                }
            }
        ),
        encoding="utf-8",
    )
    return directory


def _latest(root: Path, *, reject_mismatched: bool = False) -> Path | None:
    return discovery.latest_checkpoint(
        root,
        run_id="run",
        data_sha256="data",
        source_id="source",
        reject_mismatched=reject_mismatched,
    )


def test_numeric_selection_and_explicit_foreign_identity_policy(
    tmp_path: Path,
) -> None:
    """Different digit lengths sort numerically; foreign runs are explicit."""
    assert _latest(tmp_path / "missing") is None
    _checkpoint(tmp_path, 99_999_999)
    expected = _checkpoint(tmp_path, 100_000_000)
    _checkpoint(tmp_path, 100_000_001, run_id="other")
    assert _latest(tmp_path) == expected
    with pytest.raises(ValueError, match="identity mismatch"):
        _latest(tmp_path, reject_mismatched=True)


def test_incomplete_and_ambiguous_candidates_do_not_displace_checkpoint(
    tmp_path: Path,
) -> None:
    """Interrupted writes, unexpected files, symlinks, and bad cursors lose."""
    expected = _checkpoint(tmp_path, 1)
    (tmp_path / "step-00000002").mkdir()
    extra = _checkpoint(tmp_path, 3)
    (extra / "extra").write_bytes(b"unexpected")
    bad_json = _checkpoint(tmp_path, 4)
    (bad_json / "manifest.json").write_bytes(b"{")
    wrong_cursor = _checkpoint(tmp_path, 5)
    (wrong_cursor / "manifest.json").write_text(
        '{"cursor":{"run_id":"run","data_sha256":"data",'
        '"source_id":"source","next_batch":6}}',
        encoding="utf-8",
    )
    linked_file = _checkpoint(tmp_path, 7)
    (linked_file / "state.safetensors").unlink()
    (linked_file / "state.safetensors").symlink_to(
        expected / "state.safetensors"
    )
    (tmp_path / "step-00000008").symlink_to(expected, target_is_directory=True)
    directory_asset = _checkpoint(tmp_path, 9)
    (directory_asset / "state.safetensors").unlink()
    (directory_asset / "state.safetensors").mkdir()
    assert _latest(tmp_path, reject_mismatched=True) == expected


def test_boolean_cursor_and_short_step_names_are_not_updates(
    tmp_path: Path,
) -> None:
    """Integer-like booleans and noncanonical short names are incomplete."""
    directory = _checkpoint(tmp_path, 1)
    manifest = directory / "manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["cursor"]["next_batch"] = True
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    _checkpoint(tmp_path, 2).rename(tmp_path / "step-2")
    assert _latest(tmp_path) is None
