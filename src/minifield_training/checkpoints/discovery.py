"""Select structurally complete checkpoints for a caller-supplied identity."""

import json
from pathlib import Path
import re
from typing import cast


def _cursor(directory: Path) -> dict[str, object] | None:
    """Read only a complete, regular-file checkpoint directory."""
    children = tuple(directory.iterdir())
    if {child.name for child in children} != {
        "manifest.json",
        "state.safetensors",
    } or any(child.is_symlink() or not child.is_file() for child in children):
        return None
    try:
        raw: object = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("cursor"), dict):
        return None
    return cast(dict[str, object], raw["cursor"])


def latest_checkpoint(
    root: Path,
    *,
    run_id: str,
    data_sha256: str,
    source_id: str,
    reject_mismatched: bool = False,
) -> Path | None:
    """Find the numerically newest complete matching checkpoint directory.

    Names are ``step-`` followed by at least 8 digits. The cursor's plain
    integer ``next_batch`` must match that number. Incomplete directories and
    symlinks are ignored. Tensor integrity is verified later by the state
    loader. Set ``reject_mismatched`` to reject complete checkpoints carrying
    another run/data/source identity instead of silently filtering them out.
    """
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for directory in root.iterdir():
        match = re.fullmatch(r"step-([0-9]{8,})", directory.name)
        if match is None or directory.is_symlink() or not directory.is_dir():
            continue
        cursor = _cursor(directory)
        if cursor is None or (
            # bool is an int subclass, but cursor steps require plain integers.
            # pylint: disable-next=unidiomatic-typecheck
            type(cursor.get("next_batch")) is not int
            or cursor["next_batch"] != int(match.group(1))
        ):
            continue
        if (
            cursor.get("run_id"),
            cursor.get("data_sha256"),
            cursor.get("source_id"),
        ) != (run_id, data_sha256, source_id):
            if reject_mismatched:
                raise ValueError(
                    f"Checkpoint run/data/source identity mismatch: {directory}"
                )
            continue
        candidates.append((int(match.group(1)), directory))
    return max(candidates)[1] if candidates else None
