"""Contained file inventories with exact SHA-256 and optional size checks."""

from collections.abc import Iterable
import dataclasses
from pathlib import Path
from pathlib import PureWindowsPath
import re

from minifield_training.core import json_io


@dataclasses.dataclass(frozen=True)
class FileEntry:
    """One canonical relative file path and its expected byte identity."""

    path: str
    sha256: str
    size_bytes: int | None = None


def relative_path(name: str) -> Path:
    """Reject absolute paths, alternate spellings, and parent traversal."""
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or PureWindowsPath(name).drive
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError(f"Invalid artifact path: {name!r}")
    return Path(name)


def contained_file(root: Path, name: str) -> Path:
    """Resolve an existing regular file without following artifact symlinks."""
    relative = relative_path(name)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Invalid artifact directory: {root}")
    path = root
    for component in relative.parts:
        path /= component
        if path.is_symlink():
            raise ValueError(f"Symlink artifact path: {name}")
    if not path.is_file():
        raise ValueError(f"Missing artifact file: {name}")
    return path


def verify(
    root: Path,
    entries: Iterable[FileEntry],
    *,
    expected_paths: frozenset[str] | None = None,
) -> dict[str, Path]:
    """Verify unique declared files, returning their contained local paths.

    ``expected_paths`` checks the declared inventory, not unrelated files on
    disk. Every declaration has one canonical spelling; symlink files and
    symlink parent directories are rejected. Sizes, when supplied, are exact.
    """
    result: dict[str, Path] = {}
    for entry in entries:
        if entry.path in result:
            raise ValueError(f"Duplicate artifact path: {entry.path}")
        path = contained_file(root, entry.path)
        if (
            not isinstance(entry.sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", entry.sha256) is None
        ):
            raise ValueError(f"Invalid SHA-256 for artifact: {entry.path}")
        if entry.size_bytes is not None and (
            # bool is an int subclass, but a file size requires a plain int.
            # pylint: disable-next=unidiomatic-typecheck
            type(entry.size_bytes) is not int
            or entry.size_bytes < 0
            or path.stat().st_size != entry.size_bytes
        ):
            raise ValueError(f"Artifact size mismatch: {entry.path}")
        if json_io.digest_file(path) != entry.sha256:
            raise ValueError(f"Artifact SHA-256 mismatch: {entry.path}")
        result[entry.path] = path
    if expected_paths is not None and set(result) != expected_paths:
        raise ValueError("Artifact file inventory mismatch")
    return result
