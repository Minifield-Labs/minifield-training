"""Host-safe artifact admission rejects ambiguous paths and changed bytes."""

from pathlib import Path

import pytest

from minifield_training.artifacts import files
from minifield_training.core import json_io


def test_exact_declared_files_and_optional_sizes(tmp_path: Path) -> None:
    """Multiple artifact consumers can verify nested files with one contract."""
    nested = tmp_path / "shards"
    nested.mkdir()
    shard = nested / "part.bin"
    shard.write_bytes(b"abc")
    entry = files.FileEntry(
        "shards/part.bin",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        3,
    )
    assert files.verify(
        tmp_path, [entry], expected_paths=frozenset({"shards/part.bin"})
    ) == {"shards/part.bin": shard}
    with pytest.raises(ValueError, match="Duplicate"):
        files.verify(tmp_path, [entry, entry])
    with pytest.raises(ValueError, match="inventory"):
        files.verify(tmp_path, [entry], expected_paths=frozenset())
    with pytest.raises(ValueError, match="size"):
        files.verify(tmp_path, [files.FileEntry(entry.path, entry.sha256, 4)])
    shard.write_bytes(b"abd")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        files.verify(tmp_path, [entry])


@pytest.mark.parametrize(
    "name",
    [
        "",
        "/file",
        "../file",
        "a/../file",
        "./file",
        "a//file",
        "file/",
        "a\\b",
        "C:/file",
    ],
)
def test_paths_have_one_contained_spelling(name: str) -> None:
    """Traversal and alternate relative spellings cannot alias an inventory."""
    with pytest.raises(ValueError, match="Invalid artifact path"):
        files.relative_path(name)


def test_symlinks_are_rejected_at_every_artifact_level(tmp_path: Path) -> None:
    """A valid digest never authorizes following a file or directory symlink."""
    outside = tmp_path / "outside"
    outside.mkdir()
    payload = outside / "value"
    payload.write_bytes(b"source")
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "file-link").symlink_to(payload)
    (root / "directory-link").symlink_to(outside, target_is_directory=True)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(root, target_is_directory=True)
    for base, name in (
        (root, "file-link"),
        (root, "directory-link/value"),
        (root_link, "file-link"),
    ):
        with pytest.raises(ValueError, match="[Ss]ymlink|directory"):
            files.verify(
                base, [files.FileEntry(name, json_io.digest_file(payload))]
            )
    with pytest.raises(ValueError, match="Missing"):
        files.contained_file(root, "missing")
    with pytest.raises(ValueError, match="Missing"):
        files.contained_file(tmp_path, "artifact")


def test_invalid_digest_is_rejected(tmp_path: Path) -> None:
    """A checksum declaration must be a lowercase SHA-256 hex digest."""
    (tmp_path / "data").write_bytes(b"abc")
    with pytest.raises(ValueError, match="Invalid SHA-256"):
        files.verify(tmp_path, [files.FileEntry("data", "sha256:abc")])
