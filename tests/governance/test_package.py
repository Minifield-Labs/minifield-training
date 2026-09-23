"""Exercise source-archive admission against real Git ignore rules."""

import io
import pathlib
import subprocess
import tarfile

import pytest

from scripts import check_package


def _repository(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create an independent checkout without commits or remote access."""
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    return root


def _archive(tmp_path: pathlib.Path, paths: list[str]) -> pathlib.Path:
    """Write a source archive with explicit fixture members."""
    archive = tmp_path / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for path in paths:
            member = tarfile.TarInfo(f"minifield_training-0.1.0/{path}")
            member.size = 7
            output.addfile(member, io.BytesIO(b"fixture"))
    return archive


@pytest.mark.parametrize("rules", [".gitignore", ".git/info/exclude"])
def test_ignored_archive_members_fail(
    tmp_path: pathlib.Path, rules: str
) -> None:
    """Local and committed exclusions both bar scratch files from artifacts."""
    root = _repository(tmp_path)
    (root / rules).write_text("docs/local/\n", encoding="utf-8")
    archive = _archive(tmp_path, ["README.md", "docs/local/draft notes.md"])
    with pytest.raises(RuntimeError, match="docs/local/draft notes.md"):
        check_package.check_source_archive(root, archive)


def test_tracked_and_new_source_with_generated_metadata_pass(
    tmp_path: pathlib.Path,
) -> None:
    """Tracked exceptions, new source and generated egg-info remain valid."""
    root = _repository(tmp_path)
    (root / "README.md").write_text("# Package\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    (root / ".gitignore").write_text(
        "README.md\n*.egg-info/\n", encoding="utf-8"
    )
    archive = _archive(
        tmp_path,
        [
            "README.md",
            "src/minifield_training/core/new.py",
            "src/minifield_training.egg-info/PKG-INFO",
            "PKG-INFO",
        ],
    )
    check_package.check_source_archive(root, archive)


def test_checkout_is_required(tmp_path: pathlib.Path) -> None:
    """The quality gate fails explicitly when Git cannot inspect a checkout."""
    archive = _archive(tmp_path, ["README.md"])
    with pytest.raises(RuntimeError, match="Cannot check source archive"):
        check_package.check_source_archive(tmp_path, archive)
