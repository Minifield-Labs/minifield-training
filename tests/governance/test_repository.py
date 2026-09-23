"""Test bounded repository rules without depending on the real documentation."""

import pathlib

import pytest

from scripts import check_repository


def _repository(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build the smallest valid independent repository fixture."""
    module = tmp_path / "src" / "minifield_training" / "core"
    module.mkdir(parents=True)
    (module / "README.md").write_text(
        "# Core\n\n"
        "Core owns stable metadata and neutral parameter contracts for the "
        "worker. Consumers use these contracts to exchange explicit values "
        "without importing model implementations. New changes must preserve "
        "documented compatibility and include tests for accepted values, "
        "invalid values, deterministic serialization and changes to the "
        "public interfaces.\n",
        encoding="utf-8",
    )
    (tmp_path / "architecture.toml").write_text(
        'source_root = "src/minifield_training"\n[modules.core]\n',
        encoding="utf-8",
    )
    return module


def test_valid_repository_and_supported_links_pass(
    tmp_path: pathlib.Path,
) -> None:
    """Resolve local references, spaces, escaped and balanced parentheses."""
    _repository(tmp_path)
    (tmp_path / "a file (1).md").write_text("# Target\n", encoding="utf-8")
    (tmp_path / "target(2).md").write_text("# Target\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[space](<a file (1).md>)\n"
        "[encoded](a%20file%20%281%29.md#heading)\n"
        "[balanced](target(2).md)\n"
        "[escaped](target\\(2\\).md)\n"
        "![image][reference]\n"
        '[reference]: target(2).md "Title"\n',
        encoding="utf-8",
    )
    assert check_repository.inspect(tmp_path) == []


def test_example_and_external_links_are_skipped(tmp_path: pathlib.Path) -> None:
    """Code examples, URLs and fragments don't need local targets."""
    _repository(tmp_path)
    (tmp_path / "README.md").write_text(
        "[remote](https://example.invalid/missing)\n"
        "[mail](mailto:person@example.invalid)\n[heading](#local-heading)\n"
        "`[inline](missing-inline.md)`\n"
        "```markdown\n[fenced](missing-fenced.md)\n```\n"
        "~~~markdown\n[fenced](missing-tilde.md)\n~~~\n"
        "    [indented](missing-indented.md)\n"
        "<!-- [comment](missing-comment.md) -->\n",
        encoding="utf-8",
    )
    assert check_repository.inspect(tmp_path) == []


@pytest.mark.parametrize(
    "target", ["missing.md", "../outside/README.md", "/tmp/absolute.md"]
)
def test_missing_or_nonportable_links_fail(
    tmp_path: pathlib.Path, target: str
) -> None:
    """Every local documentation target must exist inside this repository."""
    _repository(tmp_path)
    (tmp_path / "README.md").write_text(
        f"[broken]({target})\n", encoding="utf-8"
    )
    errors = check_repository.inspect(tmp_path)
    assert len(errors) == 1
    assert "local link" in errors[0]


def test_existing_symlink_target_cannot_escape_checkout(
    tmp_path: pathlib.Path,
) -> None:
    """An existing external file is still an invalid local dependency."""
    root = tmp_path / "repository"
    _repository(root)
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\n", encoding="utf-8")
    (root / "linked.md").symlink_to(outside)
    (root / "README.md").write_text("[link](linked.md)\n", encoding="utf-8")
    errors = check_repository.inspect(root)
    assert len(errors) == 1
    assert "local link escapes repository" in errors[0]


@pytest.mark.parametrize("directory", ["src", "scripts", "tests"])
def test_future_imports_fail_in_every_code_tree(
    tmp_path: pathlib.Path, directory: str
) -> None:
    """Future-import policy applies to tooling and tests as well as source."""
    _repository(tmp_path)
    parent = tmp_path / directory
    parent.mkdir(exist_ok=True)
    (parent / "legacy.py").write_text(
        "from __future__ import annotations\n", encoding="utf-8"
    )
    errors = check_repository.inspect(tmp_path)
    assert len(errors) == 1
    assert "__future__ imports are forbidden" in errors[0]


@pytest.mark.parametrize(
    "contents",
    [None, "# Core\n", "# Core\n```\n" + "example " * 60 + "\n```\n"],
)
def test_module_readme_requires_prose(
    tmp_path: pathlib.Path, contents: str | None
) -> None:
    """A missing guide, heading or pasted code cannot satisfy ownership docs."""
    module = _repository(tmp_path)
    readme = module / "README.md"
    if contents is None:
        readme.unlink()
    else:
        readme.write_text(contents, encoding="utf-8")
    errors = check_repository.inspect(tmp_path)
    assert len(errors) == 1
    assert "README" in errors[0] or "40 prose words" in errors[0]


def test_invalid_python_is_a_diagnostic(tmp_path: pathlib.Path) -> None:
    """A broken Python file cannot evade the import-policy check."""
    module = _repository(tmp_path)
    (module / "bad.py").write_text("def invalid(\n", encoding="utf-8")
    errors = check_repository.inspect(tmp_path)
    assert len(errors) == 1
    assert "cannot parse Python source" in errors[0]
