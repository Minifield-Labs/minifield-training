"""Behavioral tests for clone detection, bounded modules and closed policy."""

import pathlib
import textwrap

import pytest

from scripts import check_duplicates


def _repository(tmp_path: pathlib.Path, *, policy: str = "") -> pathlib.Path:
    """Create an isolated source tree with deliberately small test limits."""
    source = tmp_path / "src" / "minifield_training"
    source.mkdir(parents=True)
    contents = policy or (
        "version = 1\n"
        "min_clone_statements = 3\n"
        "max_function_statements = 5\n"
        "max_module_statements = 20\n"
    )
    (tmp_path / "duplication.toml").write_text(contents, encoding="utf-8")
    return source


def _write(source: pathlib.Path, filename: str, content: str) -> None:
    (source / filename).write_text(textwrap.dedent(content), encoding="utf-8")


def _function(name: str = "first", value: int = 2) -> str:
    return (
        f"def {name}(value):\n"
        f"    '''Documentation for {name}.'''\n"
        f"    result = value + {value}\n"
        "    result *= 3\n"
        "    return result\n"
    )


def test_renamed_function_and_changed_docstring_are_duplicates(
    tmp_path: pathlib.Path,
) -> None:
    """Cosmetic changes cannot conceal a substantial function copy."""
    source = _repository(tmp_path)
    _write(source, "first.py", _function())
    _write(source, "second.py", _function("renamed"))
    errors = check_duplicates.inspect(tmp_path)
    assert len(errors) == 1
    assert "duplicate function AST" in errors[0]
    assert "first.py:1 (first)" in errors[0]
    assert "second.py:1 (renamed)" in errors[0]


def test_distinct_constants_and_tiny_boilerplate_are_allowed(
    tmp_path: pathlib.Path,
) -> None:
    """Preserve semantic constants and permit ordinary tiny wrappers."""
    source = _repository(tmp_path)
    _write(source, "first.py", _function())
    _write(source, "second.py", _function("other", value=4))
    _write(
        source,
        "tiny.py",
        "def one():\n    return 1\ndef two():\n    return 1\n",
    )
    assert check_duplicates.inspect(tmp_path) == []


@pytest.mark.parametrize(
    "changed",
    [
        "item.other",
        "item.method()",
        "getattr(item, 'other')",
    ],
)
def test_attribute_names_and_calls_remain_significant(
    tmp_path: pathlib.Path, changed: str
) -> None:
    """Different attribute access and call behavior must remain distinct."""
    source = _repository(tmp_path)
    first = _function().replace("value + 2", "item.value")
    _write(source, "first.py", first)
    _write(source, "second.py", first.replace("item.value", changed))
    assert check_duplicates.inspect(tmp_path) == []


@pytest.mark.parametrize(
    "prefix", ["class Owner:", "def outer():", "async def outer():"]
)
def test_methods_and_nested_functions_are_checked(
    tmp_path: pathlib.Path, prefix: str
) -> None:
    """Moving the duplicate into another scope cannot hide it."""
    source = _repository(tmp_path)
    _write(source, "first.py", _function())
    _write(
        source,
        "second.py",
        prefix + "\n" + textwrap.indent(_function(), "    "),
    )
    assert any(
        "duplicate function AST" in error
        for error in check_duplicates.inspect(tmp_path)
    )


def test_async_functions_are_checked_separately(tmp_path: pathlib.Path) -> None:
    """Async copies match each other, while a synchronous copy differs."""
    source = _repository(tmp_path)
    _write(source, "first.py", "async " + _function())
    _write(source, "second.py", "async " + _function("renamed"))
    _write(source, "sync.py", _function())
    errors = check_duplicates.inspect(tmp_path)
    assert len(errors) == 1
    assert "sync.py" not in errors[0]


def test_all_three_clone_locations_are_reported(tmp_path: pathlib.Path) -> None:
    """Report every consumer in a single diagnostic for each clone group."""
    source = _repository(tmp_path)
    for name in ("first", "second", "third"):
        _write(source, name + ".py", _function(name))
    errors = check_duplicates.inspect(tmp_path)
    assert len(errors) == 1
    assert all(
        name + ".py" in errors[0] for name in ("first", "second", "third")
    )


def test_independent_test_oracles_are_outside_production_gate(
    tmp_path: pathlib.Path,
) -> None:
    """Independent test implementations mustn't share production kernels."""
    source = _repository(tmp_path)
    _write(source, "implementation.py", _function())
    tests = tmp_path / "tests"
    tests.mkdir()
    _write(tests, "oracle.py", _function())
    assert check_duplicates.inspect(tmp_path) == []


@pytest.mark.parametrize("count, fails", [(5, False), (6, True)])
def test_function_size_limit_is_inclusive(
    tmp_path: pathlib.Path, count: int, fails: bool
) -> None:
    """Allow exactly the function limit and reject the next statement."""
    source = _repository(tmp_path)
    _write(
        source,
        "size.py",
        "def large():\n    '''Ignored.'''\n" + "    pass\n" * count,
    )
    errors = check_duplicates.inspect(tmp_path)
    assert bool(errors) is fails
    if fails:
        assert "function has 6 statements; limit is 5" in errors[0]


@pytest.mark.parametrize("count, fails", [(20, False), (21, True)])
def test_module_size_limit_is_inclusive(
    tmp_path: pathlib.Path, count: int, fails: bool
) -> None:
    """Ignore comments and docstrings while enforcing the module boundary."""
    source = _repository(tmp_path)
    _write(
        source, "size.py", "'''Ignored.'''\n# Comment.\n\n" + "pass\n" * count
    )
    errors = check_duplicates.inspect(tmp_path)
    assert bool(errors) is fails
    if fails:
        assert "module has 21 statements; limit is 20" in errors[0]


def test_syntax_failure_is_a_diagnostic(tmp_path: pathlib.Path) -> None:
    """Unparseable production source must fail the guard."""
    source = _repository(tmp_path)
    _write(source, "broken.py", "def unfinished(\n")
    errors = check_duplicates.inspect(tmp_path)
    assert len(errors) == 1
    assert "cannot parse source" in errors[0]


def test_nested_body_counts_toward_outer_function_size(
    tmp_path: pathlib.Path,
) -> None:
    """Nested scopes cannot hide a function's accumulated size."""
    source = _repository(tmp_path)
    body = "def outer():\n    def inner():\n" + "        pass\n" * 5
    _write(source, "nested.py", body)
    errors = check_duplicates.inspect(tmp_path)
    assert len(errors) == 1
    assert "(outer): function has 6 statements" in errors[0]


@pytest.mark.parametrize("directory", [False, True])
def test_symlinked_source_is_rejected(
    tmp_path: pathlib.Path, directory: bool
) -> None:
    """Source symlinks cannot silently escape scanning."""
    source = _repository(tmp_path)
    target = tmp_path / "elsewhere"
    if directory:
        target.mkdir()
    else:
        target.write_text(_function(), encoding="utf-8")
    (source / "linked.py").symlink_to(target, target_is_directory=directory)
    assert check_duplicates.inspect(tmp_path)


def test_invalid_source_encoding_is_a_diagnostic(
    tmp_path: pathlib.Path,
) -> None:
    """Non-UTF8 source must fail without crashing the quality runner."""
    source = _repository(tmp_path)
    (source / "encoding.py").write_bytes(b"\xff")
    errors = check_duplicates.inspect(tmp_path)
    assert len(errors) == 1
    assert "cannot parse source" in errors[0]


@pytest.mark.parametrize(
    "policy",
    [
        "unknown = 1\n",
        "version = 2\nmin_clone_statements = 3\n"
        "max_function_statements = 5\nmax_module_statements = 20\n",
        "version = true\nmin_clone_statements = 3\n"
        "max_function_statements = 5\nmax_module_statements = 20\n",
        "version = 1\nmin_clone_statements = 0\n"
        "max_function_statements = 5\nmax_module_statements = 20\n",
        "version = 1\nmin_clone_statements = 9\n"
        "max_function_statements = 5\nmax_module_statements = 20\n",
        "version = 1\nmin_clone_statements = 3\n"
        "max_function_statements = 25\nmax_module_statements = 20\n",
        "version = [\n",
    ],
)
def test_invalid_policy_fails_closed(
    tmp_path: pathlib.Path, policy: str
) -> None:
    """Malformed, unsupported or misleading policies cannot disable checks."""
    _repository(tmp_path, policy=policy)
    errors = check_duplicates.inspect(tmp_path)
    assert len(errors) == 1
    assert errors[0].startswith("duplication.toml:")


def test_missing_policy_or_source_is_rejected(tmp_path: pathlib.Path) -> None:
    """Absent inputs mustn't appear to be an empty passing repository."""
    assert "duplication.toml:" in check_duplicates.inspect(tmp_path)[0]
    source = _repository(tmp_path)
    source.rmdir()
    assert (
        "production source directory" in check_duplicates.inspect(tmp_path)[0]
    )
