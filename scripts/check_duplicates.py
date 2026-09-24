"""Reject copied contracts, substantial AST clones and oversized source."""

import argparse
import ast
import copy
import dataclasses
import hashlib
import pathlib
import sys
import tomllib


@dataclasses.dataclass(frozen=True)
class Policy:
    """Repository-reviewed limits measured in parsed Python statements."""

    min_clone_statements: int
    max_function_statements: int
    max_module_statements: int


@dataclasses.dataclass(frozen=True)
class CloneCandidate:
    """One definition's immutable diagnostic and comparison information."""

    kind: str
    location: str
    fingerprint: str


def _load_policy(root: pathlib.Path) -> Policy:
    """Read the small, closed policy schema, rejecting permissive typos."""
    with (root / "duplication.toml").open("rb") as source:
        values: dict[str, object] = tomllib.load(source)
    expected = {
        "version",
        "min_clone_statements",
        "max_function_statements",
        "max_module_statements",
    }
    if set(values) != expected:
        raise ValueError(
            "policy keys must be exactly " + ", ".join(sorted(expected))
        )
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in values.values()
    ):
        raise ValueError("all policy values must be integers (not booleans)")
    if values["version"] != 1:
        raise ValueError("unsupported duplication policy version")
    limits = {
        name: value
        for name, value in values.items()
        if name != "version" and isinstance(value, int)
    }
    if any(value < 1 for value in limits.values()):
        raise ValueError("statement limits must be positive")
    if not (
        limits["min_clone_statements"]
        <= limits["max_function_statements"]
        <= limits["max_module_statements"]
    ):
        raise ValueError("limits must satisfy clone <= function <= module")
    return Policy(**limits)


def _has_docstring(node: ast.AST) -> bool:
    """Whether a scope starts with Python's syntactic docstring form."""
    if not isinstance(
        node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ):
        return False
    return (
        bool(node.body)
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    )


def _statement_count(node: ast.AST) -> int:
    """Count all descendant statements except leading scope docstrings."""
    docstrings = {
        id(child.body[0])
        for child in ast.walk(node)
        if isinstance(
            child,
            ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        )
        and _has_docstring(child)
    }
    return sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.stmt)
        and child is not node
        and id(child) not in docstrings
    )


def _fingerprint(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Ignore only the root function's name, leading docstring and positions."""
    normalized = copy.deepcopy(node)
    normalized.name = "_function"
    if _has_docstring(normalized):
        normalized.body = normalized.body[1:]
    representation = ast.dump(normalized, include_attributes=False)
    return hashlib.sha256(representation.encode("utf-8")).hexdigest()


def _inspect_module(
    path: pathlib.Path, root: pathlib.Path, policy: Policy
) -> tuple[list[str], list[CloneCandidate]]:
    """Return size/syntax diagnostics, functions and named record contracts."""
    relative = path.relative_to(root).as_posix()
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        return [
            f"{relative}: production source must be a local regular file"
        ], []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, UnicodeError, SyntaxError) as error:
        return [f"{relative}: cannot parse source: {error}"], []
    errors: list[str] = []
    module_statements = _statement_count(tree)
    if module_statements > policy.max_module_statements:
        errors.append(
            f"{relative}: module has {module_statements} statements; "
            f"limit is {policy.max_module_statements}"
        )
    candidates: list[CloneCandidate] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            fields = sorted(
                member.target.id
                for member in node.body
                if isinstance(member, ast.AnnAssign)
                and isinstance(member.target, ast.Name)
            )
            if fields:
                field_names = ", ".join(fields)
                candidates.append(
                    CloneCandidate(
                        "record contract",
                        f"{relative}:{node.lineno} ({node.name})",
                        f"{node.name}({field_names})",
                    )
                )
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        location = f"{relative}:{node.lineno} ({node.name})"
        statements = _statement_count(node)
        if statements > policy.max_function_statements:
            errors.append(
                f"{location}: function has {statements} statements; "
                f"limit is {policy.max_function_statements}"
            )
        if statements >= policy.min_clone_statements:
            candidates.append(
                CloneCandidate("function AST", location, _fingerprint(node))
            )
    return errors, candidates


def inspect(root: pathlib.Path) -> list[str]:
    """Return all policy, parse, size, contract and function-clone violations.

    Args:
        root: Repository root containing duplication.toml and production source.

    Returns:
        Stable, human-readable diagnostics. An empty list means the gate passed.
    """
    root = root.resolve()
    try:
        policy = _load_policy(root)
    except (OSError, ValueError) as error:
        return [f"duplication.toml: {error}"]
    source = root / "src" / "minifield_training"
    if not source.is_dir() or source.is_symlink():
        return [
            "src/minifield_training: production source directory is invalid"
        ]
    errors: list[str] = []
    groups: dict[tuple[str, str], list[CloneCandidate]] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink() and path.is_dir():
            relative = path.relative_to(root).as_posix()
            errors.append(
                f"{relative}: production directories cannot be symlinked"
            )
    for path in sorted(source.rglob("*.py")):
        module_errors, candidates = _inspect_module(path, root, policy)
        errors.extend(module_errors)
        for candidate in candidates:
            groups.setdefault(
                (candidate.kind, candidate.fingerprint), []
            ).append(candidate)
    for (kind, fingerprint), candidates in groups.items():
        if len(candidates) > 1:
            locations = "; ".join(
                candidate.location for candidate in candidates
            )
            errors.append(
                f"duplicate {kind} [{fingerprint}]: {locations}; "
                "extract a shared owner or document distinct behavior"
            )
    return sorted(errors)


def main() -> int:
    """Run the gate against this checkout or an explicitly selected root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    options = parser.parse_args()
    errors = inspect(options.root)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print("Production clone and size checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
