"""Check documentation links, module guides and style bans.

Markdown checks cover inline links/images and reference definitions. Heading
fragments and remote resources aren't fetched. Code fences, indented examples,
inline code and HTML comments are excluded from documentation link inspection.
"""

import argparse
import ast
import pathlib
import re
import sys
import tomllib
import urllib.parse


def _prose(contents: str) -> str:
    """Remove fenced/indented code, inline code and HTML comments."""
    lines: list[str] = []
    fence = ""
    for line in contents.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            delimiter = marker.group(1)
            if not fence:
                fence = delimiter
            elif delimiter[0] == fence[0] and len(delimiter) >= len(fence):
                fence = ""
            lines.append("")
        elif fence or line.startswith(("    ", "\t")):
            lines.append("")
        else:
            lines.append(line)
    prose = "\n".join(lines)
    prose = re.sub(r"<!--.*?-->", "", prose, flags=re.DOTALL)
    return re.sub(r"(`+).*?\1", "", prose, flags=re.DOTALL)


def _destination(text: str) -> str:
    """Extract a Markdown destination, including balanced path parentheses."""
    text = text.lstrip()
    if text.startswith("<"):
        end = text.find(">")
        return text[1:end] if end > 0 else ""
    depth = 0
    result: list[str] = []
    escaped = False
    for char in text:
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char.isspace() or (char == ")" and depth == 0):
            break
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        result.append(char)
    return "".join(result)


def _destinations(contents: str) -> list[str]:
    """Find supported inline link/image and reference-definition targets."""
    prose = _prose(contents)
    targets = [
        _destination(prose[match.end() :])
        for match in re.finditer(r"(?<!!)\[[^\]\n]*\]\(|!\[[^\]\n]*\]\(", prose)
    ]
    targets.extend(
        _destination(match.group(1))
        for match in re.finditer(
            r"^ {0,3}\[[^\]\n]+\]:\s*(.+)$", prose, re.MULTILINE
        )
    )
    return targets


def _link_errors(path: pathlib.Path, root: pathlib.Path) -> list[str]:
    """Reject absent local targets and filesystem links outside the checkout."""
    errors: list[str] = []
    relative = path.relative_to(root).as_posix()
    for target in _destinations(path.read_text(encoding="utf-8")):
        if not target or target.startswith("#"):
            continue
        parsed = urllib.parse.urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue
        local = urllib.parse.unquote(parsed.path)
        destination = (path.parent / local).resolve()
        if (
            not destination.is_relative_to(root)
            or pathlib.Path(local).is_absolute()
        ):
            errors.append(
                f"{relative}: local link escapes repository: {target}"
            )
        elif not destination.exists():
            errors.append(f"{relative}: local link target is missing: {target}")
    return errors


def _documentation_errors(root: pathlib.Path) -> list[str]:
    """Check repository-owned Markdown while ignoring generated environments."""
    paths = list(root.glob("*.md"))
    for directory in ("docs", "src", "scripts", "tests"):
        paths.extend((root / directory).rglob("*.md"))
    errors: list[str] = []
    for path in sorted(set(paths)):
        try:
            errors.extend(_link_errors(path, root))
        except (OSError, ValueError) as error:
            errors.append(
                f"{path.relative_to(root)}: cannot read links: {error}"
            )
    return errors


def _module_guide_errors(root: pathlib.Path) -> list[str]:
    """Require substantive local README prose for every registered module."""
    with (root / "architecture.toml").open("rb") as source:
        policy = tomllib.load(source)
    source_root = policy.get("source_root")
    modules = policy.get("modules")
    if not isinstance(source_root, str) or not isinstance(modules, dict):
        raise ValueError("architecture needs source_root and modules")
    source_path = root / source_root
    if not source_path.resolve().is_relative_to(root):
        raise ValueError("architecture source_root must stay inside repository")
    errors: list[str] = []
    for name in modules:
        if not isinstance(name, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", name
        ):
            raise ValueError("architecture module names must be identifiers")
        path = source_path / name / "README.md"
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            errors.append(f"{relative}: registered modules require a README")
            continue
        words = re.findall(
            r"\b[A-Za-z][A-Za-z'-]*\b", _prose(path.read_text(encoding="utf-8"))
        )
        if len(words) < 40:
            errors.append(
                f"{relative}: module guide needs at least 40 prose words"
            )
    return errors


def _future_errors(root: pathlib.Path) -> list[str]:
    """Enforce the existing ban without importing any inspected Python file."""
    errors: list[str] = []
    for directory in ("src", "scripts", "tests"):
        for path in sorted((root / directory).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, SyntaxError) as error:
                errors.append(
                    f"{relative}: cannot parse Python source: {error}"
                )
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "__future__"
                ):
                    errors.append(
                        f"{relative}:{node.lineno}: "
                        "__future__ imports are forbidden"
                    )
    return errors


def inspect(root: pathlib.Path) -> list[str]:
    """Return repository documentation and style violations.

    Args:
        root: Repository root containing the architecture policy.

    Returns:
        Sorted, human-readable diagnostics, or an empty list when checks pass.
    """
    root = root.resolve()
    errors = _future_errors(root) + _documentation_errors(root)
    try:
        errors.extend(_module_guide_errors(root))
    except (OSError, ValueError) as error:
        errors.append(f"repository policy: {error}")
    return sorted(errors)


def main() -> int:
    """Run the checks for this checkout or another explicit repository root."""
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
    print("Repository documentation and style checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
