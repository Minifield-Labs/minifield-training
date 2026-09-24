"""Run the complete fail-fast quality gate used locally and in CI."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    """Check structure, style, strict types, behavior, and installation."""
    paths = ["src", "scripts", "tests", "examples"]
    modules = sorted(
        str(path.relative_to(ROOT))
        for directory in paths
        for path in (ROOT / directory).rglob("*.py")
    )
    production = [path for path in modules if path.startswith("src/")]
    commands = [
        ["uv", "lock", "--check"],
        [sys.executable, "scripts/check_structure.py"],
        [
            sys.executable,
            "-m",
            "pyink",
            "--workers=1",
            "--check",
            "--diff",
            *modules,
        ],
        [sys.executable, "-m", "ruff", "check", *modules],
        [sys.executable, "-m", "pylint", "--jobs=1", *modules],
        [
            sys.executable,
            "-m",
            "pylint",
            "--jobs=1",
            "--disable=all",
            "--enable=duplicate-code",
            "--min-similarity-lines=8",
            "--ignore-imports=yes",
            *production,
        ],
        [sys.executable, "-m", "mypy"],
        [sys.executable, "-m", "pytest"],
        [sys.executable, "scripts/check_package.py"],
    ]
    for arguments in commands:
        label = " ".join(arguments[:4])
        print(f"Checking {label}", flush=True)
        subprocess.run(arguments, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
