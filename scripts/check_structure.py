"""Run the fast architecture, duplication, and repository checks."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    """Fail on the first invalid structural contract."""
    for name in ("architecture", "duplicates", "repository"):
        print(f"Checking {name}", flush=True)
        subprocess.run(
            [sys.executable, f"scripts/check_{name}.py"], cwd=ROOT, check=True
        )


if __name__ == "__main__":
    main()
