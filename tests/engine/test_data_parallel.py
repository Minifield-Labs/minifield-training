"""Run multi-device numerical checks in a fresh, CPU-only JAX process."""

import os
from pathlib import Path
import subprocess
import sys


def test_eight_device_contracts() -> None:
    """Exercise actual collectives without changing the main test backend."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/engine/data_parallel_cases.py",
            "-q",
        ],
        cwd=root,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "8"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
