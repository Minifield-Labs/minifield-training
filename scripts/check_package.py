"""Build distributions and verify an installed wheel outside the checkout."""

import json
import pathlib
import subprocess
import sys
import tempfile
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = """
import importlib
import importlib.metadata
import json
import pathlib
import sys

expected_version, expected_files, host_modules = sys.argv[1:]
distribution = importlib.metadata.distribution('minifield-training')
assert distribution.version == expected_version
installed = {str(path) for path in distribution.files}
assert set(json.loads(expected_files)) <= installed
for name in json.loads(host_modules):
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).is_relative_to(sys.prefix), name
numerical = {'jax', 'jaxlib', 'torch', 'tensorflow', 'triton', 'cupy'}
assert not numerical & sys.modules.keys()
print('Independent wheel installation and host imports passed.')
"""


def _host_modules() -> list[str]:
    """List implemented modules that require host-only import safety."""
    policy = tomllib.loads((ROOT / "architecture.toml").read_text())
    result = ["minifield_training"]
    package = ROOT / "src/minifield_training"
    for layer, rules in policy["modules"].items():
        if rules["host_safe"]:
            for path in sorted((package / layer).rglob("*.py")):
                parts = list(
                    path.relative_to(ROOT / "src").with_suffix("").parts
                )
                if parts[-1] == "__init__":
                    parts.pop()
                result.append(".".join(parts))
    return result


def main() -> None:
    """Install the built wheel in a fresh environment and probe it with -I."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    expected_files = sorted(
        path.relative_to(ROOT / "src").as_posix()
        for path in (ROOT / "src").rglob("*")
        if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
    )
    with tempfile.TemporaryDirectory(prefix="minifield-wheel-") as directory:
        temporary = pathlib.Path(directory)
        output = temporary / "dist"
        subprocess.run(
            [
                "uv",
                "build",
                "--quiet",
                "--no-sources",
                "--out-dir",
                str(output),
            ],
            cwd=ROOT,
            check=True,
        )
        wheels = list(output.glob("*.whl"))
        if len(wheels) != 1 or len(list(output.glob("*.tar.gz"))) != 1:
            raise RuntimeError("Expected exactly one wheel and source archive")
        environment = temporary / "environment"
        subprocess.run(
            ["uv", "venv", str(environment), "--python", sys.executable],
            cwd=temporary,
            check=True,
        )
        python = environment / "bin/python"
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), str(wheels[0])],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                PROBE,
                project["project"]["version"],
                json.dumps(expected_files),
                json.dumps(_host_modules()),
            ],
            cwd=temporary,
            check=True,
        )


if __name__ == "__main__":
    main()
