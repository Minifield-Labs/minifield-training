"""Build distributions and verify an installed wheel outside the checkout."""

import json
import pathlib
import subprocess
import sys
import tarfile
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


def check_source_archive(root: pathlib.Path, archive: pathlib.Path) -> None:
    """Reject Git-ignored checkout files selected by the source manifest.

    Setuptools generates egg-info metadata itself. All other archive files
    must be eligible for version control in the checkout running this gate.
    Untracked, nonignored files are allowed during local development.
    """
    with tarfile.open(archive, "r:gz") as source:
        paths = [
            pathlib.PurePosixPath(member.name)
            .relative_to(pathlib.PurePosixPath(member.name).parts[0])
            .as_posix()
            for member in source.getmembers()
            if member.isfile()
        ]
    paths = [
        path
        for path in paths
        if not path.startswith("src/minifield_training.egg-info/")
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--stdin", "-z"],
        cwd=root,
        input="\0".join(paths) + "\0",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Cannot check source archive: {result.stderr}")
    ignored = sorted(filter(None, result.stdout.split("\0")))
    if ignored:
        raise RuntimeError(
            "Source archive contains Git-ignored files: " + ", ".join(ignored)
        )


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
        archives = list(output.glob("*.tar.gz"))
        if len(wheels) != 1 or len(archives) != 1:
            raise RuntimeError("Expected exactly one wheel and source archive")
        check_source_archive(ROOT, archives[0])
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
