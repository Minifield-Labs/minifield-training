"""Exercise real import forms and fail-closed architecture policy handling."""

import pathlib
import shutil

import pytest

from scripts import check_architecture


@pytest.fixture(name="repository")
def repository_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create an isolated repository using the shipping policy."""
    policy = pathlib.Path(__file__).resolve().parents[2] / "architecture.toml"
    shutil.copyfile(policy, tmp_path / "architecture.toml")
    _write(tmp_path, "__init__.py", '"""Synthetic package."""\n')
    return tmp_path


def _write(root: pathlib.Path, module: str, content: str) -> None:
    path = root / "src/minifield_training" / module
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _messages(root: pathlib.Path) -> str:
    return "\n".join(str(item) for item in check_architecture.inspect(root))


def test_allowed_shared_layers_and_metadata(repository: pathlib.Path) -> None:
    """Shared arithmetic reaches families without pulling models into core."""
    _write(repository, "core/metadata.py", "import dataclasses\n")
    _write(repository, "kernels/types.py", "import jax\n")
    _write(repository, "layers/linear.py", "from ..kernels import types\n")
    _write(repository, "models/contracts.py", "from ..kernels import types\n")
    _write(repository, "models/lfm2/block.py", "from ...layers import linear\n")
    _write(
        repository, "models/falcon/block.py", "from ...layers import linear\n"
    )
    _write(repository, "objectives/loss.py", "from ..models import contracts\n")
    _write(
        repository, "strategies/packed.py", "from ..models.lfm2 import block\n"
    )
    assert not check_architecture.inspect(repository)


@pytest.mark.parametrize(
    "statement",
    [
        "import minifield_training.models.lfm2 as family",
        "from minifield_training.models import lfm2",
        "from ..models import lfm2",
        "from ..models.lfm2 import forward",
        "if False:\n    from ..models import lfm2",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
        "    from ..models import lfm2",
        "def load():\n    from ..models import lfm2",
    ],
)
def test_forbidden_edges_include_lazy_and_type_only_imports(
    repository: pathlib.Path, statement: str
) -> None:
    """Moving a dependency behind a conditional does not hide ownership."""
    _write(repository, "models/lfm2.py", "def forward():\n    pass\n")
    _write(repository, "kernels/linear.py", statement + "\n")
    assert "crosses ownership" in _messages(repository)


@pytest.mark.parametrize(
    "source",
    [
        "models/lfm2.py",
        "models/contracts.py",
        "engine/loop.py",
        "objectives/loss.py",
    ],
)
def test_concrete_models_only_reached_by_own_family_or_strategies(
    repository: pathlib.Path, source: str
) -> None:
    """Generic orchestration cannot absorb a concrete model family."""
    _write(repository, "models/falcon.py", "VALUE = 1\n")
    _write(repository, source, "from minifield_training.models import falcon\n")
    assert "crosses ownership" in _messages(repository)


def test_reverse_dependency_rejected(repository: pathlib.Path) -> None:
    """Kernels cannot reach high-level layer composition."""
    _write(repository, "layers/linear.py", "VALUE = 1\n")
    _write(repository, "kernels/dense.py", "from ..layers import linear\n")
    assert "kernels cannot depend on layers" in _messages(repository)


def test_host_dependency_closure_rejects_hidden_numerical_edge(
    repository: pathlib.Path,
) -> None:
    """Inspection covers dependencies even if the entrypoint contains no JAX."""
    _write(repository, "supervisor/cli.py", "from ..artifacts import inspect\n")
    _write(repository, "artifacts/inspect.py", "from ..core import metadata\n")
    _write(repository, "core/metadata.py", "import jax as accelerator\n")
    assert "core/metadata.py:1: host-safe core" in _messages(repository)


@pytest.mark.parametrize(
    "statement",
    [
        "import numpy",
        "def load():\n    import jax",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import torch",
        "import unknown_transitive_dependency",
    ],
)
def test_host_external_imports_fail_closed(
    repository: pathlib.Path, statement: str
) -> None:
    """Unknown third-party packages need an explicit reviewed host allowlist."""
    _write(repository, "artifacts/inspect.py", statement + "\n")
    assert "host-safe artifacts cannot import external" in _messages(repository)


def test_reviewed_cpu_dependency_can_be_allowed(
    repository: pathlib.Path,
) -> None:
    """CPU arrays may be admitted without permitting accelerator imports."""
    path = repository / "architecture.toml"
    before, marker, after = path.read_text().partition("[modules.datasets]")
    path.write_text(
        before
        + marker
        + after.replace("external = []", 'external = ["numpy"]', 1)
    )
    _write(repository, "datasets/tokens.py", "import numpy\n")
    assert not check_architecture.inspect(repository)


@pytest.mark.parametrize(
    "statement",
    [
        "import importlib\nimportlib.import_module('jax')",
        "from importlib import import_module as load\nload('jax')",
        "loader = __import__\nloader('jax')",
        "exec('import jax')",
        "import runpy",
        "import sys as system\nsystem.path.insert(0, '../other/src')",
        "from sys import path as lookup",
        "import sys\ngetattr(sys, 'path').append('other')",
        "import sys\nvars(sys)['path'].append('other')",
        "import sys\nsys.__dict__['path'].append('other')",
        "import os\nos.environ['PYTHONPATH'] = '/other'",
        "import ctypes",
    ],
)
def test_dynamic_and_path_bypasses_rejected(
    repository: pathlib.Path, statement: str
) -> None:
    """Loader APIs and import-path mutations cannot bypass static edges."""
    _write(repository, "strategies/bypass.py", statement + "\n")
    assert check_architecture.inspect(repository)


@pytest.mark.parametrize(
    "module, statement",
    [
        ("misc.py", "VALUE = 1"),
        ("unowned/data.py", "VALUE = 1"),
        ("numerical_types.py", "import jax"),
        ("__init__.py", "import jax"),
        ("core/__init__.py", "from .metadata import Metadata"),
        ("core/data.py", "from ...outside import name"),
        ("core/data.py", "from .. import *"),
        ("core/data.py", "import minifield_training"),
        ("core/data.py", "from ..unowned import missing"),
        ("core/data.py", "from ..core import nonexistent"),
    ],
)
def test_unowned_or_opaque_modules_rejected(
    repository: pathlib.Path, module: str, statement: str
) -> None:
    """All source and imports need a concrete permitted owner."""
    _write(repository, module, statement + "\n")
    assert check_architecture.inspect(repository)


def test_initializer_cannot_disguise_missing_module(
    repository: pathlib.Path,
) -> None:
    """A package's existence does not prove a referenced child exists."""
    _write(repository, "core/__init__.py", '"""Core."""\n')
    _write(repository, "core/data.py", "from . import missing\n")
    assert "does not resolve" in _messages(repository)


def test_other_source_package_rejected(repository: pathlib.Path) -> None:
    """A second src package cannot escape ownership classification."""
    (repository / "src/unowned.py").write_text("import jax\n", encoding="utf-8")
    assert "outside the owned package" in _messages(repository)


def test_symlinked_source_rejected(repository: pathlib.Path) -> None:
    """A source link cannot bypass checkout independence."""
    target = repository / "elsewhere.py"
    target.write_text("import jax\n", encoding="utf-8")
    (repository / "src/minifield_training/shortcut.py").symlink_to(target)
    assert "symlinked source" in _messages(repository)


def test_missing_policy_fails_closed(repository: pathlib.Path) -> None:
    """Removing policy cannot disable enforcement."""
    (repository / "architecture.toml").unlink()
    assert "architecture.toml" in _messages(repository)


def test_symlinked_source_root_rejected(repository: pathlib.Path) -> None:
    """The entire src directory cannot secretly resolve to another checkout."""
    source = repository / "src"
    alternate = repository / "elsewhere"
    source.rename(alternate)
    source.symlink_to(alternate, target_is_directory=True)
    assert "symlinked source" in _messages(repository)


@pytest.mark.parametrize(
    "before, after",
    [
        ("version = 1", "version = 2"),
        ('source_root = "src/minifield_training"', 'source_root = "elsewhere"'),
        ("host_safe = true", "host_safe = 'yes'"),
        ("depends = []", 'depends = ["missing"]'),
        ("depends = []", 'depends = ["core"]'),
        ('depends = ["core"]', 'depends = ["layers"]'),
        ("external = []", 'external = ["jax"]'),
        ("external = []", 'external = ["pathlib", "pathlib"]'),
        ("model_consumers =", "unknown_key ="),
        ("[modules.core]", "[modules.missing]"),
        (
            "lazy_imports = []",
            'lazy_imports = [{ source = "execution.*", '
            'target = "strategies.*" }]',
        ),
    ],
)
def test_invalid_policy_fails_closed(
    repository: pathlib.Path, before: str, after: str
) -> None:
    """Malformed rules, bypass allowlists and dependency cycles are errors."""
    path = repository / "architecture.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(before, after, 1),
        encoding="utf-8",
    )
    assert "architecture.toml" in _messages(repository)


def _enable_dispatch(repository: pathlib.Path) -> None:
    path = repository / "architecture.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "lazy_imports = []",
            'lazy_imports = [{ source = "execution.dispatch", '
            'target = "strategies.registry" }]',
        ),
        encoding="utf-8",
    )
    _write(repository, "strategies/registry.py", "import jax\n")


def test_exact_lazy_dispatch_boundary(repository: pathlib.Path) -> None:
    """An explicit runtime adapter can cross the host/numerical boundary."""
    _enable_dispatch(repository)
    _write(
        repository,
        "execution/dispatch.py",
        "def run():\n    from ..strategies import registry\n",
    )
    assert not check_architecture.inspect(repository)


def test_dispatch_import_must_be_inside_function(
    repository: pathlib.Path,
) -> None:
    """An approved runtime edge cannot eagerly initialize the accelerator."""
    _enable_dispatch(repository)
    _write(
        repository,
        "execution/dispatch.py",
        "from ..strategies import registry\n",
    )
    assert "must be function-local" in _messages(repository)


def test_unused_dispatch_allowance_rejected(repository: pathlib.Path) -> None:
    """Allowances must disappear with their adapter."""
    _enable_dispatch(repository)
    assert "unused lazy import edge" in _messages(repository)


def test_syntax_error_becomes_diagnostic(repository: pathlib.Path) -> None:
    """Broken source fails visibly without importing any code."""
    _write(repository, "core/broken.py", "def incomplete(\n")
    assert "cannot parse source" in _messages(repository)
