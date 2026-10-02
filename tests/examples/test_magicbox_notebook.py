"""Offline execution of the direct-kernel notebook and pinned source setup."""

import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


def _cells() -> list[str]:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/kaggle_magicbox_lfm350m_tpu_v5e_8.ipynb"
    )
    return [
        "".join(cell["source"])
        for cell in json.loads(path.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]


def _execute(source: str, namespace: dict[str, Any]) -> None:
    # Execute notebook orchestration with injected synthetic boundaries.
    # pylint: disable-next=exec-used
    exec(compile(source, "magicbox-notebook-cell", "exec"), namespace)


def _settings(tmp_path: Path, mode: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    code = _cells()[0].replace(
        "IS_KAGGLE = Path('/kaggle/working').is_dir()", "IS_KAGGLE = False"
    )
    code = code.replace("Path('/content')", f"Path({str(tmp_path)!r})")
    code = code.replace("RUN_MODE = 'smoke'", f"RUN_MODE = {mode!r}")
    _execute(code, namespace)
    return namespace


@pytest.mark.parametrize(
    ("count", "local", "hosts", "platform", "valid"),
    (
        (1, 1, 1, "tpu", True),
        (8, 8, 1, "tpu", True),
        (8, 4, 2, "tpu", False),
        (1, 1, 1, "cpu", False),
    ),
)
def test_direct_hardware_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    local: int,
    hosts: int,
    platform: str,
    valid: bool,
) -> None:
    """Probe actual kernel imports and reject unsupported topology."""
    namespace = _settings(tmp_path, "smoke")
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(
            __version__="0.7.2",
            devices=lambda: [
                SimpleNamespace(platform=platform, device_kind="test")
            ]
            * count,
            process_count=lambda: hosts,
            local_device_count=lambda: local,
        ),
    )
    if valid:
        _execute(_cells()[3], namespace)
        assert namespace["ROWS"] == count * namespace["ROWS_PER_DEVICE"]
        assert namespace["DEVICES"] == count
    else:
        with pytest.raises(RuntimeError):
            _execute(_cells()[3], namespace)


@pytest.mark.parametrize("mode", ("smoke", "full"))
def test_direct_training_resume_bounds(tmp_path: Path, mode: str) -> None:
    """Direct training cells preserve resume offsets and update limits."""
    namespace = _settings(tmp_path, mode)
    calls: list[SimpleNamespace] = []

    def run(
        *_args: object, **kwargs: Any
    ) -> tuple[dict[str, object], SimpleNamespace]:
        config = _args[4]
        assert isinstance(config, SimpleNamespace)
        calls.append(config)
        return {}, SimpleNamespace(
            next_batch=kwargs["cursor"].next_batch + config.max_steps
        )

    namespace.update(
        OUTPUT=tmp_path,
        current={},
        update=object(),
        run=SimpleNamespace(
            inventory=object(),
            stream=SimpleNamespace(total_updates=300),
            optimizer_id="test",
            corpus=SimpleNamespace(pointer_records=lambda *_args: iter(())),
        ),
        cursor=SimpleNamespace(next_batch=0),
        checkpoints=tmp_path,
        evaluator=SimpleNamespace(callback=lambda *_args: None),
        diagnostics=SimpleNamespace(
            monitor=lambda *_args: contextlib.nullcontext()
        ),
        training_state=SimpleNamespace(verify_roundtrip=lambda *_args: None),
        training_run=SimpleNamespace(
            run=run,
            RunConfig=lambda *args, **kwargs: SimpleNamespace(
                args=args, **kwargs
            ),
        ),
    )
    _execute(_cells()[12], namespace)
    _execute(_cells()[14], namespace)
    assert [call.max_steps for call in calls] == (
        [2, 8] if mode == "smoke" else [2, 298]
    )
    _execute(_cells()[12], namespace)
    _execute(_cells()[14], namespace)
    assert len(calls) == 2


def test_compile_is_a_separate_direct_stage(tmp_path: Path) -> None:
    """Compilation acts on the same state and first update the runner uses."""
    calls = []
    full_state, microbatches, active = object(), object(), object()
    executable = SimpleNamespace(memory_analysis=lambda: "stats")

    def compile_step() -> object:
        calls.append("compile")
        return executable

    def lower(
        actual_state: object, actual_microbatches: object, actual_active: object
    ) -> SimpleNamespace:
        assert actual_state is full_state
        assert actual_microbatches is microbatches
        assert actual_active is active
        calls.append("lower")
        return SimpleNamespace(compile=compile_step)

    namespace = dict(
        OUTPUT=tmp_path,
        first_update=SimpleNamespace(microbatches=microbatches, active=active),
        current=full_state,
        update=SimpleNamespace(lower=lower),
        diagnostics=SimpleNamespace(
            monitor=lambda *_args: contextlib.nullcontext()
        ),
    )
    _execute(_cells()[10], namespace)
    assert calls == ["lower"]
    _execute(_cells()[11], namespace)
    assert calls == ["lower", "compile"]
    assert namespace["compiled_step"] is executable


@pytest.mark.parametrize("compatible", [True, False])
def test_install_keeps_packages_the_kernel_already_imported(
    tmp_path: Path, compatible: bool
) -> None:
    """A preloaded numpy is kept; only an incompatible one needs a restart."""
    import numpy  # pylint: disable=import-outside-toplevel

    calls: list[list[str]] = []

    def check_call(command: list[str]) -> None:
        calls.append(command)
        if "--output-file" in command:
            Path(command[command.index("--output-file") + 1]).write_text(
                "numpy==2.2.6\n    # via jax\n"
                "scipy==1.16.0 ; python_version >= '3.12'\n",
                encoding="utf-8",
            )

    def run(command: list[str]) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0 if compatible else 1)

    kernel = SimpleNamespace(
        version_info=(3, 12, 0),
        modules={"numpy": SimpleNamespace(__version__=numpy.__version__)},
        executable="python",
        path=[],
    )
    namespace: dict[str, Any] = dict(
        sys=kernel,
        subprocess=SimpleNamespace(check_call=check_call, run=run),
        SCRATCH=tmp_path,
        CHECKOUT=tmp_path / "checkout",
    )
    if not compatible:
        with pytest.raises(RuntimeError, match="Restart the session"):
            _execute(_cells()[2], namespace)
        assert str(tmp_path / "notebook-requirements.txt") in calls[-1]
        return
    _execute(_cells()[2], namespace)
    kept = (tmp_path / "notebook-requirements-kept.txt").read_text()
    assert "numpy" not in kept and "scipy==1.16.0" in kept
    assert (tmp_path / "notebook-preloaded.txt").read_text() == (
        f"numpy=={numpy.__version__}\n"
    )
    assert "-c" in calls[-1]
    assert kernel.path == [str(tmp_path / "checkout")]


def test_no_training_subprocess() -> None:
    """Only Git and package installation may create subprocesses."""
    for cell in _cells()[3:]:
        assert "subprocess" not in cell
        assert "run_child" not in cell
        assert "examples.magicbox.train" not in cell


def _git(directory: Path, *arguments: str) -> str:
    """Run only synthetic repositories under the test's isolated Git config."""
    return subprocess.run(
        ["git", *arguments],
        cwd=directory,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture(name="local_source")
def isolated_git_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Isolate Git configuration, identity, and hooks for local-only clones."""
    for name in tuple(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    hooks = tmp_path / "empty-hooks"
    hooks.mkdir()
    for name, value in {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": str(hooks),
        "GIT_AUTHOR_NAME": "Notebook fixture",
        "GIT_AUTHOR_EMAIL": "notebook@example.invalid",
        "GIT_COMMITTER_NAME": "Notebook fixture",
        "GIT_COMMITTER_EMAIL": "notebook@example.invalid",
    }.items():
        monkeypatch.setenv(name, value)
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "--initial-branch=main")
    return source


def _publish(source: Path, content: str) -> str:
    """Publish distinct trainer bytes without executing model code."""
    trainer = source / "examples/magicbox/train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text(content, encoding="utf-8")
    _git(source, "add", "examples/magicbox/train.py")
    _git(source, "commit", "-m", "test: publish synthetic trainer")
    return _git(source, "rev-parse", "HEAD")


def _checkout_settings(
    tmp_path: Path, local_source: Path, revision: str
) -> dict[str, Any]:
    namespace = _settings(tmp_path, "smoke")
    namespace.update(
        SOURCE_REPO_URL=local_source.as_uri(), SOURCE_REVISION=revision
    )
    return namespace


def test_checkout_uses_exact_pin_and_reruns_without_changing_it(
    tmp_path: Path, local_source: Path
) -> None:
    """A newer default branch never changes the selected detached revision."""
    revision = _publish(local_source, "pinned trainer\n")
    newest = _publish(local_source, "newer default branch\n")
    namespace = _checkout_settings(tmp_path, local_source, revision)
    legacy = namespace["SCRATCH"] / "training"
    legacy.mkdir()
    preserved = legacy / "existing.txt"
    preserved.write_text("previous extracted source", encoding="utf-8")
    for _ in range(2):
        _execute(_cells()[1], namespace)
        checkout = namespace["CHECKOUT"]
        assert _git(checkout, "rev-parse", "HEAD") == revision
        assert _git(checkout, "rev-parse", "origin/main") == newest
        assert _git(checkout, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
        assert _git(checkout, "status", "--porcelain") == ""
        assert (checkout / "examples/magicbox/train.py").read_text(
            encoding="utf-8"
        ) == "pinned trainer\n"
        assert (
            preserved.read_text(encoding="utf-8") == "previous extracted source"
        )


def test_existing_checkout_fetches_a_newly_published_exact_pin(
    tmp_path: Path, local_source: Path
) -> None:
    """An existing clone fetches an absent commit from the configured source."""
    initial = _publish(local_source, "initial trainer\n")
    namespace = _checkout_settings(tmp_path, local_source, initial)
    _execute(_cells()[1], namespace)
    checkout = namespace["CHECKOUT"]
    published = _publish(local_source, "published after clone\n")
    with pytest.raises(subprocess.CalledProcessError):
        _git(checkout, "cat-file", "-e", f"{published}^{{commit}}")
    namespace["SOURCE_REVISION"] = published
    _execute(_cells()[1], namespace)
    assert _git(checkout, "rev-parse", "HEAD") == published
    assert (checkout / "examples/magicbox/train.py").read_text(
        encoding="utf-8"
    ) == "published after clone\n"
    assert _git(checkout, "status", "--porcelain") == ""


@pytest.mark.parametrize("change", ("tracked", "staged", "untracked"))
def test_checkout_refuses_to_overwrite_local_changes(
    tmp_path: Path, local_source: Path, change: str
) -> None:
    """Refusing a revision switch preserves edited bytes and index state."""
    initial = _publish(local_source, "initial trainer\n")
    namespace = _checkout_settings(tmp_path, local_source, initial)
    _execute(_cells()[1], namespace)
    checkout = namespace["CHECKOUT"]
    modified = checkout / (
        "local-note.txt"
        if change == "untracked"
        else "examples/magicbox/train.py"
    )
    modified.write_text("local changes\n", encoding="utf-8")
    if change == "staged":
        _git(checkout, "add", str(modified))
    status = _git(checkout, "status", "--porcelain")
    namespace["SOURCE_REVISION"] = _publish(
        local_source, "replacement trainer\n"
    )
    with pytest.raises(RuntimeError, match="Checkout contains local changes"):
        _execute(_cells()[1], namespace)
    assert modified.read_text(encoding="utf-8") == "local changes\n"
    assert _git(checkout, "rev-parse", "HEAD") == initial
    assert _git(checkout, "status", "--porcelain") == status


def test_checkout_refuses_an_existing_non_git_directory(
    tmp_path: Path, local_source: Path
) -> None:
    """An occupied checkout path is preserved rather than replaced by clone."""
    revision = _publish(local_source, "initial trainer\n")
    namespace = _checkout_settings(tmp_path, local_source, revision)
    checkout = namespace["CHECKOUT"]
    checkout.mkdir()
    preserved = checkout / "existing.txt"
    preserved.write_text("keep existing files", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Expected a clean checkout path"):
        _execute(_cells()[1], namespace)
    assert preserved.read_text(encoding="utf-8") == "keep existing files"
    assert not (checkout / ".git").exists()
