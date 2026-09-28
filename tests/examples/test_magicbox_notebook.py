"""Offline execution of notebook topology, bounds, and download wiring."""

from collections.abc import Callable
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
    ]


def _execute(source: str, namespace: dict[str, Any]) -> None:
    # The notebook runs only against temporary paths and injected external I/O.
    # pylint: disable-next=exec-used
    exec(compile(source, "magicbox-notebook-cell", "exec"), namespace)


def _settings(tmp_path: Path, mode: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    code = _cells()[1].replace(
        "IS_KAGGLE = Path('/kaggle/working').is_dir()", "IS_KAGGLE = False"
    )
    code = code.replace("Path('/content')", f"Path({str(tmp_path)!r})")
    code = code.replace("RUN_MODE = 'smoke'", f"RUN_MODE = {mode!r}")
    _execute(code, namespace)
    return namespace


def _runner(
    namespace: dict[str, Any], devices: int, calls: list[tuple[str, ...]]
) -> Callable[..., None]:
    def run(*args: str, **_kwargs: object) -> None:
        calls.append(args)
        if "-c" in args and args[-1].endswith("hardware.json"):
            Path(args[-1]).write_text(
                json.dumps({"devices": devices}), encoding="utf-8"
            )
        if "examples.magicbox.train" in args:
            output = Path(args[args.index("--output") + 1])
            manifests = sorted(output.glob("checkpoints/step-*/manifest.json"))
            previous = (
                json.loads(manifests[-1].read_text())["cursor"]["next_batch"]
                if manifests
                else 0
            )
            additional = (
                int(args[args.index("--max-steps") + 1])
                if "--max-steps" in args
                else 100
            )
            step = previous + additional
            checkpoint = output / "checkpoints" / f"step-{step:08d}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "manifest.json").write_text(
                json.dumps({"cursor": {"next_batch": step}})
            )

    namespace["run_child"] = run
    return run


@pytest.mark.parametrize("devices", (1, 8))
@pytest.mark.parametrize("mode", ("smoke", "full"))
def test_detected_topology_and_bounded_resume(
    tmp_path: Path, devices: int, mode: str
) -> None:
    """Scale global requests and keep repeated smoke invocations bounded."""
    namespace = _settings(tmp_path, mode)
    calls: list[tuple[str, ...]] = []
    _runner(namespace, devices, calls)
    _execute(_cells()[3], namespace)
    assert namespace["DEVICES"] == devices
    assert namespace["REQUESTS"] == devices
    assert namespace["OUTPUT"].name == f"magicbox-{mode}-{devices}dev"
    namespace.update(
        DATASET=tmp_path / "data", SOURCE_TOKENS=1024, SCHEMA_TOKENS=512
    )
    _execute(_cells()[7], namespace)
    _execute(_cells()[9], namespace)
    commands = [cmd for cmd in calls if "examples.magicbox.train" in cmd]
    assert len(commands) == 2
    startup, continuation = commands
    assert startup[startup.index("--max-steps") + 1] == "2"
    for command in commands:
        assert command[command.index("--devices") + 1] == str(devices)
        assert command[command.index("--requests") + 1] == str(devices)
        assert command[command.index("--source-tokens") + 1] == "1024"
        assert command[command.index("--schema-tokens") + 1] == "512"
        for option in ("--microbatches", "--row-chunk"):
            assert command[command.index(option) + 1] == (
                "1" if mode == "smoke" else "4"
            )
    if mode == "smoke":
        assert continuation[continuation.index("--max-steps") + 1] == "8"
        assert namespace["committed_updates"]() == 10
    else:
        assert "--max-steps" not in continuation
        assert continuation[continuation.index("--final-records") + 1] == "0"
    _execute(_cells()[7], namespace)
    _execute(_cells()[9], namespace)
    assert len([cmd for cmd in calls if "examples.magicbox.train" in cmd]) == (
        2 if mode == "smoke" else 3
    )


def test_one_update_smoke_and_explicit_topology_guard(tmp_path: Path) -> None:
    """Respect a smaller smoke cap and reject mismatched device requests."""
    namespace = _settings(tmp_path, "smoke")
    calls: list[tuple[str, ...]] = []
    _runner(namespace, 1, calls)
    namespace["DEVICES"] = 8
    with pytest.raises(ValueError, match="runtime exposes 1"):
        _execute(_cells()[3], namespace)
    namespace["DEVICES"] = 1
    _execute(_cells()[3], namespace)
    namespace.update(
        DATASET=tmp_path / "data",
        SOURCE_TOKENS=1024,
        SCHEMA_TOKENS=512,
        SMOKE_STEPS=1,
    )
    _execute(_cells()[7], namespace)
    _execute(_cells()[9], namespace)
    assert namespace["committed_updates"]() == 1
    assert len([cmd for cmd in calls if "examples.magicbox.train" in cmd]) == 1


@pytest.mark.parametrize(
    ("count", "local_count", "hosts", "platform", "valid"),
    (
        (1, 1, 1, "tpu", True),
        (8, 8, 1, "tpu", True),
        (8, 4, 2, "tpu", False),
        (8, 4, 1, "tpu", False),
        (1, 1, 1, "cpu", False),
    ),
)
def test_probe_requires_local_tpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    local_count: int,
    hosts: int,
    platform: str,
    valid: bool,
) -> None:
    """Execute the child probe against synthetic JAX hardware discovery."""
    namespace = _settings(tmp_path, "smoke")
    _runner(namespace, count, [])
    _execute(_cells()[3], namespace)
    fake_jax = SimpleNamespace(
        __version__="0.7.2",
        devices=lambda: [
            SimpleNamespace(platform=platform, device_kind="synthetic")
            for _ in range(count)
        ],
        process_count=lambda: hosts,
        local_device_count=lambda: local_count,
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    destination = tmp_path / "probe.json"
    monkeypatch.setattr(sys, "argv", ["probe", str(destination)])
    if valid:
        _execute(namespace["probe"], {})
        assert json.loads(destination.read_text())["devices"] == count
    else:
        with pytest.raises(RuntimeError):
            _execute(namespace["probe"], {})
        assert not destination.exists()


@pytest.mark.parametrize("local_dataset", (False, True))
def test_dataset_pin_and_local_override(
    tmp_path: Path, local_dataset: bool
) -> None:
    """Fetch the immutable processed dataset or honor a local directory."""
    namespace = _settings(tmp_path, "smoke")
    calls: list[tuple[str, ...]] = []
    run = _runner(namespace, 1, calls)
    _execute(_cells()[3], namespace)
    data = (
        tmp_path / "attached"
        if local_dataset
        else namespace["SCRATCH"] / "dataset"
    )
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "format": "minifield.magicbox/1.0",
                "complete": True,
                "mode": "full",
                "tokenizer": {"source_limit": 1024, "schema_limit": 512},
                "shards": [{"split": "train", "rows": 446751}],
            }
        )
    )
    if local_dataset:
        namespace["DATASET"] = data
    calls.clear()
    namespace["run_child"] = run
    _execute(_cells()[5], namespace)
    assert namespace["DATASET"] == data
    assert len(calls) == (1 if local_dataset else 2)
    if not local_dataset:
        assert calls[0][-3:] == (
            "protodotdesign/magicbox-v1",
            "f074bb549f16ea091fd8ece12e79652b8082871f",
            str(data),
        )


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
        _execute(_cells()[2], namespace)
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
    _execute(_cells()[2], namespace)
    checkout = namespace["CHECKOUT"]
    published = _publish(local_source, "published after clone\n")
    with pytest.raises(subprocess.CalledProcessError):
        _git(checkout, "cat-file", "-e", f"{published}^{{commit}}")
    namespace["SOURCE_REVISION"] = published
    _execute(_cells()[2], namespace)
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
    _execute(_cells()[2], namespace)
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
        _execute(_cells()[2], namespace)
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
        _execute(_cells()[2], namespace)
    assert preserved.read_text(encoding="utf-8") == "keep existing files"
    assert not (checkout / ".git").exists()
