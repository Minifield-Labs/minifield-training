"""Offline execution of notebook topology, bounds, and download wiring."""

from collections.abc import Callable
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest


def _cells() -> list[str]:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/magicbox/notebook_template.json"
    )
    return [cell["source"] for cell in json.loads(path.read_text())["cells"]]


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
