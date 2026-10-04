"""Offline execution of the tool-call curriculum notebook's own cells."""

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import jax
import pytest

from examples.magicbox import bundle as magicbox_bundle
from examples.magicbox import smoke
from examples.magicbox import tokenizer
from examples.toolcalls import composition
from examples.toolcalls import train as curriculum
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.magicbox import pointer


def _cells() -> list[str]:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/colab_toolcalls_lfm350m_tpu.ipynb"
    )
    return [
        "".join(cell["source"])
        for cell in json.loads(path.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]


def _execute(source: str, namespace: dict[str, Any]) -> None:
    # Execute notebook orchestration with injected synthetic boundaries.
    # pylint: disable-next=exec-used
    exec(compile(source, "toolcalls-notebook-cell", "exec"), namespace)


def _settings(tmp_path: Path, mode: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    code = _cells()[0].replace("Path('/content')", f"Path({str(tmp_path)!r})")
    code = code.replace("RUN_MODE = 'smoke'", f"RUN_MODE = {mode!r}")
    _execute(code, namespace)
    return namespace


@pytest.mark.parametrize(("mode", "cap"), (("smoke", 10), ("full", None)))
def test_curriculum_cell_runs_every_stage_with_the_mode_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, cap: int | None
) -> None:
    """Smoke caps each stage; full runs them; progress lands in the log."""
    namespace = _settings(tmp_path, mode)
    calls: list[dict[str, Any]] = []

    def run_curriculum(
        base: Any, root: Path, output: Path, **kwargs: Any
    ) -> dict[int, Path]:
        calls.append({"base": base, "root": root, **kwargs})
        kwargs["report"]({"event": "stage", "stage": 0})
        return {stage: output / f"stage{stage}" for stage in kwargs["stages"]}

    monkeypatch.setattr(curriculum, "run_curriculum", run_curriculum)
    namespace.update(
        encoder=encoder,
        DATASET=tmp_path / "dataset",
        OUTPUT=tmp_path / "out",
        DEVICES=1,
        ROWS=4,
    )
    _execute(_cells()[5], namespace)
    call = calls[0]
    assert len(calls) == 1
    assert call["stages"] == (0, 1, 2, 3)
    assert call["max_updates"] == cap
    assert call["base"].quantizer == "nf4"
    assert sorted(namespace["bundles"]) == [0, 1, 2, 3]
    assert json.loads((tmp_path / "out" / "progress.jsonl").read_text()) == {
        "event": "stage",
        "stage": 0,
    }


def test_settings_reject_unknown_stages(tmp_path: Path) -> None:
    """Only the 4 published stages can be selected."""
    code = _cells()[0].replace("Path('/content')", f"Path({str(tmp_path)!r})")
    with pytest.raises(ValueError, match="STAGES"):
        _execute(code.replace("STAGES = (0, 1, 2, 3)", "STAGES = (4,)"), {})


def test_copy_cell_keeps_bundles_and_results_per_stage(tmp_path: Path) -> None:
    """COPY_TO gets each stage's bundles and results, never its checkpoints."""
    output = tmp_path / "toolcalls-full-1dev"
    stage = output / "stage0"
    (stage / "checkpoints" / "step-00000010").mkdir(parents=True)
    (stage / "metrics").mkdir()
    (stage / "metrics" / "step.json").write_text("{}")
    (stage / "bundle-x").mkdir()
    (stage / "bundle-x" / "model.safetensors").write_text("weights")
    for name in ("run.json", "final-ood.json"):
        (stage / name).write_text("{}")
    (output / "progress.jsonl").write_text("{}\n")
    namespace: dict[str, Any] = dict(
        Path=Path,
        OUTPUT=output,
        COPY_TO=str(tmp_path / "drive"),
        STAGES=(0, 1),
    )
    _execute(_cells()[6], namespace)
    copied = tmp_path / "drive" / output.name
    assert sorted(path.name for path in (copied / "stage0").iterdir()) == [
        "bundle-x",
        "final-ood.json",
        "metrics",
        "run.json",
    ]
    assert (copied / "progress.jsonl").is_file()
    assert not (copied / "stage1").exists()


def test_predict_cell_runs_a_folded_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last bundle answers both questions through the plain forward."""
    cfg, _, encoder_params = smoke.tiny()
    head = pointer.Config(encoder_width=cfg.hidden_size, pointer_width=8)
    params = {
        name: value
        for name, value in encoder_params.items()
        if name.startswith("lfm2.")
    }
    params.update(pointer.initialize(head, jax.random.PRNGKey(3)))
    folded = composition.with_markers(composition.initialize(params))
    loads: list[tuple[Path, tuple[str, ...]]] = []

    def load_pointer(directory: Path, templates: tuple[str, ...]) -> Any:
        loads.append((directory, templates))
        return cfg, head, folded, {"presence_threshold": 0.5}

    monkeypatch.setattr(magicbox_bundle, "load_pointer", load_pointer)
    monkeypatch.setattr(
        tokenizer, "Adapter", lambda _: SimpleNamespace(encode=smoke.toy_encode)
    )
    namespace: dict[str, Any] = {
        "json": json,
        "bundles": {0: tmp_path / "stage0", 3: tmp_path / "stage3"},
    }
    _execute(_cells()[7], namespace)
    assert loads == [(tmp_path / "stage3", ("toolcall-pointer/1",))]
    assert set(namespace["predictions"]) == {"next_tool", "pod_name"}


def _git(directory: Path, *arguments: str) -> str:
    """Run only synthetic repositories under the test's isolated Git config."""
    return subprocess.run(
        ["git", *arguments],
        cwd=directory,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_checkout_requires_the_tool_call_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pinned checkout is accepted only when it carries this trainer."""
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
    (source / "examples/magicbox").mkdir(parents=True)
    _git(source, "init", "--initial-branch=main")
    (source / "examples/magicbox/train.py").write_text("magicbox only\n")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "test: magicbox only")
    without = _git(source, "rev-parse", "HEAD")
    (source / "examples/toolcalls").mkdir()
    (source / "examples/toolcalls/train.py").write_text("tool calls\n")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "test: tool-call trainer")
    with_trainer = _git(source, "rev-parse", "HEAD")
    namespace = _settings(tmp_path, "smoke")
    namespace.update(SOURCE_REPO_URL=source.as_uri(), SOURCE_REVISION=without)
    with pytest.raises(RuntimeError, match="tool-call trainer"):
        _execute(_cells()[1], namespace)
    namespace["SOURCE_REVISION"] = with_trainer
    _execute(_cells()[1], namespace)
    assert _git(namespace["CHECKOUT"], "rev-parse", "HEAD") == with_trainer
