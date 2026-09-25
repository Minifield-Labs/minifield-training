"""Notebook syntax and offline QAT/dense command composition."""

import ast
from collections.abc import Callable
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "examples/colab_polyomino_classifier_tpu_v5e_8.ipynb"
)


def _cells() -> list[str]:
    """Return committed notebook cell source without executing setup."""
    raw = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in raw["cells"]]


def _download_runner(
    model_dir: Path, calls: list[tuple[str, ...]]
) -> Callable[..., str]:
    """Create an offline recorder for the notebook's model download cell."""

    def fake_run(*args: str, **_kwargs: object) -> str:
        """Materialize only filenames requested by the download."""
        calls.append(args)
        if "-c" in args:
            patterns = json.loads(args[-2])
            model_dir.mkdir(parents=True, exist_ok=True)
            for name in patterns:
                (model_dir / name).touch()
        return ""

    return fake_run


def test_code_cells_and_model_download(tmp_path: Path) -> None:
    """QAT downloads metadata only; dense setup still obtains Base weights."""
    cells = _cells()
    for index, source in enumerate(cells):
        if index in (1, 2, 3, 4, 6, 8, 10):
            compile(source, f"notebook-cell-{index}", "exec")
    for mode in ("dense", "qat"):
        model_dir = tmp_path / mode / "base-model"
        calls: list[tuple[str, ...]] = []

        namespace: dict[str, Any] = {
            "Path": Path,
            "json": json,
            "ROOT": tmp_path / mode,
            "MODEL_DIR": model_dir,
            "TRAINING_MODE": mode,
            "QAT_QUANTIZATION": "nf4",
            "UV": tmp_path / "uv",
            "PYTHON": tmp_path / "python",
            "CHECKOUT": tmp_path,
            "BASE_REVISION": "fixture-revision",
            "run_child": _download_runner(model_dir, calls),
        }
        # Notebook code runs with external calls replaced by a local recorder.
        # pylint: disable-next=exec-used
        exec(compile(cells[4], "download-cell", "exec"), namespace)
        assert (model_dir / "config.json").is_file()
        assert (model_dir / "tokenizer.json").is_file()
        assert (model_dir / "model.safetensors").is_file() == (mode == "dense")
        assert len(calls) == 2


def test_stale_bundle_fetches_pin_from_public_repo(tmp_path: Path) -> None:
    """An old uploaded bundle can't hide the pinned QAT source revision."""
    checkout = tmp_path / "checkout"
    bundle = tmp_path / "old.bundle"
    bundle.touch()
    revision = "a" * 40
    repo_url = "https://github.com/Minifield-Labs/minifield-training.git"
    commands: list[tuple[str, ...]] = []

    class FakeSubprocess:
        """Report the pinned commit only after fetching the public repo."""

        fetched = False

        def run(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            """Stand in for the notebook's read-only git cat-file probe."""
            return SimpleNamespace(returncode=0 if self.fetched else 1)

    fake_subprocess = FakeSubprocess()

    def fake_run(*args: str, **_kwargs: object) -> str:
        """Record the clone/fetch/checkout path without network access."""
        commands.append(args)
        if args[:2] == ("git", "clone"):
            (checkout / ".git").mkdir(parents=True)
            (checkout / "examples/polyomino").mkdir(parents=True)
            (checkout / "examples/polyomino/train.py").touch()
        elif args[:2] == ("git", "fetch"):
            fake_subprocess.fetched = True
        elif args[:3] == ("git", "rev-parse", "HEAD"):
            return revision
        return ""

    namespace: dict[str, Any] = {
        "CHECKOUT": checkout,
        "SOURCE_BUNDLE": bundle,
        "SOURCE_REPO_URL": repo_url,
        "SOURCE_REVISION": revision,
        "run_child": fake_run,
        "subprocess": fake_subprocess,
    }
    # Execute only the checkout cell with all git and network calls stubbed.
    # pylint: disable-next=exec-used
    exec(compile(_cells()[2], "checkout-cell", "exec"), namespace)
    assert ("git", "clone", str(bundle), str(checkout)) in commands
    assert (
        "git",
        "fetch",
        repo_url,
        "feat/quantization-strategies",
    ) in commands


@pytest.mark.parametrize("mode", ("dense", "qat"))
def test_smoke_and_long_run_choose_warm_then_resume(
    tmp_path: Path, mode: str
) -> None:
    """QAT starts from dense masters once, then resumes only QAT state."""
    cells = _cells()
    config_tree = ast.parse(cells[6])
    start_function = next(
        node
        for node in config_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "start_mode_args"
    )
    commands: list[tuple[str, ...]] = []

    def fake_run(*args: str, **_kwargs: object) -> str:
        """Publish bounded fake checkpoint manifests for command assertions."""
        commands.append(args)
        if "examples.polyomino.train" in args:
            root = Path(args[args.index("--checkpoint-root") + 1])
            step = 4 if (root / "step-00000002").exists() else 2
            directory = root / f"step-{step:08d}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "manifest.json").write_text(
                json.dumps({"cursor": {"next_batch": step}}),
                encoding="utf-8",
            )
            return '{"checkpoint_roundtrip_verified":1.0}'
        return ""

    namespace: dict[str, Any] = {
        "Path": Path,
        "json": json,
        "ROOT": tmp_path / "scratch",
        "CHECKOUT": tmp_path,
        "PYTHON": tmp_path / "python",
        "MODEL_DIR": tmp_path / "model",
        "DATASET_CACHE": tmp_path / "dataset",
        "TRAINING_MODE": mode,
        "QUANTIZATION": "nf4" if mode == "qat" else "dense",
        "DENSE_CHECKPOINT_DIR": Path("/kaggle/input/prior"),
        "DENSE_CURSOR": {"run_id": "prior-run", "source_id": "prior-source"},
        "DENSE_TENSOR_FILE": "model.safetensors",
        "recipe_args": ["--quantization", "nf4" if mode == "qat" else "dense"],
        "recipe_signature": "fixture-recipe",
        "TOTAL_UPDATES": 4,
        "run_child": fake_run,
    }
    # The extracted function reads only the supplied fixture globals.
    # pylint: disable-next=exec-used
    exec(
        compile(
            ast.Module(body=[start_function], type_ignores=[]),
            "mode-args",
            "exec",
        ),
        namespace,
    )
    # pylint: disable-next=exec-used
    exec(compile(cells[8], "smoke-cell", "exec"), namespace)
    smoke_train = next(
        cmd for cmd in commands if "examples.polyomino.train" in cmd
    )
    assert "--resume-latest" not in smoke_train
    assert ("--warm-start-checkpoint" in smoke_train) == (mode == "qat")
    if mode == "qat":
        assert "--warm-start-source-id" in smoke_train
        assert "model.safetensors" in smoke_train
    long_tree = ast.parse(cells[10])
    assert isinstance(long_tree.body[0], ast.AnnAssign)
    long_code = compile(
        ast.Module(body=long_tree.body[1:], type_ignores=[]),
        "long-cell",
        "exec",
    )
    namespace["PERSISTENT_ROOT"] = tmp_path / "outputs"
    namespace["PERSISTENT_ROOT"].mkdir()
    # Reexecute with fake I/O to verify the next invocation resumes QAT.
    # pylint: disable-next=exec-used
    exec(long_code, namespace)
    # pylint: disable-next=exec-used
    exec(long_code, namespace)
    long_trains = [
        cmd
        for cmd in commands
        if "examples.polyomino.train" in cmd and "--max-hours" in cmd
    ]
    assert len(long_trains) == 2
    assert "--resume-latest" not in long_trains[0]
    assert ("--warm-start-checkpoint" in long_trains[0]) == (mode == "qat")
    assert "--resume-latest" in long_trains[1]
    assert "--warm-start-checkpoint" not in long_trains[1]
