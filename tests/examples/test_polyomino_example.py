"""Published decision schema, resume rules, and example entrypoints."""

import argparse
from pathlib import Path
import subprocess
import sys

from datasets import Dataset  # type: ignore[import-untyped]
import jax.numpy as jnp
import pytest
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]

from examples.polyomino import engine
from examples.polyomino import serialize
from examples.polyomino import source
from examples.polyomino import train
from minifield_training.batching import classification as batching
from minifield_training.batching import contracts
from minifield_training.batching import dense
from minifield_training.checkpoints import training_state
from minifield_training.core import parameters
from minifield_training.models.lfm2_5 import model
from minifield_training.optimizers import adamw

ROOT = Path(__file__).resolve().parents[2]


def _tokenizer() -> Tokenizer:
    """Use a deterministic small tokenizer for fixture packing."""
    tokenizer = Tokenizer(WordLevel({"[UNK]": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def _rows() -> list[dict[str, object]]:
    """Build saved decisions whose game state can be resumed."""
    game = engine.Game(17)
    rows = []
    for index in range(4):
        rows.append(
            {
                "game_id": index,
                "state": game.snapshot(),
                "expert_action": [0, 1, 0, 0, 0, 0, 0, 0],
            }
        )
        game.step(engine.LEFT)
    return rows


def test_saved_state_replays_and_prompt_uses_11_columns() -> None:
    """A restored decision advances exactly like its source game."""
    game = engine.Game(17)
    restored = engine.Game.from_state(game.snapshot())
    prompt = serialize.prompt_text(restored)
    assert prompt.startswith(
        "<|startoftext|><|im_start|>system\n"
        "You are playing a polyomino game."
    )
    assert prompt.endswith("<|im_end|>\n<|im_start|>assistant\naction:")
    board = prompt.split("<board>\n", 1)[1].split("\n</board>", 1)[0]
    assert len(board.splitlines()) == 22
    assert all(len(row.split()) == 11 for row in board.splitlines())
    for action in (engine.LEFT, engine.ROTATE_CW, engine.HARD_DROP):
        game.step(action)
        restored.step(action)
        assert game.snapshot() == restored.snapshot()


def test_one_hot_and_nonrepeating_resume_batches() -> None:
    """Update cursor skips old decisions and rejects malformed actions."""
    rows = _rows()
    tokenized = source.sequence_from_row(rows[0], _tokenizer())
    assert tokenized.label == engine.LEFT
    rows[0]["expert_action"] = [0] * 8
    with pytest.raises(ValueError, match="one-hot"):
        source.sequence_from_row(rows[0], _tokenizer())
    rows[0]["expert_action"] = [0, 1, 0, 0, 0, 0, 0, 0]
    data = Dataset.from_list(rows)
    strategy = dense.DenseBatchStrategy(
        contracts.BatchShape(1, 2, 512, 0, 100),
        batching.ClassTargets((True,) * 7 + (False,), 7),
    )
    batches = list(
        source.HFDatasetBatchSource(data, _tokenizer(), strategy, seed=17)(0)
    )
    resumed = list(
        source.HFDatasetBatchSource(data, _tokenizer(), strategy, seed=17)(1)
    )
    assert len(batches) == 2
    assert resumed[0].example_ids == batches[1].example_ids
    assert len(set(batches[0].example_ids + batches[1].example_ids)) == 4


def test_source_rejects_incompatible_batch_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strategy that splits chunks cannot silently corrupt HF resume."""
    strategy = dense.DenseBatchStrategy(
        contracts.BatchShape(1, 2, 512, 0, 100),
        batching.ClassTargets((True,) * 7 + (False,), 7),
    )
    monkeypatch.setattr(
        dense.DenseBatchStrategy, "update_count", lambda *_args: 2
    )
    batches = source.HFDatasetBatchSource(
        Dataset.from_list(_rows()), _tokenizer(), strategy
    )(0)
    with pytest.raises(ValueError, match="exactly one update"):
        next(batches)


def test_block_policy_binds_checkpoint_source() -> None:
    """Different gradient paths cannot silently share a checkpoint."""
    cfg = model.Config(4, 8, 1, 1, 8, ("conv",))
    args = argparse.Namespace(
        head_seed=6,
        data_seed=17,
        microbatches=4,
        rows=2,
        no_remat=False,
        fuse_accumulation=False,
        devices=1,
    )
    checkpointed = train.source_identity(args, cfg, sequence_length=512)
    args.no_remat = True
    plain = train.source_identity(args, cfg, sequence_length=512)
    assert checkpointed != plain
    args.no_remat = False
    args.devices = 8
    assert train.source_identity(args, cfg, sequence_length=512) != checkpointed


def test_smoke_rejects_changed_checkpoint_tensor(tmp_path: Path) -> None:
    """The smoke checks restored values against the live update."""
    inventory = parameters.build_inventory(
        {"weight": (2,)},
        format_id="smoke/1",
        decayed_names=frozenset(),
    )
    current = adamw.initialize_state(
        {"weight": jnp.asarray([1.0, 2.0], dtype=jnp.float32)},
        inventory,
    )
    cursor = training_state.Cursor("smoke", "data", "source", 0)
    directory = tmp_path / "checkpoint"
    training_state.save(
        directory,
        current,
        inventory,
        optimizer_id="adamw",
        cursor=cursor,
    )
    train.verify_checkpoint_roundtrip(
        directory, current, cursor, inventory, "adamw"
    )
    current["params"]["weight"] = jnp.asarray([1.0, 3.0], dtype=jnp.float32)
    with pytest.raises(RuntimeError, match="params/weight"):
        train.verify_checkpoint_roundtrip(
            directory, current, cursor, inventory, "adamw"
        )


@pytest.mark.parametrize("module", ("train", "evaluate"))
def test_clone_entrypoint(module: str) -> None:
    """Example commands work from an ordinary repository checkout."""
    result = subprocess.run(
        [sys.executable, "-m", f"examples.polyomino.{module}", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--help" in result.stdout
