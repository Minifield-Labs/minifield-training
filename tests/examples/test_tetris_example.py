"""Repository-owned Tetris example import and decision contracts."""

from pathlib import Path
import subprocess
import sys

import pytest

from examples.tetris import engine
from examples.tetris import expert
from examples.tetris import serialize

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module", ("prepare", "train", "evaluate"))
def test_clone_entrypoint(module: str) -> None:
    """Each example command works from an ordinary repository checkout."""
    result = subprocess.run(
        [sys.executable, "-m", f"examples.tetris.{module}", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--help" in result.stdout


def test_complete_action_prompt_and_expert_tick() -> None:
    """The prompt keeps the supplied framing and a real engine action."""
    game = engine.TetrisEngine(engine.mulberry32(17))
    prompt = serialize.prompt_text(game)
    assert prompt.startswith(
        "<|startoftext|><|im_start|>system\n"
        "You are playing tetris. The board state is below.\n"
        ". is an empty cell\n# is a settled cell\n"
        "% is your falling piece<|im_end|>\n"
        "<|im_start|>user\n<board>\n"
    )
    assert prompt.endswith("<|im_end|>\n<|im_start|>assistant\naction:")
    board = prompt.split("<board>\n", 1)[1].split("\n</board>", 1)[0]
    assert len(board.splitlines()) == engine.ROWS
    assert all(len(row.split()) == engine.COLS for row in board.splitlines())
    action = expert.expert_action(game, expert.DEFAULT_WEIGHTS)
    assert 0 <= action < 7
    game.queue_input(action)
    game.tick()
    assert game.ticks == 1
