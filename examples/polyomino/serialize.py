"""Serialize a 22-row polyomino board into the action-only ChatML prompt."""

from typing import cast

import tokenizers  # type: ignore[import-untyped]  # Native package lacks stubs.

from examples.polyomino import engine

LEGEND = (
    "You are playing a polyomino game. The board state is below.\n"
    ". is an empty cell\n"
    "# is a settled cell\n"
    "% is your falling piece"
)


def column_heights(cells: list[int]) -> list[int]:
    """Return per-column stack heights, 0-22."""
    heights = []
    for x in range(engine.WIDTH):
        height = 0
        for y in range(engine.HEIGHT):
            if bool(cells[y] & (1 << x)):
                height = engine.HEIGHT - y
                break
        heights.append(height)
    return heights


def hole_count(cells: list[int]) -> int:
    """Count empty cells with at least one occupied cell above in-column."""
    holes = 0
    for x in range(engine.WIDTH):
        capped = False
        for y in range(engine.HEIGHT):
            occupied = bool(cells[y] & (1 << x))
            if occupied:
                capped = True
            elif capped:
                holes += 1
    return holes


def board_rows(state: engine.Game) -> list[str]:
    """Render the 22-row grid with the falling piece overlaid as ``%``."""
    grid = [["."] * engine.WIDTH for _ in range(engine.HEIGHT)]
    for y in range(engine.HEIGHT):
        for x in range(engine.WIDTH):
            if state.board[y] & (1 << x):
                grid[y][x] = "#"
    for cx, cy in engine.SHAPES[state.kind][state.rotation]:
        x = state.x + cx
        y = state.y + cy
        if 0 <= x < engine.WIDTH and 0 <= y < engine.HEIGHT:
            grid[y][x] = "%"
    return [" " + " ".join(row) for row in grid]


def state_text(state: engine.Game) -> str:
    """Return the user-turn board serialization (no ChatML framing)."""
    heights = column_heights(state.board)
    height_text = " ".join(f"{h:02d}" for h in heights)
    drop = engine.drop_distance(
        state.board, state.kind, state.rotation, state.x, state.y
    )
    fields = (
        f"piece:{state.kind} "
        f"rot:{state.rotation} "
        f"x:{state.x:02d} y:{state.y:02d}\n"
        f"next:{state.bag[-1]}\n"
        f"heights:{height_text}\n"
        f"holes:{hole_count(state.board):02d} "
        f"drop:{drop:02d}"
    )
    rows = "\n".join(board_rows(state))
    return f"<board>\n{rows}\n</board>\n{fields}"


def _wrap(user_text: str) -> str:
    """Frame serialized state text in ChatML ending at ``action:``."""
    return (
        "<|startoftext|><|im_start|>system\n"
        f"{LEGEND}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_text}<|im_end|>\n"
        "<|im_start|>assistant\naction:"
    )


def prompt_text(state: engine.Game) -> str:
    """Return the full ChatML prompt ending at the decision position."""
    return _wrap(state_text(state))


def encode(
    state: engine.Game,
    tokenizer: tokenizers.Tokenizer,
) -> list[int]:
    """Tokenize one decision prompt; the last token is the readout slot."""
    return cast(
        list[int],
        tokenizer.encode(prompt_text(state), add_special_tokens=False).ids,
    )
