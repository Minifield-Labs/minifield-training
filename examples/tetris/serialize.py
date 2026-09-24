"""Serialize a 22-row Tetris board into the action-only ChatML prompt."""

from typing import cast

import tokenizers  # type: ignore[import-untyped]  # Native package lacks stubs.

from examples.tetris import engine as tetris

LEGEND = (
    "You are playing tetris. The board state is below.\n"
    ". is an empty cell\n"
    "# is a settled cell\n"
    "% is your falling piece"
)


def column_heights(cells: bytes | bytearray) -> list[int]:
    """Return per-column stack heights, 0-22."""
    heights = []
    for x in range(tetris.COLS):
        height = 0
        for y in range(tetris.ROWS):
            if cells[y * tetris.COLS + x] != 0:
                height = tetris.ROWS - y
                break
        heights.append(height)
    return heights


def hole_count(cells: bytes | bytearray) -> int:
    """Count empty cells with at least one occupied cell above in-column."""
    holes = 0
    for x in range(tetris.COLS):
        capped = False
        for y in range(tetris.ROWS):
            occupied = cells[y * tetris.COLS + x] != 0
            if occupied:
                capped = True
            elif capped:
                holes += 1
    return holes


def board_rows(state: tetris.TetrisEngine) -> list[str]:
    """Render the 22-row grid with the falling piece overlaid as ``%``."""
    grid = [["."] * tetris.COLS for _ in range(tetris.ROWS)]
    for y in range(tetris.ROWS):
        for x in range(tetris.COLS):
            if state.cells[y * tetris.COLS + x] != 0:
                grid[y][x] = "#"
    for cx, cy in tetris.SHAPES[state.piece_kind][state.piece_rot]:
        x = state.piece_x + cx
        y = state.piece_y + cy
        if 0 <= x < tetris.COLS and 0 <= y < tetris.ROWS:
            grid[y][x] = "%"
    return [" " + " ".join(row) for row in grid]


def state_text(state: tetris.TetrisEngine) -> str:
    """Return the user-turn board serialization (no ChatML framing)."""
    heights = column_heights(state.cells)
    height_text = " ".join(f"{h:02d}" for h in heights)
    fields = (
        f"piece:{tetris.PIECE_NAMES[state.piece_kind]} "
        f"rot:{state.piece_rot} "
        f"x:{state.piece_x:02d} y:{state.piece_y:02d}\n"
        f"next:{tetris.PIECE_NAMES[state.next_kind]}\n"
        f"heights:{height_text}\n"
        f"holes:{hole_count(state.cells):02d} "
        f"drop:{state.drop_distance():02d}"
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


def prompt_text(state: tetris.TetrisEngine) -> str:
    """Return the full ChatML prompt ending at the decision position."""
    return _wrap(state_text(state))


def encode(
    state: tetris.TetrisEngine,
    tokenizer: tokenizers.Tokenizer,
) -> list[int]:
    """Tokenize one decision prompt; the last token is the readout slot."""
    return cast(
        list[int],
        tokenizer.encode(prompt_text(state), add_special_tokens=False).ids,
    )
