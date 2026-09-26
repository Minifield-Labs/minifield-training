"""Scripted Tetris expert that scores landings and labels tick actions."""

from collections.abc import Sequence

import numpy as np

from examples.tetris import engine as tetris
from examples.tetris import serialize

DEFAULT_WEIGHTS = (9.0, 0.9, 0.25, 0.35)


def column_heights(cells: bytes | bytearray) -> list[int]:
    """Height of each column measured from the top occupied cell."""
    heights = []
    for x in range(tetris.COLS):
        top = 0
        for y in range(tetris.ROWS):
            if cells[y * tetris.COLS + x]:
                top = tetris.ROWS - y
                break
        heights.append(top)
    return heights


def simulate_lock(
    cells: bytearray, kind: int, rot: int, px: int, py: int
) -> tuple[bytearray, int]:
    """Lock a piece at an exact pose; return post-clear cells, n cleared."""
    board = bytearray(cells)
    for cx, cy in tetris.SHAPES[kind][rot]:
        by = py + cy
        if by >= 0:
            board[by * tetris.COLS + px + cx] = kind + 1
    cleared = 0
    for y in range(tetris.ROWS):
        if all(board[y * tetris.COLS + x] for x in range(tetris.COLS)):
            cleared += 1
    if cleared:
        kept = [
            bytes(board[y * tetris.COLS : (y + 1) * tetris.COLS])
            for y in range(tetris.ROWS)
            if not all(board[y * tetris.COLS + x] for x in range(tetris.COLS))
        ]
        board = bytearray(b"".join([bytes(tetris.COLS * cleared)] + kept))
    return board, cleared


def evaluate(
    cells: bytearray,
    kind: int,
    rot: int,
    px: int,
    py: int,
    weights: Sequence[float],
) -> float | None:
    """Score a candidate landing pose; None when it tops out."""
    if any(py + cy < 0 for _, cy in tetris.SHAPES[kind][rot]):
        return None
    board, cleared = simulate_lock(cells, kind, rot, px, py)
    heights = column_heights(board)
    bumpiness = sum(
        abs(heights[x] - heights[x + 1]) for x in range(tetris.COLS - 1)
    )
    w_clear, w_holes, w_height, w_bump = weights
    return (
        w_clear * cleared
        - w_holes * serialize.hole_count(board)
        - w_height * sum(heights)
        - w_bump * bumpiness
    )


def plan(
    engine: tetris.TetrisEngine, weights: Sequence[float]
) -> tuple[int, int, int] | None:
    """Choose (rot, x, landing_y) for the active piece, if any exists."""
    best_score = -np.inf
    best_pose = None
    for rot in range(4):
        for x in range(-4, tetris.COLS):
            if engine.collides(engine.piece_kind, rot, x, 0):
                continue
            dy = 0
            while not engine.collides(engine.piece_kind, rot, x, dy + 1):
                dy += 1
            score = evaluate(
                engine.cells, engine.piece_kind, rot, x, dy, weights
            )
            if score is not None and score > best_score:
                best_score = score
                best_pose = (rot, x, dy)
    return best_pose


def expert_action(engine: tetris.TetrisEngine, weights: Sequence[float]) -> int:
    """Pick the next action stepping toward the planned landing."""
    planned = plan(engine, weights)
    if planned is None:
        return tetris.Action.NONE
    rot, target_x = planned[0], planned[1]
    cw_turns = (rot - engine.piece_rot) % 4
    if cw_turns in (1, 2):
        return tetris.Action.ROTATE_CW
    if cw_turns == 3:
        return tetris.Action.ROTATE_CCW
    if engine.piece_x < target_x:
        return tetris.Action.RIGHT
    if engine.piece_x > target_x:
        return tetris.Action.LEFT
    return tetris.Action.HARD_DROP
