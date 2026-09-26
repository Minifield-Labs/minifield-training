"""Seeded 22-by-10 Tetris engine for expert data and learned-policy games."""

from collections.abc import Callable
from typing import Any

COLS = 10
ROWS = 22
HIDDEN_ROWS = 2
VISIBLE_ROWS = ROWS - HIDDEN_ROWS


class Action:
    """Engine action ids; order matches the TypeScript enum."""

    NONE = 0
    LEFT = 1
    RIGHT = 2
    ROTATE_CW = 3
    ROTATE_CCW = 4
    SOFT_DROP = 5
    HARD_DROP = 6


ACTION_COUNT = 8  # seven real actions plus the padded head slot


class PieceKind:
    """Piece kind ids; order matches the TypeScript enum."""

    I = 0  # noqa: E741  (names mirror the TypeScript enum)
    O = 1  # noqa: E741
    T = 2
    S = 3
    Z = 4
    J = 5
    L = 6


PIECE_NAMES = "iotszjl"
LINE_SCORES = (0, 100, 300, 500, 800)

_BASE: tuple[tuple[int, tuple[tuple[int, int], ...]], ...] = (
    (4, ((0, 1), (1, 1), (2, 1), (3, 1))),
    (2, ((0, 0), (1, 0), (0, 1), (1, 1))),
    (3, ((1, 0), (0, 1), (1, 1), (2, 1))),
    (3, ((1, 0), (2, 0), (0, 1), (1, 1))),
    (3, ((0, 0), (1, 0), (1, 1), (2, 1))),
    (3, ((0, 0), (0, 1), (1, 1), (2, 1))),
    (3, ((2, 0), (0, 1), (1, 1), (2, 1))),
)

KICKS = (0, -1, 1, -2, 2)


def _shapes() -> tuple[tuple[tuple[tuple[int, int], ...], ...], ...]:
    """Derive all four rotations per kind, matching the TS derivation."""
    all_kinds = []
    for size, cells in _BASE:
        rotations = [tuple(cells)]
        for _ in range(1, 4):
            rotations.append(tuple((size - 1 - y, x) for x, y in rotations[-1]))
        all_kinds.append(tuple(rotations))
    return tuple(all_kinds)


SHAPES = _shapes()


def _int32(value: int) -> int:
    """Wrap an integer to the signed int32 range (JS ``| 0``)."""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


class Mulberry32:
    """Seeded PRNG used by the TS engine (``Math.imul`` semantics).

    ``state`` is the full generator position: snapshot/restore it to
    reproduce the engine's piece stream exactly.
    """

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def __call__(self) -> float:
        """Advance the generator and return a float in [0, 1)."""
        a = _int32(self.state + 0x6D2B79F5)
        self.state = a & 0xFFFFFFFF
        t = _int32(a ^ ((a & 0xFFFFFFFF) >> 15))
        t = _int32(t * _int32(a | 1))
        mixed = _int32(
            t + _int32((t ^ ((t & 0xFFFFFFFF) >> 7)) * _int32(t | 61))
        )
        t = _int32(mixed ^ t)
        return ((t ^ ((t & 0xFFFFFFFF) >> 14)) & 0xFFFFFFFF) / 4294967296.0


def mulberry32(seed: int) -> Mulberry32:
    """Return the seeded PRNG used by the TS engine (``Math.imul``)."""
    return Mulberry32(seed)


class TetrisEngine:
    """Faithful port of ``TetrisEngine``; see engine.ts for the reference."""

    __slots__ = (
        "cells",
        "piece_kind",
        "piece_rot",
        "piece_x",
        "piece_y",
        "next_kind",
        "score",
        "lines",
        "pieces",
        "ticks",
        "games",
        "_bag",
        "_inputs",
        "_head",
        "_rng",
    )

    def __init__(self, rng: Callable[[], float] | None = None) -> None:
        self.cells = bytearray(COLS * ROWS)
        self.piece_kind = PieceKind.I
        self.piece_rot = 0
        self.piece_x = 0
        self.piece_y = 0
        self.next_kind = PieceKind.I
        self.score = 0
        self.lines = 0
        self.pieces = 0
        self.ticks = 0
        self.games = 1
        self._bag: list[int] = []
        self._inputs: list[int] = []
        self._head = 0
        self._rng = rng if rng is not None else mulberry32(0)
        self._spawn()

    @property
    def queue_depth(self) -> int:
        """Number of queued inputs not yet consumed by ticks."""
        return len(self._inputs) - self._head

    def queue_input(self, action: int) -> None:
        """Append an action to the input queue (one consumed per tick)."""
        self._inputs.append(action)

    def _take_input(self) -> int:
        if self._head < len(self._inputs):
            action = self._inputs[self._head]
            self._head += 1
        else:
            action = Action.NONE
        if self._head == len(self._inputs):
            self._inputs.clear()
            self._head = 0
        elif self._head >= 64:
            del self._inputs[: self._head]
            self._head = 0
        return action

    def _refill_bag(self) -> None:
        if self._bag:
            return
        self._bag = [0, 1, 2, 3, 4, 5, 6]
        for i in range(len(self._bag) - 1, 0, -1):
            j = int(self._rng() * (i + 1))
            self._bag[i], self._bag[j] = self._bag[j], self._bag[i]

    def _spawn(self) -> None:
        self._refill_bag()
        kind = self._bag.pop()
        size = _BASE[kind][0]
        self.piece_kind = kind
        self.piece_rot = 0
        self.piece_x = (COLS - size) >> 1
        self.piece_y = 0
        self._refill_bag()
        self.next_kind = self._bag[-1]
        if self.collides(kind, 0, self.piece_x, self.piece_y):
            self._reset()

    def _reset(self) -> None:
        self.cells = bytearray(COLS * ROWS)
        self._inputs.clear()
        self._head = 0
        self.games += 1
        size = _BASE[self.piece_kind][0]
        self.piece_rot = 0
        self.piece_x = (COLS - size) >> 1
        self.piece_y = 0

    def collides(self, kind: int, rot: int, px: int, py: int) -> bool:
        """Check whether a piece placement is blocked or out of bounds."""
        for cx, cy in SHAPES[kind][rot]:
            x = px + cx
            y = py + cy
            if x < 0 or x >= COLS or y >= ROWS:
                return True
            if y >= 0 and self.cells[y * COLS + x] != 0:
                return True
        return False

    def _try_move(self, dx: int, dy: int) -> bool:
        if self.collides(
            self.piece_kind,
            self.piece_rot,
            self.piece_x + dx,
            self.piece_y + dy,
        ):
            return False
        self.piece_x += dx
        self.piece_y += dy
        return True

    def _try_rotate(self, direction: int) -> bool:
        next_rot = (self.piece_rot + direction + 4) % 4
        for dx in KICKS:
            if not self.collides(
                self.piece_kind, next_rot, self.piece_x + dx, self.piece_y
            ):
                self.piece_rot = next_rot
                self.piece_x += dx
                return True
        return False

    def drop_distance(self) -> int:
        """Return how many cells the active piece can still descend."""
        dy = 0
        while not self.collides(
            self.piece_kind,
            self.piece_rot,
            self.piece_x,
            self.piece_y + dy + 1,
        ):
            dy += 1
        return dy

    def _lock(self) -> None:
        for cx, cy in SHAPES[self.piece_kind][self.piece_rot]:
            by = self.piece_y + cy
            if by >= 0:
                self.cells[by * COLS + self.piece_x + cx] = self.piece_kind + 1
        self.pieces += 1
        self._clear_lines()
        self._spawn()

    def _clear_lines(self) -> None:
        write = ROWS - 1
        cleared = 0
        for read in range(ROWS - 1, -1, -1):
            full = all(self.cells[read * COLS + x] != 0 for x in range(COLS))
            if full:
                cleared += 1
                continue
            if write != read:
                self.cells[write * COLS : (write + 1) * COLS] = self.cells[
                    read * COLS : (read + 1) * COLS
                ]
            write -= 1
        if cleared > 0:
            self.cells[: (write + 1) * COLS] = bytes((write + 1) * COLS)
            self.lines += cleared
            self.score += LINE_SCORES[cleared]

    def tick(self) -> None:
        """Consume one queued input, apply it, and advance gravity."""
        action = self._take_input()
        settled = False
        if action == Action.LEFT:
            self._try_move(-1, 0)
        elif action == Action.RIGHT:
            self._try_move(1, 0)
        elif action == Action.ROTATE_CW:
            self._try_rotate(1)
        elif action == Action.ROTATE_CCW:
            self._try_rotate(-1)
        elif action == Action.SOFT_DROP:
            if self._try_move(0, 1):
                self.score += 1
            else:
                self._lock()
                settled = True
        elif action == Action.HARD_DROP:
            dist = self.drop_distance()
            self.piece_y += dist
            self.score += dist * 2
            self._lock()
            settled = True
        if not settled and not self._try_move(0, 1):
            self._lock()
        self.ticks += 1

    def snapshot(self) -> dict[str, Any]:
        """Capture the complete state needed to resume this exact point.

        Covers the board, piece pose, queue, counters, bag contents and
        the rng position. The input queue is transient and cleared.
        """
        rng_state = (
            self._rng.state if isinstance(self._rng, Mulberry32) else None
        )
        return {
            "cells": bytes(self.cells),
            "piece_kind": self.piece_kind,
            "piece_rot": self.piece_rot,
            "piece_x": self.piece_x,
            "piece_y": self.piece_y,
            "next_kind": self.next_kind,
            "score": self.score,
            "lines": self.lines,
            "pieces": self.pieces,
            "ticks": self.ticks,
            "games": self.games,
            "bag": list(self._bag),
            "rng_state": rng_state,
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Load a ``snapshot()`` back into this engine.

        Restores the board, piece pose, queue, counters, bag and rng
        position so subsequent ticks continue the recorded point
        exactly; the pending input queue is cleared.
        """
        self.cells = bytearray(state["cells"])
        self.piece_kind = state["piece_kind"]
        self.piece_rot = state["piece_rot"]
        self.piece_x = state["piece_x"]
        self.piece_y = state["piece_y"]
        self.next_kind = state["next_kind"]
        self.score = state["score"]
        self.lines = state["lines"]
        self.pieces = state["pieces"]
        self.ticks = state["ticks"]
        self.games = state["games"]
        self._bag = list(state["bag"])
        self._inputs.clear()
        self._head = 0
        if state["rng_state"] is not None and isinstance(self._rng, Mulberry32):
            self._rng.state = state["rng_state"]
