"""Rules for replaying 11-column polyomino decision states."""

from collections.abc import Mapping, Sequence
from typing import cast

type Cells = tuple[tuple[int, int], ...]
type Board = Sequence[int]

WIDTH = 11
HEIGHT = 22
FULL_ROW = (1 << WIDTH) - 1

ACTION_NAMES = (
    "NONE",
    "LEFT",
    "RIGHT",
    "ROTATE_CW",
    "ROTATE_CCW",
    "SOFT_DROP",
    "HARD_DROP",
    "PAD",
)
NONE, LEFT, RIGHT, ROTATE_CW, ROTATE_CCW, SOFT_DROP, HARD_DROP, PAD = range(8)
BASE_SHAPES: dict[str, Cells] = {
    "I4": ((0, 0), (1, 0), (2, 0), (3, 0)),
    "O4": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "T4": ((0, 0), (1, 0), (2, 0), (1, 1)),
    "S4": ((1, 0), (2, 0), (0, 1), (1, 1)),
    "Z4": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "J4": ((0, 0), (0, 1), (1, 1), (2, 1)),
    "L4": ((2, 0), (0, 1), (1, 1), (2, 1)),
    "I3": ((0, 0), (1, 0), (2, 0)),
    "L3": ((0, 0), (0, 1), (1, 1)),
    "I5": ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
    "P5": ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2)),
}
PIECES = tuple(BASE_SHAPES)


def rotations(cells: Cells) -> tuple[Cells, ...]:
    """Normalize the four orientations of a polyomino."""
    result: list[Cells] = []
    current = tuple(cells)
    for _ in range(4):
        min_x = min(x for x, _ in current)
        min_y = min(y for _, y in current)
        normalized = tuple(sorted((x - min_x, y - min_y) for x, y in current))
        result.append(normalized)
        current = tuple((-y, x) for x, y in normalized)
    return tuple(result)


SHAPES = {name: rotations(cells) for name, cells in BASE_SHAPES.items()}


class Rng32:
    """Small PRNG whose complete state fits in one JSON integer."""

    def __init__(self, state: int) -> None:
        """Require a nonzero 32-bit seed."""
        if not 0 < state < 2**32:
            raise ValueError("RNG state must be a nonzero uint32")
        self.state = state

    def next_u32(self) -> int:
        """Advance the complete xorshift32 state."""
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state


def collides(board: Board, kind: str, rotation: int, px: int, py: int) -> bool:
    """Check a placement against the board."""
    for dx, dy in SHAPES[kind][rotation]:
        x, y = px + dx, py + dy
        if x < 0 or x >= WIDTH or y >= HEIGHT:
            return True
        if y >= 0 and board[y] & (1 << x):
            return True
    return False


def drop_distance(
    board: Board, kind: str, rotation: int, x: int, y: int
) -> int:
    """Count rows until the piece collides."""
    distance = 0
    while not collides(board, kind, rotation, x, y + distance + 1):
        distance += 1
    return distance


def locked_board(
    board: Board, kind: str, rotation: int, x: int, y: int
) -> tuple[tuple[int, ...], int] | None:
    """Settle a piece and clear complete rows."""
    result = list(board)
    for dx, dy in SHAPES[kind][rotation]:
        cell_y = y + dy
        if cell_y < 0:
            return None
        result[cell_y] |= 1 << (x + dx)
    kept = [row for row in result if row != FULL_ROW]
    cleared = HEIGHT - len(kept)
    return tuple([0] * cleared + kept), cleared


def board_rows(board: Board) -> list[str]:
    """Render settled cells as compact strings."""
    return [
        "".join("#" if row & (1 << x) else "." for x in range(WIDTH))
        for row in board
    ]


def parse_board(rows: Sequence[str]) -> list[int]:
    """Decode compact rows into board bitmasks."""
    if len(rows) != HEIGHT or any(
        len(row) != WIDTH or set(row) - {".", "#"} for row in rows
    ):
        raise ValueError("Invalid saved board")
    return [
        sum((1 << x) for x, cell in enumerate(row) if cell == "#")
        for row in rows
    ]


class Game:
    """Replay a frozen decision or play a new game."""

    def __init__(self, seed: int) -> None:
        """Start from a seed and spawn the first piece."""
        self.board = [0] * HEIGHT
        self.rng = Rng32(seed)
        self.bag: list[str] = []
        self.score = 0
        self.lines = 0
        self.pieces = 0
        self.tick = 0
        self.game_over = False
        self._spawn()

    def _refill_bag(self) -> None:
        """Shuffle the next bag."""
        if self.bag:
            return
        self.bag = list(PIECES)
        for index in range(len(self.bag) - 1, 0, -1):
            swap = self.rng.next_u32() % (index + 1)
            self.bag[index], self.bag[swap] = self.bag[swap], self.bag[index]

    def _spawn(self) -> None:
        """Draw and spawn the next piece."""
        self._refill_bag()
        self.kind = self.bag.pop()
        self._refill_bag()  # bag[-1] is the next piece in every saved state
        self.rotation = 0
        shape_width = 1 + max(x for x, _ in SHAPES[self.kind][0])
        self.x = (WIDTH - shape_width) // 2
        self.y = 0
        self.game_over = collides(
            self.board, self.kind, self.rotation, self.x, self.y
        )

    def snapshot(self) -> dict[str, object]:
        """Save every variable needed to resume."""
        if self.game_over:
            raise ValueError("Terminal games have no decision state")
        return {
            "board": board_rows(self.board),
            "active": {
                "piece": self.kind,
                "rotation": self.rotation,
                "x": self.x,
                "y": self.y,
            },
            "bag": list(self.bag),
            "rng_state": self.rng.state,
            "score": self.score,
            "lines": self.lines,
            "pieces": self.pieces,
            "tick": self.tick,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "Game":
        """Resume a saved game state."""
        game = cls.__new__(cls)
        game.board = parse_board(cast(Sequence[str], state["board"]))
        active = cast(Mapping[str, object], state["active"])
        game.kind = cast(str, active["piece"])
        game.rotation = cast(int, active["rotation"])
        game.x = cast(int, active["x"])
        game.y = cast(int, active["y"])
        game.bag = list(cast(Sequence[str], state["bag"]))
        game.rng = Rng32(cast(int, state["rng_state"]))
        game.score = cast(int, state["score"])
        game.lines = cast(int, state["lines"])
        game.pieces = cast(int, state["pieces"])
        game.tick = cast(int, state["tick"])
        game.game_over = False
        if (
            game.kind not in SHAPES
            or not 0 <= game.rotation < 4
            or not game.bag
            or any(piece not in SHAPES for piece in game.bag)
            or collides(game.board, game.kind, game.rotation, game.x, game.y)
        ):
            raise ValueError("Invalid saved active piece or bag")
        return game

    def _lock(self) -> None:
        """Settle the active piece and clear rows."""
        result = locked_board(
            self.board, self.kind, self.rotation, self.x, self.y
        )
        if result is None:
            self.game_over = True
            return
        locked, cleared = result
        self.board = list(locked)
        self.lines += cleared
        self.score += 100 * cleared * cleared
        self.pieces += 1
        self._spawn()

    def step(self, action: int) -> None:
        """Apply one action and one gravity tick."""
        if self.game_over or action not in range(PAD):
            raise ValueError("Invalid action or terminal game")
        settled = False
        if action in (LEFT, RIGHT):
            next_x = self.x + (-1 if action == LEFT else 1)
            if not collides(
                self.board, self.kind, self.rotation, next_x, self.y
            ):
                self.x = next_x
        elif action in (ROTATE_CW, ROTATE_CCW):
            next_rotation = (
                self.rotation + (1 if action == ROTATE_CW else -1)
            ) % 4
            for shift in (0, -1, 1, -2, 2):
                if not collides(
                    self.board, self.kind, next_rotation, self.x + shift, self.y
                ):
                    self.rotation = next_rotation
                    self.x += shift
                    break
        elif action == SOFT_DROP:
            if not collides(
                self.board, self.kind, self.rotation, self.x, self.y + 1
            ):
                self.y += 1
                self.score += 1
            else:
                self._lock()
                settled = True
        elif action == HARD_DROP:
            distance = drop_distance(
                self.board, self.kind, self.rotation, self.x, self.y
            )
            self.y += distance
            self.score += 2 * distance
            self._lock()
            settled = True
        if not settled:
            if collides(
                self.board, self.kind, self.rotation, self.x, self.y + 1
            ):
                self._lock()
            else:
                self.y += 1
        self.tick += 1
