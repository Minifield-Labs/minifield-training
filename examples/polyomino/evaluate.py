"""Play complete polyomino games using only learned classifier actions."""

import argparse
from collections.abc import Callable
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tokenizers import Tokenizer  # type: ignore[import-untyped]

from examples.polyomino import engine
from examples.polyomino import serialize
from examples.polyomino import source
from minifield_training.checkpoints import training_state
from minifield_training.models.lfm2_5 import model
from minifield_training.optimizers import state as optimizer_state
from minifield_training.strategies import classification

ALLOWED = (True, True, True, True, True, True, True, False)


def make_evaluator(
    cfg: model.Config,
    tokenizer: Tokenizer,
    *,
    sequence_length: int,
    games: int,
    max_ticks: int,
    seed: int,
    replay_dir: Path | None = None,
) -> Callable[[optimizer_state.State, int], dict[str, float]]:
    """Compile greedy learned-policy scoring once for periodic game rollouts."""
    if min(sequence_length, games, max_ticks) < 1:
        raise ValueError("Invalid evaluation bounds")

    @jax.jit
    def action_fn(
        params: dict[str, jax.Array], ids: jax.Array, mask: jax.Array
    ) -> jax.Array:
        """Choose an engine action from classifier logits alone."""
        return classification.predict(
            params, ids, mask, cfg, ALLOWED, dtype=jnp.bfloat16
        )

    def evaluate(
        full_state: optimizer_state.State, step: int
    ) -> dict[str, float]:
        """Run until each seeded game dies or hits its explicit tick cap."""
        states = [engine.Game(seed + index) for index in range(games)]
        active = np.ones(games, dtype=np.bool_)
        frames: list[str] = []
        for tick in range(max_ticks):
            if not active.any():
                break
            ids = np.zeros((games, sequence_length), dtype=np.int32)
            masks = np.zeros_like(ids)
            for index, game in enumerate(states):
                if not active[index]:
                    ids[index, 0] = 1
                    masks[index, 0] = 1
                    continue
                row = serialize.encode(game, tokenizer)
                if len(row) > sequence_length:
                    raise ValueError(
                        f"Evaluation prompt exceeds {sequence_length} tokens"
                    )
                ids[index, : len(row)] = row
                masks[index, : len(row)] = 1
            actions = np.asarray(
                action_fn(
                    full_state["params"], jnp.asarray(ids), jnp.asarray(masks)
                )
            )
            for index, game in enumerate(states):
                if not active[index]:
                    continue
                action = int(actions[index])
                if index == 0:
                    frames.append(
                        f"tick={tick} action={action} "
                        f"lines={game.lines} pieces={game.pieces}\n"
                        + "\n".join(serialize.board_rows(game))
                    )
                game.step(action)
                if game.game_over:
                    active[index] = False
        if replay_dir is not None:
            replay_dir.mkdir(parents=True, exist_ok=True)
            replay = replay_dir / f"step-{step:08d}-game-0.txt"
            replay.write_text("\n\n".join(frames) + "\n", encoding="utf-8")
        return {
            "lines_per_game": float(np.mean([game.lines for game in states])),
            "pieces_per_game": float(np.mean([game.pieces for game in states])),
            "survival_ticks": float(np.mean([game.tick for game in states])),
            "completed_games": float(games - int(active.sum())),
        }

    return evaluate


def main() -> None:
    """Replay a saved policy through real games and write readable frames."""
    # The CLI uses train's validation, while train imports this evaluator.
    # pylint: disable-next=import-outside-toplevel
    from examples.polyomino import train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--max-ticks", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1 << 31)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--data-seed", type=int, default=17)
    parser.add_argument("--head-seed", type=int, default=6)
    parser.add_argument("--no-remat", action="store_true")
    parser.add_argument("--fuse-accumulation", action="store_true")
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    args = parser.parse_args()
    cfg, tokenizer = train.load_model_metadata(args.model_dir)
    inventory = classification.parameter_inventory(cfg, ALLOWED)
    optimizer = train.optimizer_config(args.learning_rate)
    source_id = train.source_identity(
        args, cfg, sequence_length=args.sequence_length
    )
    full_state, cursor = training_state.load(
        args.checkpoint,
        inventory,
        optimizer_id=optimizer.implementation_identity,
        run_id=args.run_id,
        data_sha256=source.data_identity(),
        source_id=source_id,
    )
    callback = make_evaluator(
        cfg,
        tokenizer,
        sequence_length=args.sequence_length,
        games=args.games,
        max_ticks=args.max_ticks,
        seed=args.seed,
        replay_dir=args.replay_dir,
    )
    print(json.dumps(callback(full_state, cursor.next_batch)), flush=True)


if __name__ == "__main__":
    main()
