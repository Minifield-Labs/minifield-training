"""Generate complete Base-tokenized expert decisions from seeded games."""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer  # type: ignore[import-untyped]

from examples.tetris import engine
from examples.tetris import expert
from examples.tetris import serialize
from minifield_training.core import json_io
from minifield_training.strategies import pretrained


def generator_identity() -> str:
    """Bind generated data to the complete engine, expert, and prompt source."""
    digest = hashlib.sha256()
    for name in ("engine.py", "expert.py", "serialize.py", "prepare.py"):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def generate(
    model_dir: Path,
    output: Path,
    *,
    games: int,
    max_ticks: int,
    seed: int,
    sequence_length: int,
) -> dict[str, object]:
    """Write one record per expert tick, preserving game grouping and seeds."""
    if min(games, max_ticks, sequence_length) < 1 or seed < 0:
        raise ValueError("Invalid expert generation settings")
    tokenizer_path = model_dir / "tokenizer.json"
    if (
        json_io.digest_file(tokenizer_path) != pretrained.BASE.tokenizer_sha256
        or json_io.digest_file(model_dir / "config.json")
        != pretrained.BASE.config_sha256
    ):
        raise ValueError("Expected the pinned Base config and tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos = tokenizer.token_to_id("<|startoftext|>")
    if bos is None:
        raise ValueError("Pinned tokenizer has no start-of-text token")
    rng = np.random.default_rng(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists() or output.exists():
        raise FileExistsError(output)
    samples = 0
    lines = 0
    pieces = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for game_index in range(games):
            state = engine.TetrisEngine(engine.mulberry32(seed + game_index))
            weights = (
                9.0 + rng.uniform(-1.5, 1.5),
                0.9 + rng.uniform(-0.2, 0.2),
                0.25 + rng.uniform(-0.05, 0.05),
                0.35 + rng.uniform(-0.08, 0.08),
            )
            initial_games = state.games
            initial_ticks = state.ticks
            for tick in range(max_ticks):
                if state.games != initial_games:
                    break
                prompt = serialize.prompt_text(state)
                if not prompt.startswith(
                    "<|startoftext|><|im_start|>system\n"
                ) or not prompt.endswith("<|im_start|>assistant\naction:"):
                    raise ValueError("Decision framing changed")
                ids = tokenizer.encode(prompt, add_special_tokens=False).ids
                if not ids or ids[0] != bos or ids.count(bos) != 1:
                    raise ValueError(
                        "Decision prompt must have exactly one BOS"
                    )
                if len(ids) > sequence_length:
                    raise ValueError(
                        f"Observation exceeds {sequence_length} tokens: "
                        f"game={game_index} tick={tick} length={len(ids)}"
                    )
                action = expert.expert_action(state, weights)
                record = {
                    "id": f"game-{game_index:06d}-tick-{tick:06d}",
                    "group_id": f"game-{game_index:06d}",
                    "input_ids": ids,
                    "label": action,
                }
                stream.write(json_io.canonical(record) + "\n")
                samples += 1
                state.queue_input(action)
                state.tick()
            pieces += state.pieces
            lines += state.lines
            print(
                f"expert game={game_index + 1}/{games} "
                f"ticks={state.ticks - initial_ticks} lines={state.lines}",
                flush=True,
            )
    os.replace(temporary, output)
    manifest: dict[str, object] = {
        "format": "tetris.expert-decisions/1",
        "model_id": pretrained.BASE.model_id,
        "revision": pretrained.BASE.revision,
        "tokenizer_sha256": pretrained.BASE.tokenizer_sha256,
        "data_sha256": json_io.digest_file(output),
        "games": games,
        "max_ticks": max_ticks,
        "sequence_length": sequence_length,
        "seed": seed,
        "samples": samples,
        "lines": lines,
        "pieces": pieces,
        "generator_sha256": generator_identity(),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json_io.canonical(manifest) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    """Generate bounded, full-length expert decision records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=80)
    parser.add_argument("--max-ticks", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sequence-length", type=int, default=512)
    args = parser.parse_args()
    print(
        json_io.canonical(
            generate(
                args.model_dir,
                args.output,
                games=args.games,
                max_ticks=args.max_ticks,
                seed=args.seed,
                sequence_length=args.sequence_length,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
