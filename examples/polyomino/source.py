"""Read immutable Hugging Face decision rows into physical updates."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any, cast

from datasets import Dataset  # type: ignore[import-untyped]
from datasets import load_dataset
import numpy as np
from tokenizers import Tokenizer  # type: ignore[import-untyped]

from examples.polyomino import engine
from examples.polyomino import serialize
from minifield_training.batching import classification as batching
from minifield_training.core import json_io
from minifield_training.datasets.labeled import LabeledSequence
from minifield_training.engine.classification_run import RunConfig
from minifield_training.strategies import pretrained

DATASET_ID = "protodotdesign/polyomino-decisions-v1"
DATASET_REVISION = "d1a79caa4eaeba129630f858f9c2de7d6de7533a"
DATASET_ROWS = 4_737_585


def load_decisions(cache_dir: Path) -> Dataset:
    """Cache the pinned public Parquet split as a disk-backed Arrow table."""
    if not cache_dir.is_absolute():
        raise ValueError("Dataset cache must be an absolute path")
    data = load_dataset(
        DATASET_ID,
        split="train",
        revision=DATASET_REVISION,
        cache_dir=str(cache_dir),
    )
    if not isinstance(data, Dataset) or len(data) != DATASET_ROWS:
        raise ValueError("Published dataset row count changed")
    return data


def data_identity() -> str:
    """Bind checkpoints to immutable rows and the prompt implementation."""
    settings = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "rows": DATASET_ROWS,
        "tokenizer_sha256": pretrained.BASE.tokenizer_sha256,
        "prompt_sha256": json_io.digest_file(Path(serialize.__file__)),
        "rules_sha256": json_io.digest_file(Path(engine.__file__)),
    }
    return hashlib.sha256(json_io.canonical(settings).encode()).hexdigest()


def _label(one_hot: object) -> int:
    """Decode the published 8-slot vector, rejecting malformed labels."""
    if (
        not isinstance(one_hot, list)
        or len(one_hot) != 8
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value not in (0, 1)
            for value in one_hot
        )
        or sum(one_hot) != 1
    ):
        raise ValueError("Invalid one-hot expert action")
    label = one_hot.index(1)
    if label == engine.PAD:
        raise ValueError("PAD cannot label a decision")
    return label


def sequence_from_row(
    row: Mapping[str, object], tokenizer: Tokenizer
) -> LabeledSequence:
    """Derive the observation and label from one saved game decision."""
    state = cast(Mapping[str, object], row["state"])
    game = engine.Game.from_state(state)
    ids = tuple(serialize.encode(game, tokenizer))
    game_id = row["game_id"]
    if not isinstance(game_id, int) or isinstance(game_id, bool) or game_id < 0:
        raise ValueError("Invalid game ID")
    return LabeledSequence(
        f"game-{game_id}-tick-{game.tick}",
        f"game-{game_id}",
        ids,
        _label(row["expert_action"]),
    )


@dataclass(frozen=True)
class HFDatasetBatchSource:
    """Visit each published decision once in a seeded global order."""

    data: Dataset
    tokenizer: Tokenizer
    config: RunConfig

    def __call__(
        self, start_update: int, deadline: float | None = None
    ) -> Iterator[batching.PhysicalUpdate]:
        """Resume at an update boundary without reusing earlier rows."""
        if start_update < 0:
            raise ValueError("Negative update cursor")
        capacity = self.config.microbatches * self.config.rows_per_microbatch
        order = np.random.default_rng(self.config.seed).permutation(
            len(self.data)
        )
        total = math.ceil(len(order) / capacity)
        stop = min(
            total,
            start_update + self.config.max_steps
            if self.config.max_steps is not None
            else total,
        )
        for update_index in range(start_update, stop):
            if deadline is not None and time.monotonic() >= deadline:
                return
            selected = [
                int(index)
                for index in order[
                    update_index * capacity : (update_index + 1) * capacity
                ]
            ]
            columns = cast(dict[str, list[Any]], self.data[selected])
            examples = [
                sequence_from_row(
                    {key: values[offset] for key, values in columns.items()},
                    self.tokenizer,
                )
                for offset in range(len(selected))
            ]
            yield from batching.iter_updates(
                examples,
                microbatches=self.config.microbatches,
                rows_per_microbatch=self.config.rows_per_microbatch,
                sequence_length=self.config.sequence_length,
                pad_token_id=self.config.pad_token_id,
                vocab_size=self.config.vocab_size,
                allowed_classes=self.config.allowed_classes,
                padding_label=self.config.padding_label,
                seed=self.config.seed,
            )
