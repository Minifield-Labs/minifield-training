"""Verified Parquet intake and disk-backed, deterministic training epochs."""

from collections.abc import Iterator
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time
from typing import cast

import datasets  # type: ignore[import-untyped]

from examples.magicbox import tokenizer
from minifield_training.batching import contracts
from minifield_training.batching import magicbox as batching
from minifield_training.core import json_io
from minifield_training.datasets import magicbox


class Corpus:
    """Verify immutable shards before Arrow caching; retain upstream splits."""

    def __init__(
        self, directory: Path, cache: Path, *, allow_sample: bool = False
    ):
        self.directory, self.cache = directory, cache
        manifest_path = directory / "manifest.json"
        self.identity = json_io.digest_file(manifest_path)
        self.manifest = magicbox.object_map(
            json.loads(manifest_path.read_text())
        )
        if (
            self.manifest.get("format") != magicbox.FORMAT
            or self.manifest.get("complete") is not True
        ):
            raise ValueError("Expected a completed MagicBox dataset")
        if not allow_sample and self.manifest.get("mode") != "full":
            raise ValueError("Full training requires a full dataset build")
        self.tokenizer = tokenizer.Adapter(directory / "tokenizer")
        if self.manifest.get("tokenizer") != self.tokenizer.contract:
            raise ValueError("Manifest and tokenizer contract differ")
        identity = json.loads((directory / "identity.json").read_text())
        if hashlib.sha256(
            json_io.canonical(identity).encode()
        ).hexdigest() != self.manifest.get("identity"):
            raise ValueError("Dataset identity digest mismatch")
        raw_shards = self.manifest.get("shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError("Dataset has no shards")
        self.shards = [magicbox.object_map(value) for value in raw_shards]
        self._splits: dict[str, datasets.Dataset] = {}
        seen: set[Path] = set()
        for shard in self.shards:
            path = (directory / str(shard["path"])).resolve()
            if not path.is_relative_to(directory.resolve()) or path in seen:
                raise ValueError("Invalid or duplicate shard path")
            seen.add(path)
            if (
                path.stat().st_size != shard["bytes"]
                or json_io.digest_file(path) != shard["sha256"]
            ):
                raise ValueError(f"Shard checksum mismatch: {path.name}")

    def split(self, name: str) -> datasets.Dataset:
        """Memory-map a named split's cached Arrow files."""
        if name not in self._splits:
            shards = [shard for shard in self.shards if shard["split"] == name]
            if not shards:
                raise ValueError(f"Dataset has no {name} split")
            data = datasets.Dataset.from_parquet(
                [str(self.directory / str(shard["path"])) for shard in shards],
                cache_dir=str(self.cache),
                keep_in_memory=False,
            )
            if len(data) != sum(cast(int, shard["rows"]) for shard in shards):
                raise ValueError("Shard row count mismatch")
            self._splits[name] = data
        return self._splits[name]

    def compile(self, raw: object, split: str) -> magicbox.Record:
        """Recheck token spans with the saved tokenizer."""
        value = magicbox.object_map(raw)
        if (
            value.get("format") != magicbox.FORMAT
            or value.get("split") != split
        ):
            raise ValueError("Record format or split mismatch")
        request = magicbox.object_map(json.loads(str(value["request_json"])))
        targets = magicbox.object_map(json.loads(str(value["targets_json"])))
        questions = magicbox.object_map(request["questions"])
        if not set(targets) <= set(questions):
            raise ValueError("Unknown supervision field")
        # Unlabeled independent rows cannot contribute a training gradient.
        # The builder may retain over-length questions after dropping labels.
        request["questions"] = {
            key: question
            for key, question in questions.items()
            if key in targets
        }
        record = magicbox.compile_record(
            str(value["id"]), request, targets, self.tokenizer.encode
        )
        encoded = magicbox.object_map(json.loads(str(value["encoding_json"])))
        spans = magicbox.object_map(encoded["token_spans"])
        if len(record.source.ids) != encoded["source_tokens"]:
            raise ValueError("Source token count changed")
        for field in record.fields:
            if field.span is not None and list(field.span) != spans.get(
                field.key
            ):
                raise ValueError("Gold token span changed")
        if not any(field.supervised for field in record.fields):
            raise ValueError("no_supervision")
        return record


def bucket(
    records: list[magicbox.Record], maximum: batching.Shape
) -> batching.Shape:
    """Choose token and row buckets within the admission caps."""

    def axis(required: int, limit: int, minimum: int) -> int:
        if required > limit:
            raise ValueError(
                f"Record requires {required} positions; limit is {limit}"
            )
        return min(limit, max(minimum, 1 << (required - 1).bit_length()))

    return dataclasses.replace(
        maximum,
        source_tokens=axis(
            max(len(record.source.ids) for record in records),
            maximum.source_tokens,
            128,
        ),
        schema_tokens=axis(
            max(
                len(row)
                for record in records
                for field in record.fields
                for row in field.rows
            ),
            maximum.schema_tokens,
            128,
        ),
        schema_rows=axis(
            max(
                sum(len(field.rows) for field in record.fields)
                for record in records
            ),
            maximum.schema_rows,
            4,
        ),
    )


@dataclasses.dataclass
class Stream:
    """Replay shuffle order and dropout keys from the update cursor."""

    corpus: Corpus
    shape: batching.Shape
    seed: int
    epochs: int

    @property
    def updates_per_epoch(self) -> int:
        """Account for a padded final update without dropping any examples."""
        return math.ceil(len(self.corpus.split("train")) / self.shape.capacity)

    def __call__(
        self, start_update: int, deadline: float | None, /
    ) -> Iterator[contracts.PhysicalUpdate]:
        """Resume at the next unread logical update, across epoch boundaries."""
        epoch, offset = divmod(start_update, self.updates_per_epoch)
        data = self.corpus.split("train")
        while epoch < self.epochs:
            shuffled = data.shuffle(
                seed=self.seed + epoch, keep_in_memory=False
            )
            for index in range(offset, self.updates_per_epoch):
                if deadline is not None and time.monotonic() >= deadline:
                    return
                start = index * self.shape.capacity
                records = [
                    self.corpus.compile(shuffled[position], "train")
                    for position in range(
                        start, min(start + self.shape.capacity, len(data))
                    )
                ]
                update = epoch * self.updates_per_epoch + index
                yield batching.build(
                    records,
                    bucket(records, self.shape),
                    seed=self.seed,
                    update=update,
                )
            epoch, offset = epoch + 1, 0
