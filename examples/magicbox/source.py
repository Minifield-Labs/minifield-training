"""Verified Parquet intake and disk-backed, deterministic training epochs."""

from collections.abc import Callable, Iterator, Sequence
import functools
import hashlib
import json
from pathlib import Path
from typing import cast

import datasets  # type: ignore[import-untyped]

from examples.magicbox import data as magicbox
from examples.magicbox import tokenizer
from minifield_training.artifacts import files
from minifield_training.batching import packing
from minifield_training.batching import schema_fields as batching
from minifield_training.batching import stream
from minifield_training.core import json_io
from minifield_training.datasets import fields


class Corpus:
    """Verify immutable shards before Arrow caching; retain upstream splits."""

    def __init__(
        self, directory: Path, cache: Path, *, allow_sample: bool = False
    ):
        self.directory, self.cache = directory, cache
        manifest_path = directory / "manifest.json"
        self.identity = json_io.digest_file(manifest_path)
        self.manifest = json_io.object_map(
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
        self.shards = [json_io.object_map(value) for value in raw_shards]
        self._splits: dict[str, datasets.Dataset] = {}
        files.verify(
            directory,
            [
                files.FileEntry(
                    str(shard["path"]),
                    str(shard["sha256"]),
                    cast(int, shard["bytes"]),
                )
                for shard in self.shards
            ],
        )

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

    def compile(self, raw: object, split: str) -> fields.Record:
        """Recheck token spans with the saved tokenizer."""
        value = json_io.object_map(raw)
        if (
            value.get("format") != magicbox.FORMAT
            or value.get("split") != split
        ):
            raise ValueError("Record format or split mismatch")
        request = json_io.object_map(json.loads(str(value["request_json"])))
        targets = json_io.object_map(json.loads(str(value["targets_json"])))
        questions = json_io.object_map(request["questions"])
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
        encoded = json_io.object_map(json.loads(str(value["encoding_json"])))
        spans = json_io.object_map(encoded["token_spans"])
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

    def packed_sequences(
        self, splits: Sequence[str], schema_tokens: int
    ) -> int:
        """Return the most packed schema rows any record in ``splits`` needs.

        Row text is batch-tokenized with the saved tokenizer. Sources, spans,
        and targets are checked later, when training compiles each record.
        """
        required = 0
        for split in splits:
            columns = self.split(split).select_columns(
                ["request_json", "targets_json"]
            )
            for chunk in columns.iter(batch_size=1024):
                rows = [
                    magicbox.labeled_schema_rows(
                        json.loads(request), json.loads(targets)
                    )
                    for request, targets in zip(
                        chunk["request_json"],
                        chunk["targets_json"],
                        strict=True,
                    )
                ]
                encoded = iter(
                    self.tokenizer.tokenizer.encode_batch(
                        [text for texts in rows for text in texts],
                        add_special_tokens=True,
                    )
                )
                for texts in rows:
                    lengths = [len(next(encoded).ids) for _ in texts]
                    required = max(
                        required, packing.rows_required(lengths, schema_tokens)
                    )
        return required

    def records(self, split: str, limit: int) -> Iterator[fields.Record]:
        """Apply the experiment's fixed held-out sampling policy."""
        dataset = self.split(split).shuffle(seed=1729, keep_in_memory=False)
        count = min(limit, len(dataset)) if limit else len(dataset)
        for index in range(count):
            yield self.compile(dataset[index], split)


def training_stream(
    corpus: Corpus,
    batches: batching.SchemaBatchStrategy,
    seed: int,
    epochs: int,
) -> stream.EpochStream[object, fields.Record]:
    """Bind the published Arrow order and adapter to shared cursor replay."""
    data = corpus.split("train")

    def read_epoch(epoch: int) -> Callable[[int, int], Sequence[object]]:
        shuffled = data.shuffle(seed=seed + epoch, keep_in_memory=False)
        return lambda start, stop: [
            shuffled[position] for position in range(start, stop)
        ]

    return stream.EpochStream(
        record_count=len(data),
        capacity=batches.shape.capacity,
        epochs=epochs,
        read_epoch=read_epoch,
        compile_record=functools.partial(corpus.compile, split="train"),
        pack=lambda records, update: batches.pack(
            records, seed=seed, update=update
        ),
    )
