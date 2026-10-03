"""Verified Parquet intake and disk-backed, deterministic training epochs."""

from collections.abc import Callable, Iterator, Mapping, Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Protocol, cast

import datasets  # type: ignore[import-untyped]

from examples.magicbox import data as magicbox
from examples.magicbox import tokenizer
from minifield_training.artifacts import files
from minifield_training.batching import contracts
from minifield_training.batching import packing
from minifield_training.batching import pointer as pointer_batching
from minifield_training.batching import stream
from minifield_training.core import json_io
from minifield_training.datasets import fields
from minifield_training.datasets import pointer


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

    def _admitted(
        self, raw: object, split: str
    ) -> tuple[str, dict[str, object], dict[str, object], dict[str, object]]:
        """Return a record's ID, labeled request, targets, and encoding."""
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
        # Unlabeled questions cannot contribute a training gradient.
        # The builder may retain over-length questions after dropping labels.
        request["questions"] = {
            key: question
            for key, question in questions.items()
            if key in targets
        }
        encoded = json_io.object_map(json.loads(str(value["encoding_json"])))
        return str(value["id"]), request, targets, encoded

    @staticmethod
    def _recheck(
        source: fields.Encoding,
        spans: dict[str, tuple[int, int] | None],
        encoded: dict[str, object],
    ) -> None:
        """Require the saved tokenizer to reproduce counts and gold spans."""
        saved = json_io.object_map(encoded["token_spans"])
        if len(source.ids) != encoded["source_tokens"]:
            raise ValueError("Source token count changed")
        for key, span in spans.items():
            if span is not None and list(span) != saved.get(key):
                raise ValueError("Gold token span changed")

    def compile(self, raw: object, split: str) -> fields.Record:
        """Recheck token spans with the saved tokenizer."""
        record_id, request, targets, encoded = self._admitted(raw, split)
        record = magicbox.compile_record(
            record_id, request, targets, self.tokenizer.encode
        )
        self._recheck(
            record.source,
            {field.key: field.span for field in record.fields},
            encoded,
        )
        if not any(field.supervised for field in record.fields):
            raise ValueError("no_supervision")
        return record

    def compile_pointer(
        self, raw: object, split: str, *, score_width: float = 0.0
    ) -> pointer.Record:
        """Compile the joint pointer record, rechecking saved token spans.

        ``score_width`` spreads hard score labels over nearby levels; zero
        keeps them one-hot. Soft labels always pass through unchanged.
        """
        record_id, request, targets, encoded = self._admitted(raw, split)
        record = magicbox.compile_pointer_record(
            record_id,
            request,
            targets,
            self.tokenizer.encode,
            score_width=score_width,
        )
        record = with_accepted(record, json_io.object_map(raw))
        self._recheck(
            record.source,
            {question.key: question.span for question in record.questions},
            encoded,
        )
        if not any(question.supervised for question in record.questions):
            raise ValueError("no_supervision")
        return record

    def _token_counts(
        self,
        splits: Sequence[str],
        texts: Callable[[object, object], tuple[str, ...]],
    ) -> Iterator[tuple[int, int, list[int]]]:
        """Yield source tokens, labeled questions, and each text's tokens.

        Text is batch-tokenized with the saved tokenizer. Sources, spans, and
        targets are checked later, when training compiles each record.
        """
        for split in splits:
            columns = self.split(split).select_columns(
                ["request_json", "targets_json", "encoding_json"]
            )
            for chunk in columns.iter(batch_size=1024):
                requests = [
                    json.loads(value) for value in chunk["request_json"]
                ]
                labels = [json.loads(value) for value in chunk["targets_json"]]
                rows = [
                    texts(request, targets)
                    for request, targets in zip(requests, labels, strict=True)
                ]
                encoded = iter(
                    self.tokenizer.tokenizer.encode_batch(
                        [text for items in rows for text in items],
                        add_special_tokens=True,
                    )
                )
                for items, targets, encoding in zip(
                    rows, labels, chunk["encoding_json"], strict=True
                ):
                    yield (
                        int(json.loads(encoding)["source_tokens"]),
                        len(targets),
                        [len(next(encoded).ids) for _ in items],
                    )

    def packed_sequences(
        self, splits: Sequence[str], schema_tokens: int
    ) -> int:
        """Return the most packed schema rows any record in ``splits`` needs."""
        return max(
            (
                packing.rows_required(lengths, schema_tokens)
                for _, _, lengths in self._token_counts(
                    splits, magicbox.labeled_schema_rows
                )
            ),
            default=0,
        )

    def pointer_sizes(self, split: str) -> list[tuple[int, int]]:
        """Return each record's joint tokens and labeled questions, in order.

        Sizes follow the split's stored row order, which planned packing
        indexes directly.
        """
        return [
            (source + sum(lengths), labeled)
            for source, labeled, lengths in self._token_counts(
                (split,), magicbox.labeled_pointer_texts
            )
        ]

    def pointer_extent(self, splits: Sequence[str]) -> tuple[int, int]:
        """Return the longest joint sequence and most questions per record."""
        tokens, questions = 0, 0
        for source, labeled, lengths in self._token_counts(
            splits, magicbox.labeled_pointer_texts
        ):
            tokens = max(tokens, source + sum(lengths))
            questions = max(questions, labeled)
        return tokens, questions

    def records(self, split: str, limit: int) -> Iterator[fields.Record]:
        """Apply the experiment's fixed held-out sampling policy."""
        dataset = self.split(split).shuffle(seed=1729, keep_in_memory=False)
        count = min(limit, len(dataset)) if limit else len(dataset)
        for index in range(count):
            yield self.compile(dataset[index], split)

    def pointer_sources(self, split: str) -> list[str]:
        """Return each record's source dataset, in stored order."""
        names: list[str] = []
        columns = self.split(split).select_columns(["provenance_json"])
        for chunk in columns.iter(batch_size=4096):
            names.extend(
                source_name(value) for value in chunk["provenance_json"]
            )
        return names

    def pointer_source_kinds(self, split: str) -> dict[str, set[str]]:
        """Each source dataset's question types in one split."""
        kinds: dict[str, set[str]] = {}
        columns = self.split(split).select_columns(
            ["provenance_json", "request_json"]
        )
        for chunk in columns.iter(batch_size=4096):
            for provenance, request in zip(
                chunk["provenance_json"], chunk["request_json"], strict=True
            ):
                kinds.setdefault(source_name(provenance), set()).update(
                    question_kinds(request)
                )
        return kinds

    def pointer_records(
        self,
        split: str,
        limit: int,
        source: str | None = None,
        kind: str | None = None,
    ) -> Iterator[pointer.Record]:
        """Sample held-out pointer records with their original labels.

        With ``source``, sample only that source dataset's records; with
        ``kind``, only records asking at least one question of that type.
        """
        dataset = self.split(split).shuffle(seed=1729, keep_in_memory=False)
        if source is not None:
            dataset = dataset.filter(
                lambda value: source_name(value) == source,
                input_columns="provenance_json",
                keep_in_memory=False,
            )
        if kind is not None:
            dataset = dataset.filter(
                lambda value: kind in question_kinds(value),
                input_columns="request_json",
                keep_in_memory=False,
            )
        for index in range(min(limit, len(dataset)) if limit else len(dataset)):
            yield self.compile_pointer(dataset[index], split)


def with_accepted(
    record: pointer.Record, raw: Mapping[str, object]
) -> pointer.Record:
    """Attach each question's alternative acceptable answers from provenance.

    The dataset records them as character spans of the record's text, for
    example every mention of an entity when one is the gold answer.
    """
    provenance = json_io.object_map(json.loads(str(raw["provenance_json"])))
    conversion = json_io.object_map(provenance.get("conversion", {}))
    spans = json_io.object_map(conversion.get("acceptable_spans", {}))
    if not spans:
        return record
    questions = []
    for question in record.questions:
        found = spans.get(question.key, [])
        texts = {
            record.text[int(str(start)) : int(str(end))]
            for start, end in cast(list[list[object]], found)
        }
        questions.append(
            dataclasses.replace(question, accepted=tuple(sorted(texts)))
        )
    return dataclasses.replace(record, questions=tuple(questions))


def question_kinds(request_json: str) -> set[str]:
    """The question types one request asks."""
    questions = json_io.object_map(
        json_io.object_map(json.loads(request_json))["questions"]
    )
    return {
        str(json_io.object_map(question)["type"])
        for question in questions.values()
    }


def source_name(provenance_json: str) -> str:
    """The upstream dataset a record came from, from its provenance."""
    provenance = json_io.object_map(json.loads(provenance_json))
    return str(json_io.object_map(provenance["source"])["dataset"])


class Packer[RecordT](Protocol):
    """A batch strategy that packs compiled records into one update."""

    @property
    def shape(self) -> contracts.CapacityShape:
        """Logical request capacity."""

    def pack(
        self, examples: Sequence[RecordT], *, seed: int, update: int
    ) -> contracts.PhysicalUpdate:
        """Pack one logical update."""


def training_stream[RecordT](
    corpus: Corpus,
    batches: Packer[RecordT],
    seed: int,
    epochs: int,
    compile_record: Callable[[object], RecordT],
    *,
    prefetch: int = 0,
) -> stream.EpochStream[object, RecordT]:
    """Bind the published Arrow order and a record compiler to cursor replay."""
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
        compile_record=compile_record,
        pack=lambda records, update: batches.pack(
            records, seed=seed, update=update
        ),
        prefetch=prefetch,
    )


def planned_training_stream(
    corpus: Corpus,
    batches: pointer_batching.PointerBatchStrategy,
    sizes: Sequence[tuple[int, int]],
    seed: int,
    epochs: int,
    compile_record: Callable[[object], pointer.Record],
    *,
    prefetch: int = 2,
    open_limit: int = 64,
    close_below: float = 0.05,
) -> stream.PlannedStream[object, pointer.Record]:
    """Pack whole training requests into each update's fixed rows.

    Each epoch shuffles by ``seed + epoch`` and packs requests by online
    first-fit within the row's token and question capacity, so the plan and
    the resume cursor are deterministic. ``sizes`` come from
    ``Corpus.pointer_sizes("train")``.
    """
    data = corpus.split("train")
    if len(sizes) != len(data):
        raise ValueError("Packing sizes don't match the training split")
    shape = batches.shape
    return stream.PlannedStream(
        epochs=epochs,
        plan=lambda epoch: packing.plan_updates(
            sizes,
            (shape.sequence_tokens, shape.questions),
            shape.capacity,
            seed=seed + epoch,
            open_limit=open_limit,
            close_below=close_below,
        ),
        read=lambda index: data[index],
        compile_record=compile_record,
        pack=batches.pack_rows,
        prefetch=prefetch,
    )
