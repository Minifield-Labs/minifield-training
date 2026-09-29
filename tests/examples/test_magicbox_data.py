"""CPU schema admission and private-label separation contracts."""

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import datasets  # type: ignore[import-untyped]
import jax.numpy as jnp
import numpy as np
import pytest

from examples.magicbox import data as magicbox
from examples.magicbox import smoke
from examples.magicbox import source
from examples.magicbox import tokenizer
from minifield_training.batching import pointer as pointer_batching
from minifield_training.batching import schema_fields as batching
from minifield_training.core import json_io
from minifield_training.datasets import fields
from minifield_training.datasets import pointer
from minifield_training.objectives import schema_fields as objective


def test_consumer_contract_snapshot() -> None:
    """Pin the producer document and the consumer's versioned constants."""
    root = (
        Path(__file__).resolve().parents[2]
        / "src/minifield_training/datasets/magicbox"
    )
    identity = json.loads((root / "format-v1.json").read_text())
    assert (
        identity["sha256"]
        == "a3c93f2385ee3a99ca40696c6d2384ccb2e49fa35017c846ee1250632b2d86ab"
    )
    assert json_io.digest_file(root / "format-v1.md") == identity["sha256"]
    assert (
        identity["format"],
        identity["template"],
        identity["offset_policy"],
    ) == (magicbox.FORMAT, magicbox.TEMPLATE, magicbox.OFFSET_POLICY)


def test_templates_keep_choice_order_and_score_ranks() -> None:
    """Choice IDs enter their own rows; scores preserve ordinal position."""
    keys, rows = magicbox.schema_rows(
        {
            "type": "choice",
            "instructions": "Pick.",
            "criteria": {"b": "Second", "a": "First"},
        }
    )
    assert keys == ("b", "a")
    assert rows == (
        "Type: choice\nQuestion: Pick.\nCandidate: b\nDescription: Second",
        "Type: choice\nQuestion: Pick.\nCandidate: a\nDescription: First",
    )
    _, ranks = magicbox.schema_rows(
        {"type": "score", "instructions": "Rate.", "criteria": ["Low", "High"]}
    )
    assert "Level: 1 of 2 levels, indexed from 0\nDescription: High" in ranks[1]


def test_unicode_overlaps_and_unalignable_boundaries() -> None:
    """Keep all byte tokens overlapping one Unicode character."""
    encoded = fields.Encoding(
        (1, 2, 3, 4, 5),
        ((0, 0), (0, 1), (0, 1), (1, 2), (2, 5)),
        (True, False, False, False, False),
    )
    assert fields.aligned_span(encoded, [0, 1], "é Ada") == (1, 3)
    assert fields.aligned_span(encoded, [0, 5], "é Ada") == (1, 5)
    with pytest.raises(ValueError, match="splits a token"):
        fields.aligned_span(encoded, [3, 5], "é Ada")


def test_type_balancing_partial_labels_and_no_supervision() -> None:
    """Unequal fields and microbatches still produce one equal mean per type."""
    record = smoke.fixture()
    second = dataclasses.replace(
        record, id="second", fields=(record.fields[1],)
    )
    packed = batching.build(
        [record, second],
        batching.Shape(2, 1, 16, 64, 8, 128, 0),
        seed=5,
        update=3,
        weighting=objective.balance_types,
    )
    weights = np.asarray(packed.microbatches["field_weight"])
    np.testing.assert_allclose(weights.sum(), 1)
    assert weights[0, 0, 0] == 0.25
    assert weights[0, 0, 1] == weights[1, 0, 0] == 0.125
    totals = []
    for index in range(2):
        batch = {
            key: jnp.asarray(value[index])
            for key, value in packed.microbatches.items()
        }
        outputs = {
            key: jnp.zeros((1, 8))
            for key in ("candidate", "binary", "presence")
        }
        outputs["tokens"] = jnp.zeros((1, 8, 16))
        loss, mass = objective.terms(outputs, batch)
        totals.append((float(loss), float(mass)))
    assert math.isclose(
        sum(value[0] for value in totals),
        (4 * math.log(2) + math.log(3)) / 4,
        rel_tol=1e-6,
    )
    missing = dataclasses.replace(
        record,
        fields=tuple(
            dataclasses.replace(field, supervised=False)
            for field in record.fields
        ),
    )
    with pytest.raises(ValueError, match="no_supervision"):
        batching.build(
            [missing],
            batching.Shape(1, 1, 16, 64, 8, 128, 0),
            seed=0,
            update=0,
            weighting=objective.balance_types,
        )


def test_absent_and_presence_only_extractions() -> None:
    """Absence labels all selectable tokens zero; presence-only skips tokens."""
    request = {
        "state": "Ada",
        "questions": {"x": {"type": "extract", "instructions": "Find a name."}},
    }
    absent = magicbox.compile_record(
        "absent", request, {"x": {"has_answer": False}}, smoke.toy_encode
    )
    presence = magicbox.compile_record(
        "presence", request, {"x": {"has_answer": True}}, smoke.toy_encode
    )
    assert absent.fields[0].token_supervised
    assert not presence.fields[0].token_supervised
    assert absent.fields[0].rows == presence.fields[0].rows
    with pytest.raises(ValueError, match="nonempty"):
        magicbox.schema_rows(
            {"type": "choice", "instructions": "", "criteria": {"x": "X"}}
        )


_REQUEST = {
    "state": "Ada paid 5.",
    "questions": {
        "name": {"type": "extract", "instructions": "Name."},
        "tier": {
            "type": "choice",
            "instructions": "Tier.",
            "criteria": {"gold": "Top", "basic": "Entry"},
        },
        "mood": {
            "type": "score",
            "instructions": "Mood.",
            "criteria": ["Low", "Mid", "High"],
        },
    },
}


def test_labeled_schema_rows_skip_unlabeled_questions() -> None:
    """Training encodes rows only for questions that carry supervision."""
    rows = magicbox.labeled_schema_rows(
        _REQUEST, {"tier": {"value": "gold"}, "mood": {"value": 1}}
    )
    assert rows == (
        "Type: choice\nQuestion: Tier.\nCandidate: gold\nDescription: Top",
        "Type: choice\nQuestion: Tier.\nCandidate: basic\nDescription: Entry",
        "Type: score\nQuestion: Mood.\nLevel: 0 of 3 levels, indexed from 0"
        "\nDescription: Low",
        "Type: score\nQuestion: Mood.\nLevel: 1 of 3 levels, indexed from 0"
        "\nDescription: Mid",
        "Type: score\nQuestion: Mood.\nLevel: 2 of 3 levels, indexed from 0"
        "\nDescription: High",
    )


class _WordTokenizer:
    """Count BOS plus whitespace words, standing in for the pinned tokenizer."""

    def encode_batch(
        self, texts: list[str], add_special_tokens: bool
    ) -> list[SimpleNamespace]:
        assert add_special_tokens
        return [
            SimpleNamespace(ids=[1] * (1 + len(text.split()))) for text in texts
        ]


def test_corpus_measures_packed_sequences_across_splits() -> None:
    """The scan packs each record's labeled rows and keeps the maximum."""
    corpus = object.__new__(source.Corpus)
    corpus.tokenizer = cast(
        tokenizer.Adapter, SimpleNamespace(tokenizer=_WordTokenizer())
    )
    labels = json.dumps({"tier": {"value": "gold"}, "mood": {"value": 1}})
    one_field = json.dumps({"name": {"value": None}})
    corpus._splits = {  # pylint: disable=protected-access
        split: datasets.Dataset.from_dict(
            {
                "request_json": [json.dumps(_REQUEST)] * len(targets),
                "targets_json": targets,
                "encoding_json": [json.dumps({"source_tokens": 7})]
                * len(targets),
            }
        )
        for split, targets in (
            ("train", [one_field, labels]),
            ("validation", [one_field]),
        )
    }
    # Choice rows count 9 tokens and score rows 15. At 20 tokens each score
    # row needs its own row and both choices share one; 24 fits [15, 9] twice.
    assert corpus.packed_sequences(("train",), 20) == 4
    assert corpus.packed_sequences(("validation",), 20) == 1
    assert corpus.packed_sequences(("validation", "train"), 24) == 3
    # Pointer texts: tier query 5 + options 5 + 5, mood query 5 + 3 levels of
    # 11, plus 7 source tokens. The extract-only record has 5 + 7 + 7.
    assert corpus.pointer_extent(("train",)) == (60, 2)
    assert corpus.pointer_extent(("validation",)) == (19, 1)
    assert corpus.pointer_sizes("train") == [(19, 1), (60, 2)]
    _check_planned_stream(corpus)


def _check_planned_stream(corpus: source.Corpus) -> None:
    """Both train records pack into 1 row per epoch and resume exactly."""

    def compile_record(raw: object) -> pointer.Record:
        """Stand in for real compilation with a small valid choice record."""
        targets = json_io.object_map(raw)["targets_json"]
        return pointer.Record(
            hashlib.sha256(str(targets).encode()).hexdigest()[:8],
            "x",
            fields.Encoding((1, 5), ((0, 0), (0, 1)), (True, False)),
            (
                pointer.Question(
                    "q",
                    fields.Kind.CHOICE,
                    (1, 7),
                    (pointer.Option("a", (1, 8)), pointer.Option("b", (1, 9))),
                    (1.0, 0.0),
                    True,
                ),
            ),
        )

    batches = pointer_batching.PointerBatchStrategy(
        pointer_batching.Shape(1, 2, 128, 4, 128, 0), objective.balance_types
    )
    planned = source.planned_training_stream(
        corpus,
        batches,
        corpus.pointer_sizes("train"),
        3,
        2,
        compile_record,
        prefetch=1,
    )
    assert planned.total_updates == 2
    updates = list(planned(0))
    assert [len(update.example_ids) for update in updates] == [2, 2]
    # One packed row per update leaves the second row slot as padding.
    assert not np.asarray(updates[0].microbatches["input_mask"])[0, 1].any()
    assert [update.example_ids for update in planned(1)] == [
        updates[1].example_ids
    ]
