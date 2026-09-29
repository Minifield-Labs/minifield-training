"""Hand-built pointer logits decode into every typed answer."""

import math

import pytest

from minifield_training.batching import pointer as batching
from minifield_training.core import json_io
from minifield_training.datasets import fields
from minifield_training.datasets import pointer
from minifield_training.evaluation import pointer as evaluation


def _record() -> pointer.Record:
    source = fields.Encoding(
        (1, 5, 6, 7),
        ((0, 0), (0, 3), (4, 7), (8, 12)),
        (True, False, False, False),
    )
    option = pointer.Option
    return pointer.Record(
        "r",
        "Ada Lee sent",
        source,
        (
            pointer.Question(
                "name", 0, (1,), (option("absent", (1,)),), (0.0,), True, (1, 3)
            ),
            pointer.Question(
                "tier",
                1,
                (1,),
                (option("a", (1,)), option("b", (1,))),
                (0.0, 1.0),
                True,
            ),
            pointer.Question(
                "open",
                2,
                (1,),
                (option("false", (1,)), option("true", (1,))),
                (0.2, 0.8),
                True,
            ),
            pointer.Question(
                "level",
                3,
                (1,),
                tuple(option(str(index), (1,)) for index in range(3)),
                (0.0, 0.0, 1.0),
                True,
            ),
        ),
    )


def _logits(
    record: pointer.Record,
) -> tuple[batching.Layout, list[list[float]], list[list[float]]]:
    """Point at "Ada Lee", option b, true with 3:1 odds, and level 2."""
    ids, placed = batching.layout(record)
    start = [[0.0] * len(ids) for _ in record.questions]
    end = [[0.0] * len(ids) for _ in record.questions]
    base = placed.source_start
    start[0][base + 1], end[0][base + 2] = 9.0, 9.0
    start[0][placed.options[0][0]] = end[0][placed.options[0][0]] = -9.0
    start[1][placed.options[1][1]] = end[1][placed.options[1][1]] = 5.0
    start[2][placed.options[2][1]] = end[2][placed.options[2][1]] = math.log(3)
    start[3][placed.options[3][2]] = end[3][placed.options[3][2]] = 20.0
    return placed, start, end


def test_decode_every_question_type() -> None:
    """Spans, options, binary odds, and ordinal expectations decode exactly."""
    record = _record()
    placed, start, end = _logits(record)
    decoded = {
        key: json_io.object_map(value)
        for key, value in evaluation.decode(record, placed, start, end).items()
    }
    assert decoded["name"]["value"] == "Ada Lee"
    assert decoded["name"]["token_span"] == (1, 3)
    assert decoded["tier"]["value"] == "b"
    assert decoded["open"]["value"] == pytest.approx(0.75)
    assert decoded["level"]["value"] == pytest.approx(2, abs=1e-6)
    absent = evaluation.decode(record, placed, start, end, presence_threshold=1)
    assert json_io.object_map(absent["name"])["value"] is None


def test_best_span_stays_inside_one_selectable_run() -> None:
    """A non-selectable token splits runs; ties keep the earliest span."""
    start = [0.0, 5.0, 0.0, 0.0, 4.0]
    end = [0.0, 0.0, 0.0, 6.0, 1.0]
    assert evaluation.best_span(
        start, end, [True, True, False, True, True], 0
    ) == (3, 4)
    assert evaluation.best_span(start, end, [True] * 5, 0) == (1, 4)
    assert evaluation.best_span(start, end, [False] * 5, 0) is None


def test_metrics_compare_against_supervised_targets() -> None:
    """Exact span, accuracy, Brier, and ordinal error per question type."""
    record = _record()
    placed, start, end = _logits(record)
    metrics = evaluation.Metrics(("extract", "choice", "binary", "ordinal"))
    metrics.record(
        record,
        evaluation.decode(record, placed, start, end),
        [1.0, 2.0, 3.0, 4.0],
    )
    means = metrics.means()
    assert means["extract/exact"] == 1
    assert means["extract/false_null"] == 0
    assert means["choice/accuracy"] == 1
    assert means["binary/brier"] == pytest.approx((0.75 - 0.8) ** 2)
    assert means["ordinal/mae"] == pytest.approx(0, abs=1e-6)
    assert means["loss"] == pytest.approx(2.5)
