"""Hand-built pointer logits decode into every typed answer."""

import dataclasses
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


def test_added_metrics_match_hand_derived_values() -> None:
    """Calibration, F1, IoU, and error reduction for the fixture answers."""
    record = _record()
    placed, start, end = _logits(record)
    metrics = evaluation.Metrics(("extract", "choice", "binary", "ordinal"))
    metrics.record(
        record,
        evaluation.decode(record, placed, start, end),
        [1.0, 2.0, 3.0, 4.0],
    )
    means = metrics.means()
    chosen = math.exp(5) / (1 + math.exp(5))
    assert means["extract/token_f1"] == 1
    assert means["extract/span_iou"] == 1
    assert means["choice/nll"] == pytest.approx(-math.log(chosen))
    assert means["choice/margin"] == pytest.approx(2 * chosen - 1)
    assert means["choice/ece"] == pytest.approx(1 - chosen)
    assert means["binary/nll"] == pytest.approx(
        -(0.8 * math.log(0.75) + 0.2 * math.log(0.25))
    )
    assert means["ordinal/within_1"] == 1
    # One question per type: rank metrics and null F1 have nothing to rank.
    assert "ordinal/spearman" not in means and "binary/auroc" not in means
    assert "extract/null_f1" not in means
    binary = 1 - (0.75 - 0.8) ** 2 / (0.5 - 0.8) ** 2
    assert means["binary/error_reduction"] == pytest.approx(binary)
    assert means["error_reduction"] == pytest.approx((3 + binary) / 4, abs=1e-6)


def test_rank_and_calibration_statistics() -> None:
    """Small cases with known AUROC, Spearman, ECE, F1, and degradation."""
    assert evaluation.auroc(
        [0.9, 0.8, 0.3, 0.8], [True, False, False, True]
    ) == (pytest.approx(0.875))
    assert evaluation.auroc([0.5, 0.6], [True, True]) is None
    assert evaluation.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1
    assert evaluation.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    # Bins 0.9 (outcomes 1, 0) and 0.2 (outcome 0): |1.8 - 1| + |0.2 - 0|.
    assert evaluation.expected_calibration_error(
        [0.9, 0.9, 0.2], [1.0, 0.0, 0.0]
    ) == pytest.approx(1.0 / 3)
    assert evaluation.token_f1("Ada Lee", "ada") == pytest.approx(2 / 3)
    assert evaluation.token_f1(None, None) == 1
    assert evaluation.token_f1("Ada", None) == 0
    shift = evaluation.degradation(
        {
            "choice/accuracy": 0.8,
            "binary/brier": 0.1,
            "extract/false_null": 0.001,
            "choice/loss": 0.5,
            "choice/accuracy/count": 9,
        },
        {
            "choice/accuracy": 0.6,
            "binary/brier": 0.15,
            "extract/false_null": 0.011,
            "choice/loss": 0.75,
        },
    )
    # Rates change absolutely, so a tiny base can't explode; losses relatively.
    assert shift == pytest.approx(
        {
            "choice/accuracy": 0.2,
            "binary/brier": 0.05,
            "extract/false_null": 0.01,
            "choice/loss": 0.5,
        }
    )


def test_null_f1_treats_not_stated_as_the_positive_class() -> None:
    """2 correct nulls, 1 missed null, 1 wrong null: F1 = 4 / 6."""
    record = _record()
    placed, start, end = _logits(record)
    metrics = evaluation.Metrics(("extract", "choice", "binary", "ordinal"))
    null_gold = dataclasses.replace(record.questions[0], span=None)
    cases = [(null_gold, 1.0), (null_gold, 1.0), (null_gold, 0.0)]
    cases.append((record.questions[0], 1.0))
    for question, threshold in cases:
        single = dataclasses.replace(record, questions=(question,))
        metrics.record(
            single,
            evaluation.decode(
                single, placed, start, end, presence_threshold=threshold
            ),
            [0.0],
        )
    assert metrics.means()["extract/null_f1"] == pytest.approx(4 / 6)


def test_accepted_answers_and_kl() -> None:
    """Another listed mention counts as accepted; KL is zero only at gold."""
    record = _record()
    placed, start, end = _logits(record)
    # Gold becomes "Lee sent"; the decoded "Ada Lee" is a listed alternative.
    name = dataclasses.replace(
        record.questions[0], span=(2, 4), accepted=("Ada Lee",)
    )
    shifted = dataclasses.replace(
        record, questions=(name, *record.questions[1:])
    )
    metrics = evaluation.Metrics(("extract", "choice", "binary", "ordinal"))
    metrics.record(
        shifted,
        evaluation.decode(shifted, placed, start, end),
        [1.0, 2.0, 3.0, 4.0],
    )
    means = metrics.means()
    assert means["extract/exact"] == 0 and means["extract/accepted"] == 1
    assert means["binary/kl"] == pytest.approx(
        0.2 * math.log(0.2 / 0.25) + 0.8 * math.log(0.8 / 0.75)
    )
    assert evaluation.kl_divergence([0.5, 0.5], [0.5, 0.5]) == 0


@pytest.mark.parametrize(
    ("targets", "correct"), [((0.5, 0.5), 1.0), ((1.0, 0.0), 0.0)]
)
def test_soft_choice_target_accepts_any_top_option(
    targets: tuple[float, float], correct: float
) -> None:
    """Choosing "b" is right when "b" shares the top target probability."""
    record = _record()
    placed, start, end = _logits(record)
    tier = dataclasses.replace(record.questions[1], targets=targets)
    record = dataclasses.replace(
        record, questions=(record.questions[0], tier, *record.questions[2:])
    )
    metrics = evaluation.Metrics(("extract", "choice", "binary", "ordinal"))
    metrics.record(
        record,
        evaluation.decode(record, placed, start, end),
        [1.0, 2.0, 3.0, 4.0],
    )
    assert metrics.means()["choice/accuracy"] == correct
