"""Decode pointer logits into typed answers, and score held-out questions."""

from collections.abc import Callable, Sequence
import dataclasses
import math
from typing import cast

import numpy as np

from minifield_training.batching import pointer as batching
from minifield_training.core import json_io
from minifield_training.datasets import fields
from minifield_training.datasets import pointer
from minifield_training.evaluation import field_decode
from minifield_training.evaluation import schema_fields
from minifield_training.kernels import types
from minifield_training.objectives import pointer as objective


def _option_probabilities(
    start: Sequence[float], end: Sequence[float], positions: Sequence[int]
) -> list[float]:
    """Average the start and end softmaxes over one question's options."""
    left = field_decode.probabilities([start[index] for index in positions])
    right = field_decode.probabilities([end[index] for index in positions])
    return [(a + b) / 2 for a, b in zip(left, right, strict=True)]


def best_span(
    start: Sequence[float],
    end: Sequence[float],
    selectable: Sequence[bool],
    offset: int,
) -> tuple[int, int] | None:
    """Find ``[first, last + 1)`` maximizing start + end logits.

    The span stays inside one run of selectable source tokens, so it never
    crosses a special or empty token. Ties keep the earliest, shortest span.
    """
    best, score, first = None, -math.inf, None
    for index, valid in enumerate(selectable):
        if not valid:
            first = None
            continue
        if first is None or start[offset + index] > start[offset + first]:
            first = index
        candidate = start[offset + first] + end[offset + index]
        if candidate > score:
            best, score = (first, index + 1), candidate
    return best


def decode(
    record: pointer.Record,
    placed: batching.Layout,
    start: Sequence[Sequence[float]],
    end: Sequence[Sequence[float]],
    *,
    presence_threshold: float = 0.5,
) -> dict[str, object]:
    """Return each question's typed value with its probabilities."""
    if not 0 <= presence_threshold <= 1:
        raise ValueError("Invalid presence threshold")
    result: dict[str, object] = {}
    source = record.source
    for index, question in enumerate(record.questions):
        markers = placed.options[index]
        details: dict[str, object] = {"confidence": None}
        if question.kind == fields.Kind.EXTRACT:
            allowed = [markers[0]] + [
                placed.source_start + token
                for token, valid in enumerate(source.selectable)
                if valid
            ]
            present = (
                1 - _option_probabilities(start[index], end[index], allowed)[0]
            )
            span = (
                best_span(
                    start[index],
                    end[index],
                    source.selectable,
                    placed.source_start,
                )
                if present >= presence_threshold
                else None
            )
            value: object = None
            if span is not None:
                characters = (
                    source.offsets[span[0]][0],
                    source.offsets[span[1] - 1][1],
                )
                value = record.text[characters[0] : characters[1]]
                details.update(token_span=span, character_span=characters)
            details.update(presence=present)
        else:
            labels = [option.label for option in question.options]
            probabilities = _option_probabilities(
                start[index], end[index], markers
            )
            details.update(
                probabilities=dict(zip(labels, probabilities, strict=True))
            )
            if question.kind == fields.Kind.CHOICE:
                value = labels[int(np.argmax(probabilities))]
            elif question.kind == fields.Kind.ORDINAL:
                value = sum(
                    level * probability
                    for level, probability in enumerate(probabilities)
                )
            else:
                value = probabilities[labels.index("true")]
        result[question.key] = {"value": value, **details}
    return result


class Predictor(schema_fields.SingleRequest):
    """Pack one request, run the jitted forward, and decode its answers."""

    def __init__(
        self,
        forward: Callable[
            [types.Parameters, types.DeviceBatch], types.DeviceBatch
        ],
        batches: batching.PointerBatchStrategy,
        *,
        presence_threshold: float = 0.5,
    ):
        super().__init__(forward, objective.losses)
        self.batches = batching.PointerBatchStrategy(
            dataclasses.replace(batches.shape, microbatches=1, rows=1),
            batches.weighting,
        )
        self.presence_threshold = presence_threshold

    def score(
        self,
        params: types.Parameters,
        record: pointer.Record,
        *,
        include_losses: bool = True,
    ) -> tuple[dict[str, object], list[float]]:
        """Return typed answers and optional per-question losses."""
        outputs, losses = self.run(
            params,
            self.batches.pack(
                [record], seed=0, update=0, allow_unsupervised=True
            ),
            include_losses,
        )
        decoded = decode(
            record,
            batching.layout(record)[1],
            cast(list[list[float]], outputs["start"]),
            cast(list[list[float]], outputs["end"]),
            presence_threshold=self.presence_threshold,
        )
        return decoded, losses[: len(record.questions)]


_BINS = 10
_EPSILON = 1e-12


def expected_calibration_error(
    confidences: Sequence[float], outcomes: Sequence[float]
) -> float:
    """Count-weighted gap between mean confidence and mean outcome per bin."""
    bins: dict[int, list[tuple[float, float]]] = {}
    for confidence, outcome in zip(confidences, outcomes, strict=True):
        index = min(int(confidence * _BINS), _BINS - 1)
        bins.setdefault(index, []).append((confidence, outcome))
    return sum(
        abs(sum(c for c, _ in items) - sum(o for _, o in items))
        for items in bins.values()
    ) / max(len(confidences), 1)


def auroc(scores: Sequence[float], positives: Sequence[bool]) -> float | None:
    """Probability a positive outranks a negative; ties count half."""
    ranks = _ranks(scores)
    count = sum(positives)
    negatives = len(positives) - count
    if not count or not negatives:
        return None
    total = sum(
        rank
        for rank, positive in zip(ranks, positives, strict=True)
        if positive
    )
    return (total - count * (count + 1) / 2) / (count * negatives)


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson correlation of average ranks."""
    if len(left) < 2:
        return None
    a, b = np.asarray(_ranks(left)), np.asarray(_ranks(right))
    a, b = a - a.mean(), b - b.mean()
    scale = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b)) / scale if scale else None


def _ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while (
            stop + 1 < len(order)
            and values[order[stop + 1]] == values[order[start]]
        ):
            stop += 1
        for position in range(start, stop + 1):
            ranks[order[position]] = (start + stop) / 2 + 1
        start = stop + 1
    return ranks


def token_f1(predicted: str | None, expected: str | None) -> float:
    """SQuAD-style lowercase whitespace-token F1; two nulls score 1."""
    if predicted is None or expected is None:
        return float(predicted is None and expected is None)
    guess, gold = predicted.lower().split(), expected.lower().split()
    common = sum(
        min(guess.count(token), gold.count(token)) for token in set(guess)
    )
    if not common:
        return 0.0
    precision, recall = common / len(guess), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def kl_divergence(
    targets: Sequence[float], predicted: Sequence[float]
) -> float:
    """KL(gold || predicted): the loss above the labels' own entropy."""
    return sum(
        gold * (math.log(gold) - math.log(max(guess, _EPSILON)))
        for gold, guess in zip(targets, predicted, strict=True)
        if gold > 0
    )


def _iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union else 0.0


class Metrics:
    """Compare typed answers with supervised questions, per question type.

    Besides each type's mean metrics, rank and calibration metrics keep
    per-question values until ``means``. ``<type>/error_reduction`` is
    ``1 - error / baseline`` against a trivial answer (always null, a
    uniform guess, 0.5, or the middle level); ``error_reduction`` weights
    the types equally.
    """

    def __init__(self, names: tuple[str, str, str, str]):
        self.names = names
        self._totals = schema_fields.Metrics(names=names)
        self._choice: list[tuple[float, bool]] = []
        self._binary: list[tuple[float, float]] = []
        self._ordinal: list[tuple[float, float]] = []
        self._nulls = {"both": 0, "predicted": 0, "expected": 0}
        self._errors: dict[int, list[tuple[float, float]]] = {}

    def _error(self, kind: int, error: float, baseline: float) -> None:
        self._errors.setdefault(kind, []).append((error, baseline))

    def record(
        self,
        record: pointer.Record,
        decoded: dict[str, object],
        losses: list[float],
    ) -> None:
        """Add each supervised question's loss and type-specific metrics."""
        for index, question in enumerate(record.questions):
            if not question.supervised:
                continue
            name = self.names[question.kind]
            answer = json_io.object_map(decoded[question.key])
            value = answer["value"]
            self._totals.add(name + "/loss", losses[index])
            targets = question.targets
            if question.kind == fields.Kind.EXTRACT:
                self._extraction(record, question, answer)
                continue
            labels = [option.label for option in question.options]
            probabilities = json_io.object_map(answer["probabilities"])
            predicted = [float(str(probabilities[label])) for label in labels]
            self._totals.add(name + "/kl", kl_divergence(targets, predicted))
            if question.kind == fields.Kind.CHOICE:
                gold = int(np.argmax(targets))
                correct = value == labels[gold]
                ranked = sorted(predicted, reverse=True)
                self._totals.add(name + "/accuracy", float(correct))
                self._totals.add(
                    name + "/nll", -math.log(max(predicted[gold], _EPSILON))
                )
                self._totals.add(
                    name + "/margin",
                    ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0),
                )
                self._choice.append((ranked[0], correct))
                self._error(
                    question.kind, 1 - float(correct), 1 - 1 / len(labels)
                )
            elif question.kind == fields.Kind.BINARY:
                truth = targets[labels.index("true")]
                p = min(max(float(str(value)), _EPSILON), 1 - _EPSILON)
                self._totals.add(name + "/brier", (p - truth) ** 2)
                self._totals.add(
                    name + "/nll",
                    -(truth * math.log(p) + (1 - truth) * math.log(1 - p)),
                )
                self._binary.append((p, truth))
                self._error(question.kind, (p - truth) ** 2, (0.5 - truth) ** 2)
            else:
                expectation = sum(
                    level * probability
                    for level, probability in enumerate(targets)
                )
                predicted_level = float(str(value))
                self._totals.add(
                    name + "/mae", abs(predicted_level - expectation)
                )
                self._totals.add(
                    name + "/within_1",
                    float(
                        abs(round(predicted_level) - int(np.argmax(targets)))
                        <= 1
                    ),
                )
                self._ordinal.append((predicted_level, expectation))
                self._error(
                    question.kind,
                    abs(predicted_level - expectation),
                    abs((len(labels) - 1) / 2 - expectation),
                )

    def _extraction(
        self,
        record: pointer.Record,
        question: pointer.Question,
        answer: dict[str, object],
    ) -> None:
        name = self.names[fields.Kind.EXTRACT]
        value = answer["value"]
        predicted = None if value is None else str(value)
        expected, gold_span = None, None
        if question.span is not None:
            offsets = record.source.offsets
            gold_span = (
                offsets[question.span[0]][0],
                offsets[question.span[1] - 1][1],
            )
            expected = record.text[gold_span[0] : gold_span[1]]
        self._totals.add(name + "/token_f1", token_f1(predicted, expected))
        self._nulls["both"] += predicted is None and expected is None
        self._nulls["predicted"] += predicted is None
        self._nulls["expected"] += expected is None
        self._error(
            fields.Kind.EXTRACT,
            float(predicted != expected),
            float(expected is not None),
        )
        if gold_span is None:
            self._totals.add(
                name + "/false_positive", float(predicted is not None)
            )
            return
        self._totals.add(name + "/exact", float(predicted == expected))
        self._totals.add(
            name + "/accepted",
            float(predicted == expected or predicted in question.accepted),
        )
        self._totals.add(name + "/false_null", float(predicted is None))
        span = cast(tuple[int, int] | None, answer.get("character_span"))
        self._totals.add(
            name + "/span_iou", 0.0 if span is None else _iou(span, gold_span)
        )

    def means(self) -> dict[str, float]:
        """Return question means, counts, and the equal-type loss."""
        result = self._totals.means()
        extract, choice, binary, ordinal = self.names

        def put(name: str, value: float | None, count: int) -> None:
            if value is not None and count:
                result[name] = value
                result[name + "/count"] = float(count)

        if self._choice:
            put(
                choice + "/ece",
                expected_calibration_error(
                    [c for c, _ in self._choice],
                    [float(o) for _, o in self._choice],
                ),
                len(self._choice),
            )
        if self._binary:
            put(
                binary + "/ece",
                expected_calibration_error(
                    [p for p, _ in self._binary], [t for _, t in self._binary]
                ),
                len(self._binary),
            )
            put(
                binary + "/auroc",
                auroc(
                    [p for p, _ in self._binary],
                    [t >= 0.5 for _, t in self._binary],
                ),
                len(self._binary),
            )
        put(
            ordinal + "/spearman",
            spearman(
                [p for p, _ in self._ordinal], [g for _, g in self._ordinal]
            ),
            len(self._ordinal),
        )
        denominator = self._nulls["predicted"] + self._nulls["expected"]
        put(
            extract + "/null_f1",
            2 * self._nulls["both"] / denominator if denominator else None,
            self._nulls["expected"],
        )
        reductions = []
        for kind, pairs in sorted(self._errors.items()):
            baseline = sum(b for _, b in pairs)
            if baseline:
                reduction = 1 - sum(e for e, _ in pairs) / baseline
                put(
                    self.names[kind] + "/error_reduction", reduction, len(pairs)
                )
                reductions.append(reduction)
        if reductions:
            put(
                "error_reduction",
                sum(reductions) / len(reductions),
                sum(len(pairs) for pairs in self._errors.values()),
            )
        return result


# Metrics where a larger value is worse.
LOWER_IS_BETTER = (
    "loss",
    "false_positive",
    "false_null",
    "brier",
    "mae",
    "nll",
    "ece",
    "kl",
)
# Unbounded metrics compare relatively; rates and scores in [0, 1] compare
# by absolute difference, since a relative change from near zero explodes.
RELATIVE = ("loss", "nll", "mae", "kl")


def degradation(
    in_domain: dict[str, float], shifted: dict[str, float]
) -> dict[str, float]:
    """Change from in-domain to shifted metrics; positive is worse.

    Loss, NLL, MAE and KL report the relative change; every other metric,
    a rate or score in [0, 1], reports the absolute difference.
    """
    result = {}
    for name, base in in_domain.items():
        if name.endswith("/count") or name not in shifted:
            continue
        metric = name.rsplit("/", 1)[-1]
        sign = 1 if metric in LOWER_IS_BETTER else -1
        change = shifted[name] - base
        if metric in RELATIVE:
            if not base:
                continue
            change /= abs(base)
        result[name] = sign * change
    return result
