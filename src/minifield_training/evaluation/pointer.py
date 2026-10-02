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


class Metrics:
    """Compare typed answers with supervised questions, per question type."""

    def __init__(self, names: tuple[str, str, str, str]):
        self.names = names
        self._totals = schema_fields.Metrics(names=names)

    def record(
        self,
        record: pointer.Record,
        decoded: dict[str, object],
        losses: list[float],
    ) -> None:
        """Add each supervised question's loss and type-specific metric."""
        for index, question in enumerate(record.questions):
            if not question.supervised:
                continue
            name = self.names[question.kind]
            value = json_io.object_map(decoded[question.key])["value"]
            self._totals.add(name + "/loss", losses[index])
            targets = question.targets
            if question.kind == fields.Kind.EXTRACT:
                if question.span is None:
                    self._totals.add(
                        name + "/false_positive", float(value is not None)
                    )
                    continue
                offsets = record.source.offsets
                expected = record.text[
                    offsets[question.span[0]][0] : offsets[
                        question.span[1] - 1
                    ][1]
                ]
                self._totals.add(name + "/exact", float(value == expected))
                self._totals.add(name + "/false_null", float(value is None))
            elif question.kind == fields.Kind.CHOICE:
                expected = question.options[int(np.argmax(targets))].label
                self._totals.add(name + "/accuracy", float(value == expected))
            elif question.kind == fields.Kind.BINARY:
                labels = [option.label for option in question.options]
                truth = targets[labels.index("true")]
                self._totals.add(
                    name + "/brier", (float(str(value)) - truth) ** 2
                )
            else:
                expectation = sum(
                    level * probability
                    for level, probability in enumerate(targets)
                )
                self._totals.add(
                    name + "/mae", abs(float(str(value)) - expectation)
                )

    def means(self) -> dict[str, float]:
        """Return question means, counts, and the equal-type loss."""
        return self._totals.means()
