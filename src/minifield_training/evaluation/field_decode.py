"""Deterministic typed decoding and exact source-substring extraction."""

from collections.abc import Mapping, Sequence
import math

from minifield_training.datasets import fields


def best_span(
    logits: Sequence[float], selectable: Sequence[bool]
) -> tuple[int, int] | None:
    """Find the positive maximum-sum interval; ties prefer short then early."""
    best: tuple[float, int, int] = (0, 0, 0)
    result = None
    running, start = 0.0, 0
    for index, (value, valid) in enumerate(
        zip(logits, selectable, strict=True)
    ):
        if not math.isfinite(value):
            raise ValueError("Nonfinite extraction logit")
        if not valid:
            running, start = 0.0, index + 1
            continue
        if running <= 0:
            running, start = 0.0, index
        running += value
        candidate = (running, -(index + 1 - start), -start)
        if running > 0 and candidate > best:
            best, result = candidate, (start, index + 1)
    return result


def sigmoid(value: float) -> float:
    """Compute a stable scalar Bernoulli probability."""
    if not math.isfinite(value):
        raise ValueError("Nonfinite scalar logit")
    exp = math.exp(-abs(value))
    return 1 / (1 + exp) if value >= 0 else exp / (1 + exp)


def probabilities(values: Sequence[float]) -> list[float]:
    """Normalize exactly one candidate group in FP64 host arithmetic."""
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Invalid candidate logits")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def decode(
    record: fields.Record,
    logits: Mapping[str, Sequence[float]],
    token_logits: Sequence[Sequence[float]],
    *,
    presence_threshold: float = 0.5,
) -> dict[str, object]:
    """Return typed values and raw diagnostics."""
    if not 0 <= presence_threshold <= 1:
        raise ValueError("Invalid presence threshold")
    result: dict[str, object] = {}
    row = 0
    for field in record.fields:
        count = len(field.rows)
        details: dict[str, object] = {"confidence": None}
        if field.kind in (1, 3):
            values = list(logits["candidate"][row : row + count])
            probs = probabilities(values)
            value: object = (
                field.candidates[
                    max(enumerate(probs), key=lambda item: item[1])[0]
                ]
                if field.kind == 1
                else sum(
                    index * probability
                    for index, probability in enumerate(probs)
                )
            )
            details.update(
                logits=values,
                probabilities=dict(zip(field.candidates, probs, strict=True)),
            )
        elif field.kind == 2:
            value = sigmoid(logits["binary"][row])
            details.update(logit=logits["binary"][row])
        else:
            present = sigmoid(logits["presence"][row])
            tokens = list(token_logits[row][: len(record.source.ids)])
            span = (
                best_span(tokens, record.source.selectable)
                if present >= presence_threshold
                else None
            )
            value = None
            if span is not None:
                start, end = (
                    record.source.offsets[span[0]][0],
                    record.source.offsets[span[1] - 1][1],
                )
                value = record.text[start:end]
                details.update(
                    token_span=span,
                    character_span=(start, end),
                    span_score=sum(tokens[span[0] : span[1]]),
                )
            details.update(presence=present, token_logits=tokens)
        result[field.key] = {"value": value, **details}
        row += count
    return result
