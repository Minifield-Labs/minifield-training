"""Model-neutral typed questions answered by pointing at input tokens."""

import dataclasses
import math

from minifield_training.datasets import fields


@dataclasses.dataclass(frozen=True)
class Option:
    """One answer a question may select; ``ids[0]`` is its marker token."""

    label: str
    ids: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Question:
    """A query marker, the options it may select, and optional supervision.

    Choice, ordinal, and binary questions answer with a distribution over
    ``options``. An extraction question has exactly one option, meaning the
    source doesn't answer it: ``targets[0]`` is that option's probability and
    ``span`` holds the answer's ``[start, end)`` source tokens otherwise.
    Unsupervised questions carry all-zero targets.
    """

    key: str
    kind: int
    query: tuple[int, ...]
    options: tuple[Option, ...]
    targets: tuple[float, ...]
    supervised: bool
    span: tuple[int, int] | None = None
    legend: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Record:
    """One source and every question asked about it."""

    id: str
    text: str
    source: fields.Encoding
    questions: tuple[Question, ...]

    @property
    def sequence_tokens(self) -> int:
        """Count the joint sequence: every query, option, and the source."""
        return len(self.source.ids) + sum(
            len(question.query)
            + sum(len(option.ids) for option in question.options)
            for question in self.questions
        )


def ordinal_targets(level: int, count: int, width: float) -> tuple[float, ...]:
    """Spread a hard ordinal label over nearby levels.

    Weights follow a Gaussian centred on ``level`` whose standard deviation is
    ``width`` times the scale's range, so scales with different level counts
    get comparable smoothing. Zero width or a single level stays one-hot.
    """
    if (
        count < 1
        or not 0 <= level < count
        or not math.isfinite(width)
        or width < 0
    ):
        raise ValueError("Invalid ordinal level, level count, or width")
    sigma = width * (count - 1)
    if sigma == 0:
        return tuple(float(index == level) for index in range(count))
    weights = [
        math.exp(-0.5 * ((index - level) / sigma) ** 2)
        for index in range(count)
    ]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _validate_targets(question: Question, source: fields.Encoding) -> None:
    """Check one question's distribution, span, and option inventory."""
    if len(question.targets) != len(question.options) or any(
        isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in question.targets
    ):
        raise ValueError(f"Invalid targets for question {question.key}")
    if not question.supervised:
        if any(question.targets) or question.span is not None:
            raise ValueError("Unsupervised questions carry no targets")
        return
    if question.kind == fields.Kind.EXTRACT:
        absent = question.targets[0]
        if absent not in (0, 1) or (absent == 0) != (question.span is not None):
            raise ValueError("Extraction needs a span exactly when present")
        if question.span is not None:
            start, end = question.span
            if not 0 <= start < end <= len(source.ids) or not all(
                source.selectable[start:end]
            ):
                raise ValueError("Invalid extraction token span")
    elif question.span is not None or not math.isclose(
        sum(question.targets), 1, abs_tol=1e-5
    ):
        raise ValueError("Option targets must be one distribution")


def validate(record: Record, *, vocab_size: int) -> None:
    """Admit token IDs, option inventories, and target distributions."""
    keys = [question.key for question in record.questions]
    if not record.id or not keys or len(set(keys)) != len(keys):
        raise ValueError("Record needs an identity and distinct questions")
    source = record.source
    if not source.ids or not (
        len(source.ids) == len(source.offsets) == len(source.special)
    ):
        raise ValueError("Source encoding lengths differ")
    for question in record.questions:
        labels = [option.label for option in question.options]
        if (
            not question.key
            or question.kind not in range(4)
            or isinstance(question.kind, bool)
            or not labels
            or len(set(labels)) != len(labels)
            or (question.kind == fields.Kind.EXTRACT and len(labels) != 1)
        ):
            raise ValueError(f"Invalid question or options: {question.key}")
        _validate_targets(question, source)
    for ids in (
        source.ids,
        *(question.query for question in record.questions),
        *(
            option.ids
            for question in record.questions
            for option in question.options
        ),
    ):
        if not ids or any(
            not isinstance(token, int)
            or isinstance(token, bool)
            or not 0 <= token < vocab_size
            for token in ids
        ):
            raise ValueError("Token outside configured vocabulary")
