"""Ordinal spreading and pointer record admission."""

import dataclasses
import math

import pytest

from minifield_training.datasets import fields
from minifield_training.datasets import pointer


def test_ordinal_targets_follow_a_scaled_gaussian() -> None:
    """Hand-computed weights; zero width and one level stay one-hot."""
    # 5 levels, width 0.25: sigma = 1 level, so neighbours get e^-0.5.
    expected = [math.exp(-0.5 * (index - 2) ** 2) for index in range(5)]
    total = sum(expected)
    assert pointer.ordinal_targets(2, 5, 0.25) == pytest.approx(
        [value / total for value in expected]
    )
    assert pointer.ordinal_targets(0, 3, 0.0) == (1.0, 0.0, 0.0)
    assert pointer.ordinal_targets(0, 1, 0.5) == (1.0,)
    edge = pointer.ordinal_targets(0, 5, 0.25)
    assert edge[0] > edge[1] > edge[2] and sum(edge) == pytest.approx(1)
    for level, count, width in ((3, 3, 0.1), (0, 0, 0.1), (0, 3, -1.0)):
        with pytest.raises(ValueError, match="Invalid ordinal"):
            pointer.ordinal_targets(level, count, width)


def _record() -> pointer.Record:
    source = fields.Encoding(
        (1, 5, 6), ((0, 0), (0, 3), (4, 7)), (True, False, False)
    )
    return pointer.Record(
        "r",
        "Ada Lee",
        source,
        (
            pointer.Question(
                "name",
                fields.Kind.EXTRACT,
                (1, 9),
                (pointer.Option("absent", (1, 8)),),
                (0.0,),
                True,
                (1, 3),
            ),
            pointer.Question(
                "tier",
                fields.Kind.CHOICE,
                (1, 10),
                (pointer.Option("a", (1, 11)), pointer.Option("b", (1, 12))),
                (0.25, 0.75),
                True,
            ),
        ),
    )


def test_valid_record_counts_its_joint_sequence() -> None:
    """Queries, options, and the source all enter the joint sequence."""
    record = _record()
    pointer.validate(record, vocab_size=16)
    assert record.sequence_tokens == 3 + (2 + 2) + (2 + 2 + 2)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"targets": (0.25, 0.5)}, "one distribution"),
        ({"targets": (0.25,)}, "Invalid targets"),
        ({"options": (pointer.Option("a", (1,)),) * 2}, "Invalid question"),
        ({"supervised": False}, "carry no targets"),
        ({"query": (1, 99)}, "vocabulary"),
    ],
)
def test_choice_admission(change: dict[str, object], message: str) -> None:
    """Malformed option inventories and targets fail before batching."""
    record = _record()
    broken = dataclasses.replace(
        record.questions[1], **change  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match=message):
        pointer.validate(
            dataclasses.replace(
                record, questions=(record.questions[0], broken)
            ),
            vocab_size=16,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"span": None}, "span exactly when present"),
        ({"targets": (1.0,)}, "span exactly when present"),
        ({"span": (0, 2)}, "Invalid extraction token span"),
        ({"targets": (0.5,)}, "span exactly when present"),
    ],
)
def test_extraction_admission(change: dict[str, object], message: str) -> None:
    """A present answer needs a selectable span; an absent one has none."""
    record = _record()
    broken = dataclasses.replace(
        record.questions[0], **change  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match=message):
        pointer.validate(
            dataclasses.replace(
                record, questions=(broken, record.questions[1])
            ),
            vocab_size=16,
        )
