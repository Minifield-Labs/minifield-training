"""Exhaustive short-span oracle and mocked typed output semantics."""

import dataclasses
import math
import random

import pytest

from examples.magicbox import smoke
from minifield_training.evaluation import magicbox


def test_linear_span_decoder_matches_exhaustive_oracle() -> None:
    """Compare masks, ties, and negative gaps against enumeration."""
    rng = random.Random(7)
    for _ in range(600):
        size = rng.randint(0, 12)
        values = [rng.choice([-3.0, -1.0, 0.0, 1.0, 3.0]) for _ in range(size)]
        mask = [rng.random() > 0.2 for _ in range(size)]
        candidates = [
            (sum(values[start:end]), -(end - start), -start, start, end)
            for start in range(size)
            for end in range(start + 1, size + 1)
            if all(mask[start:end]) and sum(values[start:end]) > 0
        ]
        best = max(candidates) if candidates else None
        expected = (best[3], best[4]) if best else None
        assert magicbox.best_span(values, mask) == expected
    assert magicbox.best_span([2, -1, 2], [True] * 3) == (0, 3)


def test_typed_rubric_and_original_substring() -> None:
    """An expected score of 1.9 preserves rubric labels and null confidence."""
    record = smoke.fixture()
    outputs = {
        "candidate": [0.0] * 4 + [-1000.0, math.log(0.1), math.log(0.9)],
        "binary": [0.0] * 3 + [1000.0] + [0.0] * 3,
        "presence": [1000.0] + [0.0] * 6,
    }
    tokens = [[-2.0] * len(record.source.ids) for _ in range(7)]
    tokens[0][0], tokens[0][1] = 100.0, 3.0
    predictions = magicbox.decode(record, outputs, tokens)
    result = magicbox.format_results(record, predictions)
    assert result["person"] == {
        "type": "extract",
        "extract": "Ada",
        "confidence": None,
    }
    score = result["amount"]
    assert isinstance(score, dict)
    assert score["score"] == pytest.approx(1.9)
    assert score["legend"] == {"0": "None", "1": "One", "2": "Several"}
    empty = dataclasses.replace(record, text="", source=smoke.toy_encode(""))
    null = magicbox.decode(empty, outputs, [[100.0]] * 7)
    assert isinstance(null["person"], dict) and null["person"]["value"] is None
