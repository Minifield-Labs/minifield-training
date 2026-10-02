"""Exhaustive short-span oracle and mocked typed output semantics."""

import dataclasses
import math

import pytest

from examples.magicbox import data
from examples.magicbox import smoke
from minifield_training.evaluation import field_decode as magicbox


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
    result = data.format_results(record, predictions)
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
