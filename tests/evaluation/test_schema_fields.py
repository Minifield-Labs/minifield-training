"""Analytical field evaluation with an unrelated zero-logit model."""

import dataclasses
import json
import math
from pathlib import Path

import jax.numpy as jnp
import pytest

from minifield_training.batching import schema_fields as batching
from minifield_training.datasets import fields
from minifield_training.evaluation import schema_fields as evaluation
from minifield_training.kernels import types
from minifield_training.objectives import schema_fields as objective


def _record() -> fields.Record:
    """Construct neutral fields directly, with BOS 7 and pad 10."""
    return fields.Record(
        "observation",
        "xy",
        fields.Encoding(
            (7, 2, 3), ((0, 0), (0, 1), (1, 2)), (True, False, False)
        ),
        (
            fields.Field(
                "category",
                fields.Kind.CHOICE,
                ("a", "b"),
                ((7, 4), (7, 5)),
                (0.25, 0.75),
                True,
                False,
                None,
            ),
            fields.Field(
                "binary",
                fields.Kind.BINARY,
                ("",),
                ((7, 4),),
                (0.25,),
                True,
                False,
                None,
            ),
            fields.Field(
                "rank",
                fields.Kind.ORDINAL,
                ("0", "1", "2"),
                ((7, 4), (7, 5), (7, 6)),
                (0, 0.5, 0.5),
                True,
                False,
                None,
            ),
            fields.Field(
                "presence",
                fields.Kind.EXTRACT,
                ("",),
                ((7, 4),),
                (1,),
                True,
                False,
                None,
            ),
            fields.Field(
                "absence",
                fields.Kind.EXTRACT,
                ("",),
                ((7, 5),),
                (0,),
                True,
                True,
                None,
            ),
            fields.Field(
                "missing",
                fields.Kind.BINARY,
                ("",),
                ((7, 6),),
                (0,),
                False,
                False,
                None,
            ),
        ),
    )


def _zeros(
    parameters: types.Parameters, batch: types.DeviceBatch
) -> types.DeviceBatch:
    """Use shape-only independent logits without an encoder or product model."""
    del parameters
    requests, rows = batch["schema_ids"].shape[:2]
    return {
        **{
            name: jnp.zeros((requests, rows))
            for name in ("candidate", "binary", "presence")
        },
        "tokens": jnp.zeros((requests, rows, batch["source_ids"].shape[-1])),
    }


@pytest.mark.parametrize("fixed_shape", (False, True))
def test_uneven_counts_partial_labels_and_soft_targets(
    tmp_path: Path, fixed_shape: bool
) -> None:
    """Aggregate per-field observations before averaging active task losses."""
    first = _record()
    second = dataclasses.replace(
        first,
        id="second",
        fields=(dataclasses.replace(first.fields[1], targets=(1,)),),
    )
    records = [first, second]
    predictor = evaluation.Predictor(
        _zeros,
        batching.SchemaBatchStrategy(
            batching.Shape(4, 8, 8, 8, 16, 11, 10),
            objective.balance_types,
            fixed_shape=fixed_shape,
        ),
    )
    evaluator = evaluation.Evaluator(
        lambda _split, limit: records[:limit] if limit else records, predictor
    )
    metrics = evaluator.run({}, "heldout")
    assert metrics["binary/brier"] == pytest.approx(0.15625)
    assert metrics["binary/brier/count"] == 2
    assert metrics["choice/accuracy"] == 0
    assert metrics["choice/accuracy/count"] == 1
    assert metrics["ordinal/mae"] == pytest.approx(0.5)
    assert metrics["extract/exact"] == 1
    assert metrics["extract/exact/count"] == 1
    assert metrics["extract/false_positive"] == 0
    assert metrics["extract/false_null"] == 1
    assert metrics["extract/loss/count"] == 2
    assert metrics["loss"] == pytest.approx(
        (3.5 * math.log(2) + math.log(3)) / 4
    )
    assert evaluator.run({}, "heldout", 1)["binary/brier"] == pytest.approx(
        0.0625
    )
    reports = tmp_path / "metrics"
    result = evaluator.callback(reports, 0)({"params": {}}, 3)
    assert result["validation/loss"] == metrics["loss"]
    assert (
        json.loads((reports / "validation-00000003.json").read_text())
        == metrics
    )


def test_same_predictor_handles_unlabeled_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference retains source and candidate identity with absent targets."""
    source = _record()
    record = dataclasses.replace(
        source,
        fields=tuple(
            dataclasses.replace(field, supervised=False, token_supervised=False)
            for field in source.fields
        ),
    )
    predictor = evaluation.Predictor(
        _zeros,
        batching.SchemaBatchStrategy(
            batching.Shape(1, 1, 8, 8, 16, 11, 10), objective.balance_types
        ),
    )

    def reject_loss(*_args: object) -> None:
        raise AssertionError("Inference must not compute supervised losses")

    monkeypatch.setattr(objective, "losses", reject_loss)
    values, losses = predictor.score({}, record, include_losses=False)
    assert values["category"] == {
        "value": "a",
        "confidence": None,
        "logits": [0, 0],
        "probabilities": {"a": 0.5, "b": 0.5},
    }
    assert (
        isinstance(values["presence"], dict)
        and values["presence"]["value"] is None
    )
    assert losses == []
