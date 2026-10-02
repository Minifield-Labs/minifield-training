"""Reusable typed-field metrics and evaluation over injected model execution."""

from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping
import dataclasses
import json
from pathlib import Path
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np

from minifield_training.batching import contracts
from minifield_training.batching import schema_fields as batching
from minifield_training.core import json_io
from minifield_training.datasets import fields
from minifield_training.evaluation import field_decode
from minifield_training.kernels import types
from minifield_training.objectives import schema_fields as objective


@dataclasses.dataclass
class Metrics:
    """Accumulate per-field values without averaging unequal batches."""

    names: tuple[str, str, str, str] = (
        "extract",
        "choice",
        "binary",
        "ordinal",
    )
    totals: dict[str, float] = dataclasses.field(default_factory=dict)
    counts: dict[str, int] = dataclasses.field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        """Add one field-level observation."""
        self.totals[name] = self.totals.get(name, 0.0) + value
        self.counts[name] = self.counts.get(name, 0) + 1

    def record(
        self,
        record: fields.Record,
        decoded: dict[str, object],
        losses: list[float],
    ) -> None:
        """Compare typed predictions with labeled fields only."""
        row = 0
        for field in record.fields:
            if field.supervised:
                result = json_io.object_map(decoded[field.key])
                value = result["value"]
                self.add(self.names[field.kind] + "/loss", losses[row])
                if field.kind == 0:
                    expected = (
                        None
                        if field.span is None
                        else record.text[
                            record.source.offsets[field.span[0]][
                                0
                            ] : record.source.offsets[field.span[1] - 1][1]
                        ]
                    )
                    if field.token_supervised:
                        self.add(
                            self.names[0] + "/exact", float(value == expected)
                        )
                    if field.targets[0] == 0:
                        self.add(
                            self.names[0] + "/false_positive",
                            float(value is not None),
                        )
                    else:
                        self.add(
                            self.names[0] + "/false_null", float(value is None)
                        )
                elif field.kind == 1:
                    expected_choice = field.candidates[
                        int(np.argmax(field.targets))
                    ]
                    self.add(
                        self.names[1] + "/accuracy",
                        float(value == expected_choice),
                    )
                elif field.kind == 2:
                    self.add(
                        self.names[2] + "/brier",
                        (float(str(value)) - field.targets[0]) ** 2,
                    )
                else:
                    expectation = sum(
                        index * probability
                        for index, probability in enumerate(field.targets)
                    )
                    self.add(
                        self.names[3] + "/mae",
                        abs(float(str(value)) - expectation),
                    )
            row += len(field.rows)

    def means(self) -> dict[str, float]:
        """Return field means, counts, and the equal-type held-out loss."""
        result = {
            name: value / self.counts[name]
            for name, value in self.totals.items()
        }
        type_losses = [
            value for name, value in result.items() if name.endswith("/loss")
        ]
        if type_losses:
            result["loss"] = sum(type_losses) / len(type_losses)
        result.update(
            {
                name + "/count": float(count)
                for name, count in self.counts.items()
            }
        )
        return result


class SingleRequest:
    """Jit one forward with and without per-row losses for single requests."""

    def __init__(
        self,
        forward: Callable[
            [types.Parameters, types.DeviceBatch], types.DeviceBatch
        ],
        losses: Callable[[types.DeviceBatch, types.DeviceBatch], jax.Array],
    ):
        self.forward = jax.jit(forward)

        def predict(
            params: types.Parameters, batch: types.DeviceBatch
        ) -> tuple[types.DeviceBatch, jax.Array]:
            outputs = forward(params, batch)
            return outputs, losses(outputs, batch)

        self.predict = jax.jit(predict)

    def run(
        self,
        params: types.Parameters,
        packed: contracts.PhysicalUpdate,
        include_losses: bool,
    ) -> tuple[dict[str, list[object]], list[float]]:
        """Return the first request's outputs as lists, and optional losses."""
        batch = {
            key: jnp.asarray(value[0])
            for key, value in packed.microbatches.items()
        }
        if include_losses:
            outputs, losses = self.predict(params, batch)
            request_losses = np.asarray(losses)[0].tolist()
        else:
            outputs = self.forward(params, batch)
            request_losses = []
        return {
            key: np.asarray(value)[0].tolist() for key, value in outputs.items()
        }, request_losses


class Predictor(SingleRequest):
    """Share packing, forward execution, and decoding across prediction uses."""

    def __init__(
        self,
        forward: Callable[
            [types.Parameters, types.DeviceBatch], types.DeviceBatch
        ],
        batches: batching.SchemaBatchStrategy,
        *,
        presence_threshold: float = 0.5,
    ):
        super().__init__(forward, objective.losses)
        self.batches = dataclasses.replace(
            batches,
            shape=dataclasses.replace(
                batches.shape, microbatches=1, requests=1
            ),
        )
        self.presence_threshold = presence_threshold

    def score(
        self,
        params: types.Parameters,
        record: fields.Record,
        *,
        include_losses: bool = True,
    ) -> tuple[dict[str, object], list[float]]:
        """Return typed predictions and optional per-field losses."""
        arrays, field_losses = self.run(
            params,
            self.batches.pack(
                [record], seed=0, update=0, allow_unsupervised=True
            ),
            include_losses,
        )
        decoded = field_decode.decode(
            record,
            cast(
                dict[str, list[float]],
                {
                    key: values
                    for key, values in arrays.items()
                    if key != "tokens"
                },
            ),
            cast(list[list[float]], arrays["tokens"]),
            presence_threshold=self.presence_threshold,
        )
        return decoded, field_losses


class Scorer[RecordT](Protocol):
    """Typed predictions and per-question losses for one held-out record."""

    @abstractmethod
    def score(
        self, params: types.Parameters, record: RecordT
    ) -> tuple[dict[str, object], list[float]]:
        """Predict one record."""


class Recorder[RecordT](Protocol):
    """Accumulate question metrics for one evaluated split."""

    @abstractmethod
    def record(
        self,
        record: RecordT,
        decoded: dict[str, object],
        losses: list[float],
    ) -> None:
        """Add one record's questions."""

    @abstractmethod
    def means(self) -> dict[str, float]:
        """Return metric means and counts."""


@dataclasses.dataclass
class Evaluator[RecordT]:
    """Aggregate held-out records supplied by any admitted corpus adapter.

    ``metrics`` builds the split's recorder from the type names; the default
    compares per-row schema-field predictions.
    """

    records: Callable[[str, int], Iterable[RecordT]]
    predictor: Scorer[RecordT]
    names: tuple[str, str, str, str] = (
        "extract",
        "choice",
        "binary",
        "ordinal",
    )
    metrics: Callable[[tuple[str, str, str, str]], Recorder[RecordT]] | None = (
        None
    )

    def run(
        self, params: types.Parameters, split: str, limit: int = 0
    ) -> dict[str, float]:
        """Evaluate a sample with field-level metric denominators."""
        metrics = (
            cast(Recorder[RecordT], Metrics(names=self.names))
            if self.metrics is None
            else self.metrics(self.names)
        )
        for record in self.records(split, limit):
            predictions, losses = self.predictor.score(params, record)
            metrics.record(record, predictions, losses)
        return metrics.means()

    def callback(
        self, destination: Path, limit: int
    ) -> Callable[[Mapping[str, object], int], dict[str, float]]:
        """Write validation reports after the engine commits a checkpoint."""

        def evaluate(
            current: Mapping[str, object], step: int
        ) -> dict[str, float]:
            metrics = self.run(
                cast(types.Parameters, current["params"]), "validation", limit
            )
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"validation-{step:08d}.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            return {
                "validation/" + key: value for key, value in metrics.items()
            }

        return evaluate
