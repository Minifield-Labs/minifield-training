"""Held-out loss and typed task metrics for the training notebook."""

from collections.abc import Callable
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.magicbox import source
from minifield_training.batching import magicbox as batching
from minifield_training.datasets import magicbox as data
from minifield_training.evaluation import magicbox as decoding
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import model
from minifield_training.objectives import magicbox as objective
from minifield_training.optimizers import state
from minifield_training.strategies import magicbox


@dataclasses.dataclass
class Metrics:
    """Accumulate per-field values without averaging unequal batches."""

    totals: dict[str, float] = dataclasses.field(default_factory=dict)
    counts: dict[str, int] = dataclasses.field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        """Add one field-level observation."""
        self.totals[name] = self.totals.get(name, 0.0) + value
        self.counts[name] = self.counts.get(name, 0) + 1

    def record(
        self,
        record: data.Record,
        decoded: dict[str, object],
        losses: list[float],
    ) -> None:
        """Compare typed predictions with labeled fields only."""
        row = 0
        for field in record.fields:
            if field.supervised:
                result = data.object_map(decoded[field.key])
                value = result["value"]
                self.add(data.KINDS[field.kind] + "/loss", losses[row])
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
                        self.add("extract/exact", float(value == expected))
                    if field.targets[0] == 0:
                        self.add(
                            "extract/false_positive", float(value is not None)
                        )
                    else:
                        self.add("extract/false_null", float(value is None))
                elif field.kind == 1:
                    expected_choice = field.candidates[
                        int(np.argmax(field.targets))
                    ]
                    self.add("choice/accuracy", float(value == expected_choice))
                elif field.kind == 2:
                    self.add(
                        "noul/brier",
                        (float(str(value)) - field.targets[0]) ** 2,
                    )
                else:
                    expectation = sum(
                        index * probability
                        for index, probability in enumerate(field.targets)
                    )
                    self.add("score/mae", abs(float(str(value)) - expectation))
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


class Evaluator:
    """Reuse compiled bucket forwards between checkpoint evaluations."""

    def __init__(
        self,
        corpus: source.Corpus,
        cfg: lfm.Config,
        fusion: model.Config,
        shape: batching.Shape,
        *,
        bf16: bool = True,
    ):
        self.corpus = corpus
        self.shape = dataclasses.replace(shape, microbatches=1, requests=1)

        def predict(
            params: types.Parameters, batch: types.DeviceBatch
        ) -> tuple[types.DeviceBatch, jax.Array]:
            outputs = magicbox.forward(params, cfg, fusion, batch, bf16=bf16)
            return outputs, objective.losses(outputs, batch)

        self.predict = jax.jit(predict)

    def run(
        self, params: types.Parameters, split: str, limit: int = 0
    ) -> dict[str, float]:
        """Evaluate a fixed sample; limit zero visits the full split."""
        dataset = self.corpus.split(split).shuffle(
            seed=1729, keep_in_memory=False
        )
        metrics = Metrics()
        count = min(limit, len(dataset)) if limit else len(dataset)
        for index in range(count):
            record = self.corpus.compile(dataset[index], split)
            packed = batching.build(
                [record], source.bucket([record], self.shape), seed=0, update=0
            )
            batch = {
                key: jnp.asarray(value[0])
                for key, value in packed.microbatches.items()
            }
            outputs, losses = self.predict(params, batch)
            arrays = {
                key: np.asarray(value)[0] for key, value in outputs.items()
            }
            decoded = decoding.decode(
                record,
                {
                    key: values.tolist()
                    for key, values in arrays.items()
                    if key != "tokens"
                },
                arrays["tokens"].tolist(),
            )
            metrics.record(record, decoded, np.asarray(losses)[0].tolist())
        return metrics.means()

    def callback(
        self, destination: Path, limit: int
    ) -> Callable[[state.State, int], dict[str, float]]:
        """Write validation reports after durable optimizer checkpoints."""

        def evaluate(current: state.State, step: int) -> dict[str, float]:
            metrics = self.run(current["params"], "validation", limit)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"validation-{step:08d}.json").write_text(
                json.dumps(metrics, indent=2)
            )
            return {
                "validation/" + key: value for key, value in metrics.items()
            }

        return evaluate
