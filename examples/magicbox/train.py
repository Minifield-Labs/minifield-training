"""Train the joint pointer model with frozen embeddings and exact resume.

The notebook and this CLI share one composition: ``prepare`` builds the run,
``initial_state`` loads or restores it, and ``final_evaluation`` and
``save_bundle`` finish it. Each caller only chooses settings and staging.
"""

import argparse
from collections.abc import Iterable, Mapping
import dataclasses
import functools
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from examples.magicbox import bundle as magicbox_bundle
from examples.magicbox import composition as magicbox
from examples.magicbox import data
from examples.magicbox import source
from minifield_training.batching import pointer as batching
from minifield_training.batching import stream as streams
from minifield_training.checkpoints import discovery
from minifield_training.checkpoints import training_state
from minifield_training.core import json_io
from minifield_training.core import parameters as core_parameters
from minifield_training.datasets import pointer as pointer_records
from minifield_training.engine import step as engine_step
from minifield_training.engine import training_run
from minifield_training.evaluation import pointer as evaluate_pointer
from minifield_training.evaluation import schema_fields as evaluate
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import pointer
from minifield_training.objectives import pointer as objective
from minifield_training.objectives import schema_fields as weighting
from minifield_training.optimizers import adamw
from minifield_training.optimizers import optax_adamw
from minifield_training.optimizers import state as optimizer_state
from minifield_training.strategies import pretrained
from minifield_training.strategies import schema_fields as strategy

SPLITS = ("validation", "calibration", "test", "ood")


@dataclasses.dataclass(frozen=True)
class Settings:
    """Choices that define a run; the notebook and CLI each fill these."""

    dataset: Path
    model_dir: Path
    cache: Path
    devices: int
    platform: str = "tpu"
    rows: int = 8
    microbatches: int = 4
    pack: bool = True
    questions_per_row: int = 32
    prefetch: int = 2
    # Joint tokens and questions per request; None measures every split.
    sequence_tokens: int | None = None
    questions: int | None = None
    score_width: float = 0.15
    epochs: int = 3
    learning_rate: float = 2e-5
    optimizer: str = "optax"
    seed: int = 20260927
    run_id: str = "magicbox-lfm350m-v1"
    bf16: bool = True
    allow_sample: bool = False


@dataclasses.dataclass(frozen=True)
class Run:  # pylint: disable=too-many-instance-attributes
    """Everything a session needs, bound to one dataset and identity."""

    settings: Settings
    devices: tuple[jax.Device, ...]
    mesh: jax.sharding.Mesh | None
    corpus: source.Corpus
    cfg: lfm.Config
    head: pointer.Config
    shape: batching.Shape
    batches: batching.PointerBatchStrategy
    sizes: list[tuple[int, int]] | None
    stream: (
        streams.EpochStream[object, pointer_records.Record]
        | streams.PlannedStream[object, pointer_records.Record]
    )
    inventory: core_parameters.FullParameterInventory
    optimizer: adamw.AdamWConfig
    optimizer_id: str
    transaction: engine_step.Transaction
    identity: dict[str, object]
    source_id: str

    @property
    def config_path(self) -> Path:
        """The pinned encoder config this run was verified against."""
        return self.settings.model_dir / "config.json"


def prepare(settings: Settings) -> Run:
    """Verify the data and encoder, then fix the shape, stream and identity."""
    devices = training_run.require_devices(settings.devices, settings.platform)
    if settings.rows % settings.devices or settings.epochs < 1:
        raise ValueError("Invalid request topology or epoch count")
    if settings.optimizer not in ("optax", "transactional"):
        raise ValueError("optimizer must be 'optax' or 'transactional'")
    corpus = source.Corpus(
        settings.dataset, settings.cache, allow_sample=settings.allow_sample
    )
    config_path = settings.model_dir / "config.json"
    if json_io.digest_file(config_path) != encoder.SOURCE.config_sha256:
        raise ValueError("Pretrained config changed")
    cfg = encoder.Adapter().parse_config(
        json_io.object_map(json.loads(config_path.read_text()))
    )
    head = pointer.Config(encoder_width=cfg.hidden_size)
    tokens, questions = settings.sequence_tokens, settings.questions
    if tokens is None or questions is None:
        # Evaluation packs with the training shape, so every split must fit.
        measured_tokens, measured_questions = corpus.pointer_extent(
            sorted({str(shard["split"]) for shard in corpus.shards})
        )
        tokens = tokens or -(-measured_tokens // 128) * 128
        questions = questions or measured_questions
    if settings.pack and questions > settings.questions_per_row:
        raise ValueError(
            f"questions_per_row must fit a whole request ({questions})"
        )
    shape = batching.Shape(
        settings.microbatches,
        settings.rows,
        tokens,
        settings.questions_per_row if settings.pack else questions,
        cfg.vocab_size,
        0,
    )
    if shape.sequence_tokens > encoder.MAX_SEQUENCE_LENGTH:
        raise ValueError("Requested tokens exceed the encoder context")
    batches = batching.PointerBatchStrategy(shape, weighting.balance_types)
    compile_train = functools.partial(
        corpus.compile_pointer, split="train", score_width=settings.score_width
    )
    sizes = corpus.pointer_sizes("train") if settings.pack else None
    stream: (
        streams.EpochStream[object, pointer_records.Record]
        | streams.PlannedStream[object, pointer_records.Record]
    ) = (
        source.planned_training_stream(
            corpus,
            batches,
            sizes,
            settings.seed,
            settings.epochs,
            compile_train,
            prefetch=settings.prefetch,
        )
        if sizes is not None
        else source.training_stream(
            corpus,
            batches,
            settings.seed,
            settings.epochs,
            compile_train,
            prefetch=settings.prefetch,
        )
    )
    optimizer = adamw.AdamWConfig(learning_rate=settings.learning_rate)
    optax = settings.optimizer == "optax"
    identity: dict[str, object] = {
        "source": dataclasses.asdict(encoder.SOURCE),
        "encoder": dataclasses.asdict(cfg),
        "head": dataclasses.asdict(head),
        "shape": {
            key: value
            for key, value in dataclasses.asdict(shape).items()
            if key not in ("vocab_size", "pad_token_id")
        },
        "seed": settings.seed,
        "bf16": settings.bf16,
        "devices": settings.devices,
        "contract": data.FORMAT,
        "template": data.POINTER_TEMPLATE,
        "score_width": settings.score_width,
        "packing": {
            "pack": settings.pack,
            "planner": "first-fit/1",
            "open_limit": 64,
            "close_below": 0.05,
        },
        "implementation": "magicbox-pointer-jax/1",
    }
    return Run(
        settings=settings,
        devices=devices,
        mesh=(
            jax.sharding.Mesh(np.asarray(devices), ("data",))
            if settings.devices > 1
            else None
        ),
        corpus=corpus,
        cfg=cfg,
        head=head,
        shape=shape,
        batches=batches,
        sizes=sizes,
        stream=stream,
        inventory=magicbox.pointer_inventory(cfg, head),
        optimizer=optimizer,
        optimizer_id=(
            optax_adamw.implementation_identity(optimizer)
            if optax
            else optimizer.implementation_identity
        ),
        transaction=(
            optax_adamw.make_transaction if optax else adamw.make_transaction
        ),
        identity=identity,
        source_id=hashlib.sha256(
            json_io.canonical(identity).encode()
        ).hexdigest(),
    )


def latest(run: Run, output: Path) -> Path | None:
    """The newest checkpoint in ``output`` that matches this run exactly."""
    return discovery.latest_checkpoint(
        output / "checkpoints",
        run_id=run.settings.run_id,
        data_sha256=run.corpus.identity,
        source_id=run.source_id,
        reject_mismatched=True,
    )


def initial_state(
    run: Run, resume: Path | None
) -> tuple[optimizer_state.State, training_state.Cursor]:
    """Restore ``resume`` exactly, or start from the pretrained encoder."""
    if resume is not None:
        return training_state.load(
            resume,
            run.inventory,
            optimizer_id=run.optimizer_id,
            run_id=run.settings.run_id,
            data_sha256=run.corpus.identity,
            source_id=run.source_id,
        )
    _, parameters = pretrained.load_verified(
        run.settings.model_dir, encoder.SOURCE, encoder.Adapter()
    )
    parameters.update(
        pointer.initialize(run.head, jax.random.PRNGKey(run.settings.seed))
    )
    cursor = training_state.Cursor(
        run.settings.run_id, run.corpus.identity, run.source_id, 0
    )
    return adamw.initialize_state(parameters, run.inventory), cursor


def write_run(run: Run, output: Path) -> None:
    """Record the run identity next to its checkpoints."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(
        json.dumps(
            {
                **run.identity,
                "optimizer": {
                    "transaction": run.settings.optimizer,
                    **dataclasses.asdict(run.optimizer),
                },
                "data_sha256": run.corpus.identity,
                "source_id": run.source_id,
                "total_updates": run.stream.total_updates,
            },
            indent=2,
        )
    )


def make_evaluator(run: Run) -> evaluate.Evaluator[pointer_records.Record]:
    """Held-out evaluation with the training shape and precision."""
    return evaluate.Evaluator(
        run.corpus.pointer_records,
        evaluate_pointer.Predictor(
            magicbox.bind_pointer(run.cfg, run.head, bf16=run.settings.bf16),
            run.batches,
        ),
        names=data.KINDS,
        metrics=evaluate_pointer.Metrics,
    )


def make_step(run: Run) -> engine_step.JitStep:
    """The jitted training update for this run's model and optimizer."""
    return strategy.make_step(
        magicbox.bind_pointer(run.cfg, run.head, bf16=run.settings.bf16),
        run.inventory,
        run.optimizer,
        mesh=run.mesh,
        terms=objective.terms,
        transaction=run.transaction,
    )


def combine(parts: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Pool per-source metric means by their counts."""
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for metrics in parts:
        for name, value in metrics.items():
            count = metrics.get(f"{name}/count")
            if name.endswith("/count") or not count:
                continue
            sums[name] = sums.get(name, 0.0) + value * count
            counts[name] = counts.get(name, 0) + count
    return {
        **{name: total / counts[name] for name, total in sums.items()},
        **{f"{name}/count": count for name, count in counts.items()},
    }


def final_evaluation(
    run: Run,
    evaluator: evaluate.Evaluator[pointer_records.Record],
    params: types.Parameters,
    output: Path,
    limit: int,
) -> None:
    """Evaluate each held-out source separately; ``limit`` 0 reads all.

    ``final-<split>.json`` holds each source's metrics and an ``all`` entry
    pooled by count.
    """
    for split in SPLITS:
        if not any(shard["split"] == split for shard in run.corpus.shards):
            continue
        by_source = {}
        for name in sorted(set(run.corpus.pointer_sources(split))):
            reader = functools.partial(run.corpus.pointer_records, source=name)
            by_source[name] = dataclasses.replace(
                evaluator, records=reader
            ).run(params, split, limit)
            print(
                json.dumps({"split": split, "source": name, **by_source[name]}),
                flush=True,
            )
        metrics = {"all": combine(by_source.values()), **by_source}
        (output / f"final-{split}.json").write_text(
            json.dumps(metrics, indent=2)
        )
        print(
            json.dumps({"split": split, "source": "all", **metrics["all"]}),
            flush=True,
        )


def save_bundle(
    run: Run, params: types.Parameters, output: Path, step: int
) -> Path:
    """Export the inference bundle once per run identity and step."""
    # The run identity keeps bundles from other settings in one output apart.
    bundle = output / f"bundle-{run.source_id[:12]}-{step:08d}"
    if not bundle.exists():
        magicbox_bundle.save_pointer(
            bundle,
            params,
            run.cfg,
            run.head,
            encoder_config=run.config_path,
            tokenizer=run.settings.dataset / "tokenizer",
            step=step,
        )
    return bundle


def arguments() -> argparse.Namespace:
    """Expose full-run bounds, hardware, and persistent output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "model-dir", "output", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--platform", choices=("cpu", "tpu"), default="tpu")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--questions-per-row", type=int, default=32)
    parser.add_argument("--no-pack", action="store_true")
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument(
        "--optimizer", choices=("optax", "transactional"), default="optax"
    )
    parser.add_argument("--microbatches", type=int, default=4)
    # Joint sequence tokens and questions per request; omitted means measure
    # every split and round tokens up to a multiple of 128.
    parser.add_argument("--sequence-tokens", type=int)
    parser.add_argument("--questions", type=int)
    parser.add_argument("--score-width", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-hours", type=float, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--validation-records", type=int, default=256)
    parser.add_argument("--final-records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--run-id", default="magicbox-lfm350m-v1")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--allow-sample", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Verify assets, recover state, run the shared lifecycle, and export."""
    args = arguments()
    if args.validation_records < 1 or args.final_records < 0:
        raise ValueError("Invalid evaluation bound")
    run = prepare(
        Settings(
            dataset=args.dataset,
            model_dir=args.model_dir,
            cache=args.cache,
            devices=args.devices,
            platform=args.platform,
            rows=args.rows,
            microbatches=args.microbatches,
            pack=not args.no_pack,
            questions_per_row=args.questions_per_row,
            prefetch=args.prefetch,
            sequence_tokens=args.sequence_tokens,
            questions=args.questions,
            score_width=args.score_width,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            seed=args.seed,
            run_id=args.run_id,
            bf16=not args.fp32,
            allow_sample=args.allow_sample,
        )
    )
    output = args.output
    resume = latest(run, output)
    if args.evaluate_only and not resume:
        raise ValueError("Evaluation requires a trained checkpoint")
    current, cursor = initial_state(run, resume)
    write_run(run, output)
    evaluator = make_evaluator(run)
    remaining = run.stream.total_updates - cursor.next_batch
    if args.max_steps is not None:
        remaining = min(remaining, args.max_steps)
    if remaining > 0 and not args.evaluate_only:
        with (output / "progress.jsonl").open("a", encoding="utf-8") as log:

            def report(event: dict[str, float | str]) -> None:
                line = json.dumps(event)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()

            current, cursor = training_run.run(
                None,
                current,
                make_step(run),
                run.inventory,
                training_run.RunConfig(
                    args.seed,
                    args.checkpoint_every,
                    10,
                    max_steps=remaining,
                    max_seconds=args.max_hours * 3600,
                ),
                checkpoint_root=output / "checkpoints",
                optimizer_id=run.optimizer_id,
                cursor=cursor,
                evaluate=evaluator.callback(
                    output / "metrics", args.validation_records
                ),
                report=report,
                required_platform=args.platform,
                batch_source=run.stream,
                strict_compiles=True,
            )
    completed = cursor.next_batch >= run.stream.total_updates
    # Save the bundle first, so a session that ends during evaluation keeps it.
    bundle = save_bundle(run, current["params"], output, cursor.next_batch)
    if completed or args.evaluate_only:
        final_evaluation(
            run, evaluator, current["params"], output, args.final_records
        )
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "completed_epochs": completed,
                "next_batch": cursor.next_batch,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
