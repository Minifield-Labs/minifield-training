"""Train the joint pointer model with frozen embeddings and exact resume.

The notebook and this CLI share one composition: ``prepare`` builds the run,
``initial_state`` loads or restores it, and ``final_evaluation`` and
``save_bundle`` finish it. Each caller only chooses settings and staging.
"""

import argparse
from collections.abc import Callable, Iterable, Mapping
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
from examples.magicbox import export
from examples.magicbox import source
from minifield_training.batching import pointer as batching
from minifield_training.batching import stream as streams
from minifield_training.checkpoints import discovery
from minifield_training.checkpoints import tensors
from minifield_training.checkpoints import training_state
from minifield_training.core import json_io
from minifield_training.core import parameters as core_parameters
from minifield_training.datasets import pointer as pointer_records
from minifield_training.engine import step as engine_step
from minifield_training.engine import training_run
from minifield_training.evaluation import pointer as evaluate_pointer
from minifield_training.evaluation import schema_fields as evaluate
from minifield_training.kernels import bidirectional
from minifield_training.kernels import types
from minifield_training.models import contracts
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import pointer
from minifield_training.objectives import pointer as objective
from minifield_training.objectives import schema_fields as weighting
from minifield_training.optimizers import adamw
from minifield_training.optimizers import optax_adamw
from minifield_training.optimizers import schedule as lr_schedule
from minifield_training.optimizers import state as optimizer_state
from minifield_training.strategies import pretrained
from minifield_training.strategies import quantization
from minifield_training.strategies import schema_fields as strategy

SPLITS = ("validation", "calibration", "test", "ood")

type Forward = Callable[
    [types.Parameters, types.DeviceBatch], types.DeviceBatch
]


@dataclasses.dataclass(frozen=True)
class Model:
    """What a pointer product supplies to this shared run composition.

    ``template`` and ``implementation`` join the run identity. ``corpus``
    reads and compiles the dataset; ``extra`` names trainable tensors beyond
    the encoder and pointer heads; ``bind`` builds the forward;
    ``initialize`` adds the extras to fresh pretrained weights; ``fold``
    turns trained masters back into plain pointer weights for bundles.
    ``vocabulary``, given the dataset tokenizer folder, returns the tokenizer
    JSON the model reads with and its original token IDs; bundles then ship
    that tokenizer and only those embedding rows.
    """

    template: str
    implementation: str
    corpus: Callable[..., source.Corpus] = source.Corpus
    extra: Callable[[lfm.Config], dict[str, tuple[int, ...]]] = lambda _: {}
    bind: Callable[..., Forward] = magicbox.bind_pointer
    initialize: Callable[[types.Parameters], types.Parameters] = lambda p: p
    fold: Callable[[types.Parameters], types.Parameters] = lambda p: p
    # Readable names for special tokens in exported device tokenizers.
    token_names: Mapping[str, str] = dataclasses.field(default_factory=dict)
    vocabulary: (
        Callable[[Path], tuple[dict[str, object], tuple[int, ...]]] | None
    ) = None


MAGICBOX = Model(data.POINTER_TEMPLATE, "magicbox-pointer-jax/1")


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
    # None keeps a constant learning rate; otherwise warm up for this many
    # updates, then cosine-decay to final_lr_fraction at the last update.
    warmup_updates: int | None = None
    final_lr_fraction: float = 0.1
    attention: encoder.Attention = encoder.DEFAULT_ATTENTION
    # "nf4" or "ternary" trains the same masters as a dense parent and a
    # fake-quantized student that distills from it; None trains dense only.
    quantizer: str | None = None
    quantized_weight: float = 1.0
    distill_weight: float = 1.0
    temperature: float = 2.0
    model: Model = MAGICBOX
    encoder_source: contracts.PretrainedSource = encoder.SOURCE
    # Encoder masters to start from instead of the pretrained ones, such as a
    # quantization warm-up's output.
    encoder_weights: Path | None = None


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
    plan: quantization.NamedQuantization | None
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
    corpus = settings.model.corpus(
        settings.dataset, settings.cache, allow_sample=settings.allow_sample
    )
    config_path = settings.model_dir / "config.json"
    if (
        json_io.digest_file(config_path)
        != settings.encoder_source.config_sha256
    ):
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
    schedule = (
        None
        if settings.warmup_updates is None
        else lr_schedule.WarmupCosine(
            settings.warmup_updates,
            stream.total_updates,
            settings.final_lr_fraction,
        )
    )
    if schedule is not None and not optax:
        raise ValueError("A learning-rate schedule needs the optax optimizer")
    plan = (
        None
        if settings.quantizer is None
        else magicbox.quantization_plan(cfg, settings.quantizer)
    )
    identity: dict[str, object] = {
        "source": dataclasses.asdict(settings.encoder_source),
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
        "template": settings.model.template,
        "score_width": settings.score_width,
        "packing": {
            "pack": settings.pack,
            "planner": "first-fit/1",
            "open_limit": 64,
            "close_below": 0.05,
        },
        "implementation": settings.model.implementation,
    }
    # Optional features join the identity only when on, so earlier
    # checkpoints keep theirs.
    if settings.attention != encoder.DEFAULT_ATTENTION:
        identity["attention"] = dataclasses.asdict(settings.attention)
    if settings.encoder_weights is not None:
        identity["encoder_weights"] = json_io.digest_file(
            settings.encoder_weights
        )
    if plan is not None:
        identity["qat"] = {
            "quantizer": plan.identity,
            "quantized_weight": settings.quantized_weight,
            "distill_weight": settings.distill_weight,
            "temperature": settings.temperature,
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
        inventory=magicbox.pointer_inventory(
            cfg, head, plan, settings.model.extra(cfg)
        ),
        plan=plan,
        optimizer=optimizer,
        optimizer_id=(
            optax_adamw.implementation_identity(optimizer, schedule)
            if optax
            else optimizer.implementation_identity
        ),
        transaction=(
            functools.partial(optax_adamw.make_transaction, schedule=schedule)
            if optax
            else adamw.make_transaction
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
    run: Run,
    resume: Path | None,
    warm_start: types.Parameters | None = None,
) -> tuple[optimizer_state.State, training_state.Cursor]:
    """Restore ``resume`` exactly, or start fresh optimizer state.

    A fresh start begins from ``warm_start`` masters (such as an earlier
    curriculum stage's) when given, otherwise from the pretrained encoder.
    """
    if resume is not None:
        return training_state.load(
            resume,
            run.inventory,
            optimizer_id=run.optimizer_id,
            run_id=run.settings.run_id,
            data_sha256=run.corpus.identity,
            source_id=run.source_id,
        )
    if warm_start is not None:
        if set(warm_start) != set(run.inventory.names):
            raise ValueError("Warm-start weights don't match this model")
        parameters = dict(warm_start)
    else:
        _, parameters = pretrained.load_verified(
            run.settings.model_dir,
            run.settings.encoder_source,
            encoder.Adapter(),
        )
        if run.settings.encoder_weights is not None:
            parameters = tensors.load_masters(
                run.settings.encoder_weights,
                encoder.Adapter().expected_shapes(run.cfg),
                source_dtype="F32",
                sha256=str(run.identity["encoder_weights"]),
            )
        parameters.update(
            pointer.initialize(run.head, jax.random.PRNGKey(run.settings.seed))
        )
        parameters = run.settings.model.initialize(parameters)
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


def forward(
    run: Run,
) -> Callable[[types.Parameters, types.DeviceBatch], types.DeviceBatch]:
    """The pointer model with this run's precision and attention."""
    return run.settings.model.bind(
        run.cfg,
        run.head,
        bf16=run.settings.bf16,
        attention=run.settings.attention,
    )


def make_evaluator(run: Run) -> evaluate.Evaluator[pointer_records.Record]:
    """Held-out evaluation with the training shape and precision."""
    return evaluate.Evaluator(
        run.corpus.pointer_records,
        evaluate_pointer.Predictor(forward(run), run.batches),
        names=data.KINDS,
        metrics=evaluate_pointer.Metrics,
    )


def make_step(run: Run) -> engine_step.JitStep:
    """The jitted training update for this run's model and optimizer.

    With a quantizer, each update trains the dense parent and the quantized
    student from the same masters.
    """
    if run.plan is None:
        return strategy.make_step(
            forward(run),
            run.inventory,
            run.optimizer,
            mesh=run.mesh,
            terms=objective.terms,
            transaction=run.transaction,
        )
    return strategy.make_distilled_step(
        forward(run),
        run.inventory,
        run.optimizer,
        run.plan,
        functools.partial(
            objective.distilled_terms,
            quantized_weight=run.settings.quantized_weight,
            distill_weight=run.settings.distill_weight,
            temperature=run.settings.temperature,
        ),
        mesh=run.mesh,
        transaction=run.transaction,
    )


def quantized(run: Run, params: types.Parameters) -> types.Parameters:
    """The fake-quantized weights the student runs, or ``params`` if dense."""
    return quantization.apply(params, run.inventory, run.plan)


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
    """Evaluate each held-out source separately.

    Each source samples up to ``limit`` records per question type; 0 reads
    every record.

    ``final-<split>.json`` holds each source's metrics and an ``all`` entry
    pooled by count; rank and calibration metrics pool as count-weighted
    means of each source's value. ``final-ood-degradation.json`` holds each
    pooled metric's relative change from test to OOD, positive when worse.
    A quantized run also writes ``final-<split>-<quantizer>.json`` for its
    student.
    """
    _final_evaluation(run, evaluator, params, output, limit, "")
    if run.settings.quantizer is not None:
        _final_evaluation(
            run,
            evaluator,
            quantized(run, params),
            output,
            limit,
            "-" + run.settings.quantizer,
        )


def _per_kind(
    evaluator: evaluate.Evaluator[pointer_records.Record],
    corpus: source.Corpus,
    params: types.Parameters,
    split: str,
    name: str,
    kinds: Iterable[str],
    limit: int,
) -> dict[str, float]:
    """One source's metrics, sampling up to ``limit`` records per type.

    Each question type gets its own sample of records that ask it, and keeps
    only its own metrics from that pass, so rare types aren't drowned out by
    common ones. ``loss`` and ``error_reduction`` weight the types equally.
    """
    metrics: dict[str, float] = {}
    for kind in sorted(kinds):
        reader = functools.partial(
            corpus.pointer_records, source=name, kind=kind
        )
        sampled = dataclasses.replace(evaluator, records=reader).run(
            params, split, limit
        )
        metrics.update(
            {
                key: value
                for key, value in sampled.items()
                if key.startswith(kind + "/")
            }
        )
    for aggregate in ("loss", "error_reduction"):
        values = [
            value
            for key, value in metrics.items()
            if key.endswith("/" + aggregate) and key.count("/") == 1
        ]
        if values:
            metrics[aggregate] = sum(values) / len(values)
    reductions = [
        key for key in metrics if key.endswith("/error_reduction/count")
    ]
    if "error_reduction" in metrics:
        metrics["error_reduction/count"] = sum(
            metrics[key] for key in reductions
        )
    return metrics


def _final_evaluation(
    run: Run,
    evaluator: evaluate.Evaluator[pointer_records.Record],
    params: types.Parameters,
    output: Path,
    limit: int,
    suffix: str,
) -> None:
    pooled = {}
    for split in SPLITS:
        if not any(shard["split"] == split for shard in run.corpus.shards):
            continue
        by_source = {}
        for name, kinds in sorted(
            run.corpus.pointer_source_kinds(split).items()
        ):
            by_source[name] = _per_kind(
                evaluator, run.corpus, params, split, name, kinds, limit
            )
            print(
                json.dumps(
                    {
                        "split": split + suffix,
                        "source": name,
                        **by_source[name],
                    }
                ),
                flush=True,
            )
        metrics = {"all": combine(by_source.values()), **by_source}
        pooled[split] = metrics["all"]
        (output / f"final-{split}{suffix}.json").write_text(
            json.dumps(metrics, indent=2)
        )
        print(
            json.dumps(
                {"split": split + suffix, "source": "all", **metrics["all"]}
            ),
            flush=True,
        )
    if "test" in pooled and "ood" in pooled:
        # Relative change from test to OOD for every shared metric.
        shift = evaluate_pointer.degradation(pooled["test"], pooled["ood"])
        (output / f"final-ood-degradation{suffix}.json").write_text(
            json.dumps(shift, indent=2)
        )
        print(json.dumps({"ood_degradation" + suffix: shift}), flush=True)


def save_bundle(
    run: Run, params: types.Parameters, output: Path, step: int
) -> Path:
    """Export the inference bundle once per run identity and step."""
    # The run identity keeps bundles from other settings in one output apart.
    bundle = output / f"bundle-{run.source_id[:12]}-{step:08d}"
    if run.settings.model.vocabulary is not None:
        # The model reads with its own vocabulary, so its bundle carries
        # that tokenizer and only those embedding rows.
        if not bundle.exists():
            kept = device_tokenizer(run, output)
            magicbox_bundle.save_device(
                bundle,
                export.trimmed_parameters(
                    run.settings.model.fold(params), kept
                ),
                run.cfg,
                run.head,
                vocabulary=kept,
                quantizer=None,
                encoder_config=run.config_path,
                tokenizer=output / f"device-tokenizer-{run.source_id[:12]}",
                step=step,
                template=run.settings.model.template,
            )
        return bundle
    if not bundle.exists():
        magicbox_bundle.save_pointer(
            bundle,
            run.settings.model.fold(params),
            run.cfg,
            run.head,
            encoder_config=run.config_path,
            tokenizer=run.settings.dataset / "tokenizer",
            step=step,
            template=run.settings.model.template,
        )
    return bundle


def device_tokenizer(run: Run, output: Path) -> tuple[int, ...]:
    """Write the device tokenizer once; return its original token IDs.

    A model with its own vocabulary ships that; otherwise the dataset's
    tokenizer is trimmed to every record and checked to reproduce them.
    """
    tokenizer = output / f"device-tokenizer-{run.source_id[:12]}"
    # Written last, so its presence means the trimmed tokenizer is complete.
    kept_path = tokenizer / "kept_ids.json"
    if not kept_path.exists():
        dataset_tokenizer = run.settings.dataset / "tokenizer"
        if run.settings.model.vocabulary is None:
            kept = export.write_trimmed_tokenizer(
                dataset_tokenizer,
                tokenizer,
                run.corpus,
                rename=run.settings.model.token_names,
            )
        else:
            spec, kept = run.settings.model.vocabulary(dataset_tokenizer)
            export.write_tokenizer(
                spec,
                kept,
                dataset_tokenizer,
                tokenizer,
                rename=run.settings.model.token_names,
            )
        kept_path.write_text(json.dumps(kept))
    return tuple(json.loads(kept_path.read_text()))


def export_device_bundles(
    run: Run, params: types.Parameters, output: Path, step: int
) -> list[Path]:
    """Write trimmed-vocabulary FP32 and, if quantized, packed bundles.

    The trimmed tokenizer is cut from every record in the dataset and
    checked to reproduce each of their encodings before any bundle is
    written. Existing bundles are kept.
    """
    tokenizer = output / f"device-tokenizer-{run.source_id[:12]}"
    kept = device_tokenizer(run, output)
    trimmed = export.trimmed_parameters(run.settings.model.fold(params), kept)
    bundles = []
    quantizers: list[str | None] = [None]
    if run.settings.quantizer is not None:
        quantizers.append(run.settings.quantizer)
    for quantizer in quantizers:
        name = quantizer or "fp32"
        bundle = output / f"device-{run.source_id[:12]}-{step:08d}-{name}"
        if not bundle.exists():
            magicbox_bundle.save_device(
                bundle,
                trimmed,
                run.cfg,
                run.head,
                vocabulary=kept,
                quantizer=quantizer,
                encoder_config=run.config_path,
                tokenizer=tokenizer,
                step=step,
                template=run.settings.model.template,
            )
        bundles.append(bundle)
    return bundles


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
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--final-lr-fraction", type=float, default=0.1)
    parser.add_argument(
        "--attention", choices=bidirectional.BACKENDS, default="dense"
    )
    parser.add_argument("--local-window", type=int)
    parser.add_argument("--global-every", type=int, default=3)
    parser.add_argument("--quantizer", choices=tuple(magicbox.QUANTIZERS))
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
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
            warmup_updates=args.warmup_updates,
            final_lr_fraction=args.final_lr_fraction,
            attention=encoder.Attention(
                args.attention, args.local_window, args.global_every
            ),
            quantizer=args.quantizer,
            distill_weight=args.distill_weight,
            temperature=args.temperature,
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
    if completed:
        for device in export_device_bundles(
            run, current["params"], output, cursor.next_batch
        ):
            print(json.dumps({"device_bundle": str(device)}), flush=True)
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
