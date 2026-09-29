"""Train the joint pointer model with frozen embeddings and exact resume."""

import argparse
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
from minifield_training.datasets import pointer as pointer_records
from minifield_training.engine import training_run
from minifield_training.evaluation import pointer as evaluate_pointer
from minifield_training.evaluation import schema_fields as evaluate
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.magicbox import pointer
from minifield_training.objectives import pointer as objective
from minifield_training.objectives import schema_fields as weighting
from minifield_training.optimizers import adamw
from minifield_training.optimizers import optax_adamw
from minifield_training.strategies import pretrained
from minifield_training.strategies import schema_fields as strategy


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
    devices = training_run.require_devices(args.devices, args.platform)
    if (
        args.rows % args.devices
        or args.epochs < 1
        or args.validation_records < 1
        or args.final_records < 0
    ):
        raise ValueError(
            "Invalid request topology, epoch count, or evaluation bound"
        )
    mesh = (
        jax.sharding.Mesh(np.asarray(devices), ("data",))
        if args.devices > 1
        else None
    )
    corpus = source.Corpus(
        args.dataset, args.cache, allow_sample=args.allow_sample
    )
    config_path = args.model_dir / "config.json"
    if json_io.digest_file(config_path) != encoder.SOURCE.config_sha256:
        raise ValueError("Pretrained config changed")
    cfg = encoder.Adapter().parse_config(
        json_io.object_map(json.loads(config_path.read_text()))
    )
    head = pointer.Config(encoder_width=cfg.hidden_size)
    tokens, questions = corpus.pointer_extent(
        sorted({str(shard["split"]) for shard in corpus.shards})
    )
    shape = batching.Shape(
        args.microbatches,
        args.rows,
        args.sequence_tokens or -(-tokens // 128) * 128,
        args.questions
        or (questions if args.no_pack else args.questions_per_row),
        cfg.vocab_size,
        0,
    )
    if shape.sequence_tokens > encoder.MAX_SEQUENCE_LENGTH:
        raise ValueError("Requested tokens exceed the encoder context")
    batches = batching.PointerBatchStrategy(shape, weighting.balance_types)
    compile_train = functools.partial(
        corpus.compile_pointer, split="train", score_width=args.score_width
    )
    stream: (
        streams.EpochStream[object, pointer_records.Record]
        | streams.PlannedStream[object, pointer_records.Record]
    ) = (
        source.training_stream(
            corpus,
            batches,
            args.seed,
            args.epochs,
            compile_train,
            prefetch=args.prefetch,
        )
        if args.no_pack
        else source.planned_training_stream(
            corpus,
            batches,
            corpus.pointer_sizes("train"),
            args.seed,
            args.epochs,
            compile_train,
            prefetch=args.prefetch,
        )
    )
    inventory = magicbox.pointer_inventory(cfg, head)
    optimizer = adamw.AdamWConfig(learning_rate=args.learning_rate)
    optimizer_id = (
        optax_adamw.implementation_identity(optimizer)
        if args.optimizer == "optax"
        else optimizer.implementation_identity
    )
    identity = {
        "source": dataclasses.asdict(encoder.SOURCE),
        "encoder": dataclasses.asdict(cfg),
        "head": dataclasses.asdict(head),
        "shape": {
            key: value
            for key, value in dataclasses.asdict(shape).items()
            if key not in ("vocab_size", "pad_token_id")
        },
        "seed": args.seed,
        "bf16": not args.fp32,
        "devices": args.devices,
        "contract": data.FORMAT,
        "template": data.POINTER_TEMPLATE,
        "score_width": args.score_width,
        "packing": {
            "pack": not args.no_pack,
            "planner": "first-fit/1",
            "open_limit": 64,
            "close_below": 0.05,
        },
        "implementation": "magicbox-pointer-jax/1",
    }
    source_id = hashlib.sha256(json_io.canonical(identity).encode()).hexdigest()
    checkpoints = args.output / "checkpoints"
    resume = discovery.latest_checkpoint(
        checkpoints,
        run_id=args.run_id,
        data_sha256=corpus.identity,
        source_id=source_id,
        reject_mismatched=True,
    )
    if args.evaluate_only and not resume:
        raise ValueError("Evaluation requires a trained checkpoint")
    if resume:
        current, cursor = training_state.load(
            resume,
            inventory,
            optimizer_id=optimizer_id,
            run_id=args.run_id,
            data_sha256=corpus.identity,
            source_id=source_id,
        )
    else:
        _, parameters = pretrained.load_verified(
            args.model_dir, encoder.SOURCE, encoder.Adapter()
        )
        parameters.update(
            pointer.initialize(head, jax.random.PRNGKey(args.seed))
        )
        current = adamw.initialize_state(parameters, inventory)
        cursor = training_state.Cursor(
            args.run_id, corpus.identity, source_id, 0
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(
        json.dumps(
            {
                **identity,
                "optimizer": dataclasses.asdict(optimizer),
                "data_sha256": corpus.identity,
                "source_id": source_id,
                "total_updates": stream.total_updates,
            },
            indent=2,
        )
    )
    evaluator = evaluate.Evaluator(
        corpus.pointer_records,
        evaluate_pointer.Predictor(
            magicbox.bind_pointer(cfg, head, bf16=not args.fp32), batches
        ),
        names=data.KINDS,
        metrics=evaluate_pointer.Metrics,
    )
    remaining = stream.total_updates - cursor.next_batch
    if args.max_steps is not None:
        remaining = min(remaining, args.max_steps)
    if remaining > 0 and not args.evaluate_only:
        run_config = training_run.RunConfig(
            args.seed,
            args.checkpoint_every,
            10,
            max_steps=remaining,
            max_seconds=args.max_hours * 3600,
        )
        update = strategy.make_step(
            magicbox.bind_pointer(cfg, head, bf16=not args.fp32),
            inventory,
            optimizer,
            mesh=mesh,
            terms=objective.terms,
            transaction=(
                optax_adamw.make_transaction
                if args.optimizer == "optax"
                else adamw.make_transaction
            ),
        )
        with (args.output / "progress.jsonl").open(
            "a", encoding="utf-8"
        ) as log:

            def report(event: dict[str, float | str]) -> None:
                line = json.dumps(event)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()

            current, cursor = training_run.run(
                None,
                current,
                update,
                inventory,
                run_config,
                checkpoint_root=checkpoints,
                optimizer_id=optimizer_id,
                cursor=cursor,
                evaluate=evaluator.callback(
                    args.output / "metrics", args.validation_records
                ),
                report=report,
                required_platform=args.platform,
                batch_source=stream,
                strict_compiles=True,
            )
    completed = cursor.next_batch >= stream.total_updates
    if completed or args.evaluate_only:
        for split in ("validation", "calibration", "test", "ood"):
            if any(shard["split"] == split for shard in corpus.shards):
                metrics = evaluator.run(
                    current["params"], split, args.final_records
                )
                (args.output / f"final-{split}.json").write_text(
                    json.dumps(metrics, indent=2)
                )
                print(json.dumps({"split": split, **metrics}), flush=True)
    # The run identity keeps bundles from other settings in one output apart.
    bundle = args.output / f"bundle-{source_id[:12]}-{cursor.next_batch:08d}"
    if not bundle.exists():
        magicbox_bundle.save_pointer(
            bundle,
            current["params"],
            cfg,
            head,
            encoder_config=config_path,
            tokenizer=args.dataset / "tokenizer",
            step=cursor.next_batch,
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
