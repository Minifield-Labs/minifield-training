"""Train the encoder and heads with frozen embeddings and exact resume."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from examples.magicbox import bundle as magicbox_bundle
from examples.magicbox import composition as magicbox
from examples.magicbox import data
from examples.magicbox import source
from minifield_training.batching import schema_fields as batching
from minifield_training.checkpoints import discovery
from minifield_training.checkpoints import training_state
from minifield_training.core import json_io
from minifield_training.engine import training_run
from minifield_training.evaluation import schema_fields as evaluate
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.magicbox import model
from minifield_training.objectives import schema_fields as objective
from minifield_training.optimizers import adamw
from minifield_training.strategies import pretrained
from minifield_training.strategies import schema_fields as strategy


def arguments() -> argparse.Namespace:
    """Expose full-run bounds, hardware, and persistent output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "model-dir", "output", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--platform", choices=("cpu", "tpu"), default="tpu")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--source-tokens", type=int, default=1024)
    parser.add_argument("--schema-tokens", type=int, default=512)
    parser.add_argument("--schema-rows", type=int, default=256)
    parser.add_argument("--row-chunk", type=int, default=4)
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
        args.requests % args.devices
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
    fusion = model.Config(
        encoder_width=cfg.hidden_size, row_chunk=args.row_chunk
    )
    shape = batching.Shape(
        args.microbatches,
        args.requests,
        args.source_tokens,
        args.schema_tokens,
        args.schema_rows,
        cfg.vocab_size,
        0,
    )
    if (
        max(shape.source_tokens, shape.schema_tokens)
        > encoder.MAX_SEQUENCE_LENGTH
    ):
        raise ValueError("Requested tokens exceed the encoder context")
    batches = batching.SchemaBatchStrategy(
        shape, objective.balance_types, fixed_shape=True
    )
    stream = source.training_stream(corpus, batches, args.seed, args.epochs)
    inventory = magicbox.inventory(cfg, fusion)
    optimizer = adamw.AdamWConfig(learning_rate=args.learning_rate)
    identity = {
        "source": dataclasses.asdict(encoder.SOURCE),
        "encoder": dataclasses.asdict(cfg),
        "fusion": dataclasses.asdict(fusion),
        "shape": {
            key: value
            for key, value in dataclasses.asdict(shape).items()
            if key not in ("vocab_size", "pad_token_id")
        },
        "seed": args.seed,
        "bf16": not args.fp32,
        "devices": args.devices,
        "contract": data.FORMAT,
        "batching": "fixed-shape/1",
        "implementation": "magicbox-jax/2-frozen-token-embeddings",
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
            optimizer_id=optimizer.implementation_identity,
            run_id=args.run_id,
            data_sha256=corpus.identity,
            source_id=source_id,
        )
    else:
        _, parameters = pretrained.load_verified(
            args.model_dir, encoder.SOURCE, encoder.Adapter()
        )
        parameters.update(
            model.initialize(fusion, jax.random.PRNGKey(args.seed))
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
                "updates_per_epoch": stream.updates_per_epoch,
            },
            indent=2,
        )
    )
    evaluator = evaluate.Evaluator(
        corpus.records,
        evaluate.Predictor(
            magicbox.bind(cfg, fusion, bf16=not args.fp32), batches
        ),
        names=data.KINDS,
    )
    remaining = args.epochs * stream.updates_per_epoch - cursor.next_batch
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
            magicbox.bind(cfg, fusion, training=True, bf16=not args.fp32),
            inventory,
            optimizer,
            mesh=mesh,
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
                optimizer_id=optimizer.implementation_identity,
                cursor=cursor,
                evaluate=evaluator.callback(
                    args.output / "metrics", args.validation_records
                ),
                report=report,
                required_platform=args.platform,
                batch_source=stream,
            )
    completed = cursor.next_batch >= args.epochs * stream.updates_per_epoch
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
    bundle = args.output / f"bundle-{cursor.next_batch:08d}"
    if not bundle.exists():
        magicbox_bundle.save(
            bundle,
            current["params"],
            cfg,
            fusion,
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
