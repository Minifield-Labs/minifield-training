"""Warm-start or resume the Base polyomino classifier on one host."""

import argparse
from collections.abc import Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
import re
import time
from typing import cast

import jax
import numpy as np
from tokenizers import Tokenizer  # type: ignore[import-untyped]

from examples.polyomino import source
from examples.polyomino.evaluate import ALLOWED
from examples.polyomino.evaluate import make_evaluator
from minifield_training.batching import classification as class_batching
from minifield_training.batching import contracts as batch_contracts
from minifield_training.batching import dense
from minifield_training.checkpoints import inference_output
from minifield_training.checkpoints import training_state
from minifield_training.core import json_io
from minifield_training.core import parameters as core_parameters
from minifield_training.engine import training_run
from minifield_training.kernels import quantization as quant_kernels
from minifield_training.models.lfm2_5 import model
from minifield_training.models.lfm2_5 import pretrained as model_source
from minifield_training.optimizers import adamw
from minifield_training.optimizers import state as optimizer_state
from minifield_training.strategies import classification
from minifield_training.strategies import pretrained
from minifield_training.strategies import quantization


def load_model_metadata(model_dir: Path) -> tuple[model.Config, Tokenizer]:
    """Verify the pinned Base config and native tokenizer without weights."""
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer.json"
    if (
        json_io.digest_file(config_path) != model_source.BASE.config_sha256
        or json_io.digest_file(tokenizer_path)
        != model_source.BASE.tokenizer_sha256
    ):
        raise ValueError("Model config/tokenizer differs from pinned Base")
    raw: object = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Model config must be an object")
    cfg = model.Config.from_dict(cast(dict[str, object], raw))
    return cfg, Tokenizer.from_file(str(tokenizer_path))


def source_identity(
    args: argparse.Namespace,
    cfg: model.Config,
    *,
    sequence_length: int,
    quantization_kind: str | None = None,
) -> str:
    """Bind the pretrained release, head seed, and replayable batch order."""
    settings = {
        "model": dataclasses.asdict(model_source.BASE),
        "config": dataclasses.asdict(cfg),
        "head_seed": args.head_seed,
        "data_seed": args.data_seed,
        "sequence_length": sequence_length,
        "microbatches": args.microbatches,
        "rows": args.rows,
        "allowed": ALLOWED,
    }
    if args.no_remat:
        settings["rematerialize_blocks"] = False
    if args.fuse_accumulation:
        settings["fuse_accumulation"] = True
    if args.devices > 1:
        settings["data_parallel_devices"] = args.devices
    if quantization_kind is not None:
        settings["quantization"] = quantization_kind
    return hashlib.sha256(json_io.canonical(settings).encode()).hexdigest()


def quantization_strategy(
    cfg: model.Config, kind: str
) -> quantization.NamedQuantization | None:
    """Select the model's exact projection names for one QAT algorithm."""
    if kind == "dense":
        return None
    identities = {
        "ternary": "ternary-g128-absmax-f16-v1",
        "nf4": "nf4-g128-absmax-f16-v1",
    }
    return quantization.NamedQuantization(
        quant_kernels.Group128Quantizer(identities[kind]),
        classification.projection_names(cfg),
    )


def optimizer_config(learning_rate: float) -> adamw.AdamWConfig:
    """Use the shared AdamW implementation with one explicit learning rate."""
    return adamw.AdamWConfig(learning_rate=learning_rate)


def verify_checkpoint_roundtrip(
    directory: Path,
    current: optimizer_state.State,
    cursor: training_state.Cursor,
    inventory: core_parameters.FullParameterInventory,
    optimizer_id: str,
) -> None:
    """Check every saved parameter and moment against live device state."""
    with jax.default_device(jax.devices("cpu")[0]):
        restored, restored_cursor = training_state.load(
            directory,
            inventory,
            optimizer_id=optimizer_id,
            run_id=cursor.run_id,
            data_sha256=cursor.data_sha256,
            source_id=cursor.source_id,
        )
    if restored_cursor != cursor or not np.array_equal(
        np.asarray(current["step"]), np.asarray(restored["step"])
    ):
        raise RuntimeError("Checkpoint cursor or optimizer step changed")

    def compare_group(
        group: str,
        live: Mapping[str, jax.Array],
        saved: Mapping[str, jax.Array],
    ) -> None:
        """Identify the first tensor changed by serialization."""
        for name in inventory.names:
            if not np.array_equal(
                np.asarray(live[name]),
                np.asarray(saved[name]),
            ):
                raise RuntimeError(f"Checkpoint changed {group}/{name}")

    compare_group("params", current["params"], restored["params"])
    compare_group("m", current["m"], restored["m"])
    compare_group("v", current["v"], restored["v"])


def latest_checkpoint(
    root: Path, *, run_id: str, data_sha256: str, source_id: str
) -> Path | None:
    """Find the newest complete directory with this run's cursor identity."""
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for directory in root.iterdir():
        match = re.fullmatch(r"step-([0-9]{8,})", directory.name)
        if match is None or directory.is_symlink() or not directory.is_dir():
            continue
        if {child.name for child in directory.iterdir()} != {
            "manifest.json",
            "state.safetensors",
        }:
            continue
        if any(child.is_symlink() for child in directory.iterdir()):
            continue
        try:
            raw: object = json.loads(
                (directory / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        cursor = raw.get("cursor")
        if not isinstance(cursor, dict):
            continue
        if (
            cursor.get("run_id") == run_id
            and cursor.get("data_sha256") == data_sha256
            and cursor.get("source_id") == source_id
            # bool is an int subclass, but cursor steps require plain integers.
            # pylint: disable-next=unidiomatic-typecheck
            and type(cursor.get("next_batch")) is int
            and cursor["next_batch"] == int(match.group(1))
        ):
            candidates.append((cursor["next_batch"], directory))
    return max(candidates)[1] if candidates else None


def main(
    output_strategy: inference_output.OutputStrategy | None = None,
) -> None:
    """Run a short startup smoke or a bounded resumed training session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", type=Path)
    resume.add_argument("--resume-latest", action="store_true")
    resume.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--warm-start-run-id")
    parser.add_argument("--skip-if-resumed", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--platform", choices=("tpu", "cpu"), default="tpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument(
        "--rows",
        type=int,
        default=2,
        help="Global rows per microbatch; must divide evenly across devices",
    )
    parser.add_argument("--data-seed", type=int, default=17)
    parser.add_argument("--head-seed", type=int, default=6)
    parser.add_argument("--no-remat", action="store_true")
    parser.add_argument("--fuse-accumulation", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument(
        "--quantization", choices=("dense", "ternary", "nf4"), default="dense"
    )
    parser.add_argument("--output-weights", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--verify-checkpoint", action="store_true")
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=0)
    parser.add_argument("--eval-max-ticks", type=int, default=2000)
    parser.add_argument("--eval-seed", type=int, default=1 << 31)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if not args.checkpoint_root.is_absolute():
        raise ValueError("Checkpoint root must be an explicit absolute path")
    if args.profile_dir is not None and (
        not args.profile_dir.is_absolute()
        or args.max_steps is None
        or not 4 <= args.max_steps <= 50
        or args.max_hours is not None
    ):
        raise ValueError("Profiling needs a separate 4-50 update run")
    devices = training_run.require_devices(args.devices, args.platform)
    if args.rows < 1 or args.rows % args.devices:
        raise ValueError("Global rows must be divisible by device count")
    mesh = (
        jax.sharding.Mesh(np.asarray(devices), ("data",))
        if args.devices > 1
        else None
    )
    cfg, tokenizer = load_model_metadata(args.model_dir)
    data_sha256 = source.data_identity()
    data = source.load_decisions(args.dataset_cache)
    qat = quantization_strategy(cfg, args.quantization)
    inventory = classification.parameter_inventory(cfg, ALLOWED, qat)
    optimizer = optimizer_config(args.learning_rate)
    source_id = source_identity(
        args,
        cfg,
        sequence_length=args.sequence_length,
        quantization_kind=qat.identity if qat is not None else None,
    )
    resume_path = (
        latest_checkpoint(
            args.checkpoint_root,
            run_id=args.run_id,
            data_sha256=data_sha256,
            source_id=source_id,
        )
        if args.resume_latest
        else args.resume
    )
    if args.warm_start_checkpoint is not None:
        if qat is None or not args.warm_start_run_id:
            raise ValueError(
                "QAT warm start needs dense run ID and quantization"
            )
        dense_inventory = classification.parameter_inventory(cfg, ALLOWED)
        dense_masters = training_state.load_warm_start_masters(
            args.warm_start_checkpoint,
            dense_inventory,
            run_id=args.warm_start_run_id,
            data_sha256=data_sha256,
            source_id=source_identity(
                args, cfg, sequence_length=args.sequence_length
            ),
        )
        full_state = adamw.initialize_state(dense_masters, inventory)
        cursor = training_state.Cursor(args.run_id, data_sha256, source_id, 0)
    elif resume_path is None:
        loaded_cfg, backbone = pretrained.load_verified(
            args.model_dir, model_source.BASE, model_source.Adapter()
        )
        if loaded_cfg != cfg:
            raise ValueError("Verified Base config changed during warm start")
        params = classification.initialize_from_backbone(
            backbone, cfg, ALLOWED, head_seed=args.head_seed
        )
        full_state = adamw.initialize_state(params, inventory)
        cursor = training_state.Cursor(args.run_id, data_sha256, source_id, 0)
    else:
        full_state, cursor = training_state.load(
            resume_path,
            inventory,
            optimizer_id=optimizer.implementation_identity,
            run_id=args.run_id,
            data_sha256=data_sha256,
            source_id=source_id,
        )
        if args.skip_if_resumed:
            print(
                json_io.canonical(
                    {
                        "existing_checkpoint": str(resume_path),
                        "step": cursor.next_batch,
                    }
                ),
                flush=True,
            )
            return
    update = classification.make_lfm2_5_streaming_step(
        cfg,
        ALLOWED,
        inventory,
        optimizer,
        attention_backend="dense",
        rematerialize_blocks=not args.no_remat,
        fuse_accumulation=args.fuse_accumulation,
        mesh=mesh,
        quantization_strategy=qat,
    )
    batch_strategy = dense.DenseBatchStrategy(
        batch_contracts.BatchShape(
            args.microbatches,
            args.rows,
            args.sequence_length,
            0,
            cfg.vocab_size,
        ),
        class_batching.ClassTargets(ALLOWED, 7),
    )
    config = training_run.RunConfig(
        seed=args.data_seed,
        checkpoint_every=args.checkpoint_every,
        report_every=args.report_every,
        max_steps=args.max_steps,
        max_seconds=(
            args.max_hours * 3600 if args.max_hours is not None else None
        ),
    )
    evaluator = (
        make_evaluator(
            cfg,
            tokenizer,
            sequence_length=args.sequence_length,
            games=args.eval_games,
            max_ticks=args.eval_max_ticks,
            seed=args.eval_seed,
            replay_dir=args.checkpoint_root / "replays",
            inventory=inventory,
            quantization_strategy=qat,
        )
        if args.eval_games > 0
        else None
    )
    if args.verify_checkpoint:
        game_evaluator = evaluator

        def verify_evaluator(
            current: optimizer_state.State, step: int
        ) -> dict[str, float]:
            """Reload the saved smoke checkpoint before optional gameplay."""
            verify_checkpoint_roundtrip(
                args.checkpoint_root / f"step-{step:08d}",
                current,
                training_state.Cursor(
                    args.run_id, data_sha256, source_id, step
                ),
                inventory,
                optimizer.implementation_identity,
            )
            metrics = {"checkpoint_roundtrip_verified": 1.0}
            if game_evaluator is not None:
                metrics.update(game_evaluator(current, step))
            return metrics

        evaluator = verify_evaluator

    tracing = False

    def report(message: dict[str, float | str]) -> None:
        """Emit scalar progress and start a bounded post-compile trace."""
        nonlocal tracing
        print(json_io.canonical(message), flush=True)
        if args.profile_dir is not None and "first_update_seconds" in message:
            jax.profiler.start_trace(
                args.profile_dir, create_perfetto_trace=False
            )
            tracing = True

    report(
        {
            "event": "first_update_ready",
            "step": float(cursor.next_batch),
            "microbatches": float(args.microbatches),
            "rows": float(args.rows),
            "devices": float(args.devices),
            "rows_per_device": args.rows / args.devices,
            "decisions_per_update": float(args.microbatches * args.rows),
            "sequence_length": float(args.sequence_length),
            "rematerialize_blocks": float(not args.no_remat),
            "fuse_accumulation": float(args.fuse_accumulation),
        }
    )
    run_started = time.monotonic()
    try:
        final_state, final_cursor = training_run.run(
            None,
            full_state,
            update,
            inventory,
            config,
            checkpoint_root=args.checkpoint_root,
            optimizer_id=optimizer.implementation_identity,
            cursor=cursor,
            evaluate=evaluator,
            report=report,
            required_platform=args.platform,
            annotate_steps=args.profile_dir is not None,
            batch_source=source.HFDatasetBatchSource(
                data, tokenizer, batch_strategy, seed=args.data_seed
            ),
        )
    finally:
        if tracing:
            jax.profiler.stop_trace()  # type: ignore[no-untyped-call]
    elapsed = time.monotonic() - run_started
    if args.output_weights is not None:
        effective = quantization.apply(final_state["params"], inventory, qat)
        writer = output_strategy or inference_output.DenseEffectiveOutput()
        writer.write(
            args.output_weights,
            effective,
            inventory,
            source_model=model_source.BASE.model_id,
            source_revision=model_source.BASE.revision,
        )
    updates = final_cursor.next_batch - cursor.next_batch
    report(
        {
            "event": "run_complete",
            "updates": float(updates),
            "elapsed_seconds": elapsed,
            "end_to_end_updates_per_second": updates / elapsed,
        }
    )
    if args.profile_dir is not None:
        report({"profile": str(args.profile_dir)})


if __name__ == "__main__":
    main()
