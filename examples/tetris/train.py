"""Warm-start or resume the Base Tetris decision classifier on one device."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import re
from typing import cast

from tokenizers import Tokenizer  # type: ignore[import-untyped]

from examples.tetris.evaluate import ALLOWED
from examples.tetris.evaluate import make_evaluator
from examples.tetris.prepare import generator_identity
from minifield_training.checkpoints import training_state
from minifield_training.core import json_io
from minifield_training.datasets.labeled import LabeledSequence
from minifield_training.engine import classification_run
from minifield_training.models.lfm2_5 import model
from minifield_training.optimizers import adamw
from minifield_training.strategies import classification
from minifield_training.strategies import pretrained


def load_model_metadata(model_dir: Path) -> tuple[model.Config, Tokenizer]:
    """Verify the pinned Base config and native tokenizer without weights."""
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer.json"
    if (
        json_io.digest_file(config_path) != pretrained.BASE.config_sha256
        or json_io.digest_file(tokenizer_path)
        != pretrained.BASE.tokenizer_sha256
    ):
        raise ValueError("Model config/tokenizer differs from pinned Base")
    raw: object = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Model config must be an object")
    cfg = model.Config.from_dict(cast(dict[str, object], raw))
    return cfg, Tokenizer.from_file(str(tokenizer_path))


def dataset_identity(path: Path) -> str:
    """Verify the prepared expert source and bytes before consuming records."""
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    raw: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "format",
        "model_id",
        "revision",
        "tokenizer_sha256",
        "data_sha256",
        "games",
        "max_ticks",
        "sequence_length",
        "seed",
        "samples",
        "lines",
        "pieces",
        "generator_sha256",
    }:
        raise ValueError("Invalid prepared Tetris dataset manifest")
    manifest = cast(dict[str, object], raw)
    data_sha256 = json_io.digest_file(path)
    if (
        manifest["format"] != "tetris.expert-decisions/1"
        or manifest["model_id"] != pretrained.BASE.model_id
        or manifest["revision"] != pretrained.BASE.revision
        or manifest["tokenizer_sha256"] != pretrained.BASE.tokenizer_sha256
        or manifest["data_sha256"] != data_sha256
        or manifest["generator_sha256"] != generator_identity()
        or any(
            # bool is an int subclass; manifest counts require plain ints.
            # pylint: disable-next=unidiomatic-typecheck
            type(manifest[key]) is not int or cast(int, manifest[key]) < 0
            for key in (
                "games",
                "max_ticks",
                "sequence_length",
                "seed",
                "samples",
                "lines",
                "pieces",
            )
        )
    ):
        raise ValueError("Prepared Tetris dataset identity mismatch")
    return data_sha256


def source_identity(
    args: argparse.Namespace, cfg: model.Config, *, sequence_length: int
) -> str:
    """Bind the pretrained release, head seed, and replayable batch order."""
    settings = {
        "model": dataclasses.asdict(pretrained.BASE),
        "config": dataclasses.asdict(cfg),
        "head_seed": args.head_seed,
        "data_seed": args.data_seed,
        "sequence_length": sequence_length,
        "microbatches": args.microbatches,
        "rows": args.rows,
        "allowed": ALLOWED,
    }
    return hashlib.sha256(json_io.canonical(settings).encode()).hexdigest()


def optimizer_config(learning_rate: float) -> adamw.AdamWConfig:
    """Use the shared AdamW implementation with one explicit learning rate."""
    return adamw.AdamWConfig(learning_rate=learning_rate)


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


def load_dataset(path: Path) -> tuple[str, list[LabeledSequence]]:
    """Keep complete tokenized records on the host for batch transfer."""
    data_sha256 = dataset_identity(path)
    examples: list[LabeledSequence] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            raw: object = json.loads(line)
            if not isinstance(raw, dict) or set(raw) != {
                "id",
                "group_id",
                "input_ids",
                "label",
            }:
                raise ValueError(
                    f"Invalid decision record at line {line_number}"
                )
            item = cast(dict[str, object], raw)
            ids = item["input_ids"]
            if (
                not isinstance(item["id"], str)
                or not isinstance(item["group_id"], str)
                or not isinstance(ids, list)
                or not isinstance(item["label"], int)
            ):
                raise ValueError(
                    f"Invalid decision types at line {line_number}"
                )
            examples.append(
                LabeledSequence(
                    item["id"],
                    item["group_id"],
                    tuple(ids),
                    item["label"],
                )
            )
    return data_sha256, examples


def main() -> None:
    """Run a short startup smoke or a bounded resumed training session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", type=Path)
    resume.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--skip-if-resumed", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--platform", choices=("tpu", "cpu"), default="tpu")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--data-seed", type=int, default=17)
    parser.add_argument("--head-seed", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=2)
    parser.add_argument("--eval-max-ticks", type=int, default=2000)
    parser.add_argument("--eval-seed", type=int, default=900)
    args = parser.parse_args()
    if not args.checkpoint_root.is_absolute():
        raise ValueError("Checkpoint root must be an explicit absolute path")
    classification_run.require_single_device(args.platform)
    cfg, tokenizer = load_model_metadata(args.model_dir)
    data_sha256, examples = load_dataset(args.dataset)
    inventory = classification.parameter_inventory(cfg, ALLOWED)
    optimizer = optimizer_config(args.learning_rate)
    source_id = source_identity(args, cfg, sequence_length=args.sequence_length)
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
    if resume_path is None:
        loaded_cfg, backbone = pretrained.load_verified(args.model_dir)
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
    update = classification.make_lfm2_5_step(
        cfg, ALLOWED, inventory, optimizer, attention_backend="dense"
    )
    config = classification_run.RunConfig(
        microbatches=args.microbatches,
        rows_per_microbatch=args.rows,
        sequence_length=args.sequence_length,
        pad_token_id=0,
        vocab_size=cfg.vocab_size,
        allowed_classes=ALLOWED,
        padding_label=7,
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
        )
        if args.eval_games > 0
        else None
    )

    def report(message: dict[str, float | str]) -> None:
        """Emit bounded scalar progress and artifact paths as JSON."""
        print(json_io.canonical(message), flush=True)

    classification_run.run(
        examples,
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
    )


if __name__ == "__main__":
    main()
