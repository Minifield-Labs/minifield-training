"""Train the tool-call model through its curriculum stages, in order.

Each stage is its own run: its own dataset folder, output folder,
checkpoints, learning-rate schedule and fresh optimizer state. A stage that
hasn't started begins from the previous stage's final weights, held in
memory when this session finished that stage, otherwise read from its last
checkpoint. One deadline covers the whole session; rerunning resumes the
unfinished stage.
"""

from collections.abc import Callable, Sequence
import dataclasses
import json
from pathlib import Path
import time

from examples.magicbox import train
from examples.toolcalls import composition
from examples.toolcalls import data
from minifield_training.checkpoints import training_state
from minifield_training.engine import training_run
from minifield_training.kernels import types
from minifield_training.optimizers import state as optimizer_state

STAGES = (0, 1, 2, 3)
type Report = Callable[[dict[str, object]], None]


def stage_settings(
    base: train.Settings, root: Path, stage: int
) -> train.Settings:
    """One stage's settings: its dataset folder, run ID and the tool model."""
    return dataclasses.replace(
        base,
        dataset=data.stage_directory(root, stage),
        run_id=f"toolcalls-stage{stage}",
        model=composition.MODEL,
    )


def finished_weights(run: train.Run, output: Path) -> types.Parameters | None:
    """A completed stage's final masters, from its newest checkpoint."""
    resume = train.latest(run, output)
    if resume is None:
        return None
    current, cursor = train.initial_state(run, resume)
    if cursor.next_batch < run.stream.total_updates:
        return None
    return current["params"]


def train_stage(  # pylint: disable=too-many-arguments
    run: train.Run,
    output: Path,
    warm_start: types.Parameters | None,
    *,
    deadline: float,
    checkpoint_every: int,
    validation_records: int,
    keep_checkpoints: int,
    report: Report,
    target: int,
) -> tuple[optimizer_state.State, training_state.Cursor]:
    """Resume or start one stage; train to update ``target`` or the deadline."""
    resume = train.latest(run, output)
    current, cursor = train.initial_state(
        run, resume, None if resume else warm_start
    )
    train.write_run(run, output)
    remaining = target - cursor.next_batch
    seconds = deadline - time.monotonic()
    if remaining > 0 and seconds > 0:
        current, cursor = training_run.run(
            None,
            current,
            train.make_step(run),
            run.inventory,
            training_run.RunConfig(
                run.settings.seed,
                checkpoint_every,
                10,
                max_steps=remaining,
                max_seconds=seconds,
                keep_checkpoints=keep_checkpoints,
            ),
            checkpoint_root=output / "checkpoints",
            optimizer_id=run.optimizer_id,
            cursor=cursor,
            evaluate=train.make_evaluator(run).callback(
                output / "metrics", validation_records
            ),
            report=lambda event: report(dict(event)),
            required_platform=run.settings.platform,
            batch_source=run.stream,
            strict_compiles=True,
        )
    return current, cursor


def run_curriculum(  # pylint: disable=too-many-arguments,too-many-locals
    base: train.Settings,
    root: Path,
    output: Path,
    *,
    stages: Sequence[int] = STAGES,
    session_seconds: float,
    checkpoint_every: int = 250,
    validation_records: int = 256,
    final_records: int = 1000,
    keep_checkpoints: int = 2,
    export_last: bool = True,
    max_updates: int | None = None,
    report: Report = lambda event: print(json.dumps(event), flush=True),
) -> dict[int, Path]:
    """Train ``stages`` in order; return each finished stage's bundle.

    ``max_updates`` caps every stage, for smoke runs; a capped stage counts
    as finished.

    Every finished stage gets a bundle and a final evaluation in its own
    folder; the last stage also exports device bundles. The run stops at the
    first stage that doesn't finish before the deadline.
    """
    deadline = time.monotonic() + session_seconds
    weights: types.Parameters | None = None
    bundles: dict[int, Path] = {}
    for position, stage in enumerate(stages):
        stage_output = output / f"stage{stage}"
        run = train.prepare(stage_settings(base, root, stage))
        if weights is None and position > 0:
            previous = train.prepare(
                stage_settings(base, root, stages[position - 1])
            )
            weights = finished_weights(
                previous, output / f"stage{stages[position - 1]}"
            )
            if weights is None:
                raise ValueError(
                    f"Stage {stages[position - 1]} hasn't finished"
                )
        target = run.stream.total_updates
        if max_updates is not None:
            target = min(target, max_updates)
        report({"event": "stage", "stage": stage, "updates": target})
        current, cursor = train_stage(
            run,
            stage_output,
            weights,
            deadline=deadline,
            checkpoint_every=checkpoint_every,
            validation_records=validation_records,
            keep_checkpoints=keep_checkpoints,
            report=report,
            target=target,
        )
        if cursor.next_batch < target:
            report(
                {
                    "event": "stopped",
                    "stage": stage,
                    "next_batch": cursor.next_batch,
                }
            )
            break
        bundles[stage] = train.save_bundle(
            run, current["params"], stage_output, cursor.next_batch
        )
        if not (stage_output / "final-test.json").exists():
            train.final_evaluation(
                run,
                train.make_evaluator(run),
                current["params"],
                stage_output,
                final_records,
            )
        if export_last and position == len(stages) - 1:
            for device in train.export_device_bundles(
                run, current["params"], stage_output, cursor.next_batch
            ):
                report({"event": "device_bundle", "path": str(device)})
        weights = current["params"]
    return bundles
