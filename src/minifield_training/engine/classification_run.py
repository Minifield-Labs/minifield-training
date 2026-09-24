"""Bounded single-device lifecycle for hard-label sequence updates."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
from pathlib import Path
import time

import jax

from minifield_training.batching import classification as batching
from minifield_training.checkpoints import training_state
from minifield_training.core import parameters as core_parameters
from minifield_training.datasets.labeled import LabeledSequence
from minifield_training.engine import step
from minifield_training.optimizers import adamw
from minifield_training.optimizers import state

type Evaluator = Callable[[state.State, int], Mapping[str, float]]
type Reporter = Callable[[dict[str, float | str]], None]


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Fixed physical shapes, replay seed, and bounded stop/cadence."""

    microbatches: int
    rows_per_microbatch: int
    sequence_length: int
    pad_token_id: int
    vocab_size: int
    allowed_classes: tuple[bool, ...]
    padding_label: int
    seed: int
    checkpoint_every: int
    report_every: int
    max_steps: int | None = None
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        """Reject unbounded runs and invalid fixed update shapes."""
        if (
            min(
                self.microbatches,
                self.rows_per_microbatch,
                self.sequence_length,
                self.vocab_size,
                self.checkpoint_every,
                self.report_every,
            )
            < 1
            or self.seed < 0
            or self.max_steps is None
            and self.max_seconds is None
            or self.max_steps is not None
            and self.max_steps < 1
            or self.max_seconds is not None
            and (not math.isfinite(self.max_seconds) or self.max_seconds <= 0)
        ):
            raise ValueError("Invalid or unbounded classification run")


def require_single_device(platform: str | None = None) -> jax.Device:
    """Require exactly one device, optionally enforcing its platform."""
    devices = jax.devices()
    if (
        len(devices) != 1
        or platform is not None
        and devices[0].platform != platform
    ):
        label = platform if platform is not None else "local"
        raise RuntimeError(
            f"Expected one {label} device, found "
            f"{[(device.platform, device.id) for device in devices]}"
        )
    return devices[0]


def run(
    examples: Sequence[LabeledSequence],
    initial_state: state.State,
    update: step.LogicalStep | step.StreamingStep,
    inventory: core_parameters.FullParameterInventory,
    config: RunConfig,
    *,
    checkpoint_root: Path,
    optimizer_id: str,
    cursor: training_state.Cursor,
    evaluate: Evaluator | None = None,
    report: Reporter | None = None,
    required_platform: str | None = None,
) -> tuple[state.State, training_state.Cursor]:
    """Compile and run bounded updates, saving after committed boundaries.

    The full dataset stays on the host. Only one fixed physical batch is
    transferred per step. A restored cursor resumes the seeded update order.
    The caller provides persistent checkpoint storage and optional gameplay
    evaluation, so this loop contains no product-specific behavior.
    """
    require_single_device(required_platform)
    if not examples:
        raise ValueError("Classification run requires labeled examples")
    adamw.validate_full_weight_state(initial_state, inventory)
    if int(initial_state["step"]) != cursor.next_batch:
        raise ValueError("Optimizer step and data cursor disagree")
    capacity = config.microbatches * config.rows_per_microbatch
    updates_per_epoch = math.ceil(len(examples) / capacity)
    compiled = (
        update if isinstance(update, step.StreamingStep) else jax.jit(update)
    )
    current = initial_state
    started = time.monotonic()
    committed = 0
    last_saved = -1
    while True:
        if config.max_steps is not None and committed >= config.max_steps:
            break
        if (
            config.max_seconds is not None
            and committed > 0
            and time.monotonic() - started >= config.max_seconds
        ):
            break
        epoch, offset = divmod(cursor.next_batch, updates_per_epoch)
        batches = batching.iter_updates(
            examples,
            microbatches=config.microbatches,
            rows_per_microbatch=config.rows_per_microbatch,
            sequence_length=config.sequence_length,
            pad_token_id=config.pad_token_id,
            vocab_size=config.vocab_size,
            allowed_classes=config.allowed_classes,
            padding_label=config.padding_label,
            seed=config.seed + epoch,
            start_update=offset,
        )
        for batch in batches:
            result = compiled(current, batch.microbatches, batch.active)
            if not bool(result.committed):
                raise RuntimeError(
                    f"Classification update rejected, code={int(result.code)}"
                )
            current = result.state
            cursor = dataclasses.replace(
                cursor, next_batch=cursor.next_batch + 1
            )
            committed += 1
            if report is not None and committed % config.report_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                report(
                    {
                        "step": float(cursor.next_batch),
                        "loss": float(result.loss),
                        "updates_per_second": committed / elapsed,
                    }
                )
            if cursor.next_batch % config.checkpoint_every == 0:
                destination = checkpoint_root / f"step-{cursor.next_batch:08d}"
                training_state.save(
                    destination,
                    current,
                    inventory,
                    optimizer_id=optimizer_id,
                    cursor=cursor,
                )
                last_saved = cursor.next_batch
                if report is not None:
                    report({"checkpoint": str(destination)})
                if evaluate is not None:
                    metrics = evaluate(current, cursor.next_batch)
                    if report is not None:
                        report({"step": float(cursor.next_batch), **metrics})
            if config.max_steps is not None and committed >= config.max_steps:
                break
            if (
                config.max_seconds is not None
                and time.monotonic() - started >= config.max_seconds
            ):
                break
        if (config.max_steps is not None and committed >= config.max_steps) or (
            config.max_seconds is not None
            and time.monotonic() - started >= config.max_seconds
        ):
            break
    if last_saved != cursor.next_batch:
        destination = checkpoint_root / f"step-{cursor.next_batch:08d}"
        training_state.save(
            destination,
            current,
            inventory,
            optimizer_id=optimizer_id,
            cursor=cursor,
        )
        if report is not None:
            report({"checkpoint": str(destination)})
        if evaluate is not None:
            metrics = evaluate(current, cursor.next_batch)
            if report is not None:
                report({"step": float(cursor.next_batch), **metrics})
    return current, cursor
