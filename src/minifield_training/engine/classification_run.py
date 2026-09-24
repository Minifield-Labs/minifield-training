"""Bounded single-device lifecycle for hard-label sequence updates."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from contextlib import nullcontext
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

_PROFILE_WARMUP_UPDATES = 3
_now: Callable[[], float] = time.monotonic


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


def _check_profile_window(
    config: RunConfig,
    cursor: training_state.Cursor,
    profile_dir: Path | None,
    profile_updates: int,
) -> None:
    """Keep a requested trace bounded and clear of checkpoint work."""
    if profile_dir is None:
        if profile_updates:
            raise ValueError("Profiling needs an explicit output directory")
        return
    if (
        not profile_dir.is_absolute()
        or profile_updates < 1
        or config.max_steps is None
        or config.max_steps < _PROFILE_WARMUP_UPDATES + profile_updates
        or config.max_seconds is not None
    ):
        raise ValueError("Profiling needs a bounded warm-update window")
    first_traced = cursor.next_batch + _PROFILE_WARMUP_UPDATES + 1
    last_traced = cursor.next_batch + _PROFILE_WARMUP_UPDATES + profile_updates
    next_checkpoint = (
        (first_traced + config.checkpoint_every - 1)
        // config.checkpoint_every
        * config.checkpoint_every
    )
    if next_checkpoint <= last_traced:
        raise ValueError("Profile window crosses a checkpoint boundary")


def _save_and_evaluate(
    current: state.State,
    cursor: training_state.Cursor,
    inventory: core_parameters.FullParameterInventory,
    checkpoint_root: Path,
    optimizer_id: str,
    evaluate: Evaluator | None,
    report: Reporter | None,
) -> None:
    """Publish a complete checkpoint before optional gameplay."""
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
    profile_dir: Path | None = None,
    profile_updates: int = 0,
) -> tuple[state.State, training_state.Cursor]:
    """Compile and run bounded updates, saving after committed boundaries.

    The full dataset stays on the host. Only one fixed physical batch is
    transferred per step. A restored cursor resumes the seeded update order.
    The caller provides persistent checkpoint storage and optional gameplay
    evaluation, so this loop contains no product-specific behavior.
    """
    device = require_single_device(required_platform)
    if not examples:
        raise ValueError("Classification run requires labeled examples")
    adamw.validate_full_weight_state(initial_state, inventory)
    if int(initial_state["step"]) != cursor.next_batch:
        raise ValueError("Optimizer step and data cursor disagree")
    _check_profile_window(config, cursor, profile_dir, profile_updates)
    capacity = config.microbatches * config.rows_per_microbatch
    updates_per_epoch = math.ceil(len(examples) / capacity)
    compiled = (
        update if isinstance(update, step.StreamingStep) else jax.jit(update)
    )
    # Loaded arrays may be physically on this device but uncommitted. The
    # first JIT result is committed; starting committed keeps one compilation
    # signature across the warm-start and resumed updates.
    current = jax.device_put(initial_state, device)
    started = _now()
    committed = 0
    warm_seconds = 0.0
    warm_updates = 0
    traced_updates = 0
    tracing = False
    last_saved = -1
    with ExitStack() as profile_stack:
        while True:
            if config.max_steps is not None and committed >= config.max_steps:
                break
            if (
                config.max_seconds is not None
                and committed > 0
                and _now() - started >= config.max_seconds
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
                if (
                    profile_dir is not None
                    and committed == _PROFILE_WARMUP_UPDATES
                    and not tracing
                    and not traced_updates
                ):
                    profile_stack.enter_context(
                        jax.profiler.trace(
                            profile_dir, create_perfetto_trace=True
                        )
                    )
                    tracing = True
                update_started = _now()
                annotation = (
                    jax.profiler.StepTraceAnnotation(
                        "train", step_num=cursor.next_batch + 1
                    )
                    if tracing
                    else nullcontext()
                )
                with annotation:
                    result = compiled(current, batch.microbatches, batch.active)
                    accepted = bool(result.committed)
                update_seconds = _now() - update_started
                if not accepted:
                    code = int(result.code)
                    raise RuntimeError(
                        f"Classification update rejected, code={code}"
                    )
                current = result.state
                cursor = dataclasses.replace(
                    cursor, next_batch=cursor.next_batch + 1
                )
                committed += 1
                if committed == 1:
                    if report is not None:
                        report(
                            {
                                "step": float(cursor.next_batch),
                                "first_update_seconds": update_seconds,
                            }
                        )
                else:
                    warm_seconds += update_seconds
                    warm_updates += 1
                if tracing:
                    traced_updates += 1
                    if traced_updates == profile_updates:
                        profile_stack.close()
                        tracing = False
                        if report is not None and profile_dir is not None:
                            report({"profile": str(profile_dir)})
                if report is not None and committed % config.report_every == 0:
                    report(
                        {
                            "step": float(cursor.next_batch),
                            "loss": float(result.loss),
                            "last_update_seconds": update_seconds,
                            "warm_updates_per_second": (
                                warm_updates / warm_seconds
                                if warm_seconds > 0
                                else 0.0
                            ),
                        }
                    )
                if cursor.next_batch % config.checkpoint_every == 0:
                    _save_and_evaluate(
                        current,
                        cursor,
                        inventory,
                        checkpoint_root,
                        optimizer_id,
                        evaluate,
                        report,
                    )
                    last_saved = cursor.next_batch
                if (
                    config.max_steps is not None
                    and committed >= config.max_steps
                ):
                    break
                if (
                    config.max_seconds is not None
                    and _now() - started >= config.max_seconds
                ):
                    break
            if (
                config.max_steps is not None and committed >= config.max_steps
            ) or (
                config.max_seconds is not None
                and _now() - started >= config.max_seconds
            ):
                break
    if last_saved != cursor.next_batch:
        _save_and_evaluate(
            current,
            cursor,
            inventory,
            checkpoint_root,
            optimizer_id,
            evaluate,
            report,
        )
    return current, cursor
