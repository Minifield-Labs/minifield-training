"""Bounded single-host lifecycle over caller-supplied batch strategies."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
import dataclasses
import math
from pathlib import Path
import time

import jax

from minifield_training.batching import contracts as batching
from minifield_training.checkpoints import training_state
from minifield_training.core import parameters as core_parameters
from minifield_training.engine import step
from minifield_training.optimizers import adamw
from minifield_training.optimizers import state

type Evaluator = Callable[[state.State, int], Mapping[str, float]]
type Reporter = Callable[[dict[str, float | str]], None]

_now: Callable[[], float] = time.monotonic


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Replay seed, checkpoint/report cadence, and bounded run duration."""

    seed: int
    checkpoint_every: int
    report_every: int
    max_steps: int | None = None
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        """Reject unbounded runs and invalid cadence or seed."""
        if (
            min(self.checkpoint_every, self.report_every) < 1
            or self.seed < 0
            or self.max_steps is None
            and self.max_seconds is None
            or self.max_steps is not None
            and self.max_steps < 1
            or self.max_seconds is not None
            and (not math.isfinite(self.max_seconds) or self.max_seconds <= 0)
        ):
            raise ValueError("Invalid or unbounded training run")


def require_single_device(platform: str | None = None) -> jax.Device:
    """Require exactly one device, optionally enforcing its platform."""
    return require_devices(1, platform)[0]


def require_devices(
    count: int, platform: str | None = None
) -> tuple[jax.Device, ...]:
    """Require an exact visible device count on one host, without fallback."""
    if count < 1:
        raise ValueError("Device count must be positive")
    devices = jax.devices()
    if (
        jax.process_count() != 1
        or len(devices) != count
        or platform is not None
        and any(device.platform != platform for device in devices)
    ):
        label = platform if platform is not None else "local"
        quantity = "one" if count == 1 else str(count)
        raise RuntimeError(
            f"Expected {quantity} {label} device(s) on one host, found "
            f"{[(device.platform, device.id) for device in devices]}"
        )
    return tuple(devices)


def _state_placement(
    update: step.LogicalStep | step.StreamingStep, platform: str | None
) -> jax.Device | jax.sharding.NamedSharding:
    """Admit the step's device topology and place one logical state."""
    mesh = update.mesh if isinstance(update, step.StreamingStep) else None
    if mesh is None:
        return require_single_device(platform)
    devices = require_devices(mesh.size, platform)
    if set(mesh.devices.flat) != set(devices):
        raise ValueError("Training mesh must contain all visible devices")
    # JAX 0.7.2's PartitionSpec constructor has no type annotations.
    replicated = jax.sharding.PartitionSpec()  # type: ignore[no-untyped-call]
    return jax.sharding.NamedSharding(mesh, replicated)


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


def _close_if_supported(iterator: object) -> None:
    """Release a producer iterator when it owns external resources."""
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _epoch_update_count[RecordT](
    examples: Sequence[RecordT] | None,
    strategy: batching.BatchStrategy[RecordT] | None,
    source: batching.BatchSource | None,
) -> int:
    """Admit exactly one input mode and resolve its finite epoch length."""
    if source is not None:
        if examples is not None or strategy is not None:
            raise ValueError(
                "Specify examples with a batch strategy or a batch source"
            )
        return 0
    if not examples or strategy is None:
        raise ValueError("Run requires examples and a batch strategy")
    updates = strategy.update_count(examples)
    if updates < 1:
        raise ValueError("Batch strategy must produce at least one update")
    return updates


def run[RecordT](
    examples: Sequence[RecordT] | None,
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
    annotate_steps: bool = False,
    batch_strategy: batching.BatchStrategy[RecordT] | None = None,
    batch_source: batching.BatchSource | None = None,
) -> tuple[state.State, training_state.Cursor]:
    """Compile and run bounded updates, saving after committed boundaries.

    The finite dataset stays on the host and resumes seeded epoch order. A
    batch source instead starts at the global next-batch cursor and must yield
    deterministic, non-repeating updates until the run bound is reached. Only
    one logical update's arrays are transferred at a time. The caller provides
    persistent checkpoints and optional gameplay evaluation.
    """
    placement = _state_placement(update, required_platform)
    updates_per_epoch = _epoch_update_count(
        examples, batch_strategy, batch_source
    )
    adamw.validate_full_weight_state(initial_state, inventory)
    if int(initial_state["step"]) != cursor.next_batch:
        raise ValueError("Optimizer step and data cursor disagree")
    compiled = (
        update if isinstance(update, step.StreamingStep) else jax.jit(update)
    )
    # Loaded arrays may be physically on this device but uncommitted. The
    # first JIT result is committed; starting committed keeps one compilation
    # signature across the warm-start and resumed updates.
    current = jax.device_put(initial_state, placement)
    started = _now()
    deadline = started + config.max_seconds if config.max_seconds else None
    committed = 0
    warm_seconds = 0.0
    warm_updates = 0
    last_saved = -1
    source_batches = (
        batch_source(cursor.next_batch, deadline)
        if batch_source is not None
        else None
    )
    try:
        while True:
            if config.max_steps is not None and committed >= config.max_steps:
                break
            if (
                config.max_seconds is not None
                and committed > 0
                and _now() - started >= config.max_seconds
            ):
                break
            if batch_source is None:
                assert examples is not None and batch_strategy is not None
                epoch, offset = divmod(cursor.next_batch, updates_per_epoch)
                batches = batch_strategy.iter_updates(
                    examples, seed=config.seed + epoch, start_update=offset
                )
            else:
                assert source_batches is not None
                batches = source_batches
            for batch in batches:
                if (
                    deadline is not None
                    and committed > 0
                    and _now() >= deadline
                ):
                    break
                update_started = _now()
                annotation = (
                    jax.profiler.StepTraceAnnotation(
                        "train", step_num=cursor.next_batch + 1
                    )
                    if annotate_steps
                    else nullcontext()
                )
                with annotation:
                    result = compiled(current, batch.microbatches, batch.active)
                update_seconds = _now() - update_started
                if not bool(result.committed):
                    code = int(result.code)
                    raise RuntimeError(f"Training update rejected, code={code}")
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
            else:
                if batch_source is not None and (
                    deadline is None or _now() < deadline
                ):
                    raise RuntimeError(
                        "Training batch source exhausted before run limit"
                    )
            if (
                config.max_steps is not None and committed >= config.max_steps
            ) or (
                config.max_seconds is not None
                and _now() - started >= config.max_seconds
            ):
                break
        if committed > 0 and last_saved != cursor.next_batch:
            _save_and_evaluate(
                current,
                cursor,
                inventory,
                checkpoint_root,
                optimizer_id,
                evaluate,
                report,
            )
    finally:
        _close_if_supported(source_batches)
    return current, cursor
