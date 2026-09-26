"""Single-device runner checkpoint, callback, and rejection lifecycle."""

from collections.abc import Callable, Generator, Iterator
from contextlib import nullcontext
import dataclasses
from itertools import count as indices
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from minifield_training.batching import classification as batching
from minifield_training.checkpoints import training_state
from minifield_training.core import parameters
from minifield_training.datasets.labeled import LabeledSequence
from minifield_training.engine import classification_run
from minifield_training.engine import step
from minifield_training.optimizers import adamw
from minifield_training.optimizers import state


def _config(steps: int) -> classification_run.RunConfig:
    """Return one-row fixed batches with a masked padding class."""
    return classification_run.RunConfig(
        microbatches=1,
        rows_per_microbatch=1,
        sequence_length=3,
        pad_token_id=0,
        vocab_size=8,
        allowed_classes=(True, True, False),
        padding_label=2,
        seed=4,
        checkpoint_every=2,
        report_every=1,
        max_steps=steps,
    )


def _update(
    full_state: state.State,
    batch: dict[str, jax.Array],
    active: jax.Array,
) -> adamw.CommitResult:
    """Deterministic fake commit that observes the physical batch."""
    delta = jnp.sum(batch["valid_rows"] & active[:, None]).astype(jnp.float32)
    next_state: state.State = {
        **full_state,
        "params": {"weight": full_state["params"]["weight"] + delta},
        "step": full_state["step"] + jnp.int32(1),
    }
    return adamw.CommitResult(
        next_state,
        jnp.asarray(True),
        jnp.int32(0),
        delta,
        delta,
        delta,
        delta,
    )


def _stream_source(
    seen: list[str],
    starts: list[int],
    closed: list[bool],
    *,
    end: int | None = None,
    failure: int | None = None,
) -> Callable[[int, float | None], Generator[batching.PhysicalUpdate]]:
    """Generate fresh game records from a global logical update index."""

    def produce(
        start: int, deadline: float | None
    ) -> Generator[batching.PhysicalUpdate]:
        """Resume at exactly the requested game and release on close."""
        assert deadline is None or deadline > 0
        starts.append(start)
        try:
            for index in indices(start):
                if end is not None and index == end:
                    return
                if failure is not None and index == failure:
                    raise ValueError("producer failed")
                records = [
                    LabeledSequence(
                        f"{index}-{row}",
                        f"game-{index}",
                        (index % 7 + 1,),
                        index % 2,
                    )
                    for row in range(index % 2 + 1)
                ]
                batch = next(
                    batching.iter_updates(
                        records,
                        microbatches=1,
                        rows_per_microbatch=2,
                        sequence_length=3,
                        pad_token_id=0,
                        vocab_size=8,
                        allowed_classes=(True, True, False),
                        padding_label=2,
                        seed=4,
                    )
                )
                seen.extend(batch.example_ids)
                yield batch
        finally:
            closed.append(True)

    return produce


def test_checkpoint_then_resume_at_next_batch(tmp_path: Path) -> None:
    """A resumed run advances without replaying an earlier decision batch."""
    inventory = parameters.build_inventory(
        {"weight": (1,)}, format_id="runner/1", decayed_names=frozenset()
    )
    initial = adamw.initialize_state(
        {"weight": jnp.zeros((1,), dtype=jnp.float32)}, inventory
    )
    examples = [
        LabeledSequence("a", "episode", (1, 2), 1),
        LabeledSequence("b", "episode", (2, 3), 0),
        LabeledSequence("c", "episode", (3, 4), 1),
    ]
    messages: list[dict[str, float | str]] = []
    evaluations: list[int] = []

    def evaluate(_: state.State, index: int) -> dict[str, float]:
        """Record evaluation boundaries without product-specific logic."""
        evaluations.append(index)
        return {"lines": 0.0}

    current, cursor = classification_run.run(
        examples,
        initial,
        _update,
        inventory,
        _config(2),
        checkpoint_root=tmp_path,
        optimizer_id="optimizer",
        cursor=training_state.Cursor("run", "data", "source", 0),
        evaluate=evaluate,
        report=messages.append,
        required_platform="cpu",
    )
    assert int(current["step"]) == 2
    assert cursor.next_batch == 2
    assert evaluations == [2]
    assert any("checkpoint" in message for message in messages)
    restored, restored_cursor = training_state.load(
        tmp_path / "step-00000002",
        inventory,
        optimizer_id="optimizer",
        run_id="run",
        data_sha256="data",
        source_id="source",
    )
    resumed, end = classification_run.run(
        examples,
        restored,
        _update,
        inventory,
        _config(1),
        checkpoint_root=tmp_path,
        optimizer_id="optimizer",
        cursor=restored_cursor,
        required_platform="cpu",
    )
    assert end.next_batch == 3
    assert int(resumed["step"]) == 3
    assert float(resumed["params"]["weight"][0]) == 3.0
    assert (tmp_path / "step-00000003" / "manifest.json").exists()


def test_stream_source_uses_fresh_games_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long run consumes fresh games and retains reporting."""
    inventory = parameters.build_inventory(
        {"weight": (1,)}, format_id="runner/1", decayed_names=frozenset()
    )
    initial = adamw.initialize_state(
        {"weight": jnp.zeros((1,), dtype=jnp.float32)}, inventory
    )
    seen: list[str] = []
    starts: list[int] = []
    closed: list[bool] = []
    steps: list[int] = []
    messages: list[dict[str, float | str]] = []
    evaluations: list[int] = []

    def fake_annotation(name: str, *, step_num: int) -> object:
        """Record the same global numbering used for finite input."""
        assert name == "train"
        steps.append(step_num)
        return nullcontext()

    def evaluate(_: state.State, index: int) -> dict[str, float]:
        """Record checkpoints reached by the source-backed run."""
        evaluations.append(index)
        return {"lines": float(index)}

    monkeypatch.setattr(jax.profiler, "StepTraceAnnotation", fake_annotation)
    config = dataclasses.replace(_config(5), rows_per_microbatch=2)
    current, cursor = classification_run.run(
        None,
        initial,
        _update,
        inventory,
        config,
        checkpoint_root=tmp_path,
        optimizer_id="optimizer",
        cursor=training_state.Cursor("run", "data", "source", 0),
        evaluate=evaluate,
        report=messages.append,
        required_platform="cpu",
        annotate_steps=True,
        batch_source=_stream_source(seen, starts, closed),
    )
    assert starts == [0]
    assert closed == [True]
    assert steps == [1, 2, 3, 4, 5]
    assert [name.split("-")[0] for name in seen] == [
        "0",
        "1",
        "1",
        "2",
        "3",
        "3",
        "4",
    ]
    assert len(seen) == len(set(seen))
    assert cursor.next_batch == int(current["step"]) == 5
    assert float(current["params"]["weight"][0]) == 7.0
    assert evaluations == [2, 4, 5]
    assert any("warm_updates_per_second" in message for message in messages)
    assert (tmp_path / "step-00000005" / "manifest.json").exists()


def test_final_checkpoint_precedes_stream_shutdown(tmp_path: Path) -> None:
    """A slow producer shutdown cannot strand the last committed update."""
    inventory = parameters.build_inventory(
        {"weight": (1,)}, format_id="runner/1", decayed_names=frozenset()
    )
    initial = adamw.initialize_state(
        {"weight": jnp.zeros((1,), dtype=jnp.float32)}, inventory
    )
    checkpoint = tmp_path / "step-00000001" / "manifest.json"
    checkpoint_visible_at_close: list[bool] = []

    def source(
        start: int, deadline: float | None
    ) -> Iterator[batching.PhysicalUpdate]:
        """Record whether the final update survived before source teardown."""
        wrapped = _stream_source([], [], [])(start, deadline)
        try:
            yield next(wrapped)
        finally:
            checkpoint_visible_at_close.append(checkpoint.is_file())
            wrapped.close()

    current, cursor = classification_run.run(
        None,
        initial,
        _update,
        inventory,
        dataclasses.replace(
            _config(1), rows_per_microbatch=2, checkpoint_every=100
        ),
        checkpoint_root=tmp_path,
        optimizer_id="optimizer",
        cursor=training_state.Cursor("run", "data", "source", 0),
        required_platform="cpu",
        batch_source=source,
    )
    assert int(current["step"]) == cursor.next_batch == 1
    assert checkpoint_visible_at_close == [True]


def test_stream_deadline_stops_without_treating_source_as_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A producer that stops at the time bound preserves its final update."""
    inventory = parameters.build_inventory(
        {"weight": (1,)}, format_id="runner/1", decayed_names=frozenset()
    )
    initial = adamw.initialize_state(
        {"weight": jnp.zeros((1,), dtype=jnp.float32)}, inventory
    )
    clock = [0.0]
    monkeypatch.setattr(classification_run, "_now", lambda: clock[0])

    def source(
        start: int, deadline: float | None
    ) -> Iterator[batching.PhysicalUpdate]:
        """Consume one update, then hit the deadline before the next shard."""
        assert deadline == 1.0
        wrapped = _stream_source([], [], [])(start, deadline)
        try:
            yield next(wrapped)
            clock[0] = 2.0
        finally:
            wrapped.close()

    _, cursor = classification_run.run(
        None,
        initial,
        _update,
        inventory,
        dataclasses.replace(
            _config(4),
            rows_per_microbatch=2,
            checkpoint_every=100,
            max_seconds=1.0,
        ),
        checkpoint_root=tmp_path,
        optimizer_id="optimizer",
        cursor=training_state.Cursor("run", "data", "source", 0),
        required_platform="cpu",
        batch_source=source,
    )
    assert cursor.next_batch == 1
    assert (tmp_path / "step-00000001" / "manifest.json").is_file()


def test_stream_resume_matches_uninterrupted_state(tmp_path: Path) -> None:
    """Restoring the global cursor reproduces the exact unread games."""
    inventory = parameters.build_inventory(
        {"weight": (1,)}, format_id="runner/1", decayed_names=frozenset()
    )
    initial = adamw.initialize_state(
        {"weight": jnp.zeros((1,), dtype=jnp.float32)}, inventory
    )
    config = dataclasses.replace(_config(6), rows_per_microbatch=2)
    full_seen: list[str] = []
    full_starts: list[int] = []
    full_closed: list[bool] = []
    full_state, full_cursor = classification_run.run(
        None,
        initial,
        _update,
        inventory,
        config,
        checkpoint_root=tmp_path / "full",
        optimizer_id="optimizer",
        cursor=training_state.Cursor("run", "data", "source", 0),
        required_platform="cpu",
        batch_source=_stream_source(full_seen, full_starts, full_closed),
    )
    split_seen: list[str] = []
    split_starts: list[int] = []
    split_closed: list[bool] = []
    split_root = tmp_path / "split"
    classification_run.run(
        None,
        initial,
        _update,
        inventory,
        dataclasses.replace(config, max_steps=3),
        checkpoint_root=split_root,
        optimizer_id="optimizer",
        cursor=training_state.Cursor("run", "data", "source", 0),
        required_platform="cpu",
        batch_source=_stream_source(split_seen, split_starts, split_closed),
    )
    restored, restored_cursor = training_state.load(
        split_root / "step-00000003",
        inventory,
        optimizer_id="optimizer",
        run_id="run",
        data_sha256="data",
        source_id="source",
    )
    resumed, end = classification_run.run(
        None,
        restored,
        _update,
        inventory,
        dataclasses.replace(config, max_steps=3),
        checkpoint_root=split_root,
        optimizer_id="optimizer",
        cursor=restored_cursor,
        required_platform="cpu",
        batch_source=_stream_source(split_seen, split_starts, split_closed),
    )
    assert full_starts == [0]
    assert split_starts == [0, 3]
    assert full_closed == [True]
    assert split_closed == [True, True]
    assert split_seen == full_seen
    assert len(split_seen) == len(set(split_seen))
    assert end == full_cursor
    for resumed_leaf, full_leaf in zip(
        jax.tree.leaves(resumed), jax.tree.leaves(full_state), strict=True
    ):
        assert jnp.array_equal(resumed_leaf, full_leaf)


def test_stream_source_exhaustion_and_failure_are_errors(
    tmp_path: Path,
) -> None:
    """A broken source cannot be mistaken for a completed bounded run."""
    inventory = parameters.build_inventory(
        {"weight": (1,)}, format_id="runner/1", decayed_names=frozenset()
    )
    initial = adamw.initialize_state(
        {"weight": jnp.zeros((1,), dtype=jnp.float32)}, inventory
    )
    config = dataclasses.replace(_config(3), rows_per_microbatch=2)
    closed: list[bool] = []
    with pytest.raises(RuntimeError, match="exhausted before run limit"):
        classification_run.run(
            None,
            initial,
            _update,
            inventory,
            config,
            checkpoint_root=tmp_path / "short",
            optimizer_id="optimizer",
            cursor=training_state.Cursor("run", "data", "source", 0),
            required_platform="cpu",
            batch_source=_stream_source([], [], closed, end=2),
        )
    assert closed == [True]
    assert (tmp_path / "short" / "step-00000002").is_dir()
    assert not (tmp_path / "short" / "step-00000003").exists()

    failed_closed: list[bool] = []
    with pytest.raises(ValueError, match="producer failed"):
        classification_run.run(
            None,
            initial,
            _update,
            inventory,
            config,
            checkpoint_root=tmp_path / "failed",
            optimizer_id="optimizer",
            cursor=training_state.Cursor("run", "data", "source", 0),
            required_platform="cpu",
            batch_source=_stream_source([], [], failed_closed, failure=1),
        )
    assert failed_closed == [True]


def test_stream_source_is_required_for_missing_examples(tmp_path: Path) -> None:
    """A caller must select exactly one input mode."""
    inventory = parameters.build_inventory(
        {"weight": (1,)}, format_id="runner/1", decayed_names=frozenset()
    )
    initial = adamw.initialize_state(
        {"weight": jnp.zeros((1,), dtype=jnp.float32)}, inventory
    )
    with pytest.raises(ValueError, match="requires labeled examples"):
        classification_run.run(
            None,
            initial,
            _update,
            inventory,
            _config(1),
            checkpoint_root=tmp_path,
            optimizer_id="optimizer",
            cursor=training_state.Cursor("run", "data", "source", 0),
            required_platform="cpu",
        )
    with pytest.raises(ValueError, match="Specify examples or a batch source"):
        classification_run.run(
            [LabeledSequence("one", "game", (1,), 0)],
            initial,
            _update,
            inventory,
            _config(1),
            checkpoint_root=tmp_path,
            optimizer_id="optimizer",
            cursor=training_state.Cursor("run", "data", "source", 0),
            required_platform="cpu",
            batch_source=_stream_source([], [], []),
        )


def test_requested_tpu_fails_on_cpu() -> None:
    """A TPU request never silently falls back to CPU."""
    with pytest.raises(RuntimeError, match="Expected one tpu"):
        classification_run.require_single_device("tpu")


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_time_limit_is_rejected(seconds: float) -> None:
    """A malformed time bound cannot turn a run into an endless job."""
    with pytest.raises(ValueError, match="Invalid or unbounded"):
        classification_run.RunConfig(
            microbatches=1,
            rows_per_microbatch=1,
            sequence_length=3,
            pad_token_id=0,
            vocab_size=8,
            allowed_classes=(True, True),
            padding_label=1,
            seed=0,
            checkpoint_every=1,
            report_every=1,
            max_seconds=seconds,
        )


def test_rejected_step_publishes_no_checkpoint(tmp_path: Path) -> None:
    """A failed update cannot advance the persistent data cursor."""
    inventory = parameters.build_inventory(
        {"weight": (1,)}, format_id="runner/1", decayed_names=frozenset()
    )
    initial = adamw.initialize_state(
        {"weight": jnp.zeros((1,), dtype=jnp.float32)}, inventory
    )

    def reject(
        full_state: state.State,
        batch: dict[str, jax.Array],
        active: jax.Array,
    ) -> adamw.CommitResult:
        """Return the shared transaction's rejected-state shape."""
        assert batch["input_ids"].shape in ((1, 1, 3), (1, 2, 3))
        assert active.shape == (1,)
        zero = jnp.float32(0)
        return adamw.CommitResult(
            full_state,
            jnp.asarray(False),
            jnp.int32(2),
            zero,
            zero,
            zero,
            zero,
        )

    with pytest.raises(RuntimeError, match="rejected"):
        classification_run.run(
            [LabeledSequence("one", "episode", (1, 2), 0)],
            initial,
            reject,
            inventory,
            _config(1),
            checkpoint_root=tmp_path,
            optimizer_id="optimizer",
            cursor=training_state.Cursor("run", "data", "source", 0),
            required_platform="cpu",
        )
    assert not list(tmp_path.iterdir())
    assert int(initial["step"]) == 0

    closed: list[bool] = []
    with pytest.raises(RuntimeError, match="rejected"):
        classification_run.run(
            None,
            initial,
            reject,
            inventory,
            dataclasses.replace(_config(1), rows_per_microbatch=2),
            checkpoint_root=tmp_path / "stream",
            optimizer_id="optimizer",
            cursor=training_state.Cursor("run", "data", "source", 0),
            required_platform="cpu",
            batch_source=_stream_source([], [], closed),
        )
    assert closed == [True]


def test_streamed_runner_reuses_first_compilation(tmp_path: Path) -> None:
    """Warm-start arrays use the same committed signature as updated state."""
    # Inspect JAX's cache directly to catch a costly second-step recompile.
    # pylint: disable=protected-access
    inventory = parameters.build_inventory(
        {"weight": (1,)}, format_id="runner/1", decayed_names=frozenset()
    )
    initial = adamw.initialize_state(
        {"weight": jnp.zeros((1,), dtype=jnp.float32)}, inventory
    )

    def terms(
        params: dict[str, jax.Array], batch: dict[str, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Return one finite decision gradient per physical batch."""
        loss = params["weight"][0] * jnp.sum(
            batch["input_ids"], dtype=jnp.float32
        )
        count = jnp.sum(batch["valid_rows"], dtype=jnp.float32)
        return loss, count

    update = step.make_streaming_step(terms, inventory, adamw.AdamWConfig(0.01))
    current, cursor = classification_run.run(
        [
            LabeledSequence("a", "episode", (1, 2), 0),
            LabeledSequence("b", "episode", (2, 3), 1),
        ],
        initial,
        update,
        inventory,
        _config(2),
        checkpoint_root=tmp_path,
        optimizer_id="optimizer",
        cursor=training_state.Cursor("run", "data", "source", 0),
        required_platform="cpu",
    )
    assert cursor.next_batch == int(current["step"]) == 2
    assert update.gradient._cache_size() == 1  # type: ignore[attr-defined]
    assert update.normalize._cache_size() == 1  # type: ignore[attr-defined]
    assert update.transition._cache_size() == 1  # type: ignore[attr-defined]


def test_warm_update_rate_excludes_first_compile_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report update-call time with a controlled clock and clear denominator."""
    ticks = iter((0.0, 1.0, 11.0, 20.0, 21.0))
    monkeypatch.setattr(classification_run, "_now", lambda: next(ticks))
    inventory = parameters.build_inventory(
        {"weight": (1,)}, format_id="runner/1", decayed_names=frozenset()
    )
    initial = adamw.initialize_state(
        {"weight": jnp.zeros((1,), dtype=jnp.float32)}, inventory
    )
    messages: list[dict[str, float | str]] = []
    classification_run.run(
        [
            LabeledSequence("a", "episode", (1,), 0),
            LabeledSequence("b", "episode", (2,), 1),
        ],
        initial,
        _update,
        inventory,
        _config(2),
        checkpoint_root=tmp_path,
        optimizer_id="optimizer",
        cursor=training_state.Cursor("run", "data", "source", 0),
        report=messages.append,
        required_platform="cpu",
    )
    assert messages[0]["first_update_seconds"] == 10.0
    assert messages[1]["warm_updates_per_second"] == 0.0
    assert messages[2]["last_update_seconds"] == 1.0
    assert messages[2]["warm_updates_per_second"] == 1.0


def test_step_annotations_leave_profile_capture_to_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Annotate every numbered update without pausing to export a trace."""
    steps: list[int] = []

    def unexpected_trace(*_args: object, **_kwargs: object) -> None:
        """Fail if the runner starts or stops profile capture."""
        raise AssertionError(
            "The runner must leave profile capture to its caller"
        )

    def fake_annotation(name: str, *, step_num: int) -> object:
        """Record numbered train steps for an external profiler."""
        assert name == "train"
        steps.append(step_num)
        return nullcontext()

    monkeypatch.setattr(jax.profiler, "trace", unexpected_trace)
    monkeypatch.setattr(jax.profiler, "start_trace", unexpected_trace)
    monkeypatch.setattr(jax.profiler, "stop_trace", unexpected_trace)
    monkeypatch.setattr(jax.profiler, "StepTraceAnnotation", fake_annotation)
    inventory = parameters.build_inventory(
        {"weight": (1,)}, format_id="runner/1", decayed_names=frozenset()
    )
    initial = adamw.initialize_state(
        {"weight": jnp.zeros((1,), dtype=jnp.float32)}, inventory
    )
    examples = [
        LabeledSequence(str(index), "episode", (index + 1,), 0)
        for index in range(5)
    ]
    classification_run.run(
        examples,
        initial,
        _update,
        inventory,
        _config(1),
        checkpoint_root=tmp_path / "default",
        optimizer_id="optimizer",
        cursor=training_state.Cursor("run", "data", "source", 0),
        required_platform="cpu",
    )
    assert not steps

    current, cursor = classification_run.run(
        examples,
        initial,
        _update,
        inventory,
        _config(5),
        checkpoint_root=tmp_path / "checkpoints",
        optimizer_id="optimizer",
        cursor=training_state.Cursor("run", "data", "source", 0),
        required_platform="cpu",
        annotate_steps=True,
    )
    assert steps == [1, 2, 3, 4, 5]
    assert (tmp_path / "checkpoints" / "step-00000004").is_dir()

    classification_run.run(
        examples,
        current,
        _update,
        inventory,
        _config(2),
        checkpoint_root=tmp_path / "checkpoints",
        optimizer_id="optimizer",
        cursor=cursor,
        required_platform="cpu",
        annotate_steps=True,
    )
    assert steps == [1, 2, 3, 4, 5, 6, 7]
