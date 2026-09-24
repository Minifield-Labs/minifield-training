"""Single-device runner checkpoint, callback, and rejection lifecycle."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from minifield_training.checkpoints import training_state
from minifield_training.core import parameters
from minifield_training.datasets.labeled import LabeledSequence
from minifield_training.engine import classification_run
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


def test_requested_tpu_fails_on_cpu() -> None:
    """A TPU request never silently falls back to CPU."""
    with pytest.raises(RuntimeError, match="Expected one tpu"):
        classification_run.require_single_device("tpu")


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
        assert batch["input_ids"].shape == (1, 1, 3)
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
