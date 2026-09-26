"""Checkpoint continuation and tamper rejection."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from minifield_training.checkpoints import training_state
from minifield_training.core import parameters
from minifield_training.optimizers import adamw


def test_complete_state_and_cursor_round_trip(tmp_path: Path) -> None:
    """Restore all moments and the next unread batch exactly."""
    inventory = parameters.build_inventory(
        {"weight": (2,)},
        format_id="synthetic/1",
        decayed_names=frozenset(),
    )
    original = adamw.initialize_state(
        {"weight": jnp.asarray([1.0, -2.0], dtype=jnp.float32)},
        inventory,
    )
    original["m"]["weight"] = jnp.asarray([0.25, 0.5], dtype=jnp.float32)
    original["v"]["weight"] = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    original["step"] = jnp.asarray(3, dtype=jnp.int32)
    cursor = training_state.Cursor("run", "data-hash", "source", 17)
    destination = tmp_path / "step-000003"
    training_state.save(
        destination,
        original,
        inventory,
        optimizer_id="adamw-config",
        cursor=cursor,
    )
    restored, restored_cursor = training_state.load(
        destination,
        inventory,
        optimizer_id="adamw-config",
        run_id="run",
        data_sha256="data-hash",
        source_id="source",
    )
    assert restored_cursor == cursor
    assert int(restored["step"]) == 3
    np.testing.assert_array_equal(restored["params"]["weight"], [1, -2])
    np.testing.assert_array_equal(restored["m"]["weight"], [0.25, 0.5])
    np.testing.assert_array_equal(restored["v"]["weight"], [0.125, 0.25])
    with pytest.raises(ValueError, match="identity"):
        training_state.load(
            destination,
            inventory,
            optimizer_id="adamw-config",
            run_id="other",
            data_sha256="data-hash",
            source_id="source",
        )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tensor_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA-256"):
        training_state.load(
            destination,
            inventory,
            optimizer_id="adamw-config",
            run_id="run",
            data_sha256="data-hash",
            source_id="source",
        )
