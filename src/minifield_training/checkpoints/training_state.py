"""Atomic, hash-bound safetensors checkpoints for exact update continuation."""

import dataclasses
import json
import os
from pathlib import Path
import tempfile
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from safetensors.numpy import load_file
from safetensors.numpy import save_file

from minifield_training.core import json_io
from minifield_training.core import parameters as core_parameters
from minifield_training.optimizers import adamw
from minifield_training.optimizers import state

_FORMAT = "minifield.full-training-state/1"


@dataclasses.dataclass(frozen=True)
class Cursor:
    """Identity and next unread batch for a deterministic input stream."""

    run_id: str
    data_sha256: str
    source_id: str
    next_batch: int

    def __post_init__(self) -> None:
        """Reject ambiguous or negative resume coordinates."""
        if (
            not self.run_id
            or not self.data_sha256
            or not self.source_id
            or self.next_batch < 0
        ):
            raise ValueError("Invalid checkpoint cursor")


def _arrays(full_state: state.State) -> dict[str, npt.NDArray[np.generic]]:
    """Flatten the exact optimizer state into safe CPU arrays."""
    result: dict[str, npt.NDArray[np.generic]] = {}
    for group in ("params", "m", "v"):
        for name, value in full_state[group].items():
            result[f"{group}/{name}"] = np.asarray(value, dtype=np.float32)
    result["step"] = np.asarray(full_state["step"], dtype=np.int32)
    return result


def save(
    directory: Path,
    full_state: state.State,
    inventory: core_parameters.FullParameterInventory,
    *,
    optimizer_id: str,
    cursor: Cursor,
) -> None:
    """Publish a complete checkpoint under a new path atomically.

    Call only after a committed update. The caller owns durable storage; a
    Colab runtime-local directory cannot be considered persistent.
    """
    if directory.exists():
        raise FileExistsError(directory)
    if not optimizer_id:
        raise ValueError("Optimizer identity is required")
    adamw.validate_full_weight_state(full_state, inventory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{directory.name}-", dir=directory.parent
    ) as temporary:
        staging = Path(temporary)
        tensor_path = staging / "state.safetensors"
        save_file(_arrays(full_state), str(tensor_path))
        manifest = {
            "format": _FORMAT,
            "inventory_sha256": inventory.sha256,
            "optimizer_id": optimizer_id,
            "cursor": dataclasses.asdict(cursor),
            "tensor_sha256": json_io.digest_file(tensor_path),
        }
        (staging / "manifest.json").write_text(
            json_io.canonical(manifest) + "\n", encoding="utf-8"
        )
        if directory.exists():
            raise FileExistsError(directory)
        os.rename(staging, directory)


def load(
    directory: Path,
    inventory: core_parameters.FullParameterInventory,
    *,
    optimizer_id: str,
    run_id: str,
    data_sha256: str,
    source_id: str,
) -> tuple[state.State, Cursor]:
    """Restore exact FP32 state only for matching run, source, and data."""
    raw: object = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(raw, dict) or set(raw) != {
        "format",
        "inventory_sha256",
        "optimizer_id",
        "cursor",
        "tensor_sha256",
    }:
        raise ValueError("Invalid checkpoint manifest")
    manifest = cast(dict[str, object], raw)
    if (
        manifest["format"] != _FORMAT
        or manifest["inventory_sha256"] != inventory.sha256
        or manifest["optimizer_id"] != optimizer_id
    ):
        raise ValueError("Incompatible checkpoint state identity")
    cursor_raw = manifest["cursor"]
    if not isinstance(cursor_raw, dict) or set(cursor_raw) != {
        "run_id",
        "data_sha256",
        "source_id",
        "next_batch",
    }:
        raise ValueError("Invalid checkpoint cursor")
    if (
        not isinstance(cursor_raw["run_id"], str)
        or not isinstance(cursor_raw["data_sha256"], str)
        or not isinstance(cursor_raw["source_id"], str)
        or not isinstance(cursor_raw["next_batch"], int)
    ):
        raise ValueError("Invalid checkpoint cursor types")
    cursor = Cursor(
        cursor_raw["run_id"],
        cursor_raw["data_sha256"],
        cursor_raw["source_id"],
        cursor_raw["next_batch"],
    )
    if (cursor.run_id, cursor.data_sha256, cursor.source_id) != (
        run_id,
        data_sha256,
        source_id,
    ):
        raise ValueError("Checkpoint run/data/source identity mismatch")
    tensor_path = directory / "state.safetensors"
    if json_io.digest_file(tensor_path) != manifest["tensor_sha256"]:
        raise ValueError("Checkpoint tensor SHA-256 mismatch")
    arrays = load_file(str(tensor_path))
    expected = {
        f"{group}/{name}"
        for group in ("params", "m", "v")
        for name in inventory.names
    } | {"step"}
    if set(arrays) != expected:
        raise ValueError("Checkpoint tensor inventory mismatch")
    groups: dict[str, dict[str, jax.Array]] = {}
    shapes = {spec.name: spec.shape for spec in inventory.specs}
    for group in ("params", "m", "v"):
        group_values: dict[str, jax.Array] = {}
        for name in inventory.names:
            value = arrays[f"{group}/{name}"]
            if value.dtype != np.float32 or value.shape != shapes[name]:
                raise ValueError(f"Invalid checkpoint tensor: {group}/{name}")
            group_values[name] = jnp.asarray(value)
        groups[group] = group_values
    step = arrays["step"]
    if step.dtype != np.int32 or step.shape:
        raise ValueError("Invalid checkpoint step")
    restored: state.State = {
        "params": groups["params"],
        "m": groups["m"],
        "v": groups["v"],
        "step": jnp.asarray(step),
    }
    adamw.validate_full_weight_state(restored, inventory)
    return restored, cursor
