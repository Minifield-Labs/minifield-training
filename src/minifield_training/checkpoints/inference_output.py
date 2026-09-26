"""Final dense inference weight assets, separate from resumable state."""

from collections.abc import Mapping
import os
from pathlib import Path
import tempfile
from typing import Protocol

import jax
import numpy as np
import numpy.typing as npt
from safetensors.numpy import save_file

from minifield_training.core import parameters


class OutputStrategy(Protocol):
    """Publish final inference weights from an effective parameter set."""

    def write(
        self,
        path: Path,
        effective: Mapping[str, jax.Array],
        inventory: parameters.FullParameterInventory,
        *,
        source_model: str,
        source_revision: str,
    ) -> None:
        """Write one immutable inference asset."""


class DenseEffectiveOutput:
    """Store effective FP32 tensors for a runtime dense weight loader."""

    def write(
        self,
        path: Path,
        effective: Mapping[str, jax.Array],
        inventory: parameters.FullParameterInventory,
        *,
        source_model: str,
        source_revision: str,
    ) -> None:
        """Atomically write contiguous FP32 values and provenance metadata."""
        if path.exists():
            raise FileExistsError(path)
        if not source_model or not source_revision:
            raise ValueError("Output source identity is required")
        if set(effective) != set(inventory.names):
            raise ValueError("Output tensor inventory mismatch")
        arrays: dict[str, npt.NDArray[np.float32]] = {}
        for spec in inventory.specs:
            value = np.asarray(effective[spec.name])
            if value.shape != spec.shape or value.dtype != np.float32:
                raise ValueError(f"Invalid output tensor: {spec.name}")
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite output tensor: {spec.name}")
            arrays[spec.name] = np.ascontiguousarray(value)
        metadata = {
            "format": "pt",
            "source_model": source_model,
            "source_revision": source_revision,
            "inventory_sha256": inventory.sha256,
            "producer": "minifield-training-dense-effective-v1",
        }
        if inventory.quantization_profile is not None:
            metadata["quantization_profile"] = inventory.quantization_profile
            metadata["quantized_names"] = ",".join(
                spec.name for spec in inventory.specs if spec.quantized
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{path.name}-", dir=path.parent
        ) as temporary:
            staged = Path(temporary) / path.name
            save_file(arrays, str(staged), metadata=metadata)
            if path.exists():
                raise FileExistsError(path)
            os.rename(staged, path)
