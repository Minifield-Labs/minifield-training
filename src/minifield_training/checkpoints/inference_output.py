"""Final inference weight assets, separate from resumable state."""

from collections.abc import Mapping
import os
from pathlib import Path
import tempfile
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from safetensors import deserialize
from safetensors.numpy import save_file

from minifield_training.core import json_io
from minifield_training.core import parameters
from minifield_training.kernels import quantization

# Runtime storage contracts (runtime/tools/converters): codes per byte.
PACKED_FORMATS = {
    "nf4-g128-absmax-f16-v1": ("minifield.nf4.v1", 2),
    "ternary-g128-absmax-f16-v1": ("minifield.ternary.v1", 4),
}


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


def _atomic_save(
    path: Path,
    arrays: Mapping[str, npt.NDArray[np.generic]],
    metadata: dict[str, str],
) -> None:
    """Write a safetensors file that appears only when complete.

    safetensors stores each array's raw memory, and device transfers can
    return column-major arrays, so every array is made row-major first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{path.name}-", dir=path.parent
    ) as temporary:
        staged = Path(temporary) / path.name
        save_file(
            {
                name: np.ascontiguousarray(value)
                for name, value in arrays.items()
            },
            str(staged),
            metadata=metadata,
        )
        if path.exists():
            raise FileExistsError(path)
        os.rename(staged, path)


def pack(codes: npt.NDArray[np.uint8], per_byte: int) -> npt.NDArray[np.uint8]:
    """Pack row-major codes into bytes, first code in the lowest bits."""
    bits = 8 // per_byte
    grouped = codes.reshape(codes.shape[0], -1, per_byte)
    packed = np.zeros(grouped.shape[:2], dtype=np.uint8)
    for index in range(per_byte):
        packed |= (grouped[:, :, index] << (index * bits)).astype(np.uint8)
    return packed


def unpack(
    packed: npt.NDArray[np.uint8], per_byte: int
) -> npt.NDArray[np.uint8]:
    """Invert ``pack``."""
    bits = 8 // per_byte
    mask = (1 << bits) - 1
    columns = [(packed >> (index * bits)) & mask for index in range(per_byte)]
    return np.stack(columns, axis=-1).reshape(packed.shape[0], -1)


class PackedGroup128Output:
    """Store QAT matrices as the runtime's packed codes and F16 scales.

    Each quantized master ``name`` becomes ``name.codes`` (U8, several codes
    per byte, first in the lowest bits) and ``name.scales`` (F16 ``[N,
    K/128]``) under the runtime's ``minifield.nf4.v1`` or
    ``minifield.ternary.v1`` contract. Codes come from the same quantizer as
    training, so decoding reproduces the QAT forward weights exactly. Other
    tensors stay FP32.
    """

    def write(
        self,
        path: Path,
        effective: Mapping[str, jax.Array],
        inventory: parameters.FullParameterInventory,
        *,
        source_model: str,
        source_revision: str,
    ) -> None:
        """Atomically write packed and dense tensors with provenance."""
        if path.exists():
            raise FileExistsError(path)
        profile = inventory.quantization_profile
        if profile not in PACKED_FORMATS:
            raise ValueError("Packed output needs a group-128 QAT inventory")
        if not source_model or not source_revision:
            raise ValueError("Output source identity is required")
        if set(effective) != set(inventory.names):
            raise ValueError("Output tensor inventory mismatch")
        storage, per_byte = PACKED_FORMATS[profile]
        arrays: dict[str, npt.NDArray[np.generic]] = {}
        for spec in inventory.specs:
            value = jnp.asarray(effective[spec.name])
            if value.shape != spec.shape or value.dtype != jnp.float32:
                raise ValueError(f"Invalid output tensor: {spec.name}")
            if not bool(jnp.isfinite(value).all()):
                raise ValueError(f"Nonfinite output tensor: {spec.name}")
            if spec.quantized:
                codes, scales = quantization.codes(value, profile)
                arrays[spec.name + ".codes"] = pack(np.asarray(codes), per_byte)
                arrays[spec.name + ".scales"] = np.asarray(scales)
            else:
                arrays[spec.name] = np.asarray(value)
        _atomic_save(
            path,
            arrays,
            {
                "format": "pt",
                "source_model": source_model,
                "source_revision": source_revision,
                "inventory_sha256": inventory.sha256,
                "producer": "minifield-training-packed-g128-v1",
                "quantization_profile": profile,
                "quantization_format": storage,
                "quantized_names": ",".join(
                    spec.name for spec in inventory.specs if spec.quantized
                ),
            },
        )


def load_packed(
    path: Path, inventory: parameters.FullParameterInventory, *, sha256: str
) -> dict[str, jax.Array]:
    """Decode a packed output back to the FP32 weights the model runs."""
    if json_io.digest_file(path) != sha256:
        raise ValueError("Packed tensor SHA-256 mismatch")
    profile = inventory.quantization_profile
    if profile not in PACKED_FORMATS:
        raise ValueError("Packed output needs a group-128 QAT inventory")
    _, per_byte = PACKED_FORMATS[profile]
    decoded = deserialize(path.read_bytes())  # type: ignore[no-untyped-call]
    stored = {name: item for name, item in decoded}
    expected: dict[str, tuple[str, tuple[int, ...]]] = {}
    for spec in inventory.specs:
        rows, columns = (spec.shape + (0,))[:2]
        if spec.quantized:
            expected[spec.name + ".codes"] = (
                "uint8",
                (rows, columns // per_byte),
            )
            expected[spec.name + ".scales"] = (
                "float16",
                (rows, columns // 128),
            )
        else:
            expected[spec.name] = ("float32", spec.shape)
    if set(stored) != set(expected):
        raise ValueError("Packed tensor inventory mismatch")
    arrays = {}
    for name, (dtype, shape) in expected.items():
        item = stored[name]
        if (
            tuple(item["shape"]) != shape
            or item["dtype"]
            != {"uint8": "U8", "float16": "F16", "float32": "F32"}[dtype]
        ):
            raise ValueError(f"Packed tensor shape/dtype mismatch: {name}")
        arrays[name] = np.frombuffer(
            cast(bytes, item["data"]), dtype=dtype
        ).reshape(shape)
    result: dict[str, jax.Array] = {}
    for spec in inventory.specs:
        if spec.quantized:
            result[spec.name] = quantization.decode(
                jnp.asarray(unpack(arrays[spec.name + ".codes"], per_byte)),
                jnp.asarray(arrays[spec.name + ".scales"]),
                profile,
            )
        else:
            result[spec.name] = jnp.asarray(arrays[spec.name])
    return result
