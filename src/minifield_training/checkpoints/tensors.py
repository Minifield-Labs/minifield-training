"""Strict safetensors admission into FP32 master tensors."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import deserialize

from minifield_training.core import json_io

_DTYPES = {"BF16", "F16", "F32"}


def load_masters(
    path: Path,
    expected_shapes: Mapping[str, tuple[int, ...]],
    *,
    source_dtype: str,
    sha256: str,
) -> dict[str, jax.Array]:
    """Verify an immutable source and cast its exact tensor set to FP32.

    The maintained safetensors parser validates the file format. Its NumPy
    adapter can't represent BF16, so only that value conversion is local.
    """
    if json_io.digest_file(path) != sha256:
        raise ValueError("Pretrained tensor SHA-256 mismatch")
    if source_dtype not in _DTYPES:
        raise ValueError("Unsupported pretrained source dtype")
    decoded = deserialize(path.read_bytes())  # type: ignore[no-untyped-call]
    values = {name: item for name, item in decoded}
    if set(values) != set(expected_shapes):
        raise ValueError(
            "Pretrained tensor inventory mismatch; "
            f"missing={sorted(set(expected_shapes) - set(values))}, "
            f"extra={sorted(set(values) - set(expected_shapes))}"
        )
    result: dict[str, jax.Array] = {}
    for name, shape in expected_shapes.items():
        item = values[name]
        if tuple(item["shape"]) != shape or item["dtype"] != source_dtype:
            raise ValueError(f"Pretrained tensor shape/dtype mismatch: {name}")
        raw = cast(bytes, item["data"])
        if source_dtype == "BF16":
            bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            value = bits.view(np.float32).reshape(shape)
        else:
            native = "<f2" if source_dtype == "F16" else "<f4"
            value = np.frombuffer(raw, dtype=native).astype(np.float32)
            value = value.reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite pretrained tensor: {name}")
        result[name] = jnp.asarray(value, dtype=jnp.float32)
    return result
