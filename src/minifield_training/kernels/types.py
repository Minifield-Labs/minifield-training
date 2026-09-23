"""Shared array aliases and parameter-dict structure contracts."""

from collections.abc import Mapping

import jax
import jax.typing
import numpy as np
import numpy.typing as npt

type Parameters = dict[str, jax.Array]
type DeviceBatch = dict[str, jax.Array]
type HostArray = npt.NDArray[np.int32] | npt.NDArray[np.float32]
type HostBatch = dict[str, HostArray]
type DType = jax.typing.DTypeLike


def slice_parameters(
    parameters: Mapping[str, jax.Array], prefix: str
) -> Parameters:
    """Return the sub-dict under ``prefix`` with the prefix stripped."""
    return {
        key[len(prefix) :]: value
        for key, value in parameters.items()
        if key.startswith(prefix)
    }


def validate_parameter_structure(
    parameters: Mapping[str, jax.Array],
    expected: Mapping[str, tuple[int, ...]],
) -> None:
    """Check key, shape, and FP32-master dtype without reading leaf values.

    This structural check is safe to invoke while tracing because it only
    reads static pytree metadata. Use ``validate_parameter_masters`` at
    explicit eager load/update boundaries when finite-value validation is
    required.
    """
    actual = set(parameters)
    if actual != set(expected):
        lora = sorted(key for key in actual if ".lora_" in key)
        if lora:
            raise ValueError(
                "Full-weight parameters cannot contain LoRA leaves: "
                + ", ".join(lora)
            )
        missing = sorted(set(expected).difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(
            "Parameter inventory mismatch; missing="
            + repr(missing)
            + ", extra="
            + repr(extra)
        )
    for name, shape in expected.items():
        value = parameters[name]
        if tuple(value.shape) != shape:
            raise ValueError(f"Parameter has wrong shape: {name}")
        try:
            dtype = np.dtype(value.dtype)
        except TypeError as error:
            raise ValueError(f"Parameter has no valid dtype: {name}") from error
        if dtype != np.dtype(np.float32):
            raise ValueError(f"Parameter is not float32: {name}")


def validate_parameter_masters(
    parameters: Mapping[str, jax.Array],
    expected: Mapping[str, tuple[int, ...]],
) -> None:
    """Eagerly reject non-finite masters at load or update boundaries.

    Finite checking deliberately transfers leaves to the host; do not call
    this from a jitted forward or gradient transform.
    """
    validate_parameter_structure(parameters, expected)
    for name, value in parameters.items():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"Parameter master is non-finite: {name}")
