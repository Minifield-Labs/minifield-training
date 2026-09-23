"""Immutable parameter metadata records for host-safe inventory contracts."""

from collections.abc import Mapping
import dataclasses
import hashlib
import math

from minifield_training.core import json_io


@dataclasses.dataclass(frozen=True)
class FullParameterSpec:
    """Caller-supplied metadata for one stored parameter row."""

    name: str
    shape: tuple[int, ...]
    source_dtype: str
    master_dtype: str
    trainable: bool
    decayed: bool
    quantized: bool


@dataclasses.dataclass(frozen=True)
class FullParameterInventory:
    """Caller-supplied inventory metadata with derived membership views."""

    specs: tuple[FullParameterSpec, ...]
    parameter_count: int
    source_dtype: str
    master_dtype: str
    quantization_profile: str | None
    sha256: str

    @property
    def names(self) -> tuple[str, ...]:
        """Return supplied leaf names in order, including duplicates."""
        return tuple(spec.name for spec in self.specs)

    @property
    def trainable_names(self) -> tuple[str, ...]:
        """Return names of rows marked trainable, in supplied order."""
        return tuple(spec.name for spec in self.specs if spec.trainable)

    @property
    def frozen_names(self) -> tuple[str, ...]:
        """Return names of rows marked frozen, in supplied order."""
        return tuple(spec.name for spec in self.specs if not spec.trainable)

    @property
    def trainable_parameter_count(self) -> int:
        """Return the scalar count over leaves that receive updates."""
        return sum(
            math.prod(spec.shape) for spec in self.specs if spec.trainable
        )


def build_inventory(
    shapes: Mapping[str, tuple[int, ...]],
    *,
    format_id: str,
    source_dtype: str = "bfloat16",
    master_dtype: str = "float32",
    quantization_profile: str | None = None,
    quantized_names: frozenset[str] = frozenset(),
    frozen_names: frozenset[str] = frozenset(),
) -> FullParameterInventory:
    """Bind parameter shape, dtype, trainability, and decay metadata.

    There is no implicit quantization policy: a caller must name a versioned
    quantization profile and its exact eligible leaf set before any row
    reports itself as quantized. ``frozen_names`` declares stored leaves that
    never receive a gradient or an optimizer update; they still load, run
    forward, checkpoint, and export like every other master, and they are
    excluded from the decay mask. Matrix-only decay is an explicit optimizer
    mask, not a trainability mask.
    """
    if source_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Unsupported checkpoint source dtype")
    if master_dtype != "float32":
        raise ValueError("Full-weight masters must be float32")
    if quantization_profile is None and quantized_names:
        raise ValueError("Quantized leaves require a declared QAT profile")
    if quantization_profile is not None and not quantization_profile:
        raise ValueError("Quantization profile cannot be empty")
    unknown_quantized = quantized_names.difference(shapes)
    if unknown_quantized:
        raise ValueError(
            "Quantization profile names unknown tensors: "
            + ", ".join(sorted(unknown_quantized))
        )
    unknown_frozen = frozen_names.difference(shapes)
    if unknown_frozen:
        raise ValueError(
            "Frozen set names unknown tensors: "
            + ", ".join(sorted(unknown_frozen))
        )
    specs = tuple(
        FullParameterSpec(
            name=name,
            shape=shape,
            source_dtype=source_dtype,
            master_dtype=master_dtype,
            trainable=name not in frozen_names,
            decayed=len(shape) == 2 and name not in frozen_names,
            quantized=name in quantized_names,
        )
        for name, shape in sorted(shapes.items())
    )
    payload = {
        "format": format_id,
        "source_dtype": source_dtype,
        "master_dtype": master_dtype,
        "quantization_profile": quantization_profile,
        "specs": [
            {
                "name": spec.name,
                "shape": list(spec.shape),
                "source_dtype": spec.source_dtype,
                "master_dtype": spec.master_dtype,
                "trainable": spec.trainable,
                "decayed": spec.decayed,
                "quantized": spec.quantized,
            }
            for spec in specs
        ],
    }
    encoded = json_io.canonical(payload).encode("utf-8")
    return FullParameterInventory(
        specs=specs,
        parameter_count=sum(math.prod(spec.shape) for spec in specs),
        source_dtype=source_dtype,
        master_dtype=master_dtype,
        quantization_profile=quantization_profile,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )
