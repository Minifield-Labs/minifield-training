"""Process-local inference schema cache bound to immutable encoder leaves."""

import dataclasses
import hashlib

import jax
import numpy as np

from minifield_training.kernels import types
from minifield_training.models.magicbox import model


def _signature(
    parameters: types.Parameters,
    batch: types.DeviceBatch,
    context: tuple[str, ...],
) -> tuple[object, ...]:
    """Bind weights, revisions, and the packed schema encoder inputs.

    JAX arrays are immutable. Leaf identities reject any replaced encoder
    parameter without reading 1.4 GB of weight bytes on every lookup. This
    cache is deliberately process-local and cannot be restored from disk.
    """
    weights = tuple(
        (name, id(value))
        for name, value in sorted(parameters.items())
        if not name.startswith("magicbox.")
    )
    digest = hashlib.sha256()
    for name in (
        "packed_schema_ids",
        "packed_schema_segments",
        "packed_schema_positions",
    ):
        value = np.asarray(batch[name])
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return weights, context, digest.hexdigest(), "BOS:0"


@dataclasses.dataclass(frozen=True)
class SchemaCache:
    """Detached encoder tokens for inference."""

    signature: tuple[object, ...]
    hidden: jax.Array

    @classmethod
    def build(
        cls,
        parameters: types.Parameters,
        batch: types.DeviceBatch,
        encode: model.Encoder,
        *,
        context: tuple[str, ...],
    ) -> "SchemaCache":
        """Cache complete schema token representations before projection."""
        if not context or any(not part for part in context):
            raise ValueError(
                "Cache requires tokenizer/template/precision revisions"
            )
        hidden = model.encode_schema(parameters, encode, batch)
        return cls(
            _signature(parameters, batch, context),
            jax.lax.stop_gradient(hidden),
        )

    def get(
        self,
        parameters: types.Parameters,
        batch: types.DeviceBatch,
        *,
        context: tuple[str, ...],
        training: bool = False,
    ) -> jax.Array:
        """Reject training, changed schema, and stale encoder weights."""
        if training:
            raise ValueError("Detached schema caches cannot enter training")
        if self.signature != _signature(parameters, batch, context):
            raise ValueError("Stale schema cache")
        return self.hidden
