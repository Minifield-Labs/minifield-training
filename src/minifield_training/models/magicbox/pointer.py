"""One joint encoder pass over questions and source, with pointer answers.

Every question answers by pointing at tokens of its own request: an option
marker for choice, ordinal, and binary questions, or a source span (or the
"absent" marker) for extraction. Two start/end query-key projections replace
task-specific heads, fusion blocks, and per-row loops.
"""

from collections.abc import Callable
import dataclasses
import math

import jax
import jax.numpy as jnp

from minifield_training.kernels import linear
from minifield_training.kernels import types

type Encoder = Callable[[types.Parameters, jax.Array, jax.Array], jax.Array]


@dataclasses.dataclass(frozen=True)
class Config:
    """Encoder output width and the pointer query/key width."""

    encoder_width: int = 1024
    pointer_width: int = 256

    def __post_init__(self) -> None:
        """Validate widths before parameters are allocated."""
        if min(self.encoder_width, self.pointer_width) < 1:
            raise ValueError("Pointer widths must be positive")


def shapes(cfg: Config) -> dict[str, tuple[int, ...]]:
    """Name the 4 projection matrices; nothing depends on question counts."""
    return {
        f"magicbox.pointer.{end}_{role}": (
            cfg.pointer_width,
            cfg.encoder_width,
        )
        for end in ("start", "end")
        for role in ("query", "key")
    }


def initialize(cfg: Config, key: jax.Array) -> types.Parameters:
    """Draw fan-in scaled FP32 projection matrices."""
    return {
        name: jax.random.normal(
            jax.random.fold_in(key, index), shape, dtype=jnp.float32
        )
        / math.sqrt(shape[-1])
        for index, (name, shape) in enumerate(sorted(shapes(cfg).items()))
    }


def forward(
    parameters: types.Parameters,
    cfg: Config,
    encode: Encoder,
    batch: types.DeviceBatch,
) -> types.DeviceBatch:
    """Return FP32 ``[requests, questions, tokens]`` start and end logits.

    Queries are the encoder states at each question's marker token. Masking
    to allowed tokens belongs to the objective and decoder.
    """
    ids, queries = batch["input_ids"], batch["query_index"]
    if (
        ids.ndim != 2
        or batch["input_mask"].shape != ids.shape
        or queries.ndim != 2
        or queries.shape[0] != ids.shape[0]
    ):
        raise ValueError("Inconsistent joint sequence or query positions")
    hidden = encode(parameters, ids, batch["input_mask"]).astype(jnp.float32)
    asked = jnp.take_along_axis(hidden, queries[..., None], axis=1)
    params = types.slice_parameters(parameters, "magicbox.pointer.")
    scale = 1 / math.sqrt(cfg.pointer_width)
    return {
        end: jnp.einsum(
            "rqd,rtd->rqt",
            linear.full_linear(asked, params[end + "_query"]),
            linear.full_linear(hidden, params[end + "_key"]),
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
        for end in ("start", "end")
    }
