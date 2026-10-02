"""Shared encoder, token fusion, and raw typed logits without decoding."""

from collections.abc import Callable
import dataclasses
import math
from typing import cast

import jax
import jax.numpy as jnp

from minifield_training.kernels import linear
from minifield_training.kernels import types
from minifield_training.layers import schema_fusion

# encode(params, ids, mask, segment_ids, positions); packed rows pass segment
# IDs (0 for padding) and per-segment positions, unpacked rows pass None.
type Encoder = Callable[
    [
        types.Parameters,
        jax.Array,
        jax.Array,
        jax.Array | None,
        jax.Array | None,
    ],
    jax.Array,
]


@dataclasses.dataclass(frozen=True)
class Config:
    """Fusion dimensions independent of question and candidate counts."""

    encoder_width: int = 1024
    width: int = 256
    layers: int = 2
    heads: int = 4
    ffn_multiplier: int = 2
    match_width: int = 128
    dropout: float = 0.1
    row_chunk: int = 4

    def __post_init__(self) -> None:
        """Validate architecture before initializing parameters."""
        if (
            min(
                self.encoder_width,
                self.width,
                self.layers,
                self.heads,
                self.ffn_multiplier,
                self.match_width,
                self.row_chunk,
            )
            < 1
        ):
            raise ValueError("MagicBox dimensions must be positive")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("Invalid head width or dropout")


def shapes(cfg: Config) -> dict[str, tuple[int, ...]]:
    """Describe task tensors; candidate counts never enter the inventory."""
    width, inner = cfg.width, cfg.width * cfg.ffn_multiplier
    result: dict[str, tuple[int, ...]] = {}
    if cfg.encoder_width != width:
        result["projection"] = (width, cfg.encoder_width)
    for name in ("source_norm", "readout_norm"):
        result[name + ".gain"], result[name + ".bias"] = (width,), (width,)
    for name in ("candidate", "binary", "presence"):
        result[name + ".weight"], result[name + ".bias"] = (1, width), (1,)
    result.update(
        {
            "span_query": (cfg.match_width, width),
            "span_source": (cfg.match_width, width),
            "span_bias": (1,),
        }
    )
    for index in range(cfg.layers):
        prefix = f"fusion.{index}."
        for name in ("self_norm", "cross_norm", "ffn_norm"):
            result[prefix + name + ".gain"] = (width,)
            result[prefix + name + ".bias"] = (width,)
        for name in ("self", "cross"):
            for matrix in ("q", "k", "v", "out"):
                result[prefix + name + "." + matrix] = (width, width)
        result.update(
            {
                prefix + "up": (inner, width),
                prefix + "down": (width, inner),
                prefix + "up_bias": (inner,),
                prefix + "down_bias": (width,),
            }
        )
    return {"magicbox." + name: shape for name, shape in result.items()}


def initialize(cfg: Config, key: jax.Array) -> types.Parameters:
    """Initialize affine norms, zero biases, and fan-in scaled matrices."""
    result = {}
    for index, (name, shape) in enumerate(sorted(shapes(cfg).items())):
        if name.endswith("gain"):
            result[name] = jnp.ones(shape, jnp.float32)
        elif len(shape) == 1:
            result[name] = jnp.zeros(shape, jnp.float32)
        else:
            result[name] = jax.random.normal(
                jax.random.fold_in(key, index), shape, dtype=jnp.float32
            ) / math.sqrt(shape[-1])
    return result


def project(hidden: jax.Array, params: types.Parameters) -> jax.Array:
    """Use the same projection for both encoder calls, or the identity."""
    return (
        linear.full_linear(hidden, params["projection"])
        if "projection" in params
        else hidden
    )


def encode_schema(
    parameters: types.Parameters,
    encode: Encoder,
    batch: types.DeviceBatch,
    chunk: int | None = None,
) -> jax.Array:
    """Encode every request's packed schema rows, ``chunk`` at a time.

    Returns ``[requests * packed_sequences, schema_tokens, encoder_width]``.
    ``chunk`` bounds how many packed sequences the encoder holds at once;
    ``None`` encodes them all in one call.
    """
    length = batch["packed_schema_ids"].shape[-1]
    segments = batch["packed_schema_segments"].reshape(-1, length)
    inputs = (
        batch["packed_schema_ids"].reshape(-1, length),
        (segments != 0).astype(jnp.int32),
        segments,
        batch["packed_schema_positions"].reshape(-1, length),
    )
    if chunk is None:
        return encode(parameters, *inputs)

    def one(item: tuple[jax.Array, ...]) -> jax.Array:
        ids, mask, segment, position = (value[None] for value in item)
        return encode(parameters, ids, mask, segment, position)[0]

    return cast(jax.Array, jax.lax.map(one, inputs, batch_size=chunk))


def forward(
    parameters: types.Parameters,
    cfg: Config,
    encode: Encoder,
    batch: types.DeviceBatch,
    *,
    training: bool = False,
    schema_hidden: jax.Array | None = None,
) -> types.DeviceBatch:
    """Encode sources once, schema rows packed, and fuse rows in chunks.

    All arrays retain a leading request axis for device sharding. Schema rows
    are encoded packed, then gathered back to ``[rows, schema_tokens]`` views
    for fusion. Inside the row map source_owner is explicit; source tensors
    stay in the same graph. Keys are attached to rows by the batch compiler
    for exact replay.
    """
    if training and schema_hidden is not None:
        raise ValueError("Detached schema caches cannot enter training")
    source_ids, packed = batch["source_ids"], batch["packed_schema_ids"]
    requests, rows, width = batch["schema_mask"].shape
    if (
        source_ids.ndim != 2
        or packed.ndim != 3
        or source_ids.shape[0] != requests
        or packed.shape[0] != requests
        or batch["source_mask"].shape != source_ids.shape
        or batch["packed_schema_segments"].shape != packed.shape
        or batch["packed_schema_positions"].shape != packed.shape
        or batch["schema_token_index"].shape != batch["schema_mask"].shape
        or batch["row_seed"].shape != (requests, rows)
        or min(*source_ids.shape, *packed.shape, rows, width) < 1
    ):
        raise ValueError("Inconsistent source/schema ownership or masks")
    if schema_hidden is not None and schema_hidden.shape != (
        packed.shape[0] * packed.shape[1],
        packed.shape[2],
        cfg.encoder_width,
    ):
        raise ValueError("Cached schema tensor shape mismatch")
    params = types.slice_parameters(parameters, "magicbox.")
    source = encode(
        parameters, batch["source_ids"], batch["source_mask"], None, None
    )
    memory = schema_fusion.norm(project(source, params), params, "source_norm")
    matched = linear.full_linear(memory, params["span_source"])
    encoded = (
        encode_schema(parameters, encode, batch, cfg.row_chunk)
        if schema_hidden is None
        else schema_hidden
    )
    # Rows gather from the packed projection inside the chunked map, so the
    # padded [rows, schema_tokens] view never exists for all rows at once.
    flat = project(encoded, params).reshape(
        requests, packed.shape[1] * packed.shape[2], -1
    )
    indices = batch["schema_token_index"].reshape(-1, width)
    masks = batch["schema_mask"].reshape(-1, width)
    owners = jnp.repeat(jnp.arange(requests), rows)
    seeds = batch["row_seed"].reshape(-1)

    def row_forward(
        inputs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, jax.Array]:
        """Keep complete token sequences through every interaction block."""
        token_index, row_mask, owner, seed = inputs
        hidden = flat[owner][token_index] * row_mask[:, None].astype(flat.dtype)
        for index in range(cfg.layers):
            hidden = schema_fusion.block(
                hidden,
                memory[owner],
                row_mask,
                batch["source_mask"][owner],
                types.slice_parameters(params, f"fusion.{index}."),
                jax.random.fold_in(jax.random.PRNGKey(seed), index),
                heads=cfg.heads,
                dropout=cfg.dropout if training else 0,
            )
        readout = schema_fusion.norm(hidden[0], params, "readout_norm")
        query = linear.full_linear(readout, params["span_query"])
        tokens = (
            jnp.einsum(
                "h,nh->n",
                query.astype(jnp.float32),
                matched[owner].astype(jnp.float32),
            )
            / math.sqrt(cfg.match_width)
            + params["span_bias"][0]
        )
        return readout, tokens

    readouts, token_logits = jax.lax.map(
        # JAX 0.7.2 exports checkpoint without a public typing declaration.
        jax.checkpoint(row_forward),  # type: ignore[attr-defined]
        (indices, masks, owners, seeds),
        batch_size=cfg.row_chunk,
    )
    result = {
        name: (
            linear.full_linear(readouts, params[name + ".weight"]).astype(
                jnp.float32
            )
            + params[name + ".bias"]
        ).reshape(requests, rows)
        for name in ("candidate", "binary", "presence")
    }
    result["tokens"] = token_logits.reshape(requests, rows, -1)
    return result
