"""Row-local pre-norm self attention and read-only source cross attention."""

import jax

from minifield_training.kernels import bidirectional
from minifield_training.kernels import linear
from minifield_training.kernels import types


def norm(value: jax.Array, params: types.Parameters, name: str) -> jax.Array:
    """Apply one named affine layer norm."""
    return bidirectional.layer_norm(
        value, params[name + ".gain"], params[name + ".bias"]
    )


def attend(
    query: jax.Array,
    memory: jax.Array,
    mask: jax.Array,
    params: types.Parameters,
    heads: int,
) -> jax.Array:
    """Project one schema row and attend over a masked token sequence."""
    width = query.shape[-1]
    q = linear.full_linear(query, params["q"]).reshape(
        1, -1, heads, width // heads
    )
    k = linear.full_linear(memory, params["k"]).reshape(
        1, -1, heads, width // heads
    )
    v = linear.full_linear(memory, params["v"]).reshape(
        1, -1, heads, width // heads
    )
    result = bidirectional.attention(q, k, v, mask[None]).reshape(query.shape)
    return linear.full_linear(result, params["out"])


def block(
    row: jax.Array,
    memory: jax.Array,
    row_mask: jax.Array,
    source_mask: jax.Array,
    params: types.Parameters,
    key: jax.Array,
    *,
    heads: int,
    dropout: float,
) -> jax.Array:
    """Refine schema tokens without changing source tokens or other rows."""
    keys = jax.random.split(key, 3)
    normalized = norm(row, params, "self_norm")
    update = attend(
        normalized,
        normalized,
        row_mask,
        types.slice_parameters(params, "self."),
        heads,
    )
    row = (row + bidirectional.dropout(update, keys[0], dropout)) * row_mask[
        :, None
    ]
    update = attend(
        norm(row, params, "cross_norm"),
        memory,
        source_mask,
        types.slice_parameters(params, "cross."),
        heads,
    )
    row = (row + bidirectional.dropout(update, keys[1], dropout)) * row_mask[
        :, None
    ]
    update = jax.nn.gelu(
        linear.full_linear(norm(row, params, "ffn_norm"), params["up"])
        + params["up_bias"].astype(row.dtype),
        approximate=False,
    )
    update = linear.full_linear(update, params["down"]) + params[
        "down_bias"
    ].astype(row.dtype)
    return (row + bidirectional.dropout(update, keys[2], dropout)) * row_mask[
        :, None
    ]
