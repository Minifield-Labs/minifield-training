"""The tool-call model: MagicBox's pointer encoder with trainable markers.

The token embedding table stays frozen. Ten marker vectors train in its
place: each forward writes them into the marker tokens' rows, so gradients
reach only the markers. They start as copies of ``<|startoftext|>``'s
vector, which the encoder already handles, so training begins where a
BOS-marked model would and each marker learns its own role.
"""

import jax.numpy as jnp

from examples.magicbox import composition as magicbox
from examples.magicbox import train
from examples.toolcalls import data
from examples.toolcalls import vocabulary
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import pointer

MARKERS = "toolcalls.markers"
EMBEDDINGS = "lfm2.embed_tokens.weight"
_IDS = tuple(data.MARKER_IDS.values())


def extra(cfg: lfm.Config) -> dict[str, tuple[int, ...]]:
    """One trainable vector per marker."""
    return {MARKERS: (len(_IDS), cfg.hidden_size)}


def initialize(parameters: types.Parameters) -> types.Parameters:
    """Start every marker as a copy of the BOS embedding."""
    bos = parameters[EMBEDDINGS][data.BOS]
    return {**parameters, MARKERS: jnp.tile(bos[None, :], (len(_IDS), 1))}


def with_markers(parameters: types.Parameters) -> types.Parameters:
    """Write the marker vectors into their embedding rows.

    Bundles store this folded form, so a runtime sees an ordinary pointer
    model whose marker rows hold the trained vectors.
    """
    table = parameters[EMBEDDINGS]
    folded = table.at[jnp.asarray(_IDS)].set(
        parameters[MARKERS].astype(table.dtype)
    )
    return {
        **{
            name: value for name, value in parameters.items() if name != MARKERS
        },
        EMBEDDINGS: folded,
    }


def bind(
    cfg: lfm.Config,
    head: pointer.Config,
    *,
    bf16: bool = True,
    attention: encoder.Attention = encoder.DEFAULT_ATTENTION,
) -> train.Forward:
    """MagicBox's pointer forward over the marker-folded weights."""
    base = magicbox.bind_pointer(cfg, head, bf16=bf16, attention=attention)

    def apply(
        params: types.Parameters, batch: types.DeviceBatch
    ) -> types.DeviceBatch:
        return base(with_markers(params), batch)

    return apply


def unfold(parameters: types.Parameters) -> types.Parameters:
    """Recover trainable masters from a folded bundle, for warm starts."""
    table = parameters[EMBEDDINGS]
    return {**parameters, MARKERS: jnp.asarray(table)[jnp.asarray(_IDS)]}


MODEL = train.Model(
    template=data.TEMPLATE,
    implementation="toolcall-pointer-jax/1",
    corpus=data.Corpus,
    extra=extra,
    bind=bind,
    initialize=initialize,
    fold=with_markers,
    token_names=data.TOKEN_NAMES,
    vocabulary=vocabulary.tokenizer_spec,
)
