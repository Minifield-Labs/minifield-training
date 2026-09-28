"""Bind the selected LFM encoder to the MagicBox model and parameter names."""

from collections.abc import Callable
import functools

import jax

from minifield_training.core import parameters as core_parameters
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import model


def inventory(
    cfg: lfm.Config, fusion: model.Config
) -> core_parameters.FullParameterInventory:
    """Train all backbone and task parameters, decaying matrix weights."""
    if cfg.hidden_size != fusion.encoder_width:
        raise ValueError("Encoder and fusion widths differ")
    shapes = {**encoder.Adapter().expected_shapes(cfg), **model.shapes(fusion)}
    return core_parameters.build_inventory(
        shapes,
        decayed_names=frozenset(
            name for name, shape in shapes.items() if len(shape) == 2
        ),
        format_id="minifield.magicbox.parameters/1",
        source_dtype="float32",
        master_dtype="float32",
    )


def forward(
    parameters: types.Parameters,
    cfg: lfm.Config,
    fusion: model.Config,
    batch: types.DeviceBatch,
    *,
    training: bool = False,
    bf16: bool = True,
) -> types.DeviceBatch:
    """Inject one shared encoder into the architecture."""
    encode = functools.partial(encoder.encode, cfg=cfg, bf16=bf16)

    def apply(
        params: types.Parameters, ids: jax.Array, mask: jax.Array
    ) -> jax.Array:
        """Bind the encoder's keyword config without copying parameters."""
        return encode(params, ids=ids, mask=mask)

    return model.forward(parameters, fusion, apply, batch, training=training)


def bind(
    cfg: lfm.Config,
    fusion: model.Config,
    *,
    training: bool = False,
    bf16: bool = True,
) -> Callable[[types.Parameters, types.DeviceBatch], types.DeviceBatch]:
    """Supply the selected model as a neutral two-argument forward callback."""

    def apply(
        params: types.Parameters, batch: types.DeviceBatch
    ) -> types.DeviceBatch:
        return forward(params, cfg, fusion, batch, training=training, bf16=bf16)

    return apply
