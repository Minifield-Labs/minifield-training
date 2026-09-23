"""Dense LFM2.5 causal SFT composition for logical updates."""

import jax
import jax.numpy as jnp

from minifield_training.core import parameters as core_parameters
from minifield_training.engine import step
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import model
from minifield_training.objectives import loss
from minifield_training.optimizers import adamw


def make_lfm2_5_step(
    cfg: model.Config,
    inventory: core_parameters.FullParameterInventory,
    optimizer: adamw.AdamWConfig,
    *,
    dtype: types.DType = jnp.bfloat16,
) -> step.LogicalStep:
    """Compose dense model forward with summed causal NLL and AdamW."""

    def loss_terms(
        parameters: types.Parameters, batch: types.DeviceBatch
    ) -> tuple[jax.Array, jax.Array]:
        """Return supervised next-token loss and count for one batch."""
        logits = model.forward(
            parameters,
            batch["input_ids"],
            batch["attention_mask"],
            cfg,
            dtype=dtype,
        )
        return loss.causal_loss_terms(
            logits,
            batch["input_ids"],
            batch["loss_mask"],
            batch["attention_mask"],
        )

    return step.make_step(loss_terms, inventory, optimizer)
