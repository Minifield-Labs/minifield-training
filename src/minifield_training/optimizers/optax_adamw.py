"""Optional optax AdamW commit over the shared full-weight state layout.

This transaction uses optax's clipped AdamW for the update and skips a whole
update when the loss, count, or gradients aren't finite. Unlike the default
transaction in ``adamw``, it doesn't recheck every incoming master and moment
or every candidate value inside the step: the training runner validates state
eagerly at startup, restore, and every checkpoint. That keeps the compiled
program much smaller on TPU. State keeps the ``params``/``m``/``v``/``step``
layout, so checkpoints, discovery, and resume work unchanged; a separate
implementation identity keeps the two transactions' checkpoints apart.
"""

from collections.abc import Callable
import dataclasses
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[import-untyped]

from minifield_training.core import json_io
from minifield_training.core import parameters as core_parameters
from minifield_training.kernels import types
from minifield_training.optimizers import adamw
from minifield_training.optimizers import schedule as lr_schedule
from minifield_training.optimizers import state

_IMPLEMENTATION_ID = "minifield.optax-adamw/1-skip-nonfinite"
_INT32_MAX = np.iinfo(np.int32).max


def implementation_identity(
    config: adamw.AdamWConfig, schedule: lr_schedule.WarmupCosine | None = None
) -> str:
    """Bind checkpoints to this transaction, its settings, and optax.

    A schedule joins the identity only when present, so constant-rate
    checkpoints keep their existing identity.
    """
    payload: dict[str, object] = {
        "implementation": _IMPLEMENTATION_ID,
        "optax": optax.__version__,
        **dataclasses.asdict(config),
    }
    if schedule is not None:
        payload["schedule"] = {
            "kind": "warmup-cosine/1",
            **dataclasses.asdict(schedule),
        }
    return hashlib.sha256(
        json_io.canonical(payload).encode("utf-8")
    ).hexdigest()


def make_transaction(
    inventory: core_parameters.FullParameterInventory,
    config: adamw.AdamWConfig,
    schedule: lr_schedule.WarmupCosine | None = None,
) -> Callable[
    [state.State, types.Parameters, jax.Array, jax.Array], adamw.CommitResult
]:
    """Build a pure optax AdamW commit with the shared calling convention.

    Gradients cover exactly the trainable leaves. Frozen masters and moments
    pass through unchanged. Decay follows each leaf's inventory flag.
    ``update_norm`` is reported as NaN; computing it would add a full pass.
    With ``schedule``, the update at committed step ``n`` uses
    ``learning_rate * schedule.factor(n)``; weight decay scales with it.
    """
    names = inventory.trainable_names
    rate = (
        config.learning_rate
        if schedule is None
        else lambda count: config.learning_rate * schedule.factor(count)
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.clip_norm),
        optax.adamw(
            rate,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            mask={
                spec.name: spec.decayed
                for spec in inventory.specs
                if spec.trainable
            },
        ),
    )

    def transition(
        full_state: state.State,
        gradients: types.Parameters,
        active_loss: jax.Array,
        accumulation_valid: jax.Array,
    ) -> adamw.CommitResult:
        """Apply one clipped AdamW update, or keep the incoming state."""
        adamw.validate_full_weight_state_structure(full_state, inventory)
        adamw.validate_gradient_tree(gradients, inventory)
        adamw.validate_transition_inputs(active_loss, accumulation_valid)
        step = full_state["step"]
        params = {name: full_state["params"][name] for name in names}
        opt_state = optax.tree.set(
            optimizer.init(params),
            count=step,
            mu={name: full_state["m"][name] for name in names},
            nu={name: full_state["v"][name] for name in names},
        )
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        candidate = {
            "params": optax.apply_updates(params, updates),
            "m": optax.tree.get(opt_state, "mu"),
            "v": optax.tree.get(opt_state, "nu"),
        }
        loss = active_loss.astype(jnp.float32)
        gradient_norm = optax.tree.norm(gradients)
        code = jnp.where(
            ~accumulation_valid,
            jnp.int32(adamw.CommitCode.ACCUMULATION_INVALID),
            jnp.where(
                ~jnp.isfinite(loss),
                jnp.int32(adamw.CommitCode.NONFINITE_LOSS),
                jnp.where(
                    ~jnp.isfinite(gradient_norm),
                    jnp.int32(adamw.CommitCode.NONFINITE_GRADIENT),
                    jnp.where(
                        step >= _INT32_MAX,
                        jnp.int32(adamw.CommitCode.STEP_OVERFLOW),
                        jnp.int32(adamw.CommitCode.COMMITTED),
                    ),
                ),
            ),
        )
        committed = code == jnp.int32(adamw.CommitCode.COMMITTED)

        def publish(
            old: types.Parameters, new: types.Parameters
        ) -> types.Parameters:
            """Select each trainable leaf; frozen leaves pass through."""
            return {
                name: jnp.where(committed, new[name], value)
                if name in new
                else value
                for name, value in old.items()
            }

        nan = jnp.float32(jnp.nan)
        return adamw.CommitResult(
            {
                "params": publish(full_state["params"], candidate["params"]),
                "m": publish(full_state["m"], candidate["m"]),
                "v": publish(full_state["v"], candidate["v"]),
                "step": jnp.where(committed, step + 1, step),
            },
            committed,
            code,
            loss,
            jnp.where(committed, gradient_norm, nan),
            jnp.where(
                committed,
                jnp.minimum(gradient_norm, jnp.float32(config.clip_norm)),
                nan,
            ),
            nan,
        )

    return transition
