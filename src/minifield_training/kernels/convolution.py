"""Causal gated depthwise convolution and continuation-state kernels."""

import jax
import jax.numpy as jnp


def gated_depthwise_convolution(
    b_gate: jax.Array,
    c_gate: jax.Array,
    values: jax.Array,
    weights: jax.Array,
    *,
    kernel_size: int,
) -> jax.Array:
    """Return the gated depthwise convolution output before projection."""
    gated_values = b_gate * values
    padded = jnp.pad(gated_values, ((0, 0), (kernel_size - 1, 0), (0, 0)))
    mixed = sum(
        padded[:, tap : tap + values.shape[1], :] * weights[:, tap]
        for tap in range(kernel_size)
    )
    return c_gate * mixed


def gated_depthwise_convolution_segmented(
    b_gate: jax.Array,
    c_gate: jax.Array,
    values: jax.Array,
    weights: jax.Array,
    segment_ids: jax.Array,
    *,
    kernel_size: int,
) -> jax.Array:
    """Return the segmented gated depthwise convolution output.

    A shifted tap contributes only when its segment id equals the current
    token's nonzero segment id, so packed-example boundaries act exactly like
    the left padding an independently processed example would see.
    """
    gated_values = b_gate * values
    padded = jnp.pad(gated_values, ((0, 0), (kernel_size - 1, 0), (0, 0)))
    padded_segments = jnp.pad(segment_ids, ((0, 0), (kernel_size - 1, 0)))
    current = segment_ids
    length = values.shape[1]
    mixed = sum(
        padded[:, tap : tap + length, :]
        * weights[:, tap]
        * (
            (padded_segments[:, tap : tap + length] == current) & (current != 0)
        )[:, :, None].astype(gated_values.dtype)
        for tap in range(kernel_size)
    )
    return c_gate * mixed


def state_tail(
    gated_values: jax.Array,
    valid_lengths: jax.Array,
    history_size: int,
) -> jax.Array:
    """Return the final ``history_size`` valid gated inputs per row.

    The result is the convolution boundary state a continuation needs: the
    ``kernel_size - 1`` gated columns ending at each row's valid length, with
    zeros where a short row has no earlier column. Positions at or beyond the
    valid length never contribute.
    """
    if history_size <= 0:
        return jnp.zeros(
            (gated_values.shape[0], 0, gated_values.shape[-1]),
            dtype=gated_values.dtype,
        )
    offsets = jnp.arange(history_size, dtype=jnp.int32)
    positions = valid_lengths[:, None] - history_size + offsets[None, :]
    clipped = jnp.clip(positions, 0, jnp.maximum(valid_lengths[:, None] - 1, 0))
    selected = jnp.take_along_axis(gated_values, clipped[:, :, None], axis=1)
    return jnp.where(
        (positions >= 0)[:, :, None], selected, jnp.zeros_like(selected)
    )


def gated_depthwise_convolution_with_history(
    b_gate: jax.Array,
    c_gate: jax.Array,
    values: jax.Array,
    weights: jax.Array,
    history: jax.Array,
    *,
    kernel_size: int,
) -> jax.Array:
    """Return the continuation of a gated depthwise convolution.

    ``history`` holds the ``kernel_size - 1`` gated ``b_gate * values`` columns
    immediately preceding ``values`` so continuation outputs match one
    uninterrupted causal convolution. A single-row history broadcasts across
    the batch.
    """
    if history.shape[1] != kernel_size - 1:
        raise ValueError("Convolution history must hold kernel_size - 1 taps")
    if history.shape[0] != values.shape[0]:
        if history.shape[0] != 1:
            raise ValueError("Convolution history batch is incompatible")
        history = jnp.broadcast_to(
            history, (values.shape[0],) + history.shape[1:]
        )
    window = jnp.concatenate((history, b_gate * values), axis=1)
    mixed = sum(
        window[:, tap : tap + values.shape[1], :] * weights[:, tap]
        for tap in range(kernel_size)
    )
    return c_gate * mixed
