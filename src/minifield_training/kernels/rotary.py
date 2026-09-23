"""Unscaled full-head rotary embeddings with split-half rotation."""

import jax
import jax.numpy as jnp


def apply_rotary(
    x: jax.Array,
    *,
    rope_theta: float,
    head_dim: int,
    positions: jax.Array | None = None,
) -> jax.Array:
    """Apply unscaled rotary position embeddings to ``[B, T, H, D]`` values."""
    if positions is None:
        positions = jnp.arange(x.shape[1], dtype=jnp.int32)
    position_array = jnp.asarray(positions, dtype=jnp.float32)
    if position_array.ndim == 0:
        position_array = jnp.full(
            (x.shape[0], x.shape[1]), position_array, dtype=jnp.float32
        )
    elif position_array.ndim == 1:
        position_array = jnp.broadcast_to(
            position_array[None, :], (x.shape[0], x.shape[1])
        )
    elif position_array.ndim != 2:
        raise ValueError("RoPE positions must be scalar, [T], or [B, T]")
    if position_array.shape != x.shape[:2]:
        raise ValueError("RoPE positions must match the token dimensions")
    frequencies = rope_theta ** (
        -jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    )
    angles = position_array[..., None, None] * frequencies[None, None, None, :]
    angles = jnp.concatenate((angles, angles), axis=-1)
    first, second = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate((-second, first), axis=-1)
    return x * jnp.cos(angles).astype(x.dtype) + rotated * jnp.sin(
        angles
    ).astype(x.dtype)
