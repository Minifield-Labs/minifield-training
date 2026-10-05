"""Learning-rate warm-up and cosine decay over committed updates."""

import dataclasses
import math

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class WarmupCosine:
    """Scale the base learning rate by committed update count.

    Update ``n`` (0-based) uses ``(n + 1) / warmup_updates`` during warm-up,
    so the first update already moves. Cosine decay then runs from 1 to
    ``final_fraction`` at ``total_updates`` and holds there.
    """

    warmup_updates: int
    total_updates: int
    final_fraction: float = 0.1

    def __post_init__(self) -> None:
        """Reject schedules that can't reach their peak or end."""
        if self.warmup_updates < 0 or self.total_updates < 1:
            raise ValueError("Schedule needs nonnegative warm-up and updates")
        if self.warmup_updates >= self.total_updates:
            raise ValueError("Warm-up must end before the last update")
        if not 0 <= self.final_fraction <= 1:
            raise ValueError("final_fraction must be in [0, 1]")

    def factor(self, step: jax.Array) -> jax.Array:
        """Return the FP32 multiplier for the update at committed ``step``."""
        step = jnp.asarray(step, jnp.float32)
        warmup = jnp.float32(max(self.warmup_updates, 1))
        rising = jnp.minimum((step + 1) / warmup, 1.0)
        span = jnp.float32(self.total_updates - self.warmup_updates)
        progress = jnp.clip((step - self.warmup_updates) / span, 0.0, 1.0)
        cosine = 0.5 * (1 + jnp.cos(jnp.float32(math.pi) * progress))
        decayed = self.final_fraction + (1 - self.final_fraction) * cosine
        return jnp.where(step < self.warmup_updates, rising, decayed)
