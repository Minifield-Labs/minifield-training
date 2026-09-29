"""Pointer logits against an independent NumPy computation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from minifield_training.kernels import types
from minifield_training.models.magicbox import pointer


def test_logits_are_scaled_query_key_products() -> None:
    """Gather each question's marker state, then compare with every token."""
    cfg = pointer.Config(encoder_width=6, pointer_width=4)
    params = pointer.initialize(cfg, jax.random.PRNGKey(0))
    assert set(params) == set(pointer.shapes(cfg))
    assert all(value.dtype == jnp.float32 for value in params.values())
    rng = np.random.default_rng(1)
    hidden = rng.normal(size=(2, 5, 6)).astype(np.float32)

    def encode(
        weights: types.Parameters, ids: jax.Array, mask: jax.Array
    ) -> jax.Array:
        del weights, mask
        assert ids.shape == (2, 5)
        return jnp.asarray(hidden)

    batch = {
        "input_ids": jnp.zeros((2, 5), jnp.int32),
        "input_mask": jnp.ones((2, 5), jnp.int32),
        "query_index": jnp.asarray([[0, 3], [4, 1]]),
    }
    outputs = pointer.forward(params, cfg, encode, batch)
    for end in ("start", "end"):
        query = np.asarray(params[f"magicbox.pointer.{end}_query"])
        key = np.asarray(params[f"magicbox.pointer.{end}_key"])
        expected = np.zeros((2, 2, 5))
        for request, positions in enumerate(([0, 3], [4, 1])):
            for question, position in enumerate(positions):
                q = query @ hidden[request, position]
                expected[request, question] = (hidden[request] @ key.T) @ q / 2
        np.testing.assert_allclose(outputs[end], expected, rtol=1e-5)
        assert outputs[end].dtype == jnp.float32


def test_rejects_misaligned_queries() -> None:
    """Query positions must share the request axis with the sequence."""
    cfg = pointer.Config(encoder_width=6, pointer_width=4)
    params = pointer.initialize(cfg, jax.random.PRNGKey(0))
    batch = {
        "input_ids": jnp.zeros((2, 5), jnp.int32),
        "input_mask": jnp.ones((2, 5), jnp.int32),
        "query_index": jnp.zeros((3, 2), jnp.int32),
    }
    with pytest.raises(ValueError, match="query positions"):
        pointer.forward(
            params, cfg, lambda weights, ids, mask: ids.astype(float), batch
        )
