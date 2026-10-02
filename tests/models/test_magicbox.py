"""Tiny shared-encoder structural, gradient, masking, and cache checks."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.magicbox import smoke
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.magicbox import cache
from minifield_training.models.magicbox import model


def _batch(rows: int, requests: int = 1, padding: int = 0) -> types.DeviceBatch:
    """Build disjoint source/schema token IDs with request ownership."""
    source_ids = jnp.tile(jnp.asarray([[1, 7, 8, 9]]), (requests, 1))
    schema_ids = jnp.tile(jnp.asarray([1, 20, 21, 22]), (requests, rows, 1))
    schema_ids = schema_ids.at[:, :, 1].set(jnp.arange(rows) % 50 + 20)
    return {
        "source_ids": jnp.pad(source_ids, ((0, 0), (0, padding))),
        "source_mask": jnp.pad(
            jnp.ones_like(source_ids), ((0, 0), (0, padding))
        ),
        "schema_ids": jnp.pad(schema_ids, ((0, 0), (0, 0), (0, padding))),
        "schema_mask": jnp.pad(
            jnp.ones_like(schema_ids), ((0, 0), (0, 0), (0, padding))
        ),
        "row_seed": jnp.arange(requests * rows, dtype=jnp.uint32).reshape(
            requests, rows
        ),
    }


@pytest.mark.parametrize("count", [2, 3, 17, 65])
def test_dynamic_candidates_and_isolation(count: int) -> None:
    """An unchanged head accepts arbitrary candidate inventories and order."""
    cfg, fusion, params = smoke.tiny()

    def encode(
        weights: types.Parameters, ids: jax.Array, mask: jax.Array
    ) -> jax.Array:
        return encoder.encode(weights, cfg, ids, mask, bf16=False)

    batch = _batch(count)
    output = model.forward(params, fusion, encode, batch)
    permutation = jnp.arange(count)[::-1]
    permuted = {
        key: value[:, permutation]
        if key.startswith("schema") or key == "row_seed"
        else value
        for key, value in batch.items()
    }
    changed = model.forward(
        params, dataclasses.replace(fusion, row_chunk=1), encode, permuted
    )
    np.testing.assert_allclose(
        output["candidate"][:, permutation],
        changed["candidate"],
        rtol=2e-5,
        atol=2e-5,
    )
    smaller = model.forward(params, fusion, encode, _batch(2))
    np.testing.assert_allclose(
        output["candidate"][:, :2], smaller["candidate"], rtol=2e-5, atol=2e-5
    )
    assert output["tokens"].shape == (1, count, 4)
    # JAX 0.7.2 omits a type declaration for cache cleanup.
    jax.clear_caches()  # type: ignore[no-untyped-call]


def test_padding_batch_ownership_and_shared_gradient() -> None:
    """Both encoder paths share embeddings; requests stay isolated."""
    cfg, fusion, params = smoke.tiny()

    def encode(
        weights: types.Parameters, ids: jax.Array, mask: jax.Array
    ) -> jax.Array:
        return encoder.encode(weights, cfg, ids, mask, bf16=False)

    batch = _batch(3, requests=2)
    batch["source_ids"] = (
        batch["source_ids"].at[1, 1:].set(jnp.asarray([10, 11, 12]))
    )
    output = model.forward(params, fusion, encode, batch)
    for index in range(2):
        single = {key: value[index : index + 1] for key, value in batch.items()}
        expected = model.forward(params, fusion, encode, single)
        np.testing.assert_allclose(
            output["candidate"][index],
            expected["candidate"][0],
            rtol=1e-5,
            atol=1e-5,
        )
    padded = model.forward(params, fusion, encode, _batch(3, padding=4))
    np.testing.assert_allclose(
        output["candidate"][0], padded["candidate"][0], rtol=2e-5, atol=2e-5
    )
    gradient = jax.jit(
        jax.grad(
            lambda weights: model.forward(weights, fusion, encode, batch)[
                "candidate"
            ].sum()
        )
    )(params)
    embedding = np.asarray(gradient["lfm2.embed_tokens.weight"])
    assert np.linalg.norm(embedding[7:13]) > 0
    assert np.linalg.norm(embedding[20:23]) > 0
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(gradient))
    # JAX 0.7.2 omits a type declaration for cache cleanup.
    jax.clear_caches()  # type: ignore[no-untyped-call]


def test_cached_schema_matches_and_rejects_stale_training() -> None:
    """Cached full tokens agree and cannot silently detach encoder training."""
    cfg, fusion, params = smoke.tiny()

    def encode(
        weights: types.Parameters, ids: jax.Array, mask: jax.Array
    ) -> jax.Array:
        return encoder.encode(weights, cfg, ids, mask, bf16=False)

    batch = _batch(3)
    context = ("tokenizer-v1", "templates-v1", "float32")
    stored = cache.SchemaCache.build(params, batch, encode, context=context)
    hidden = stored.get(params, batch, context=context)
    fresh = model.forward(params, fusion, encode, batch)
    cached = model.forward(params, fusion, encode, batch, schema_hidden=hidden)
    np.testing.assert_allclose(
        fresh["candidate"], cached["candidate"], rtol=1e-5, atol=1e-5
    )
    with pytest.raises(ValueError, match="training"):
        model.forward(
            params, fusion, encode, batch, training=True, schema_hidden=hidden
        )
    changed = {
        **params,
        "lfm2.embed_tokens.weight": params["lfm2.embed_tokens.weight"] + 0.1,
    }
    with pytest.raises(ValueError, match="Stale"):
        stored.get(changed, batch, context=context)
    with pytest.raises(ValueError, match="Stale"):
        stored.get(params, batch, context=("new-tokenizer",))
    # JAX 0.7.2 omits a type declaration for cache cleanup.
    jax.clear_caches()  # type: ignore[no-untyped-call]


def test_rejects_inconsistent_source_ownership() -> None:
    """A schema batch cannot silently index the wrong source request."""
    cfg, fusion, params = smoke.tiny()

    def encode(
        weights: types.Parameters, ids: jax.Array, mask: jax.Array
    ) -> jax.Array:
        return encoder.encode(weights, cfg, ids, mask, bf16=False)

    batch = _batch(2)
    batch["source_ids"] = jnp.tile(batch["source_ids"], (2, 1))
    batch["source_mask"] = jnp.tile(batch["source_mask"], (2, 1))
    with pytest.raises(ValueError, match="ownership"):
        model.forward(params, fusion, encode, batch)
    assert all(
        value.dtype == jnp.float32
        for value in model.initialize(fusion, jax.random.PRNGKey(0)).values()
    )
