"""Published encoder inventory and tiny bidirectional numerical behavior."""

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.magicbox import smoke
from minifield_training.core import json_io
from minifield_training.kernels import normalization
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model


def test_pinned_checkpoint_inventory() -> None:
    """Match all 148 actual checkpoint shapes, with no unused LM head."""
    directory = Path(__file__).parent / "fixtures"
    path = directory / "encoder_config.json"
    assert json_io.digest_file(path) == encoder.SOURCE.config_sha256
    cfg = encoder.Adapter().parse_config(
        json_io.object_map(json.loads(path.read_text()))
    )
    published = json.loads((directory / "encoder_inventory.json").read_text())
    expected = {
        name: tuple(value["shape"]) for name, value in published.items()
    }
    assert encoder.Adapter().expected_shapes(cfg) == expected
    assert sum(np.prod(shape) for shape in expected.values()) == 354483968
    assert {value["dtype"] for value in published.values()} == {"F32"}


@pytest.mark.parametrize("bf16", [False, True])
def test_encoder_reads_future_and_preserves_padding(bf16: bool) -> None:
    """The first position sees later tokens; right padding doesn't change it."""
    cfg, _, params = smoke.tiny()
    ids = jnp.asarray([[1, 5, 6, 7]])
    mask = jnp.ones_like(ids)
    original = encoder.encode(params, cfg, ids, mask, bf16=bf16)
    changed = encoder.encode(params, cfg, ids.at[0, 3].set(8), mask, bf16=bf16)
    assert not np.allclose(
        np.asarray(original[:, 0], dtype=np.float32),
        np.asarray(changed[:, 0], dtype=np.float32),
    )
    padded = encoder.encode(
        params,
        cfg,
        jnp.pad(ids, ((0, 0), (0, 3))),
        jnp.pad(mask, ((0, 0), (0, 3))),
        bf16=bf16,
    )
    np.testing.assert_allclose(
        np.asarray(original, dtype=np.float32),
        np.asarray(padded[:, :4], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(padded[:, 4:], np.zeros((1, 3, 16)))


@pytest.mark.parametrize("bf16", [False, True])
def test_packed_segments_encode_as_separate_sequences(bf16: bool) -> None:
    """Two packed sequences match their independently padded encodings."""
    cfg, _, params = smoke.tiny()
    first, second = [1, 5, 6, 7, 9], [1, 8, 3]
    packed = jnp.asarray([first + second + [0, 0]])
    segments = jnp.asarray([[1] * 5 + [2] * 3 + [0, 0]])
    positions = jnp.asarray([list(range(5)) + list(range(3)) + [0, 0]])
    actual = encoder.encode(
        params,
        cfg,
        packed,
        segments != 0,
        bf16=bf16,
        segment_ids=segments,
        positions=positions,
    )
    separate = encoder.encode(
        params,
        cfg,
        jnp.asarray([first, second + [0, 0]]),
        jnp.asarray([[1] * 5, [1] * 3 + [0, 0]]),
        bf16=bf16,
    )
    np.testing.assert_allclose(
        np.asarray(actual[0, :5], dtype=np.float32),
        np.asarray(separate[0], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual[0, 5:8], dtype=np.float32),
        np.asarray(separate[1, :3], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(actual[0, 8:], np.zeros((2, 16)))
    with pytest.raises(ValueError, match="segment IDs and positions"):
        encoder.encode(params, cfg, packed, segments != 0, segment_ids=segments)


def test_encoder_owned_context_boundary() -> None:
    """Admit 8192 positions and reject 8193 before model computation."""
    cfg = model.Config(4, 8, 1, 1, 2, ("conv",))
    params = {
        name: jnp.ones(shape)
        for name, shape in encoder.Adapter().expected_shapes(cfg).items()
    }
    ids = jnp.zeros((1, 8192), dtype=jnp.int32)
    assert encoder.MAX_SEQUENCE_LENGTH == 8192
    assert encoder.encode(
        params, cfg, ids, jnp.ones_like(ids), bf16=False
    ).shape == (1, 8192, 4)
    ids = jnp.zeros((1, 8193), dtype=jnp.int32)
    with pytest.raises(ValueError, match="trained context limit"):
        encoder.encode(params, cfg, ids, jnp.ones_like(ids))


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize(
    "layout",
    [
        ("conv",) * 3,
        ("full_attention",) * 3,
        ("conv", "conv", "full_attention", "conv", "full_attention"),
        ("full_attention", "conv", "full_attention", "conv", "conv"),
    ],
)
def test_scanned_layers_match_unrolled_outputs_and_gradients(
    layout: tuple[str, ...], bf16: bool
) -> None:
    """Check routing and all master gradients against the old layer schedule.

    The independent reference retains the old unrolled execution order and
    shares only the unchanged block equations. Repeated kinds have distinct
    random weights, so routing every occurrence to one slot cannot pass.
    """
    cfg = model.Config(8, 16, 2, 1, 32, layout)
    rng = np.random.default_rng(31)
    params = {
        name: jnp.asarray(
            rng.normal(1 if len(shape) == 1 else 0, 0.08, shape),
            dtype=jnp.float32,
        )
        for name, shape in encoder.Adapter().expected_shapes(cfg).items()
    }
    ids = jnp.asarray([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = jnp.asarray([[1, 1, 1, 0], [1, 1, 0, 0]])
    coefficients = jnp.asarray(rng.normal(size=(2, 4, 8)), dtype=jnp.float32)

    def unrolled(weights: types.Parameters) -> jax.Array:
        """Apply the pre-scan schedule with no stacked parameter routing."""
        hidden = weights["lfm2.embed_tokens.weight"][ids].astype(
            jnp.bfloat16 if bf16 else jnp.float32
        )
        for index, kind in enumerate(layout):
            # Isolate scheduling from the unchanged private block equations.
            block = functools.partial(
                encoder._block, cfg=cfg, kind=kind  # pylint: disable=protected-access
            )
            # Preserve the pre-scan checkpoint policy in the reference.
            rematerialize = jax.checkpoint  # type: ignore[attr-defined]
            hidden = rematerialize(block)(
                hidden,
                mask,
                types.slice_parameters(weights, f"lfm2.layers.{index}."),
            )
        return (
            normalization.rms_norm(
                hidden, weights["lfm2.embedding_norm.weight"], cfg.norm_eps
            )
            * mask[..., None]
        )

    def evaluate(
        weights: types.Parameters, *, reference: bool
    ) -> tuple[jax.Array, jax.Array]:
        """Differentiate a nonuniform scalar probe of every output token."""
        output = (
            unrolled(weights)
            if reference
            else encoder.encode(weights, cfg, ids, mask, bf16=bf16)
        )
        return (output.astype(jnp.float32) * coefficients).sum(), output

    # CPU can elide BF16 rounding across unrolled layer boundaries. Enforce
    # declared dtypes in both graphs to isolate scheduling from that rewrite.
    precision = {"xla_allow_excess_precision": False}
    (_, expected), expected_grad = jax.jit(
        jax.value_and_grad(
            lambda weights: evaluate(weights, reference=True), has_aux=True
        ),
        compiler_options=precision,
    )(params)
    (_, actual), actual_grad = jax.jit(
        jax.value_and_grad(
            lambda weights: evaluate(weights, reference=False), has_aux=True
        ),
        compiler_options=precision,
    )(params)
    tolerance = (2e-2, 2e-3) if bf16 else (1e-4, 1e-5)
    for left, right in [
        (actual, expected),
        *zip(
            jax.tree.leaves(actual_grad),
            jax.tree.leaves(expected_grad),
            strict=True,
        ),
    ]:
        np.testing.assert_allclose(
            np.asarray(left, dtype=np.float32),
            np.asarray(right, dtype=np.float32),
            rtol=tolerance[0],
            atol=tolerance[1],
        )
    assert all(value.dtype == jnp.float32 for value in actual_grad.values())
    # JAX 0.7.2 omits a type declaration for cache cleanup.
    jax.clear_caches()  # type: ignore[no-untyped-call]
