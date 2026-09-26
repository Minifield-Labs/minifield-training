"""Checkpoint loading contract for the LFM2.5 model adapter."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from minifield_training.models.lfm2_5 import model


def _config() -> model.Config:
    """Return a tiny two-layer config covering both block kinds."""
    return model.Config(
        hidden_size=16,
        intermediate_size=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        layer_types=("conv", "full_attention"),
        conv_kernel=3,
        tied_embeddings=True,
    )


def _checkpoint(cfg: model.Config, seed: int = 0) -> dict[str, jax.Array]:
    """Return FP32 tensors laid out exactly like a loaded checkpoint."""
    rng = np.random.default_rng(seed)
    return {
        name: jnp.asarray(rng.standard_normal(shape) * 0.3, dtype=jnp.float32)
        for name, shape in model.expected_shapes(cfg).items()
    }


def test_loads_full_inventory_and_runs() -> None:
    """A dict matching ``expected_shapes`` validates and decodes."""
    cfg = _config()
    params = _checkpoint(cfg)
    model.validate_parameters(params, cfg)
    ids = jnp.asarray([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)
    mask = jnp.ones((2, 4), dtype=jnp.int32)
    logits = model.forward(params, ids, mask, cfg)
    assert logits.shape == (2, 4, cfg.vocab_size)
    assert logits.dtype == jnp.float32


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_block_rematerialization_preserves_forward_and_gradients(
    dtype: type[np.generic],
) -> None:
    """Both block policies agree for a model with convolution and attention."""
    cfg = _config()
    params = _checkpoint(cfg, seed=3)
    ids = jnp.asarray([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)
    mask = jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=jnp.int32)

    def output_and_gradient(
        rematerialize_blocks: bool,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Differentiate one full-model scalar through the selected policy."""

        def scored(parameters: dict[str, jax.Array]) -> jax.Array:
            """Weight logits by token position so every row contributes."""
            values = model.forward(
                parameters,
                ids,
                mask,
                cfg,
                dtype=dtype,
                rematerialize_blocks=rematerialize_blocks,
            )
            return jnp.sum(values * jnp.arange(1, 5)[None, :, None])

        return cast(
            tuple[jax.Array, dict[str, jax.Array]],
            jax.jit(jax.value_and_grad(scored))(params),
        )

    checkpointed, checkpointed_grad = output_and_gradient(True)
    plain, plain_grad = output_and_gradient(False)
    np.testing.assert_array_equal(checkpointed, plain)
    assert checkpointed_grad.keys() == plain_grad.keys()
    maximum_relative_error = 1e-6 if dtype == jnp.float32 else 0.025
    for name in checkpointed_grad:
        original = np.asarray(checkpointed_grad[name], dtype=np.float64)
        changed = np.asarray(plain_grad[name], dtype=np.float64)
        assert np.isfinite(original).all()
        assert np.isfinite(changed).all()
        relative_error = float(np.linalg.norm(original - changed)) / max(
            float(np.linalg.norm(original)), 1e-12
        )
        assert relative_error <= maximum_relative_error, name


def test_hf_config_dict_loads() -> None:
    """``from_dict`` accepts a published-style HF configuration."""
    cfg = model.Config.from_dict(
        {
            "model_type": "lfm2",
            "hidden_size": 16,
            "intermediate_size": 36,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "layer_types": ["conv", "full_attention"],
            "vocab_size": 32,
            "block_auto_adjust_ff_dim": False,
        }
    )
    assert cfg.layer_types == ("conv", "full_attention")
    assert cfg.head_dim == 4
    assert cfg.intermediate_size == 36


def test_rejects_malformed_inventories() -> None:
    """Missing, extra, misshaped, and non-FP32 leaves all fail to load."""
    cfg = _config()
    params = _checkpoint(cfg)

    missing = dict(params)
    del missing["model.layers.1.self_attn.q_proj.weight"]
    with pytest.raises(ValueError, match="inventory mismatch"):
        model.validate_parameters(missing, cfg)

    extra = dict(params)
    extra["bogus.weight"] = jnp.zeros((1,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="inventory mismatch"):
        model.validate_parameters(extra, cfg)

    wrong_shape = dict(params)
    wrong_shape["model.embedding_norm.weight"] = jnp.zeros(
        (cfg.hidden_size + 1,), dtype=jnp.float32
    )
    with pytest.raises(ValueError, match="wrong shape"):
        model.validate_parameters(wrong_shape, cfg)

    wrong_dtype = dict(params)
    wrong_dtype["model.embed_tokens.weight"] = jnp.zeros(
        model.expected_shapes(cfg)["model.embed_tokens.weight"],
        dtype=jnp.bfloat16,
    )
    with pytest.raises(ValueError, match="not float32"):
        model.validate_parameters(wrong_dtype, cfg)


def test_rejects_adapter_leaves_with_named_error() -> None:
    """LoRA leaves get their own rejection, not a generic mismatch."""
    cfg = _config()
    params = _checkpoint(cfg)
    params["model.layers.0.feed_forward.w1.lora_a.weight"] = jnp.zeros(
        (4, 4), dtype=jnp.float32
    )
    with pytest.raises(ValueError, match="LoRA"):
        model.validate_parameters(params, cfg)


def test_masters_reject_nonfinite_leaves() -> None:
    """Structural validation passes but the master boundary catches NaN."""
    cfg = _config()
    params = _checkpoint(cfg)
    params["model.embedding_norm.weight"] = jnp.full(
        (cfg.hidden_size,), jnp.nan, dtype=jnp.float32
    )
    model.validate_parameters(params, cfg)
    with pytest.raises(ValueError, match="non-finite"):
        model.validate_masters(params, cfg)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
def test_logits_selects_head_and_preserves_projection_precision(
    tied: bool, dtype: type[np.generic]
) -> None:
    """The adapter selects the stored head and returns FP32 projected logits."""
    cfg = model.Config(
        hidden_size=2,
        intermediate_size=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=2,
        layer_types=("full_attention",),
        tied_embeddings=tied,
    )
    hidden = jnp.array([[[2.0, -4.0]]], dtype=dtype)
    parameters = {
        "model.embed_tokens.weight": jnp.array([[0.5, -0.25], [0.25, 0.5]]),
        "lm_head.weight": jnp.array([[1.0, 0.5], [-0.5, 0.25]]),
    }
    result = model.logits(hidden, parameters, cfg)
    np.testing.assert_array_equal(
        result, [[[2.0, -1.5]]] if tied else [[[0.0, -2.0]]]
    )
    assert result.dtype == jnp.float32


def test_inventory_declares_family_decay_policy() -> None:
    """Tied embedding and unfrozen matrices decay; norm and conv taps don't."""
    frozen = frozenset({"model.layers.0.conv.in_proj.weight"})
    inventory = model.parameter_inventory(_config(), frozen_names=frozen)
    expected = {
        "model.embed_tokens.weight",
        "model.layers.0.conv.out_proj.weight",
        "model.layers.0.feed_forward.w1.weight",
        "model.layers.0.feed_forward.w2.weight",
        "model.layers.0.feed_forward.w3.weight",
        "model.layers.1.feed_forward.w1.weight",
        "model.layers.1.feed_forward.w2.weight",
        "model.layers.1.feed_forward.w3.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
        "model.layers.1.self_attn.v_proj.weight",
        "model.layers.1.self_attn.out_proj.weight",
    }
    assert {spec.name for spec in inventory.specs if spec.decayed} == expected
    assert frozenset(inventory.frozen_names) == frozen
    assert "lm_head.weight" not in inventory.names
    assert inventory.sha256 == (
        "385ed95f6c8b547cdefe500b3d7d6d14f7545a7c3d6f4a6d9e7c2199ec6e8155"
    )
