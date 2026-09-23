"""Checkpoint loading contract for the LFM2.5 model adapter."""

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
