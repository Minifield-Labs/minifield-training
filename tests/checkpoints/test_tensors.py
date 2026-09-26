"""Independent synthetic coverage of pretrained tensor admission."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import numpy.typing as npt
import pytest
from safetensors import TensorSpec
from safetensors import serialize_file
from safetensors.numpy import save_file

from minifield_training.checkpoints import tensors
from minifield_training.models import contracts
from minifield_training.models.lfm2_5 import model
from minifield_training.models.lfm2_5 import pretrained as model_source
from minifield_training.strategies import pretrained


class _VectorModel:
    """A separate F32 model with its own config and checkpoint vocabulary."""

    source_dtype = "F32"

    def parse_config(self, value: Mapping[str, object]) -> int:
        """Admit a vector dimension without any transformer config."""
        size = value["size"]
        if not isinstance(size, int) or size < 1:
            raise ValueError("Invalid vector dimension")
        return size

    def expected_shapes(self, cfg: int) -> Mapping[str, tuple[int, ...]]:
        """Name the vector model's single parameter."""
        return {"vector": (cfg,)}

    def validate_masters(
        self, parameters: Mapping[str, jax.Array], cfg: int
    ) -> None:
        """Check this model's configuration reached final admission."""
        assert parameters["vector"].shape == (cfg,)
        assert np.isfinite(np.asarray(parameters["vector"])).all()


def test_pretrained_loader_accepts_an_independent_model_adapter(
    tmp_path: Path,
) -> None:
    """The shared loader delegates architecture and dtype to the protocol."""
    config_bytes = b'{"size":3}'
    tokenizer_bytes = b"{}"
    (tmp_path / "config.json").write_bytes(config_bytes)
    (tmp_path / "tokenizer.json").write_bytes(tokenizer_bytes)
    weights = tmp_path / "model.safetensors"
    save_file(
        {"vector": np.array([0.25, -0.5, 0.75], dtype=np.float32)}, weights
    )
    source = contracts.PretrainedSource(
        "synthetic/vector",
        "fixed",
        hashlib.sha256(config_bytes).hexdigest(),
        hashlib.sha256(tokenizer_bytes).hexdigest(),
        hashlib.sha256(weights.read_bytes()).hexdigest(),
    )
    adapter: contracts.PretrainedModel[int] = _VectorModel()
    cfg, params = pretrained.load_verified(tmp_path, source, adapter)
    assert cfg == 3
    assert set(params) == {"vector"}
    assert params["vector"].dtype == np.float32
    np.testing.assert_array_equal(params["vector"], [0.25, -0.5, 0.75])


def _write_bf16(path: Path, shapes: dict[str, tuple[int, ...]]) -> str:
    """Use the upstream writer for exact BF16 test tensors."""
    buffers: dict[str, npt.NDArray[np.uint16]] = {}
    specs: dict[str, TensorSpec] = {}
    for name, shape in shapes.items():
        count = int(np.prod(shape))
        buffers[name] = np.full(count, 0x3F80, dtype="<u2")
        value = buffers[name]
        specs[name] = TensorSpec(
            dtype="bfloat16",
            shape=list(shape),
            data_ptr=value.ctypes.data,
            data_len=value.nbytes,
        )
    serialize_file(specs, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verified_backbone_requires_all_original_tensors(
    tmp_path: Path,
) -> None:
    """Only the downstream head may be new; source leaves stay exact."""
    cfg = {
        "model_type": "lfm2",
        "hidden_size": 4,
        "intermediate_size": 8,
        "block_auto_adjust_ff_dim": False,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "num_hidden_layers": 1,
        "layer_types": ["conv"],
    }
    config_bytes = json.dumps(cfg).encode()
    (tmp_path / "config.json").write_bytes(config_bytes)
    tokenizer_bytes = b'{"test":true}'
    (tmp_path / "tokenizer.json").write_bytes(tokenizer_bytes)
    shapes = model.expected_shapes(model.Config.from_dict(cfg))
    weights = tmp_path / "model.safetensors"
    weights_hash = _write_bf16(weights, shapes)
    source = contracts.PretrainedSource(
        "synthetic/base",
        "fixed-revision",
        hashlib.sha256(config_bytes).hexdigest(),
        hashlib.sha256(tokenizer_bytes).hexdigest(),
        weights_hash,
    )
    loaded_cfg, params = pretrained.load_verified(
        tmp_path, source, model_source.Adapter()
    )
    assert loaded_cfg.vocab_size == 8
    assert set(params) == set(shapes)
    assert all(
        np.asarray(value).dtype == np.float32 for value in params.values()
    )
    assert float(params["model.embed_tokens.weight"][0, 0]) == 1.0
    with pytest.raises(ValueError, match="SHA-256"):
        pretrained.load_verified(
            tmp_path,
            contracts.PretrainedSource(
                source.model_id,
                source.revision,
                source.config_sha256,
                source.tokenizer_sha256,
                "0" * 64,
            ),
            model_source.Adapter(),
        )
    _write_bf16(weights, {"unexpected": (1,)})
    with pytest.raises(ValueError, match="inventory"):
        tensors.load_masters(
            weights,
            shapes,
            source_dtype="BF16",
            sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        )


def test_pinned_release_declares_published_inventory() -> None:
    """Bind the public Base release, not the instruction variant."""
    assert model_source.BASE.model_id == "LiquidAI/LFM2.5-230M-Base"
    assert len(model_source.BASE.revision) == 40
    assert len(model_source.BASE.weights_sha256) == 64
