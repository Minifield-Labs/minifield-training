"""Portable notebook, inference bundle, and tiny optimizer integration."""

import ast
import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from examples.magicbox import bundle as magicbox_bundle
from examples.magicbox import composition as magicbox
from examples.magicbox import smoke
from minifield_training.batching import schema_fields as batching
from minifield_training.core import json_io
from minifield_training.models.lfm2_5 import encoder
from minifield_training.objectives import schema_fields as objective


def test_notebook_python_cells() -> None:
    """Verify the committed notebook's executable, unexecuted Python cells."""
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/kaggle_magicbox_lfm350m_tpu_v5e_8.ipynb"
    )
    actual = json.loads(path.read_text())
    for cell in actual["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert not cell["outputs"] and cell["execution_count"] is None


def test_tiny_bundle_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen synthetic encoder config exercises the real bundle loader."""
    cfg, fusion, parameters = smoke.tiny()
    config = {
        "model_type": "lfm2",
        "architectures": ["Lfm2BidirectionalForMaskedLM"],
        "hidden_size": 16,
        "intermediate_size": 32,
        "block_auto_adjust_ff_dim": False,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 128,
        "num_hidden_layers": 2,
        "layer_types": ["conv", "full_attention"],
    }
    path = tmp_path / "encoder.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(
        encoder,
        "SOURCE",
        dataclasses.replace(
            encoder.SOURCE, config_sha256=json_io.digest_file(path)
        ),
    )
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"synthetic":true}')
    (tokenizer / "contract.json").write_text('{"offset_policy":"synthetic"}')
    bundle = tmp_path / "bundle"
    magicbox_bundle.save(
        bundle,
        parameters,
        cfg,
        fusion,
        encoder_config=path,
        tokenizer=tokenizer,
        step=4,
    )
    restored_cfg, restored_fusion, restored, decode = magicbox_bundle.load(
        bundle
    )
    assert (restored_cfg, restored_fusion) == (cfg, fusion)
    assert decode == {"presence_threshold": 0.5, "confidence": None}
    assert len(restored) == len(parameters)
    packed = batching.build(
        [smoke.fixture()],
        batching.Shape(1, 1, 16, 64, 8, 128, 0),
        seed=0,
        update=0,
        weighting=objective.balance_types,
    )
    batch = {
        key: jnp.asarray(value[0]) for key, value in packed.microbatches.items()
    }
    expected = magicbox.forward(parameters, cfg, fusion, batch, bf16=False)
    actual = magicbox.forward(restored, cfg, fusion, batch, bf16=False)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    (bundle / "tokenizer/contract.json").write_text("{}")
    with pytest.raises(
        ValueError, match="SHA-256 mismatch: tokenizer/contract.json"
    ):
        magicbox_bundle.load(bundle)


def test_tiny_optimization_and_exact_checkpoint_resume() -> None:
    """All four typed objectives learn and a resumed next update is exact."""
    report = smoke.run(80)
    assert report["checkpoint_next_update"] == "exact"
    assert float(str(report["final_loss"])) < 0.01
