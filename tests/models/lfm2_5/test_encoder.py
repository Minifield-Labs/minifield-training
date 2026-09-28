"""Published encoder inventory and tiny bidirectional numerical behavior."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from examples.magicbox import smoke
from minifield_training.core import json_io
from minifield_training.datasets import magicbox
from minifield_training.models.lfm2_5 import encoder


def test_pinned_checkpoint_inventory() -> None:
    """Match all 148 actual checkpoint shapes, with no unused LM head."""
    directory = Path(__file__).parent / "fixtures"
    path = directory / "encoder_config.json"
    assert json_io.digest_file(path) == encoder.SOURCE.config_sha256
    cfg = encoder.Adapter().parse_config(
        magicbox.object_map(json.loads(path.read_text()))
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
