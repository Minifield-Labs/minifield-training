"""Packed group-128 inference weights match the QAT forward exactly."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from minifield_training.checkpoints import inference_output
from minifield_training.core import json_io
from minifield_training.core import parameters
from minifield_training.kernels import quantization as kernels
from minifield_training.strategies import quantization


def test_pack_puts_the_first_code_in_the_lowest_bits() -> None:
    """The runtime contract: low nibble first, 2-bit ternary lowest first."""
    nf4 = np.asarray([[1, 2] * 64], np.uint8)
    assert inference_output.pack(nf4, 2)[0, 0] == 0x21
    ternary = np.asarray([[0, 1, 2, 1] * 32], np.uint8)
    assert inference_output.pack(ternary, 4)[0, 0] == 0b01_10_01_00
    for codes, per_byte in ((nf4, 2), (ternary, 4)):
        np.testing.assert_array_equal(
            inference_output.unpack(
                inference_output.pack(codes, per_byte), per_byte
            ),
            codes,
        )


@pytest.mark.parametrize(
    "kind", ["nf4-g128-absmax-f16-v1", "ternary-g128-absmax-f16-v1"]
)
def test_packed_output_decodes_to_the_qat_effective_weights(
    tmp_path: Path, kind: str
) -> None:
    """Quantized matrices round-trip to the forward values; others are exact."""
    plan = quantization.NamedQuantization(
        kernels.Group128Quantizer(kind), frozenset({"matrix"})
    )
    inventory = parameters.build_inventory(
        {"matrix": (3, 256), "vector": (5,)},
        format_id="packed-test/1",
        decayed_names=frozenset({"matrix"}),
        quantization_profile=plan.identity,
        quantized_names=frozenset({"matrix"}),
    )
    masters = {
        "matrix": jax.random.normal(jax.random.PRNGKey(0), (3, 256)),
        "vector": jnp.arange(5, dtype=jnp.float32),
    }
    masters["matrix"] = masters["matrix"].at[1, 128:].set(0)
    path = tmp_path / "model.safetensors"
    inference_output.PackedGroup128Output().write(
        path, masters, inventory, source_model="m", source_revision="r"
    )
    loaded = inference_output.load_packed(
        path, inventory, sha256=json_io.digest_file(path)
    )
    effective = quantization.apply(masters, inventory, plan)
    for name, value in effective.items():
        np.testing.assert_array_equal(loaded[name], value)


def test_column_major_device_arrays_are_written_in_row_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TPU transfers can return column-major arrays; bytes must be row order."""
    kind = "nf4-g128-absmax-f16-v1"
    plan = quantization.NamedQuantization(
        kernels.Group128Quantizer(kind), frozenset({"matrix"})
    )
    inventory = parameters.build_inventory(
        {"matrix": (4, 256)},
        format_id="packed-test/1",
        decayed_names=frozenset({"matrix"}),
        quantization_profile=plan.identity,
        quantized_names=frozenset({"matrix"}),
    )
    masters = {"matrix": jax.random.normal(jax.random.PRNGKey(1), (4, 256))}
    original = kernels.codes

    def column_major(
        weight: jax.Array, name: str
    ) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        codes, scales = original(weight, name)
        return np.asfortranarray(codes), np.asfortranarray(scales)

    monkeypatch.setattr(kernels, "codes", column_major)
    path = tmp_path / "model.safetensors"
    inference_output.PackedGroup128Output().write(
        path, masters, inventory, source_model="m", source_revision="r"
    )
    loaded = inference_output.load_packed(
        path, inventory, sha256=json_io.digest_file(path)
    )
    expected = quantization.apply(masters, inventory, plan)
    np.testing.assert_array_equal(loaded["matrix"], expected["matrix"])
