"""Dense bundle fixtures cover metadata freedom and byte integrity."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from safetensors import safe_open

from minifield_training.checkpoints import bundle
from minifield_training.core import json_io
from minifield_training.core import parameters


def _inventory() -> parameters.FullParameterInventory:
    return parameters.build_inventory(
        {"projection": (2, 2)},
        format_id="synthetic-projection/1",
        source_dtype="float32",
        decayed_names=frozenset(),
    )


def _save(directory: Path, metadata: dict[str, object]) -> None:
    asset = directory.parent / "vocabulary.json"
    asset.write_text('["red","blue"]', encoding="utf-8")
    bundle.save(
        directory,
        {"projection": jnp.asarray([[1, 2], [3, 4]], dtype=jnp.float32)},
        _inventory(),
        metadata=metadata,
        assets={"text/vocabulary.json": asset},
        source_model="independent/toy",
        source_revision="revision-7",
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"format": "toy.classifier/2", "classes": ["red", "blue"]},
        {"format": "toy.regressor/1", "range": {"min": 0, "max": 5}},
    ],
)
def test_arbitrary_model_metadata_and_dense_values_round_trip(
    tmp_path: Path, metadata: dict[str, object]
) -> None:
    """The writer knows file contracts but has no model configuration policy."""
    destination = tmp_path / "bundle"
    _save(destination, metadata)
    expected = frozenset({"model.safetensors", "text/vocabulary.json"})
    observed = bundle.inspect(destination, expected_files=expected)
    assert {
        key: value for key, value in observed.items() if key != "files"
    } == metadata
    restored = bundle.load_parameters(
        destination, _inventory(), expected_files=expected
    )
    np.testing.assert_array_equal(restored["projection"], [[1, 2], [3, 4]])
    with safe_open(
        str(destination / "model.safetensors"), framework="np"
    ) as stored:
        assert stored.metadata() == {
            "format": "pt",
            "source_model": "independent/toy",
            "source_revision": "revision-7",
            "inventory_sha256": _inventory().sha256,
            "producer": "minifield-training-dense-effective-v1",
        }
    with pytest.raises(FileExistsError):
        _save(destination, metadata)


@pytest.mark.parametrize("asset", ["model.safetensors", "text/vocabulary.json"])
def test_changed_assets_are_rejected(tmp_path: Path, asset: str) -> None:
    """Every declared file participates in admission, including side assets."""
    destination = tmp_path / "bundle"
    _save(destination, {"format": "toy/1"})
    (destination / asset).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        bundle.inspect(
            destination,
            expected_files=frozenset(
                {"model.safetensors", "text/vocabulary.json"}
            ),
        )


def test_foreign_asset_inventory_and_shapes_are_rejected(
    tmp_path: Path,
) -> None:
    """Valid bundles must match the caller's files and parameter shapes."""
    destination = tmp_path / "bundle"
    _save(destination, {"format": "toy/1"})
    with pytest.raises(ValueError, match="inventory"):
        bundle.inspect(
            destination, expected_files=frozenset({"model.safetensors"})
        )
    wrong = parameters.build_inventory(
        {"projection": (4,)}, format_id="other/1", decayed_names=frozenset()
    )
    with pytest.raises(ValueError, match="shape/dtype"):
        bundle.load_parameters(
            destination,
            wrong,
            expected_files=frozenset(
                {"model.safetensors", "text/vocabulary.json"}
            ),
        )
    metadata = json.loads(
        (destination / "config.json").read_text(encoding="utf-8")
    )
    metadata["files"]["../vocabulary.json"] = metadata["files"].pop(
        "text/vocabulary.json"
    )
    (destination / "config.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid artifact path"):
        bundle.inspect(destination, expected_files=frozenset(metadata["files"]))


@pytest.mark.parametrize(
    "name", ["../escape", "model.safetensors", "config.json"]
)
def test_invalid_asset_does_not_publish_partial_bundle(
    tmp_path: Path, name: str
) -> None:
    """Publication never leaves a usable-looking directory after bad input."""
    asset = tmp_path / "asset"
    asset.write_bytes(b"auxiliary")
    destination = tmp_path / "bundle"
    with pytest.raises(ValueError, match="artifact path|Reserved"):
        bundle.save(
            destination,
            {"projection": jnp.ones((2, 2), dtype=jnp.float32)},
            _inventory(),
            metadata={"format": "toy/1"},
            assets={name: asset},
            source_model="toy",
            source_revision="7",
        )
    assert not destination.exists()


def test_serialization_failure_does_not_publish_partial_bundle(
    tmp_path: Path,
) -> None:
    """Atomic staging is discarded when caller metadata cannot be serialized."""
    with pytest.raises(ValueError, match="JSON compliant"):
        _save(tmp_path / "bundle", {"format": "toy/1", "bad": float("nan")})
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle-*"))


def test_metadata_adapter_load_inspects_assets_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata admission adds no redundant full-bundle checksum pass."""
    destination = tmp_path / "bundle"
    _save(destination, {"format": "toy/1"})
    hashed: list[str] = []
    original_digest = json_io.digest_file

    def track_digest(path: Path) -> str:
        hashed.append(path.relative_to(destination).as_posix())
        return original_digest(path)

    def admit(metadata: dict[str, object]) -> parameters.FullParameterInventory:
        assert metadata["format"] == "toy/1"
        assert sorted(hashed) == ["model.safetensors", "text/vocabulary.json"]
        return _inventory()

    monkeypatch.setattr(json_io, "digest_file", track_digest)
    metadata, restored = bundle.load(
        destination,
        expected_files=frozenset({"model.safetensors", "text/vocabulary.json"}),
        inventory=admit,
    )
    assert metadata["format"] == "toy/1"
    np.testing.assert_array_equal(restored["projection"], [[1, 2], [3, 4]])
    assert hashed.count("text/vocabulary.json") == 1
    assert hashed.count("model.safetensors") == 2


def test_adapter_runs_only_after_artifact_integrity_passes(
    tmp_path: Path,
) -> None:
    """A metadata adapter never receives a bundle containing changed assets."""
    destination = tmp_path / "bundle"
    _save(destination, {"format": "toy/1"})
    (destination / "text/vocabulary.json").write_bytes(b"changed")

    def unexpected_adapter(
        metadata: dict[str, object],
    ) -> parameters.FullParameterInventory:
        pytest.fail(f"Adapter ran before artifact admission: {metadata}")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        bundle.load(
            destination,
            expected_files=frozenset(
                {"model.safetensors", "text/vocabulary.json"}
            ),
            inventory=unexpected_adapter,
        )


def test_adapter_cannot_replace_verified_weight_digest(tmp_path: Path) -> None:
    """Tensor admission binds to the digest verified before the adapter."""
    destination = tmp_path / "bundle"
    _save(destination, {"format": "toy/1"})

    def change_weights(
        metadata: dict[str, object],
    ) -> parameters.FullParameterInventory:
        weights = destination / "model.safetensors"
        weights.write_bytes(b"replaced after inspection")
        json_io.object_map(metadata["files"])["model.safetensors"] = (
            json_io.digest_file(weights)
        )
        return _inventory()

    with pytest.raises(ValueError, match="Pretrained tensor SHA-256 mismatch"):
        bundle.load(
            destination,
            expected_files=frozenset(
                {"model.safetensors", "text/vocabulary.json"}
            ),
            inventory=change_weights,
        )
