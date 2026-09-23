"""Independent contracts for immutable parameter metadata records."""

import dataclasses

import pytest

from minifield_training.core import parameters


def test_inventory_views_preserve_supplied_order() -> None:
    """Views keep spec order and multiplicity without sorting or validation."""
    specs = (
        parameters.FullParameterSpec(
            name="z.weight",
            shape=(2, 3),
            source_dtype="bfloat16",
            master_dtype="float32",
            trainable=True,
            decayed=True,
            quantized=False,
        ),
        parameters.FullParameterSpec(
            name="a.embedding",
            shape=(5, 7),
            source_dtype="bfloat16",
            master_dtype="float32",
            trainable=False,
            decayed=False,
            quantized=False,
        ),
        parameters.FullParameterSpec(
            name="m.scalar",
            shape=(),
            source_dtype="bfloat16",
            master_dtype="float32",
            trainable=True,
            decayed=False,
            quantized=False,
        ),
    )
    inventory = parameters.FullParameterInventory(
        specs=specs,
        parameter_count=999,
        source_dtype="bfloat16",
        master_dtype="float32",
        quantization_profile=None,
        sha256="caller-supplied",
    )
    assert inventory.specs == specs
    assert inventory.names == ("z.weight", "a.embedding", "m.scalar")
    assert inventory.trainable_names == ("z.weight", "m.scalar")
    assert inventory.frozen_names == ("a.embedding",)
    assert inventory.trainable_parameter_count == 7
    assert inventory.parameter_count == 999
    assert inventory.sha256 == "caller-supplied"


def test_empty_inventory_returns_empty_views() -> None:
    """An inventory with no specs has empty name tuples and zero count."""
    inventory = parameters.FullParameterInventory(
        specs=(),
        parameter_count=0,
        source_dtype="bfloat16",
        master_dtype="float32",
        quantization_profile=None,
        sha256="caller-supplied",
    )
    assert isinstance(inventory.names, tuple)
    assert not inventory.names
    assert not inventory.trainable_names
    assert not inventory.frozen_names
    assert inventory.trainable_parameter_count == 0


def test_frozen_only_inventory_counts_nothing_trainable() -> None:
    """Frozen rows stay in names and frozen_names but never train."""
    spec = parameters.FullParameterSpec(
        name="model.embed_tokens.weight",
        shape=(4, 4),
        source_dtype="float16",
        master_dtype="float32",
        trainable=False,
        decayed=False,
        quantized=True,
    )
    inventory = parameters.FullParameterInventory(
        specs=(spec,),
        parameter_count=16,
        source_dtype="float16",
        master_dtype="float32",
        quantization_profile="qat-int8-v1",
        sha256="d" * 64,
    )
    assert inventory.names == ("model.embed_tokens.weight",)
    assert not inventory.trainable_names
    assert inventory.frozen_names == ("model.embed_tokens.weight",)
    assert inventory.trainable_parameter_count == 0
    assert inventory.quantization_profile == "qat-int8-v1"


def test_zero_sized_dimension_contributes_zero() -> None:
    """A trainable row whose shape contains 0 adds no scalars."""
    spec = parameters.FullParameterSpec(
        name="empty.weight",
        shape=(0, 3),
        source_dtype="bfloat16",
        master_dtype="float32",
        trainable=True,
        decayed=False,
        quantized=False,
    )
    inventory = parameters.FullParameterInventory(
        specs=(spec,),
        parameter_count=0,
        source_dtype="bfloat16",
        master_dtype="float32",
        quantization_profile=None,
        sha256="caller-supplied",
    )
    assert inventory.names == ("empty.weight",)
    assert inventory.trainable_parameter_count == 0


def test_duplicate_names_remain_in_supplied_order() -> None:
    """Duplicate rows are not deduplicated and each contributes its count."""
    first = parameters.FullParameterSpec(
        name="shared.weight",
        shape=(2,),
        source_dtype="bfloat16",
        master_dtype="float32",
        trainable=True,
        decayed=False,
        quantized=False,
    )
    second = parameters.FullParameterSpec(
        name="shared.weight",
        shape=(3,),
        source_dtype="bfloat16",
        master_dtype="float32",
        trainable=True,
        decayed=True,
        quantized=False,
    )
    inventory = parameters.FullParameterInventory(
        specs=(first, second),
        parameter_count=5,
        source_dtype="bfloat16",
        master_dtype="float32",
        quantization_profile=None,
        sha256="caller-supplied",
    )
    assert inventory.names == ("shared.weight", "shared.weight")
    assert inventory.trainable_names == ("shared.weight", "shared.weight")
    assert inventory.trainable_parameter_count == 5


@pytest.mark.parametrize(
    ("trainable", "expected_views"),
    [
        (False, ((), ("any.name",), 0)),
        (True, (("any.name",), (), 2)),
    ],
)
@pytest.mark.parametrize("decayed", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_inventory_views_use_trainability_only(
    trainable: bool,
    expected_views: tuple[tuple[str, ...], tuple[str, ...], int],
    decayed: bool,
    quantized: bool,
) -> None:
    """Decay, quantization and supplied dtypes don't change membership."""
    spec = parameters.FullParameterSpec(
        name="any.name",
        shape=(2,),
        source_dtype="custom-source",
        master_dtype="custom-master",
        trainable=trainable,
        decayed=decayed,
        quantized=quantized,
    )
    assert spec.trainable is trainable
    assert spec.decayed is decayed
    assert spec.quantized is quantized
    assert spec.source_dtype == "custom-source"
    assert spec.master_dtype == "custom-master"
    inventory = parameters.FullParameterInventory(
        specs=(spec,),
        parameter_count=999,
        source_dtype="inventory-source",
        master_dtype="inventory-master",
        quantization_profile=None,
        sha256="caller-supplied",
    )
    assert inventory.names == ("any.name",)
    assert (
        inventory.trainable_names,
        inventory.frozen_names,
        inventory.trainable_parameter_count,
    ) == expected_views
    assert inventory.source_dtype == "inventory-source"
    assert inventory.master_dtype == "inventory-master"
    assert inventory.quantization_profile is None


def test_records_reject_field_assignment() -> None:
    """Both frozen dataclasses raise FrozenInstanceError on assignment."""
    spec = parameters.FullParameterSpec(
        name="x.weight",
        shape=(1,),
        source_dtype="bfloat16",
        master_dtype="float32",
        trainable=True,
        decayed=False,
        quantized=False,
    )
    inventory = parameters.FullParameterInventory(
        specs=(spec,),
        parameter_count=1,
        source_dtype="bfloat16",
        master_dtype="float32",
        quantization_profile=None,
        sha256="caller-supplied",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "renamed"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        inventory.sha256 = "replaced"  # type: ignore[misc]


def test_records_compare_by_value() -> None:
    """Equivalent records are equal; differing fields make them unequal."""
    spec = parameters.FullParameterSpec(
        name="x.weight",
        shape=(2, 3),
        source_dtype="bfloat16",
        master_dtype="float32",
        trainable=True,
        decayed=True,
        quantized=False,
    )
    same = dataclasses.replace(spec)
    different = dataclasses.replace(spec, trainable=False)
    assert spec == same
    assert spec != different
    inventory = parameters.FullParameterInventory(
        specs=(spec,),
        parameter_count=6,
        source_dtype="bfloat16",
        master_dtype="float32",
        quantization_profile=None,
        sha256="caller-supplied",
    )
    assert inventory == dataclasses.replace(inventory)
    assert inventory != dataclasses.replace(inventory, parameter_count=7)
