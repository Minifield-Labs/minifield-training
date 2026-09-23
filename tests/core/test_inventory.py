"""Host-only parameter inventory construction and explicit policy contracts."""

import pytest

from minifield_training.core import parameters


def test_builder_preserves_explicit_membership() -> None:
    """A matrix may avoid decay while a selected vector receives it."""
    inventory = parameters.build_inventory(
        {
            "stored": (2, 2),
            "scale": (2,),
            "projection": (2, 2),
            "embedding": (3, 2),
        },
        format_id="test.inventory/1",
        decayed_names=frozenset({"projection", "scale"}),
        frozen_names=frozenset({"stored"}),
    )
    assert inventory.names == ("embedding", "projection", "scale", "stored")
    assert inventory.trainable_names == ("embedding", "projection", "scale")
    assert inventory.frozen_names == ("stored",)
    assert inventory.parameter_count == 16
    assert inventory.trainable_parameter_count == 12
    assert [(spec.shape, spec.decayed) for spec in inventory.specs] == [
        ((3, 2), False),
        ((2, 2), True),
        ((2,), True),
        ((2, 2), False),
    ]
    assert all(spec.master_dtype == "float32" for spec in inventory.specs)


def test_empty_decay_set_excludes_all_parameters() -> None:
    """Explicitly excluding every leaf leaves matrix parameters trainable."""
    inventory = parameters.build_inventory(
        {"embedding": (3, 2), "projection": (2, 2)},
        format_id="test.inventory/1",
        decayed_names=frozenset(),
    )
    assert not any(spec.decayed for spec in inventory.specs)
    assert inventory.trainable_names == ("embedding", "projection")


@pytest.mark.parametrize(
    ("decayed_names", "frozen_names", "message"),
    [
        (
            frozenset({"unknown"}),
            frozenset(),
            "Decay set names unknown tensors: unknown",
        ),
        (
            frozenset({"embedding"}),
            frozenset({"embedding"}),
            "Frozen tensors cannot receive weight decay: embedding",
        ),
    ],
)
def test_builder_rejects_invalid_decay_membership(
    decayed_names: frozenset[str],
    frozen_names: frozenset[str],
    message: str,
) -> None:
    """Unknown and frozen decay names fail before metadata is returned."""
    with pytest.raises(ValueError, match=message):
        parameters.build_inventory(
            {"embedding": (3, 2)},
            format_id="test.inventory/1",
            decayed_names=decayed_names,
            frozen_names=frozen_names,
        )


def test_digest_binds_decay_and_frozen_membership() -> None:
    """Order preserves identity; changing decay or trainability changes it."""
    shapes = {"projection": (2, 2), "embedding": (3, 2)}
    inventory = parameters.build_inventory(
        shapes,
        format_id="test.inventory/1",
        decayed_names=frozenset({"projection"}),
    )
    reordered = parameters.build_inventory(
        dict(reversed(tuple(shapes.items()))),
        format_id="test.inventory/1",
        decayed_names=frozenset({"projection"}),
    )
    without_decay = parameters.build_inventory(
        shapes,
        format_id="test.inventory/1",
        decayed_names=frozenset(),
    )
    frozen_embedding = parameters.build_inventory(
        shapes,
        format_id="test.inventory/1",
        decayed_names=frozenset({"projection"}),
        frozen_names=frozenset({"embedding"}),
    )
    assert inventory == reordered
    assert (
        len({inventory.sha256, without_decay.sha256, frozen_embedding.sha256})
        == 3
    )


@pytest.mark.parametrize("master_dtype", ["bfloat16", "float16", "float64"])
def test_builder_requires_fp32_masters(master_dtype: str) -> None:
    """Explicit decay policy doesn't permit a non-FP32 master inventory."""
    with pytest.raises(ValueError, match="Full-weight masters must be float32"):
        parameters.build_inventory(
            {"projection": (2, 2)},
            format_id="test.inventory/1",
            decayed_names=frozenset({"projection"}),
            master_dtype=master_dtype,
        )
