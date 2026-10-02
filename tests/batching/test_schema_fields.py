"""Schema packing works with caller-owned token and supervision policies."""

from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np
import pytest

from minifield_training.batching import contracts
from minifield_training.batching import schema_fields
from minifield_training.datasets import fields


def _record(identifier: str = "fixture") -> fields.Record:
    """Build an unrelated 11-token fixture directly from neutral records."""
    return fields.Record(
        identifier,
        "ab",
        fields.Encoding(
            (7, 1, 2), ((0, 0), (0, 1), (1, 2)), (True, False, False)
        ),
        (
            fields.Field(
                "mention",
                fields.Kind.EXTRACT,
                (),
                ((8, 1),),
                (1.0,),
                True,
                True,
                (1, 2),
            ),
            fields.Field(
                "color",
                fields.Kind.CHOICE,
                ("red", "blue"),
                ((8, 2), (8, 3)),
                (0.25, 0.75),
                True,
                False,
                None,
            ),
            fields.Field(
                "missing",
                fields.Kind.BINARY,
                (),
                ((8, 4),),
                (0.0,),
                False,
                False,
                None,
            ),
            fields.Field(
                "rank",
                fields.Kind.ORDINAL,
                ("low", "high"),
                ((8, 5), (8, 6)),
                (0.0, 1.0),
                True,
                False,
                None,
            ),
        ),
    )


def _binary_record(identifier: str = "binary") -> fields.Record:
    """Keep a one-row task for capacity and token-policy checks."""
    original = _record(identifier)
    labeled = dataclasses.replace(original.fields[2], supervised=True)
    return dataclasses.replace(original, fields=(labeled,))


def _uniform(kinds: Sequence[int]) -> Sequence[float]:
    """Weight every labeled field equally without selecting objective policy."""
    return [1.0] * len(kinds)


def test_explicit_token_policy_masks_and_injected_field_weights() -> None:
    """Caller padding and arbitrary field weights survive physical packing."""
    observed: list[int] = []

    def weighting(kinds: Sequence[int]) -> Sequence[float]:
        observed.extend(kinds)
        return (2.0, 3.0, 5.0, 7.0)

    shape = schema_fields.Shape(2, 2, 5, 4, 7, 11, 10)
    update = schema_fields.build(
        [_record(), _binary_record()],
        shape,
        seed=3,
        update=4,
        weighting=weighting,
    )
    arrays = update.microbatches
    assert observed == [0, 1, 3, 2]
    assert update.active.tolist() == [True, False]
    assert update.example_ids == ("fixture", "binary")
    np.testing.assert_array_equal(arrays["source_ids"][0, 0], [7, 1, 2, 10, 10])
    np.testing.assert_array_equal(arrays["source_mask"][0, 0], [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(arrays["selectable"][0, 0], [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(
        arrays["schema_ids"][0, 0],
        [
            [8, 1, 10, 10],
            [8, 2, 10, 10],
            [8, 3, 10, 10],
            [8, 4, 10, 10],
            [8, 5, 10, 10],
            [8, 6, 10, 10],
            [10, 10, 10, 10],
        ],
    )
    np.testing.assert_array_equal(
        arrays["schema_mask"][0, 0], [[1, 1, 0, 0]] * 6 + [[0, 0, 0, 0]]
    )
    np.testing.assert_array_equal(
        arrays["field_owner"][0, 0], [0, 1, 1, 2, 3, 3, 0]
    )
    np.testing.assert_array_equal(arrays["kind"][0, 0], [0, 1, 1, 2, 3, 3, 0])
    np.testing.assert_array_equal(
        arrays["field_weight"][0, 0], [2, 3, 0, 0, 5, 0, 0]
    )
    np.testing.assert_array_equal(
        arrays["field_weight"][0, 1], [7, 0, 0, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        arrays["token_target"][0, 0, 0], [0, 1, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        arrays["token_supervised"][0, 0], [1, 0, 0, 0, 0, 0, 0]
    )
    assert np.all(arrays["source_ids"][1] == 10)
    assert np.all(arrays["schema_ids"][1] == 10)
    for name in ("source_mask", "schema_mask", "row_mask", "field_weight"):
        assert not np.any(arrays[name][1])


def test_large_vocabulary_is_admitted_by_configuration() -> None:
    """Token IDs above 65,536 follow the caller's explicit token policy."""
    original = _binary_record()
    record = dataclasses.replace(
        original,
        source=dataclasses.replace(
            original.source, ids=(70_001, 90_000, 99_998)
        ),
        fields=(
            dataclasses.replace(original.fields[0], rows=((80_000, 99_997),)),
        ),
    )
    batch = schema_fields.build(
        [record],
        schema_fields.Shape(1, 1, 4, 3, 1, 100_000, 99_999),
        seed=0,
        update=0,
        weighting=_uniform,
    ).microbatches
    np.testing.assert_array_equal(
        batch["source_ids"][0, 0], [70_001, 90_000, 99_998, 99_999]
    )
    np.testing.assert_array_equal(
        batch["schema_ids"][0, 0, 0], [80_000, 99_997, 99_999]
    )


def test_context_capacity_has_no_model_specific_ceiling() -> None:
    """A caller can admit 8,193 source tokens while retaining every position."""
    length = 8193
    record = dataclasses.replace(
        _binary_record(),
        text="a" * (length - 1),
        source=fields.Encoding(
            (7,) + (1,) * (length - 1),
            ((0, 0),) + tuple((i, i + 1) for i in range(length - 1)),
            (True,) + (False,) * (length - 1),
        ),
    )
    batch = schema_fields.build(
        [record],
        schema_fields.Shape(1, 1, length, 2, 1, 11, 10),
        seed=0,
        update=0,
        weighting=_uniform,
    ).microbatches
    assert batch["source_ids"].shape == (1, 1, length)
    assert np.all(batch["source_mask"] == 1)
    assert int(np.sum(batch["selectable"])) == length - 1
    assert batch["source_ids"][0, 0, -1] == 1


@pytest.mark.parametrize("bad_id", (-1, 11, True))
def test_token_admission_uses_the_supplied_vocabulary(bad_id: int) -> None:
    """Configured vocabulary bounds apply before copying source tokens."""
    record = _binary_record()
    record = dataclasses.replace(
        record, source=dataclasses.replace(record.source, ids=(7, bad_id, 2))
    )
    with pytest.raises(ValueError, match="configured vocabulary"):
        schema_fields.build(
            [record],
            schema_fields.Shape(1, 1, 4, 2, 1, 11, 10),
            seed=0,
            update=0,
            weighting=_uniform,
        )


def test_strategy_satisfies_shared_contract_and_replays_partial_update() -> (
    None
):
    """The strategy obeys BatchStrategy with caller-configured buckets."""
    strategy: contracts.BatchStrategy[fields.Record] = (
        schema_fields.SchemaBatchStrategy(
            schema_fields.Shape(1, 2, 17, 9, 9, 11, 10),
            _uniform,
            min_tokens=8,
            min_rows=4,
        )
    )
    records = [_binary_record(str(index)) for index in range(3)]
    assert strategy.shape.capacity == 2
    assert strategy.update_count(records) == 2
    updates = list(strategy.iter_updates(records, seed=5, shuffle=False))
    resumed = next(
        strategy.iter_updates(records, seed=5, shuffle=False, start_update=1)
    )
    assert [item.example_ids for item in updates] == [("0", "1"), ("2",)]
    assert resumed.example_ids == ("2",)
    assert resumed.microbatches["source_ids"].shape == (1, 2, 8)
    assert resumed.microbatches["schema_ids"].shape == (1, 2, 4, 8)
    for name, array in resumed.microbatches.items():
        np.testing.assert_array_equal(array, updates[1].microbatches[name])


def test_fixed_shape_preserves_axes_through_partial_update_and_resume() -> None:
    """Record lengths and candidate counts cannot select new array shapes."""
    shape = schema_fields.Shape(2, 2, 17, 9, 9, 11, 10)
    strategy = schema_fields.SchemaBatchStrategy(
        shape, _uniform, fixed_shape=True
    )
    records = [_record("0")] + [_binary_record(str(i)) for i in range(1, 5)]
    records[1] = dataclasses.replace(
        records[1],
        text="abcdef",
        source=fields.Encoding(
            (7, 1, 2, 3, 4, 5, 6),
            ((0, 0),) + tuple((i, i + 1) for i in range(6)),
            (True,) + (False,) * 6,
        ),
        fields=(
            dataclasses.replace(
                records[1].fields[0], rows=((8, 1, 2, 3, 4, 5),)
            ),
        ),
    )
    updates = list(strategy.iter_updates(records, seed=5, shuffle=False))
    assert [item.example_ids for item in updates] == [
        ("0", "1", "2", "3"),
        ("4",),
    ]
    signatures = [
        {
            name: (value.shape, value.dtype)
            for name, value in item.microbatches.items()
        }
        for item in updates
    ]
    assert signatures[0] == signatures[1]
    last = updates[1]
    assert last.microbatches["source_ids"].shape == (2, 2, 17)
    assert last.microbatches["schema_ids"].shape == (2, 2, 9, 9)
    assert last.microbatches["token_target"].shape == (2, 2, 9, 17)
    assert last.active.tolist() == [True, False]
    assert np.sum(last.microbatches["row_mask"]) == 1
    assert np.sum(last.microbatches["field_weight"]) == 1
    resumed = next(
        strategy.iter_updates(records, seed=5, shuffle=False, start_update=1)
    )
    for name, array in last.microbatches.items():
        np.testing.assert_array_equal(resumed.microbatches[name], array)


@pytest.mark.parametrize(
    "axis", ("source_tokens", "schema_tokens", "schema_rows")
)
def test_fixed_shape_rejects_overflow_without_truncation(axis: str) -> None:
    """The fixed envelope fails explicitly when any record axis exceeds it."""
    shape = dataclasses.replace(
        schema_fields.Shape(1, 1, 8, 8, 8, 11, 10), **{axis: 1}
    )
    strategy = schema_fields.SchemaBatchStrategy(
        shape, _uniform, fixed_shape=True
    )
    with pytest.raises(ValueError, match="exceeds configured batch shape"):
        strategy.pack([_record()], seed=0, update=0)


@pytest.mark.parametrize("weights", ((), (-1.0,), (float("nan"),)))
def test_invalid_weighting_results_fail_before_training(
    weights: tuple[float, ...],
) -> None:
    """Injected weights must have one finite nonnegative value per label."""
    with pytest.raises(ValueError, match="Invalid field weighting"):
        schema_fields.build(
            [_binary_record()],
            schema_fields.Shape(1, 1, 4, 2, 1, 11, 10),
            seed=0,
            update=0,
            weighting=lambda _kinds: weights,
        )


@pytest.mark.parametrize(
    ("kind", "targets"), ((4, (0.0,)), (fields.Kind.BINARY, ()))
)
def test_malformed_neutral_fields_fail_admission(
    kind: int, targets: tuple[float, ...]
) -> None:
    """Unsupported kinds and missing row targets cannot reach array packing."""
    record = _binary_record()
    record = dataclasses.replace(
        record,
        fields=(
            dataclasses.replace(record.fields[0], kind=kind, targets=targets),
        ),
    )
    with pytest.raises(ValueError, match="Invalid field kind or row targets"):
        schema_fields.build(
            [record],
            schema_fields.Shape(1, 1, 4, 2, 1, 11, 10),
            seed=0,
            update=0,
            weighting=_uniform,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"targets": (0.0,)},
        {"span": None},
        {"token_supervised": False},
        {"kind": fields.Kind.BINARY, "span": None},
        {"kind": 0.0},
        {"targets": (0.5,)},
    ],
)
def test_rejects_contradictory_extraction_supervision(
    changes: dict[str, Any],
) -> None:
    """Neutral callers receive the same consistent span/presence admission."""
    record = _record()
    invalid = dataclasses.replace(
        record, fields=(dataclasses.replace(record.fields[0], **changes),)
    )
    with pytest.raises(
        ValueError, match="field kind|Token supervision|Inconsistent extraction"
    ):
        schema_fields.build(
            [invalid],
            schema_fields.Shape(1, 1, 4, 4, 4, 11, 10),
            seed=0,
            update=0,
            weighting=_uniform,
        )


def test_short_rows_share_packed_encoder_sequences() -> None:
    """Six 2-token rows fill three 4-token encoder rows and gather back."""
    shape = schema_fields.Shape(2, 2, 5, 4, 7, 11, 10, schema_sequences=3)
    arrays = schema_fields.build(
        [_record()], shape, seed=3, update=4, weighting=_uniform
    ).microbatches
    np.testing.assert_array_equal(
        arrays["packed_schema_ids"][0, 0],
        [[8, 1, 8, 2], [8, 3, 8, 4], [8, 5, 8, 6]],
    )
    np.testing.assert_array_equal(
        arrays["packed_schema_segments"][0, 0],
        [[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 6, 6]],
    )
    np.testing.assert_array_equal(
        arrays["packed_schema_positions"][0, 0], [[0, 1, 0, 1]] * 3
    )
    np.testing.assert_array_equal(
        arrays["schema_token_index"][0, 0],
        [[0, 1, 0, 0], [2, 3, 0, 0], [4, 5, 0, 0], [6, 7, 0, 0]]
        + [[8, 9, 0, 0], [10, 11, 0, 0], [0, 0, 0, 0]],
    )
    gathered = arrays["packed_schema_ids"][0, 0].reshape(-1)[
        arrays["schema_token_index"][0, 0]
    ]
    mask = arrays["schema_mask"][0, 0]
    np.testing.assert_array_equal(
        gathered * mask, arrays["schema_ids"][0, 0] * mask
    )
    assert np.all(arrays["packed_schema_ids"][:, 1] == 10)
    assert not np.any(arrays["packed_schema_segments"][:, 1])
    with pytest.raises(ValueError, match="exceeds configured batch shape"):
        schema_fields.build(
            [_record()],
            dataclasses.replace(shape, schema_sequences=2),
            seed=3,
            update=4,
            weighting=_uniform,
        )


@pytest.mark.parametrize(("min_tokens", "sequences"), [(1, 8), (4, 4)])
def test_bucketing_counts_packed_sequences_at_the_row_bucket(
    min_tokens: int, sequences: int
) -> None:
    """2-token rows need 6 rows of 2 tokens or 3 rows of 4 tokens."""
    shape = schema_fields.Shape(1, 1, 8, 8, 8, 11, 10, schema_sequences=8)
    bucketed = schema_fields.bucket([_record()], shape, min_tokens=min_tokens)
    assert bucketed.schema_sequences == sequences
    assert bucketed.packed_sequences == sequences
    default = schema_fields.bucket(
        [_record()],
        dataclasses.replace(shape, schema_sequences=None),
        min_tokens=min_tokens,
    )
    assert default.schema_sequences is None
    assert default.packed_sequences == default.schema_rows == 8


def test_bucketing_widens_packed_rows_before_rejecting() -> None:
    """10 rows of 64 tokens fit 4 x 512; bucketing widens rows to fit."""
    wide = _binary_record("wide")
    rows = dataclasses.replace(
        wide.fields[0],
        kind=fields.Kind.CHOICE,
        candidates=tuple(str(index) for index in range(10)),
        rows=tuple((8, *([index + 1] * 63)) for index in range(10)),
        targets=(1.0,) + (0.0,) * 9,
    )
    record = dataclasses.replace(wide, fields=(rows,))
    shape = schema_fields.Shape(1, 1, 8, 512, 16, 11, 10, schema_sequences=4)
    bucketed = schema_fields.bucket([record], shape)
    # 64-token rows need 10 sequences and 128 need 5; 256 need 3 (bucket 4).
    assert (bucketed.schema_tokens, bucketed.schema_sequences) == (256, 4)
    narrow = schema_fields.bucket(
        [record], dataclasses.replace(shape, schema_sequences=2)
    )
    assert (narrow.schema_tokens, narrow.schema_sequences) == (512, 2)
    with pytest.raises(ValueError, match="limit is 1"):
        schema_fields.bucket(
            [record], dataclasses.replace(shape, schema_sequences=1)
        )
