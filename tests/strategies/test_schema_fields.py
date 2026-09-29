"""An independent four-scalar model learns through neutral typed supervision."""

import dataclasses
import math

import jax.numpy as jnp
import numpy as np
import pytest

from minifield_training.batching import contracts
from minifield_training.batching import schema_fields as batching
from minifield_training.core import parameters
from minifield_training.datasets import fields
from minifield_training.kernels import types
from minifield_training.objectives import schema_fields as objective
from minifield_training.optimizers import adamw
from minifield_training.strategies import schema_fields

_PARAMETERS = ("category", "binary", "presence", "token")
_GRADIENTS = {
    "category": -0.375,
    "binary": -0.125,
    "presence": -0.125,
    "token": -0.125,
}
_INITIAL_LOSS = (4 * math.log(2) + math.log(3)) / 4


def _records() -> list[fields.Record]:
    """Declare complete, partial, and absent labels with 11 token IDs."""
    source = fields.Encoding(
        (7, 1, 2), ((0, 0), (0, 1), (1, 2)), (True, False, False)
    )
    binary = fields.Field(
        "flag", fields.Kind.BINARY, (), ((8, 3),), (1.0,), True, False, None
    )
    complete = fields.Record(
        "all-kinds",
        "ab",
        source,
        (
            fields.Field(
                "mention",
                fields.Kind.EXTRACT,
                (),
                ((8, 0),),
                (1.0,),
                True,
                True,
                (1, 2),
            ),
            fields.Field(
                "color",
                fields.Kind.CHOICE,
                ("red", "blue"),
                ((8, 1), (8, 2)),
                (0.0, 1.0),
                True,
                False,
                None,
            ),
            binary,
            fields.Field(
                "rank",
                fields.Kind.ORDINAL,
                ("low", "medium", "high"),
                ((8, 4), (8, 5), (8, 6)),
                (0.0, 0.0, 1.0),
                True,
                False,
                None,
            ),
        ),
    )
    return [
        complete,
        fields.Record("one-kind", "ab", source, (binary,)),
        fields.Record(
            "unlabeled",
            "ab",
            source,
            (dataclasses.replace(binary, supervised=False, targets=(0.0,)),),
        ),
    ]


def _forward(
    params: types.Parameters, batch: types.DeviceBatch
) -> types.DeviceBatch:
    """Use independent scalar slopes and biases, with no encoder or fusion."""
    row_shape = batch["kind"].shape
    candidate_feature = batch["schema_ids"][:, :, 1].astype(jnp.float32)
    token_feature = jnp.where(batch["source_ids"] == 1, 1.0, -1.0)
    return {
        "candidate": params["category"] * candidate_feature,
        "binary": jnp.broadcast_to(params["binary"], row_shape),
        "presence": jnp.broadcast_to(params["presence"], row_shape),
        "tokens": jnp.broadcast_to(
            params["token"] * token_feature[:, None, :],
            (*row_shape, batch["source_ids"].shape[-1]),
        ),
    }


def _inventory() -> parameters.FullParameterInventory:
    """Admit exactly four unrelated trainable scalar masters."""
    return parameters.build_inventory(
        {name: () for name in _PARAMETERS},
        format_id="four-scalars/1",
        decayed_names=frozenset(),
    )


def _zeros() -> types.Parameters:
    """Allocate fresh buffers for each donated optimizer transaction."""
    return {name: jnp.zeros((), dtype=jnp.float32) for name in _PARAMETERS}


def _batch(microbatches: int, requests: int) -> contracts.PhysicalUpdate:
    """Hold the logical examples fixed while changing their physical layout."""
    return batching.build(
        _records(),
        batching.Shape(microbatches, requests, 4, 3, 8, 11, 10),
        seed=9,
        update=0,
        weighting=objective.balance_types,
    )


@pytest.mark.parametrize("layout", ((1, 3), (3, 1), (2, 2), (4, 1)))
def test_analytical_loss_gradient_and_adam_commit_across_microbatches(
    layout: tuple[int, int],
) -> None:
    """Uneven labels and padding preserve logical loss and optimizer state."""
    inventory = _inventory()
    optimizer = adamw.AdamWConfig(
        learning_rate=0.1,
        beta1=0.0,
        beta2=0.0,
        epsilon=1e-6,
        weight_decay=0.0,
        clip_norm=10.0,
    )
    step = schema_fields.make_step(_forward, inventory, optimizer)
    batch = _batch(*layout)
    full_state = adamw.initialize_state(_zeros(), inventory)
    loss_sum, mass = 0.0, 0.0
    gradient_sums = dict.fromkeys(_PARAMETERS, 0.0)
    for index in np.flatnonzero(batch.active):
        physical = {
            name: jnp.asarray(value[int(index)])
            for name, value in batch.microbatches.items()
        }
        loss, count, gradients = step.gradient(full_state["params"], physical)
        loss_sum += float(loss)
        mass += float(count)
        for name, value in gradients.items():
            gradient_sums[name] += float(value)
    assert loss_sum == pytest.approx(_INITIAL_LOSS, abs=2e-7)
    assert mass == pytest.approx(1.0, abs=1e-7)
    for name, expected in _GRADIENTS.items():
        assert gradient_sums[name] == pytest.approx(expected, abs=1e-7)
    result = step(full_state, batch.microbatches, batch.active)
    assert bool(result.committed)
    assert int(result.state["step"]) == 1
    assert float(result.loss) == pytest.approx(_INITIAL_LOSS, abs=2e-7)
    assert float(result.gradient_norm) == pytest.approx(
        math.sqrt(0.1875), abs=1e-7
    )
    for name, gradient in _GRADIENTS.items():
        assert float(result.state["m"][name]) == pytest.approx(
            gradient, abs=1e-7
        )
        assert float(result.state["v"][name]) == pytest.approx(
            gradient**2, abs=1e-7
        )
        expected_parameter = -0.1 * gradient / (abs(gradient) + 1e-6)
        assert float(result.state["params"][name]) == pytest.approx(
            expected_parameter, abs=1e-7
        )


def test_scalar_model_learns_all_four_types_with_finite_state() -> None:
    """Independent logits improve through the shared strategy and optimizer."""
    inventory = _inventory()
    batch = _batch(3, 1)
    step = schema_fields.make_step(
        _forward,
        inventory,
        adamw.AdamWConfig(0.1, weight_decay=0.0, clip_norm=10.0),
    )
    current = adamw.initialize_state(_zeros(), inventory)
    observed = []
    for iteration in range(40):
        result = step(current, batch.microbatches, batch.active)
        assert bool(result.committed)
        assert int(result.state["step"]) == iteration + 1
        assert math.isfinite(float(result.loss))
        observed.append(float(result.loss))
        current = result.state
    assert observed[0] == pytest.approx(_INITIAL_LOSS, abs=2e-7)
    assert observed[-1] < 0.15
    assert int(current["step"]) == 40
    for group in ("params", "m", "v"):
        assert all(
            np.isfinite(np.asarray(value)).all()
            for value in current[group].values()
        )
    assert all(float(value) > 2 for value in current["params"].values())


def test_per_row_losses_match_independent_closed_form() -> None:
    """Check categorical, presence, binary, and token losses independently."""
    batch = _batch(1, 3)
    physical = {
        name: jnp.asarray(value[0])
        for name, value in batch.microbatches.items()
    }
    actual = objective.losses(_forward(_zeros(), physical), physical)
    expected = [
        2 * math.log(2),
        math.log(2),
        math.log(2),
        math.log(2),
        math.log(3),
        math.log(3),
        math.log(3),
    ]
    np.testing.assert_allclose(actual[0, :7], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(
        physical["field_weight"][0], [0.25, 0.25, 0, 0.125, 0.25, 0, 0, 0]
    )
    assert float(physical["field_weight"][1, 0]) == 0.125
    assert not np.any(physical["field_weight"][2])


def test_fixed_shape_reuses_gradient_trace_with_correct_padding() -> None:
    """Length, row-count, and partial-batch changes reuse one gradient graph."""
    traces = []

    def forward(
        params: types.Parameters, batch: types.DeviceBatch
    ) -> types.DeviceBatch:
        traces.append(batch["schema_ids"].shape)
        return _forward(params, batch)

    batches = batching.SchemaBatchStrategy(
        batching.Shape(1, 3, 16, 8, 16, 11, 10),
        objective.balance_types,
        fixed_shape=True,
    )
    update = schema_fields.make_step(
        forward, _inventory(), adamw.AdamWConfig(0.1)
    )
    binary = _records()[1]
    longer = dataclasses.replace(
        binary,
        text="abcdef",
        source=fields.Encoding(
            (7, 1, 2, 3, 4, 5, 6),
            ((0, 0),) + tuple((i, i + 1) for i in range(6)),
            (True,) + (False,) * 6,
        ),
        fields=(
            dataclasses.replace(binary.fields[0], rows=((8, 3, 4, 5, 6),)),
        ),
    )
    for index, records in enumerate((_records(), [longer], [binary])):
        packed = batches.pack(records, seed=9, update=index)
        physical = {key: value[0] for key, value in packed.microbatches.items()}
        loss, mass, gradients = update.gradient(_zeros(), physical)
        expected_loss = _INITIAL_LOSS if index == 0 else math.log(2)
        expected_gradients = (
            _GRADIENTS
            if index == 0
            else {
                "category": 0.0,
                "binary": -0.5,
                "presence": 0.0,
                "token": 0.0,
            }
        )
        assert float(loss) == pytest.approx(expected_loss, abs=2e-7)
        assert float(mass) == pytest.approx(1, abs=1e-7)
        for name, expected in expected_gradients.items():
            assert float(gradients[name]) == pytest.approx(expected, abs=1e-7)
    assert traces == [(3, 16, 8)]
