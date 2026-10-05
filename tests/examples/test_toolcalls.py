"""Tool-call model: role-marked layout, trainable markers and curriculum."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.magicbox import composition as magicbox
from examples.magicbox import data as magicbox_data
from examples.magicbox import export
from examples.magicbox import smoke
from examples.magicbox import train as magicbox_train
from examples.toolcalls import composition
from examples.toolcalls import data
from examples.toolcalls import train as curriculum
from minifield_training.batching import pointer as batching
from minifield_training.datasets import pointer
from minifield_training.evaluation import pointer as evaluation
from minifield_training.models.magicbox import pointer as model
from minifield_training.objectives import pointer as objective
from minifield_training.objectives import schema_fields as weighting
from minifield_training.optimizers import adamw
from minifield_training.strategies import schema_fields as strategy

_REQUEST: dict[str, object] = {
    "state": "User: restart the pod back-end-pod",
    "questions": {
        "next_tool": {
            "type": "choice",
            "instructions": "Which tool should be called next?",
            "criteria": {
                "restart_pod": "Restart a named pod.",
                "none": "No listed tool fits.",
            },
        },
        "pod_name": {"type": "extract", "instructions": "Argument `pod_name`"},
        "force": {"type": "noul", "instructions": "Argument `force`"},
    },
}
_TARGETS = {
    "next_tool": {"choice": "restart_pod"},
    "pod_name": {"has_answer": True, "span": [22, 34], "text": "back-end-pod"},
    "force": {"probability": 0.0},
}


def _records() -> tuple[pointer.Record, pointer.Record]:
    base = magicbox_data.compile_pointer_record(
        "r", _REQUEST, _TARGETS, smoke.toy_encode
    )
    return base, data.marked(base, _REQUEST, smoke.toy_encode)


def test_marked_layout_leads_with_bos_and_names_every_role() -> None:
    """One leading BOS, then a role marker opens every run; labels unchanged."""
    base, record = _records()
    ids, placed = batching.layout(record)
    markers = data.MARKER_IDS
    assert ids[0] == data.BOS and ids.count(data.BOS) == 1
    assert [ids[i] for i in placed.queries] == [
        markers["choice_question"],
        markers["extract_question"],
        markers["noul_question"],
    ]
    assert [[ids[i] for i in options] for options in placed.options] == [
        [markers["choice_option"]] * 2,
        [markers["extract_absent"]],
        [markers["noul_false"], markers["noul_true"]],
    ]
    assert ids[placed.source_start] == markers["source"]
    choice = record.questions[0]
    expected = smoke.toy_encode("Question: Which tool should be called next?")
    assert choice.query == (markers["choice_question"], *expected.ids[1:])
    for before, after in zip(base.questions, record.questions, strict=True):
        assert (after.targets, after.span) == (before.targets, before.span)
    # The dropped "Type:" lines outweigh the leading BOS.
    assert record.sequence_tokens < base.sequence_tokens


def _tiny() -> tuple[Any, model.Config, dict[str, jax.Array]]:
    cfg, _, encoder_params = smoke.tiny()
    head = model.Config(encoder_width=cfg.hidden_size, pointer_width=8)
    params = {
        name: value
        for name, value in encoder_params.items()
        if name.startswith("lfm2.")
    }
    params.update(model.initialize(head, jax.random.PRNGKey(3)))
    return cfg, head, composition.initialize(params)


def test_markers_start_as_bos_and_fold_into_their_rows() -> None:
    """Fresh markers copy BOS; folding writes them, unfolding reads them."""
    _, _, params = _tiny()
    table = params[composition.EMBEDDINGS]
    np.testing.assert_array_equal(
        params[composition.MARKERS], np.tile(table[data.BOS], (10, 1))
    )
    params[composition.MARKERS] = (
        params[composition.MARKERS] + jnp.arange(10.0)[:, None]
    )
    folded = composition.with_markers(params)
    assert composition.MARKERS not in folded
    rows = folded[composition.EMBEDDINGS][
        jnp.asarray(list(data.MARKER_IDS.values()))
    ]
    np.testing.assert_array_equal(rows, params[composition.MARKERS])
    np.testing.assert_array_equal(
        composition.unfold(folded)[composition.MARKERS],
        params[composition.MARKERS],
    )


def test_tool_model_learns_and_only_markers_change_in_the_table() -> None:
    """The marked model overfits a tool record; the table stays put."""
    cfg, head, params = _tiny()
    inventory = magicbox.pointer_inventory(
        cfg, head, extra=composition.extra(cfg)
    )
    spec = {s.name: s for s in inventory.specs}
    assert (
        spec[composition.MARKERS].trainable
        and not spec[composition.MARKERS].decayed
    )
    assert not spec[composition.EMBEDDINGS].trainable
    _, record = _records()
    batches = batching.PointerBatchStrategy(
        batching.Shape(1, 1, record.sequence_tokens + 3, 8, 128, 0),
        weighting.balance_types,
    )
    packed = batches.pack([record], seed=0, update=0)
    update = strategy.make_step(
        composition.bind(cfg, head, bf16=False),
        inventory,
        adamw.AdamWConfig(learning_rate=0.01, weight_decay=0),
        terms=objective.terms,
    )
    # The step donates its state, so keep copies to compare against.
    table = np.asarray(params[composition.EMBEDDINGS]).copy()
    markers = np.asarray(params[composition.MARKERS]).copy()
    current = adamw.initialize_state(params, inventory)
    losses = []
    for _ in range(60):
        result = update(current, packed.microbatches, packed.active)
        losses.append(float(result.loss))
        current = result.state
    assert losses[-1] < losses[0] * 0.25
    np.testing.assert_array_equal(
        current["params"][composition.EMBEDDINGS], table
    )
    assert not np.allclose(current["params"][composition.MARKERS], markers)
    predictor = evaluation.Predictor(
        composition.bind(cfg, head, bf16=False), batches
    )
    decoded, _ = predictor.score(current["params"], record)
    assert (
        cast(dict[str, object], decoded["next_tool"])["value"] == "restart_pod"
    )
    assert (
        cast(dict[str, object], decoded["pod_name"])["value"] == "back-end-pod"
    )


def test_rename_keeps_ids_and_refuses_collisions() -> None:
    """Readable marker names replace reserved strings at the same IDs."""
    spec: dict[str, object] = {
        "added_tokens": [{"id": 17, "content": "<|reserved_7|>"}],
        "model": {"vocab": {"<|reserved_7|>": 17, "a": 30}},
    }
    renamed = export.rename_special_tokens(
        spec, {"<|reserved_7|>": "<|extract_question|>"}
    )
    assert renamed["added_tokens"] == [
        {"id": 17, "content": "<|extract_question|>"}
    ]
    assert cast(dict[str, Any], renamed["model"])["vocab"] == {
        "a": 30,
        "<|extract_question|>": 17,
    }
    with pytest.raises(ValueError, match="unused names"):
        export.rename_special_tokens(spec, {"<|reserved_7|>": "<|reserved_7|>"})


def test_curriculum_warm_starts_each_stage_and_stops_when_time_runs_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage n+1 starts from stage n's weights; an unfinished one ends it."""
    starts: list[object] = []

    def prepare(settings: Any) -> SimpleNamespace:
        return SimpleNamespace(
            settings=settings, stream=SimpleNamespace(total_updates=5)
        )

    def train_stage(
        *args: Any, **_: object
    ) -> tuple[dict[str, object], SimpleNamespace]:
        _, output, warm_start = args
        starts.append(warm_start)
        stage = int(output.name[-1])
        done = 5 if stage < 2 else 3
        return {"params": f"weights-{stage}"}, SimpleNamespace(next_batch=done)

    monkeypatch.setattr(magicbox_train, "prepare", prepare)
    monkeypatch.setattr(curriculum, "train_stage", train_stage)
    monkeypatch.setattr(
        magicbox_train,
        "save_bundle",
        lambda run, params, output, step: output / "bundle",
    )
    monkeypatch.setattr(magicbox_train, "make_evaluator", lambda run: None)
    monkeypatch.setattr(magicbox_train, "final_evaluation", lambda *args: None)
    base = SimpleNamespace()
    monkeypatch.setattr(
        curriculum,
        "stage_settings",
        lambda base, root, stage: SimpleNamespace(stage=stage),
    )
    bundles = curriculum.run_curriculum(
        cast(Any, base),
        tmp_path / "data",
        tmp_path / "out",
        session_seconds=60,
        report=lambda event: None,
    )
    assert starts == [None, "weights-0", "weights-1"]
    assert sorted(bundles) == [0, 1]
