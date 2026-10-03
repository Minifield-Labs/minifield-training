"""End-to-end joint pointer MagicBox: compile, train, decode, and format."""

from collections.abc import Callable
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.magicbox import bundle as magicbox_bundle
from examples.magicbox import composition as magicbox
from examples.magicbox import data
from examples.magicbox import smoke
from minifield_training.batching import contracts
from minifield_training.batching import pointer as batching
from minifield_training.core import json_io
from minifield_training.datasets import pointer
from minifield_training.evaluation import pointer as evaluation
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import pointer as model
from minifield_training.objectives import pointer as objective
from minifield_training.objectives import schema_fields as weighting
from minifield_training.optimizers import adamw
from minifield_training.strategies import quantization
from minifield_training.strategies import schema_fields as strategy

_REQUEST = {
    "state": "Ada sent 3 reports.",
    "questions": {
        "person": {"type": "extract", "instructions": "Who sent reports?"},
        "missing": {"type": "extract", "instructions": "Which city?"},
        "category": {
            "type": "choice",
            "instructions": "Classify the message.",
            "criteria": {"report": "Reporting", "payment": "Payments"},
        },
        "sent": {"type": "noul", "instructions": "Were reports sent?"},
        "amount": {
            "type": "score",
            "instructions": "How many reports?",
            "criteria": ["None", "One", "Several"],
        },
    },
}
_TARGETS = {
    "person": {"has_answer": True, "span": [0, 3], "text": "Ada"},
    "missing": {"has_answer": False, "span": None, "text": None},
    "category": {"choice": "report"},
    "sent": {"probability": 1.0},
    "amount": {"level": 2},
}


def _record(score_width: float = 0.0) -> pointer.Record:
    return data.compile_pointer_record(
        "tiny",
        _REQUEST,
        _TARGETS,
        smoke.toy_encode,
        score_width=score_width,
    )


def test_compiled_questions_carry_pointer_targets() -> None:
    """Options, absent markers, spans, and the spread score are explicit."""
    record = _record(score_width=0.5)
    by_key = {question.key: question for question in record.questions}
    assert [option.label for option in by_key["category"].options] == [
        "report",
        "payment",
    ]
    assert by_key["category"].targets == (1.0, 0.0)
    assert by_key["person"].targets == (0.0,)
    assert by_key["person"].span == (1, 2)
    assert by_key["missing"].targets == (1.0,)
    assert by_key["missing"].span is None
    assert by_key["sent"].targets == (0.0, 1.0)
    # Width 0.5 on 3 levels: sigma is 1 level, so weights are e^-2, e^-0.5, 1.
    weights = np.exp([-2.0, -0.5, 0.0])
    np.testing.assert_allclose(
        by_key["amount"].targets, weights / weights.sum()
    )
    assert by_key["amount"].legend == ("None", "One", "Several")
    assert all(question.query[0] == 1 for question in record.questions)
    pointer.validate(record, vocab_size=128)


def test_tiny_pointer_model_learns_every_question_type() -> None:
    """A tiny encoder overfits and decodes the gold answer of every type."""
    cfg, _, encoder_params = smoke.tiny()
    head = model.Config(encoder_width=cfg.hidden_size, pointer_width=8)
    params = {
        name: value
        for name, value in encoder_params.items()
        if name.startswith("lfm2.")
    }
    params.update(model.initialize(head, jax.random.PRNGKey(3)))
    inventory = magicbox.pointer_inventory(cfg, head)
    record = _record()
    batches = batching.PointerBatchStrategy(
        batching.Shape(1, 1, record.sequence_tokens + 3, 8, 128, 0),
        weighting.balance_types,
    )
    packed = batches.pack([record], seed=0, update=0)
    optimizer = adamw.AdamWConfig(learning_rate=0.01, weight_decay=0)
    update = strategy.make_step(
        magicbox.bind_pointer(cfg, head, bf16=False),
        inventory,
        optimizer,
        terms=objective.terms,
    )
    current = adamw.initialize_state(params, inventory)
    current = jax.device_put(current, jax.devices()[0])
    losses = []
    for _ in range(60):
        result = update(current, packed.microbatches, packed.active)
        assert bool(result.committed)
        losses.append(float(result.loss))
        current = result.state
    assert losses[-1] < losses[0] * 0.25
    predictor = evaluation.Predictor(
        magicbox.bind_pointer(cfg, head, bf16=False), batches
    )
    decoded, question_losses = predictor.score(current["params"], record)
    formatted = data.format_results(record, decoded)
    assert formatted["person"] == {
        "type": "extract",
        "extract": "Ada",
        "confidence": None,
    }
    assert formatted["missing"]["extract"] is None  # type: ignore[index]
    assert formatted["category"]["choice"] == "report"  # type: ignore[index]
    assert formatted["sent"]["noul"] > 0.9  # type: ignore[index]
    assert formatted["amount"]["score"] == pytest.approx(  # type: ignore[index]
        2, abs=0.2
    )
    assert len(question_losses) == len(record.questions)


def test_pointer_bundle_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v3 bundle restores identical logits and rejects the old loader."""
    cfg, _, encoder_params = smoke.tiny()
    head = model.Config(encoder_width=cfg.hidden_size, pointer_width=8)
    params = {
        name: value
        for name, value in encoder_params.items()
        if name.startswith("lfm2.")
    }
    params.update(model.initialize(head, jax.random.PRNGKey(3)))
    config = tmp_path / "encoder.json"
    config.write_text(
        json.dumps(
            {
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
        )
    )
    monkeypatch.setattr(
        encoder,
        "SOURCE",
        dataclasses.replace(
            encoder.SOURCE, config_sha256=json_io.digest_file(config)
        ),
    )
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"synthetic":true}')
    (tokenizer / "contract.json").write_text('{"offset_policy":"synthetic"}')
    bundle = tmp_path / "bundle"
    magicbox_bundle.save_pointer(
        bundle,
        params,
        cfg,
        head,
        encoder_config=config,
        tokenizer=tokenizer,
        step=7,
    )
    restored_cfg, restored_head, restored, decode = (
        magicbox_bundle.load_pointer(bundle)
    )
    assert (restored_cfg, restored_head) == (cfg, head)
    assert decode == {"presence_threshold": 0.5, "confidence": None}
    record = _record()
    packed = batching.build(
        [record],
        batching.Shape(1, 1, record.sequence_tokens, 8, 128, 0),
        weighting=weighting.balance_types,
    )
    batch = {
        key: jnp.asarray(value[0]) for key, value in packed.microbatches.items()
    }
    forward = magicbox.bind_pointer(cfg, head, bf16=False)
    for end, expected in forward(params, batch).items():
        np.testing.assert_array_equal(forward(restored, batch)[end], expected)
    with pytest.raises(ValueError, match="Unknown MagicBox bundle"):
        magicbox_bundle.load(bundle)


def _tiny_pointer() -> tuple[
    model.Config,
    dict[str, jax.Array],
    Callable[
        [dict[str, jax.Array], dict[str, jax.Array]], dict[str, jax.Array]
    ],
]:
    """Tiny real encoder plus pointer projections, computed in FP32."""
    cfg, _, encoder_params = smoke.tiny()
    head = model.Config(encoder_width=cfg.hidden_size, pointer_width=8)
    params = {
        name: value
        for name, value in encoder_params.items()
        if name.startswith("lfm2.")
    }
    params.update(model.initialize(head, jax.random.PRNGKey(3)))
    return head, params, magicbox.bind_pointer(cfg, head, bf16=False)


def test_packed_requests_match_separate_rows() -> None:
    """Packing changes no question's logits, loss, or any parameter gradient."""
    first = _record()
    second = data.compile_pointer_record(
        "second",
        {
            "state": "Bo paid 5 dollars.",
            "questions": {
                "payer": {"type": "extract", "instructions": "Who paid?"},
                "paid": {"type": "noul", "instructions": "Was it paid?"},
            },
        },
        {
            "payer": {"has_answer": True, "span": [0, 2], "text": "Bo"},
            "paid": {"probability": 0.8},
        },
        smoke.toy_encode,
    )
    length = max(first.sequence_tokens, second.sequence_tokens)
    separate = batching.build(
        [first, second],
        batching.Shape(1, 2, length, 5, 128, 0),
        weighting=weighting.balance_types,
    )
    packed = batching.build_rows(
        [[first, second]],
        batching.Shape(1, 1, 2 * length, 10, 128, 0),
        weighting=weighting.balance_types,
    )
    _, params, forward = _tiny_pointer()
    alone_batch, together_batch = _device(separate), _device(packed)
    alone = forward(params, alone_batch)
    together = forward(params, together_batch)
    alone_losses = np.asarray(objective.losses(alone, alone_batch))
    together_losses = np.asarray(objective.losses(together, together_batch))
    # Question j of request r is slot j of row r alone, and slot 5 * r + j
    # of the packed row, since the first request has 5 questions.
    for row, record in enumerate((first, second)):
        for question in range(len(record.questions)):
            slot = 5 * row + question
            np.testing.assert_allclose(
                together_losses[0, slot],
                alone_losses[row, question],
                rtol=1e-5,
                atol=1e-6,
            )
            for end in ("start", "end"):
                expected = np.asarray(alone[end])[row, question][
                    np.flatnonzero(alone_batch["allowed"][row, question])
                ]
                actual = np.asarray(together[end])[0, slot][
                    np.flatnonzero(together_batch["allowed"][0, slot])
                ]
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-5, atol=1e-5
                )

    def loss(
        weights: dict[str, jax.Array], batch: dict[str, jax.Array]
    ) -> jax.Array:
        total, mass = objective.terms(forward(weights, batch), batch)
        return total / mass

    alone_gradient = jax.grad(loss)(params, alone_batch)
    together_gradient = jax.grad(loss)(params, together_batch)
    for name, value in alone_gradient.items():
        difference = np.asarray(together_gradient[name]) - np.asarray(value)
        scale = max(float(np.linalg.norm(np.asarray(value))), 1e-8)
        assert float(np.linalg.norm(difference)) / scale < 1e-4, name


def _device(update: contracts.PhysicalUpdate) -> dict[str, jax.Array]:
    """The first microbatch of a physical update, on device."""
    return {
        key: jnp.asarray(value[0]) for key, value in update.microbatches.items()
    }


def _qat_encoder() -> tuple[lfm.Config, dict[str, jax.Array]]:
    """A tiny encoder whose projections fit the group-128 quantizer."""
    cfg = lfm.Config(128, 128, 2, 1, 128, ("conv", "full_attention"))
    params = {
        name: jnp.ones(shape, jnp.float32)
        if len(shape) == 1
        else jax.random.normal(jax.random.PRNGKey(index), shape) * 0.05
        for index, (name, shape) in enumerate(
            encoder.Adapter().expected_shapes(cfg).items()
        )
    }
    return cfg, params


def test_shared_weights_train_dense_and_nf4_answers_together() -> None:
    """One set of masters lowers both the dense and the NF4 pointer loss."""
    cfg, params = _qat_encoder()
    head = model.Config(encoder_width=cfg.hidden_size, pointer_width=8)
    params.update(model.initialize(head, jax.random.PRNGKey(3)))
    plan = magicbox.quantization_plan(cfg, "nf4")
    inventory = magicbox.pointer_inventory(cfg, head, plan)
    assert {spec.name for spec in inventory.specs if spec.quantized} == (
        magicbox.encoder_projections(cfg)
    )
    record = _record()
    batches = batching.PointerBatchStrategy(
        batching.Shape(1, 1, record.sequence_tokens + 3, 8, 128, 0),
        weighting.balance_types,
    )
    packed = batches.pack([record], seed=0, update=0)
    forward = magicbox.bind_pointer(cfg, head, bf16=False)
    update = strategy.make_distilled_step(
        forward,
        inventory,
        adamw.AdamWConfig(learning_rate=0.003, weight_decay=0),
        plan,
        objective.distilled_terms,
    )
    batch = {key: value[0] for key, value in packed.microbatches.items()}

    def losses(masters: dict[str, jax.Array]) -> tuple[float, float]:
        quantized = quantization.apply(masters, inventory, plan)
        return tuple(  # type: ignore[return-value]
            float(objective.terms(forward(weights, batch), batch)[0])
            for weights in (masters, quantized)
        )

    before = losses(params)
    current = adamw.initialize_state(params, inventory)
    for _ in range(40):
        result = update(current, packed.microbatches, packed.active)
        assert bool(result.committed)
        current = result.state
    dense, nf4 = losses(current["params"])
    assert dense < before[0] * 0.5
    assert nf4 < before[1] * 0.5
