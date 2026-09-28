"""Offline tiny-model overfit and checkpoint recovery before full training."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from minifield_training.batching import magicbox as batching
from minifield_training.checkpoints import training_state
from minifield_training.datasets import magicbox as data
from minifield_training.kernels import types
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import model
from minifield_training.objectives import magicbox as objective
from minifield_training.optimizers import adamw
from minifield_training.strategies import magicbox


def toy_encode(text: str) -> data.Encoding:
    """Use deterministic whitespace tokens for offline structural checks."""
    spans = [
        (match.start(), match.end()) for match in re.finditer(r"\S+|\s+", text)
    ]
    ids = [
        2
        + int.from_bytes(
            hashlib.sha256(text[start:end].encode()).digest()[:2], "little"
        )
        % 126
        for start, end in spans
    ]
    return data.Encoding(
        (1, *ids), ((0, 0), *spans), (True, *((False,) * len(ids)))
    )


def fixture() -> data.Record:
    """Cover a span, categorical choice, boolean probability, and rubric."""
    return data.compile_record(
        "tiny",
        {
            "state": "Ada sent 3 reports.",
            "questions": {
                "person": {
                    "type": "extract",
                    "instructions": "Who sent reports?",
                },
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
        },
        {
            "person": {"has_answer": True, "span": [0, 3], "text": "Ada"},
            "category": {"choice": "report"},
            "sent": {"probability": 1.0},
            "amount": {"level": 2},
        },
        toy_encode,
    )


def tiny(seed: int = 17) -> tuple[lfm.Config, model.Config, types.Parameters]:
    """Initialize a small hybrid with the real parameter layout."""
    cfg = lfm.Config(16, 32, 2, 1, 128, ("conv", "full_attention"))
    fusion = model.Config(
        encoder_width=16,
        width=16,
        heads=2,
        match_width=8,
        dropout=0,
        row_chunk=4,
    )
    parameters = {}
    for index, (name, shape) in enumerate(
        encoder.Adapter().expected_shapes(cfg).items()
    ):
        parameters[name] = (
            jnp.ones(shape, jnp.float32)
            if len(shape) == 1
            else jax.random.normal(jax.random.PRNGKey(seed + index), shape)
            * 0.08
        )
    parameters.update(model.initialize(fusion, jax.random.PRNGKey(seed)))
    return cfg, fusion, parameters


def run(steps: int = 80) -> dict[str, object]:
    """Exercise full gradients, restart, and all-four-type optimization."""
    cfg, fusion, params = tiny()
    record = fixture()
    packed = batching.build(
        [record], batching.Shape(1, 1, 16, 64, 8), seed=17, update=0
    )
    batch = {
        key: jnp.asarray(value[0]) for key, value in packed.microbatches.items()
    }
    inventory = magicbox.inventory(cfg, fusion)
    optimizer = adamw.AdamWConfig(learning_rate=0.003, weight_decay=0)
    update = magicbox.make_step(cfg, fusion, optimizer, bf16=False)
    current = adamw.initialize_state(params, inventory)
    before = objective.terms(
        magicbox.forward(params, cfg, fusion, batch, bf16=False), batch
    )[0]
    for _ in range(steps):
        result = update(current, packed.microbatches, packed.active)
        if not bool(result.committed):
            raise RuntimeError(f"Tiny optimizer rejected update: {result.code}")
        current = result.state
    after = objective.terms(
        magicbox.forward(current["params"], cfg, fusion, batch, bf16=False),
        batch,
    )[0]
    if not float(after) < float(before) * 0.25:
        raise RuntimeError(f"Tiny overfit failed: {before} -> {after}")
    cursor = training_state.Cursor(
        "tiny", "synthetic", "magicbox-tiny/1", steps
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "checkpoint"
        training_state.save(
            path,
            current,
            inventory,
            optimizer_id=optimizer.implementation_identity,
            cursor=cursor,
        )
        restored, restored_cursor = training_state.load(
            path,
            inventory,
            optimizer_id=optimizer.implementation_identity,
            run_id="tiny",
            data_sha256="synthetic",
            source_id="magicbox-tiny/1",
        )
        assert restored_cursor == cursor
        original_next = update(
            current, packed.microbatches, packed.active
        ).state
        restored_next = update(
            restored, packed.microbatches, packed.active
        ).state
        for left, right in zip(
            jax.tree.leaves(original_next),
            jax.tree.leaves(restored_next),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    counts = {"backbone": 0, "fusion": 0, "heads": 0}
    for name, value in params.items():
        group = (
            "backbone"
            if name.startswith("lfm2.")
            else "fusion"
            if "fusion." in name or "norm." in name or "projection" in name
            else "heads"
        )
        counts[group] += value.size
    return {
        "initial_loss": float(before),
        "final_loss": float(after),
        "steps": steps,
        "parameters": counts,
        "device": str(jax.devices()[0]),
        "dtype": "float32",
        "checkpoint_next_update": "exact",
    }


def main() -> None:
    """Print the bounded smoke report as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(run(args.steps), indent=2))


if __name__ == "__main__":
    main()
