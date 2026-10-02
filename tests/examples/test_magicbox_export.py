"""Device packaging: trimmed vocabulary and FP32 or NF4 bundles."""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers import decoders
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import processors
from tokenizers import trainers

from examples.magicbox import bundle as magicbox_bundle
from examples.magicbox import composition as magicbox
from examples.magicbox import export
from minifield_training.core import json_io
from minifield_training.models.lfm2_5 import encoder
from minifield_training.models.lfm2_5 import model as lfm
from minifield_training.models.magicbox import pointer
from minifield_training.strategies import quantization

_CORPUS = [
    "Ada sent 3 reports to the billing desk.",
    "Call +1 416-555-0184 for the dispatch team.",
    "Invoice total: USD 3,574.24, due on 2026-10-01.",
    "Grace Hopper reviewed the compiler release notes.",
] * 5


def _tokenizer() -> Tokenizer:
    """A small byte-level BPE with a BOS template, like the encoder's."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        _CORPUS,
        trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=["<|pad|>", "<|startoftext|>", "<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|startoftext|> $A", special_tokens=[("<|startoftext|>", 1)]
    )
    return tokenizer


def test_trimming_keeps_used_encodings_and_still_encodes_anything() -> None:
    """Seen texts encode identically; unseen text survives in smaller pieces."""
    original = _tokenizer()
    seen = _CORPUS[:2]
    spec = json.loads(original.to_str())
    trimmed_spec, kept = export.trim_tokenizer(
        spec, export.used_ids(original, seen)
    )
    trimmed = Tokenizer.from_str(json.dumps(trimmed_spec))
    assert len(kept) < original.get_vocab_size()
    assert export.check_trimmed(original, trimmed, kept, seen) == 2
    assert trimmed.encode("x").ids[0] == 1
    unseen = "Invoice total: USD 3,574.24 — reviewed 東京"
    ids = [kept[index] for index in trimmed.encode(unseen).ids]
    assert original.decode(ids) == unseen
    assert len(ids) > len(original.encode(unseen).ids)
    with pytest.raises(ValueError, match="changed an encoding"):
        export.check_trimmed(original, trimmed, kept, [_CORPUS[2]])


def _encoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> lfm.Config:
    """Pin a tiny encoder config whose projections fit group-128 NF4."""
    config = tmp_path / "encoder.json"
    config.write_text(
        json.dumps(
            {
                "model_type": "lfm2",
                "architectures": ["Lfm2BidirectionalForMaskedLM"],
                "hidden_size": 128,
                "intermediate_size": 128,
                "block_auto_adjust_ff_dim": False,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 64,
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
    return encoder.Adapter().parse_config(
        json_io.object_map(json.loads(config.read_text()))
    )


@pytest.mark.parametrize("quantizer", [None, "nf4"])
def test_device_bundle_runs_like_the_trained_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quantizer: str | None
) -> None:
    """Remapped IDs on trimmed weights give the trained model's logits."""
    cfg = _encoder(tmp_path, monkeypatch)
    head = pointer.Config(encoder_width=128, pointer_width=8)
    params = {
        name: jnp.ones(shape, jnp.float32)
        if len(shape) == 1
        else jax.random.normal(jax.random.PRNGKey(index), shape) * 0.05
        for index, (name, shape) in enumerate(
            encoder.Adapter().expected_shapes(cfg).items()
        )
    }
    params.update(pointer.initialize(head, jax.random.PRNGKey(3)))
    kept = (0, 1, 2, 5, 9, 17, 33, 40)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"synthetic":true}')
    (tokenizer / "contract.json").write_text(
        json.dumps({"trimmed_from": "pinned-digest"})
    )
    bundle = tmp_path / "bundle"
    magicbox_bundle.save_device(
        bundle,
        export.trimmed_parameters(params, kept),
        cfg,
        head,
        vocabulary=kept,
        quantizer=quantizer,
        encoder_config=tmp_path / "encoder.json",
        tokenizer=tokenizer,
        step=5,
    )
    loaded_cfg, loaded_head, loaded, _ = magicbox_bundle.load_pointer(bundle)
    assert loaded_cfg.vocab_size == len(kept) and loaded_head == head
    reference = params
    if quantizer is not None:
        plan = magicbox.quantization_plan(cfg, quantizer)
        reference = quantization.apply(
            params, magicbox.pointer_inventory(cfg, head, plan), plan
        )
    new_ids = jnp.asarray([[1, 3, 4, 6, 7, 1, 5, 2]])
    batch = {
        "input_ids": new_ids,
        "input_mask": jnp.ones_like(new_ids),
        "segment_ids": jnp.ones_like(new_ids),
        "positions": jnp.arange(8)[None],
        "query_index": jnp.asarray([[0, 5]]),
    }
    original = {**batch, "input_ids": jnp.asarray(kept)[new_ids]}
    expected = magicbox.bind_pointer(cfg, head, bf16=False)(reference, original)
    actual = magicbox.bind_pointer(loaded_cfg, head, bf16=False)(loaded, batch)
    for end in ("start", "end"):
        np.testing.assert_allclose(
            actual[end], expected[end], rtol=1e-5, atol=1e-5
        )
