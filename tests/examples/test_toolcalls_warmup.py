"""Quantization warm-up: masking, distillation progress, resume, output."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from safetensors.numpy import save_file

from examples.toolcalls import warmup
from minifield_training.checkpoints import tensors
from minifield_training.core import json_io
from minifield_training.models import contracts
from minifield_training.models.lfm2_5 import encoder
from tests.examples import test_toolcalls_vocabulary as vocabulary_test


def _settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> warmup.Settings:
    """A tiny group-128 encoder, a pinned vocabulary and a text file."""
    directory, tokenizer = vocabulary_test.pin_vocabulary(
        tmp_path, monkeypatch, " ".join(vocabulary_test.CORPUS[:3])
    )
    model = tmp_path / "model"
    model.mkdir()
    config = {
        "model_type": "lfm2",
        "architectures": ["Lfm2BidirectionalForMaskedLM"],
        "hidden_size": 128,
        "intermediate_size": 128,
        "block_auto_adjust_ff_dim": False,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 512,
        "num_hidden_layers": 2,
        "layer_types": ["conv", "full_attention"],
    }
    (model / "config.json").write_text(json.dumps(config))
    tokenizer.save(str(model / "tokenizer.json"))
    cfg = encoder.Adapter().parse_config(config)
    rng = np.random.default_rng(0)
    save_file(
        {
            name: np.ones(shape, np.float32)
            if len(shape) == 1
            else (rng.normal(size=shape) * 0.05).astype(np.float32)
            for name, shape in encoder.Adapter().expected_shapes(cfg).items()
        },
        str(model / "model.safetensors"),
    )
    source = contracts.PretrainedSource(
        "tiny",
        "test",
        json_io.digest_file(model / "config.json"),
        json_io.digest_file(model / "tokenizer.json"),
        json_io.digest_file(model / "model.safetensors"),
    )
    text = tmp_path / "text.parquet"
    pq.write_table(
        pa.table({"text": vocabulary_test.CORPUS * 4}),
        text,
        row_group_size=40,
    )
    return warmup.Settings(
        model_dir=model,
        dataset_tokenizer=directory,
        text=text,
        output=tmp_path / "warmup",
        encoder_source=source,
        tokens=30 * 4 * 32,
        rows=4,
        sequence_tokens=32,
        learning_rate=3e-3,
        log_every=10,
        checkpoint_every=10,
    )


def test_masking_spares_bos_and_padding() -> None:
    """Only real tokens after BOS are masked, about the configured share."""
    ids = np.zeros((64, 32), np.uint16)
    ids[:, 0] = 1
    ids[:, 1:20] = 7
    settings = warmup.Settings(Path(), Path(), Path(), Path())
    masked, mask, chosen = warmup.masked(ids, 3, settings, 2)
    assert not chosen[:, 0].any() and not chosen[:, 20:].any()
    assert (masked[chosen > 0] == 2).all()
    assert (mask == (ids > 0)).all()
    assert 0.15 < chosen[:, 1:20].mean() < 0.35


def test_warmup_learns_resumes_and_writes_loadable_masters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distillation loss falls; a stopped run resumes; output loads as F32."""
    settings = _settings(tmp_path, monkeypatch)
    events: list[dict[str, object]] = []
    assert warmup.run(settings, session_seconds=0, report=events.append) is None
    assert events[-1] == {"event": "warmup_stopped", "step": 0}
    result = warmup.run(settings, session_seconds=600, report=events.append)
    assert result is not None and result.exists()
    losses = [float(str(e["loss"])) for e in events if "warmup_step" in e]
    assert len(losses) == 3 and losses[-1] < losses[0]
    assert all(np.isfinite(losses))
    cfg = encoder.Adapter().parse_config(
        json.loads((settings.model_dir / "config.json").read_text())
    )
    loaded = tensors.load_masters(
        result,
        encoder.Adapter().expected_shapes(cfg),
        source_dtype="F32",
        sha256=json_io.digest_file(result),
    )
    assert warmup.EMBEDDINGS in loaded
    # A finished warm-up returns its file without training again.
    assert (
        warmup.run(settings, session_seconds=0, report=events.append) == result
    )


def test_encoder_sources_are_found_by_config() -> None:
    """Either pinned release resolves from its config digest."""
    for source in (encoder.SOURCE, encoder.SOURCE_230M):
        assert encoder.source_for_config(source.config_sha256) == source
    with pytest.raises(ValueError, match="no pinned source"):
        encoder.source_for_config("0" * 64)
