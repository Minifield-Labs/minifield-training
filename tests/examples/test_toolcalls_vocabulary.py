"""Tool-call vocabulary: pinned IDs, normalization, export and markers."""

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers import decoders
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import processors
from tokenizers import trainers

from examples.magicbox import export
from examples.toolcalls import data
from examples.toolcalls import vocabulary
from minifield_training.core import json_io

CORPUS = [
    "Restart the backend pod and scale the deployment to three replicas.",
    "Invoices are due on the first of every month for each account.",
    "It's José's turn to review the Q3 launch notes.",
] * 20
_SPECIALS = ["<|pad|>", "<|startoftext|>", "<|mask|>"] + [
    f"<|reserved_{index}|>" for index in range(7, 17)
]


def _tokenizer(directory: Path) -> Tokenizer:
    """A small byte-level BPE with the encoder's BOS template."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        CORPUS,
        trainers.BpeTrainer(
            vocab_size=420,
            special_tokens=_SPECIALS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|startoftext|> $A", special_tokens=[("<|startoftext|>", 1)]
    )
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "contract.json").write_text('{"offset_policy": "test"}')
    return tokenizer


def pin_vocabulary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keep: str = "the deployment",
) -> tuple[Path, Tokenizer]:
    """Pin the tokens of ``keep`` plus everything the merges need."""
    directory = tmp_path / "tokenizer"
    tokenizer = _tokenizer(directory)
    spec = json.loads(
        (directory / "tokenizer.json").read_text(encoding="utf-8")
    )
    _, kept = export.trim_tokenizer(spec, tokenizer.encode(keep).ids)
    pinned = tmp_path / "vocabulary.json"
    pinned.write_text(
        json.dumps(
            {
                "tokenizer_sha256": json_io.digest_file(
                    directory / "tokenizer.json"
                ),
                "normalization": {"’": "'", "“": '"', "”": '"'},
                "kept": list(kept),
            }
        )
    )
    monkeypatch.setattr(vocabulary, "PATH", pinned)
    return directory, tokenizer


def test_encodings_carry_original_ids_and_original_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rare words split into kept pieces; curly quotes keep their offsets."""
    directory, full = pin_vocabulary(tmp_path, monkeypatch)
    encoder = vocabulary.Encoder(directory)
    text = "Scale “the deployment”, it’s due"
    encoded = encoder.encode(text)
    assert encoded.ids[0] == data.BOS
    assert set(encoded.ids) <= set(encoder.kept)
    # Words outside the vocabulary split into more, kept pieces.
    plain = "Restart the backend pod"
    assert len(encoder.encode(plain).ids) > len(full.encode(plain).ids)
    pieces = [text[start:end] for start, end in encoded.offsets]
    assert "“" in pieces and "”" in pieces
    assert "".join(pieces[1:]).replace(" ", "") == text.replace(" ", "")
    assert encoder.lengths([text]) == [len(encoded.ids)]
    assert encoder.token_id("<|mask|>") == full.token_to_id("<|mask|>")


def test_rejects_another_tokenizer_or_an_open_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pinned list must match its tokenizer and contain its merges."""
    directory, _ = pin_vocabulary(tmp_path, monkeypatch)
    pinned = json.loads(vocabulary.PATH.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    # Whole-word tokens without the pieces their merges are built from.
    open_ids = sorted(
        set(tokenizer.encode("deployment replicas").ids) | set(range(13))
    )
    vocabulary.PATH.write_text(
        json.dumps({**pinned, "kept": open_ids}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="every token it needs"):
        vocabulary.Encoder(directory)
    vocabulary.PATH.write_text(
        json.dumps({**pinned, "tokenizer_sha256": "0" * 64}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="another tokenizer"):
        vocabulary.Encoder(directory)


def test_exported_tokenizer_reads_like_training_and_finds_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bundle IDs index the kept rows; markers are found by their names."""
    directory, _ = pin_vocabulary(tmp_path, monkeypatch)
    encoder = vocabulary.Encoder(directory)
    spec, kept = vocabulary.tokenizer_spec(directory)
    exported = tmp_path / "device"
    export.write_tokenizer(
        spec, kept, directory, exported, rename=data.TOKEN_NAMES
    )
    device = Tokenizer.from_file(str(exported / "tokenizer.json"))
    for text in CORPUS[:3] + ["Due “now”, it’s late"]:
        ids = device.encode(text).ids
        assert [kept[index] for index in ids] == list(encoder.encode(text).ids)
    full = Tokenizer.from_file(str(directory / "tokenizer.json"))
    assert data.device_markers(device) == {
        name: kept.index(full.token_to_id(reserved))
        for reserved, name in zip(data.TOKEN_NAMES, data.MARKERS, strict=True)
    }
    contract = json.loads(
        (exported / "contract.json").read_text(encoding="utf-8")
    )
    assert contract["vocab_size"] == len(kept)
    assert contract["renamed"] == data.TOKEN_NAMES


def test_pinned_vocabulary_fits_the_cap() -> None:
    """The committed vocabulary keeps BOS and every marker within 12,000."""
    pinned = json.loads(vocabulary.PATH.read_text(encoding="utf-8"))
    kept = pinned["kept"]
    assert len(kept) <= 12000 and kept == sorted(set(kept))
    assert {data.BOS, *data.MARKER_IDS.values()} <= set(kept)
