"""Package a trained MagicBox model for devices with a trimmed vocabulary.

The tokenizer is byte-level BPE. Trimming keeps every token our records
produce, every intermediate token their merges build through, all 256 byte
tokens and every added token. Merges that would build a removed token are
dropped, so our records tokenize exactly as before and any other text still
encodes, in smaller pieces. Embedding rows follow the kept IDs.
"""

from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import tempfile
from typing import cast

import jax.numpy as jnp
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers import pre_tokenizers

from examples.magicbox import data
from examples.magicbox import source
from examples.magicbox import tokenizer as magicbox_tokenizer
from minifield_training.core import json_io
from minifield_training.kernels import types

EMBEDDINGS = "lfm2.embed_tokens.weight"


def request_texts(request: object) -> Iterator[str]:
    """Every string the pointer compiler encodes for one request."""
    public = json_io.object_map(request)
    yield str(public["state"])
    for raw in json_io.object_map(public["questions"]).values():
        query, options, _ = data.pointer_texts(json_io.object_map(raw))
        yield query
        yield from (text for _, text in options)


def corpus_texts(corpus: source.Corpus) -> Iterator[str]:
    """Every encoded string across all of the dataset's splits."""
    for split in sorted({str(shard["split"]) for shard in corpus.shards}):
        requests = corpus.split(split).select_columns(["request_json"])
        for chunk in requests.iter(batch_size=4096):
            for value in chunk["request_json"]:
                yield from request_texts(json.loads(value))


def _encode(tokenizer: Tokenizer, texts: list[str]) -> list[list[int]]:
    return [
        encoding.ids
        for encoding in tokenizer.encode_batch(texts, add_special_tokens=True)
    ]


def used_ids(tokenizer: Tokenizer, texts: Iterable[str]) -> set[int]:
    """Token IDs the given texts encode to, special tokens included."""
    used: set[int] = set()
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) == 8192:
            used.update(i for ids in _encode(tokenizer, batch) for i in ids)
            batch = []
    used.update(i for ids in _encode(tokenizer, batch) for i in ids)
    return used


def trim_tokenizer(
    spec: dict[str, object], used: Iterable[int]
) -> tuple[dict[str, object], tuple[int, ...]]:
    """Return the trimmed tokenizer JSON and its kept original IDs, in order.

    New ID ``n`` is original ID ``kept[n]``. Kept IDs stay sorted, so low
    special IDs such as the BOS readout (1) keep their values.
    """
    model = json_io.object_map(spec["model"])
    if model.get("type") != "BPE" or model.get("ignore_merges"):
        raise ValueError("Trimming needs a BPE model that applies its merges")
    vocab = {
        str(k): int(str(v))
        for k, v in json_io.object_map(model["vocab"]).items()
    }
    tokens = {index: token for token, index in vocab.items()}
    merges = [
        (str(left), str(right))
        for left, right in cast(list[list[object]], model["merges"])
    ]
    added = [
        json_io.object_map(item)
        for item in cast(list[object], spec["added_tokens"])
    ]
    producer: dict[str, tuple[str, str]] = {}
    for left, right in merges:
        producer.setdefault(left + right, (left, right))
    keep = set(pre_tokenizers.ByteLevel.alphabet())
    stack = [tokens[index] for index in used if index in tokens]
    while stack:
        token = stack.pop()
        if token in keep:
            continue
        keep.add(token)
        stack.extend(producer.get(token, ()))
    kept = sorted(
        {vocab[token] for token in keep}
        | {int(str(item["id"])) for item in added}
    )
    renumber = {old: new for new, old in enumerate(kept)}
    trimmed = json.loads(json.dumps(spec))
    trimmed["model"]["vocab"] = {
        token: renumber[index]
        for token, index in vocab.items()
        if token in keep
    }
    trimmed["model"]["merges"] = [
        [left, right]
        for left, right in merges
        if left in keep and right in keep and left + right in keep
    ]
    for item in trimmed["added_tokens"]:
        item["id"] = renumber[item["id"]]
    _renumber_special_ids(trimmed.get("post_processor"), renumber)
    if renumber.get(1) != 1:
        raise ValueError("Trimming moved the BOS readout token")
    return trimmed, tuple(kept)


def _renumber_special_ids(processor: object, renumber: dict[int, int]) -> None:
    """Rewrite template special-token IDs in place, at any nesting depth."""
    if isinstance(processor, list):
        for item in processor:
            _renumber_special_ids(item, renumber)
    elif isinstance(processor, dict):
        for token in processor.get("special_tokens", {}).values():
            token["ids"] = [renumber[index] for index in token["ids"]]
        for value in processor.values():
            if isinstance(value, list | dict):
                _renumber_special_ids(value, renumber)


def check_trimmed(
    original: Tokenizer,
    trimmed: Tokenizer,
    kept: tuple[int, ...],
    texts: Iterable[str],
) -> int:
    """Require identical tokenization of ``texts``; return how many checked."""
    checked, batch = 0, []

    def compare(items: list[str]) -> None:
        for before, after in zip(
            _encode(original, items), _encode(trimmed, items), strict=True
        ):
            if before != [kept[index] for index in after]:
                raise ValueError("Trimmed tokenizer changed an encoding")

    for text in texts:
        batch.append(text)
        if len(batch) == 8192:
            compare(batch)
            checked += len(batch)
            batch = []
    compare(batch)
    return checked + len(batch)


def trimmed_parameters(
    params: types.Parameters, kept: tuple[int, ...]
) -> types.Parameters:
    """Keep only the embedding rows of the kept token IDs."""
    return {
        **params,
        EMBEDDINGS: params[EMBEDDINGS][jnp.asarray(kept, jnp.int32)],
    }


def write_trimmed_tokenizer(
    dataset_tokenizer: Path, destination: Path, corpus: source.Corpus
) -> tuple[int, ...]:
    """Trim to the corpus, verify every corpus text, and write the assets."""
    original = Tokenizer.from_file(str(dataset_tokenizer / "tokenizer.json"))
    spec = json.loads((dataset_tokenizer / "tokenizer.json").read_text())
    trimmed_spec, kept = trim_tokenizer(
        spec, used_ids(original, corpus_texts(corpus))
    )
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "tokenizer.json"
    with tempfile.NamedTemporaryFile(
        "w", dir=destination, suffix=".json", delete=False
    ) as handle:
        json.dump(trimmed_spec, handle, ensure_ascii=False)
    Path(handle.name).rename(path)
    check_trimmed(
        original, Tokenizer.from_file(str(path)), kept, corpus_texts(corpus)
    )
    contract = json.loads((dataset_tokenizer / "contract.json").read_text())
    contract.update(
        sha256=json_io.digest_file(path),
        trimmed_from=magicbox_tokenizer.TOKENIZER_SHA256,
        vocab_size=len(kept),
    )
    (destination / "contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    return kept
