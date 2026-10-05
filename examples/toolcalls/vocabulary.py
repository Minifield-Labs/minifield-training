"""The tool-call model's pinned vocabulary and typographic normalization.

``vocabulary-v1.json`` lists the original token IDs the model reads with:
the dataset's most frequent ASCII tokens, plus every byte, added and
merge-intermediate token, so any text still encodes (rare words in more
pieces). Curly quotes and dashes become ASCII one character for one, so
offsets still point into the original text. The experiment's
``select_vocab.py`` picked the IDs.

The encoder keeps its full embedding table while training, so encodings
carry original IDs; bundles keep only these rows and use the trimmed IDs.
"""

import json
from pathlib import Path
from typing import cast

from tokenizers import Tokenizer  # type: ignore[import-untyped]

from examples.magicbox import export
from minifield_training.core import json_io
from minifield_training.datasets import fields

PATH = Path(__file__).with_name("vocabulary-v1.json")


def tokenizer_spec(
    dataset_tokenizer: Path,
) -> tuple[dict[str, object], tuple[int, ...]]:
    """The trimmed tokenizer JSON and its original IDs, in trimmed order."""
    pinned = json_io.object_map(json.loads(PATH.read_text(encoding="utf-8")))
    source = dataset_tokenizer / "tokenizer.json"
    if json_io.digest_file(source) != pinned["tokenizer_sha256"]:
        raise ValueError("The vocabulary was picked from another tokenizer")
    wanted = tuple(cast(list[int], pinned["kept"]))
    spec, kept = export.trim_tokenizer(
        json.loads(source.read_text(encoding="utf-8")), wanted
    )
    if kept != wanted:
        raise ValueError("The vocabulary must include every token it needs")
    spec["normalizer"] = {
        "type": "Sequence",
        "normalizers": [
            {"type": "Replace", "pattern": {"String": old}, "content": new}
            for old, new in json_io.object_map(pinned["normalization"]).items()
        ],
    }
    return spec, kept


class Encoder:
    """Encode with the vocabulary, returning the encoder's original IDs."""

    def __init__(self, dataset_tokenizer: Path):
        spec, self.kept = tokenizer_spec(dataset_tokenizer)
        self.tokenizer = Tokenizer.from_str(json.dumps(spec))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    def encode(self, text: str) -> fields.Encoding:
        """Tokenize like the dataset adapter: BOS first, trimmed offsets."""
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        special = tuple(bool(value) for value in encoded.special_tokens_mask)
        return fields.Encoding(
            tuple(self.kept[index] for index in encoded.ids),
            fields.trim_offsets(text, tuple(encoded.offsets), special),
            special,
        )

    def lengths(self, texts: list[str]) -> list[int]:
        """Token counts, BOS included, for many texts at once."""
        return [
            len(encoded.ids)
            for encoded in self.tokenizer.encode_batch(
                texts, add_special_tokens=True
            )
        ]

    def token_id(self, token: str) -> int:
        """The original ID of a token the vocabulary keeps."""
        index = self.tokenizer.token_to_id(token)
        if index is None:
            raise ValueError(f"The vocabulary has no {token}")
        return self.kept[int(index)]
