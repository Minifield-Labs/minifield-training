"""Admit the dataset builder's pinned tokenizer and explicit offset policy."""

import json
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-untyped]

from minifield_training.core import json_io
from minifield_training.datasets import magicbox
from minifield_training.models.lfm2_5 import encoder

TOKENIZER_SHA256 = (
    "378b5d20a05932e01685774af0c77c018ee6689b98798b1f5743a4d01878ca44"
)


class Adapter:
    """Load the saved native IDs; apply offset policy outside the tokenizer."""

    def __init__(self, directory: Path):
        self.contract = magicbox.object_map(
            json.loads((directory / "contract.json").read_text())
        )
        expected = {
            "model": encoder.SOURCE.model_id,
            "revision": encoder.SOURCE.revision,
            "original_sha256": encoder.SOURCE.tokenizer_sha256,
            "sha256": TOKENIZER_SHA256,
            "offset_policy": magicbox.OFFSET_POLICY,
            "template": magicbox.TEMPLATE,
            "padding_side": "right",
            "readout_token_id": 1,
        }
        if any(
            self.contract.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("Dataset tokenizer contract mismatch")
        path = directory / "tokenizer.json"
        if json_io.digest_file(path) != TOKENIZER_SHA256:
            raise ValueError("Dataset tokenizer bytes changed")
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    def encode(self, text: str) -> magicbox.Encoding:
        """Trim edges of text tokens while preserving whitespace-only tokens."""
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        offsets = []
        for (start, end), special in zip(
            encoded.offsets, encoded.special_tokens_mask, strict=True
        ):
            if not special and text[start:end].strip():
                while start < end and text[start].isspace():
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
            offsets.append((start, end))
        return magicbox.Encoding(
            tuple(encoded.ids),
            tuple(offsets),
            tuple(bool(value) for value in encoded.special_tokens_mask),
        )
