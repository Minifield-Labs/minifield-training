"""Admit the dataset builder's pinned tokenizer and explicit offset policy."""

import json
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-untyped]

from examples.magicbox import data as magicbox
from minifield_training.core import json_io
from minifield_training.datasets import fields
from minifield_training.models.lfm2_5 import encoder

TOKENIZER_SHA256 = (
    "378b5d20a05932e01685774af0c77c018ee6689b98798b1f5743a4d01878ca44"
)


class Adapter:
    """Load the saved native IDs; apply offset policy outside the tokenizer."""

    def __init__(self, directory: Path):
        self.contract = json_io.object_map(
            json.loads((directory / "contract.json").read_text())
        )
        # A device bundle's tokenizer is trimmed from the dataset's pinned one
        # (examples.magicbox.export) and declares its own bytes.
        trimmed = self.contract.get("trimmed_from") == TOKENIZER_SHA256
        expected = {
            "model": encoder.SOURCE.model_id,
            "revision": encoder.SOURCE.revision,
            "original_sha256": encoder.SOURCE.tokenizer_sha256,
            "offset_policy": magicbox.OFFSET_POLICY,
            "template": magicbox.TEMPLATE,
            "padding_side": "right",
            "readout_token_id": 1,
        }
        if not trimmed:
            expected["sha256"] = TOKENIZER_SHA256
        if any(
            self.contract.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("Dataset tokenizer contract mismatch")
        path = directory / "tokenizer.json"
        if json_io.digest_file(path) != self.contract.get("sha256"):
            raise ValueError("Dataset tokenizer bytes changed")
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    def encode(self, text: str) -> fields.Encoding:
        """Trim edges of text tokens while preserving whitespace-only tokens."""
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        return fields.Encoding(
            tuple(encoded.ids),
            fields.trim_offsets(
                text,
                tuple(encoded.offsets),
                tuple(bool(value) for value in encoded.special_tokens_mask),
            ),
            tuple(bool(value) for value in encoded.special_tokens_mask),
        )
