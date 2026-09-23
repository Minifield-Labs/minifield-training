"""Deterministic JSON text and streamed file digests for host contracts."""

import hashlib
import json
import pathlib


def canonical(value: object) -> str:
    """Encode deterministic JSON, rejecting non-finite values."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_file(path: pathlib.Path) -> str:
    """Compute SHA-256 without retaining an entire artifact in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
