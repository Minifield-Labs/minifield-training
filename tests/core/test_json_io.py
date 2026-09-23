"""Independent byte contracts for canonical JSON text and file digests."""

import pathlib
from typing import BinaryIO, cast

import pytest

from minifield_training.core import json_io


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"z": 1, "a": {"y": 2, "x": "é"}}, '{"a":{"x":"é","y":2},"z":1}'),
        (
            [None, True, False, 1, 1.0, -0.0],
            "[null,true,false,1,1.0,-0.0]",
        ),
        ({"value": 'a\nb\t"\\'}, '{"value":"a\\nb\\t\\"\\\\"}'),
        ({"html": "<tag>&"}, '{"html":"<tag>&"}'),
        ((1, "x"), '[1,"x"]'),
        ({2: "b", 1: "a"}, '{"1":"a","2":"b"}'),
        ({10: "b", 2: "a"}, '{"2":"a","10":"b"}'),
        (9007199254740992, "9007199254740992"),
        ({"\U00010000": 1, "\ue000": 2}, '{"\ue000":2,"\U00010000":1}'),
        ({}, "{}"),
        ([], "[]"),
    ],
)
def test_canonical_exact_text(value: object, expected: str) -> None:
    """Encoding sorts keys recursively with minimal separators."""
    assert json_io.canonical(value) == expected


def test_canonical_returns_str_without_trailing_newline() -> None:
    """The result is plain text, not encoded or newline-terminated bytes."""
    result = json_io.canonical({"a": 1})
    assert isinstance(result, str)
    assert not result.endswith("\n")


def test_canonical_utf8_bytes_preserve_unicode() -> None:
    """UTF-8 encoding keeps non-ASCII text unescaped in sorted keys."""
    result = json_io.canonical({"a": "é"})
    assert result.encode("utf-8") == b'{"a":"\xc3\xa9"}'


def test_canonical_preserves_unicode_normalization() -> None:
    """Composed and decomposed forms remain distinct JSON text."""
    composed = json_io.canonical({"x": "é"})
    decomposed = json_io.canonical({"x": "e\u0301"})
    assert composed == '{"x":"é"}'
    assert decomposed == '{"x":"e\u0301"}'
    assert composed != decomposed


def test_canonical_retains_lone_surrogate() -> None:
    """A lone surrogate stays in the result but cannot encode to UTF-8."""
    result = json_io.canonical("\ud800")
    assert result == '"\ud800"'
    with pytest.raises(UnicodeEncodeError):
        result.encode("utf-8")


@pytest.mark.parametrize("value", [object(), {1, 2}, pathlib.Path("x")])
def test_canonical_rejects_unsupported_values(value: object) -> None:
    """Values without a JSON encoding raise TypeError."""
    with pytest.raises(TypeError):
        json_io.canonical(value)


def test_canonical_rejects_mixed_key_types() -> None:
    """Incomparable string and integer keys raise TypeError when sorted."""
    with pytest.raises(TypeError):
        json_io.canonical({1: "a", "b": 2})


def test_canonical_rejects_circular_containers() -> None:
    """A self-referencing container raises ValueError."""
    value: list[object] = []
    value.append(value)
    with pytest.raises(ValueError):
        json_io.canonical(value)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        {"nested": float("nan")},
        {"nested": float("inf")},
        {"nested": float("-inf")},
        [float("nan")],
    ],
)
def test_canonical_rejects_non_finite_numbers(value: object) -> None:
    """NaN and infinities raise ValueError at the root or nested."""
    with pytest.raises(ValueError):
        json_io.canonical(value)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            b"\x00\xff\r\n",
            "e9489f37fb3051e9efa1dc916004d7274e7b63975e3209708947267f2393a9be",
        ),
        (
            b"a" * (8 * 1024 * 1024 + 1),
            "c92697f4cc3b569dff3d484285d22487e523d4b439ae7c9a6747dc258e35b275",
        ),
    ],
    ids=["empty", "abc", "binary", "crosses-read-boundary"],
)
def test_digest_file_known_bytes(
    tmp_path: pathlib.Path, content: bytes, expected: str
) -> None:
    """SHA-256 of fixed byte fixtures matches independently known digests."""
    path = tmp_path / "blob.bin"
    path.write_bytes(content)
    assert json_io.digest_file(path) == expected


def test_digest_file_propagates_missing_file(tmp_path: pathlib.Path) -> None:
    """A missing path surfaces the ordinary FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        json_io.digest_file(tmp_path / "missing.bin")


class _RecordingStream:
    """Forward reads to a real binary stream while logging request sizes."""

    def __init__(self, stream: BinaryIO, sizes: list[int]) -> None:
        self._stream = stream
        self._sizes = sizes

    def __enter__(self) -> "_RecordingStream":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stream.close()

    def read(self, size: int = -1) -> bytes:
        self._sizes.append(size)
        return self._stream.read(size)


class _RecordingPath:
    """Stand in for Path while recording each read size the loop requests."""

    def __init__(self, path: pathlib.Path, sizes: list[int]) -> None:
        self._path = path
        self._sizes = sizes

    def open(self, mode: str) -> _RecordingStream:
        assert mode == "rb"
        return _RecordingStream(self._path.open("rb"), self._sizes)


def test_digest_file_reads_bounded_chunks(tmp_path: pathlib.Path) -> None:
    """A file larger than one chunk is consumed through 8 MiB reads."""
    path = tmp_path / "blob.bin"
    path.write_bytes(b"a" * (8 * 1024 * 1024 + 1))
    sizes: list[int] = []
    recording = cast(pathlib.Path, _RecordingPath(path, sizes))
    assert (
        json_io.digest_file(recording)
        == "c92697f4cc3b569dff3d484285d22487e523d4b439ae7c9a6747dc258e35b275"
    )
    assert sizes == [8 * 1024 * 1024, 8 * 1024 * 1024, 8 * 1024 * 1024]
