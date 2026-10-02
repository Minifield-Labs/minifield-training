"""Model-neutral token offsets and schema-conditioned typed supervision."""

import dataclasses
from enum import IntEnum
import math
from typing import cast


class Kind(IntEnum):
    """Task codes for extraction, categories, binary, and ordinal labels."""

    EXTRACT = 0
    CHOICE = 1
    BINARY = 2
    ORDINAL = 3


@dataclasses.dataclass(frozen=True)
class Encoding:
    """Native IDs and Python-character offsets, including special tokens."""

    ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    special: tuple[bool, ...]

    @property
    def selectable(self) -> tuple[bool, ...]:
        """Select only non-special tokens with nonempty source ranges."""
        return tuple(
            not special and end > start
            for (start, end), special in zip(
                self.offsets, self.special, strict=True
            )
        )


@dataclasses.dataclass(frozen=True)
class Field:
    """Public row tokens and private, separately stored field supervision."""

    key: str
    kind: int
    candidates: tuple[str, ...]
    rows: tuple[tuple[int, ...], ...]
    targets: tuple[float, ...]
    supervised: bool
    token_supervised: bool
    span: tuple[int, int] | None
    legend: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Record:
    """One source with independently encoded schema rows and gold fields."""

    id: str
    text: str
    source: Encoding
    fields: tuple[Field, ...]


def aligned_span(source: Encoding, span: object, text: str) -> tuple[int, int]:
    """Admit exact boundaries, keeping all overlapping UTF-8 byte tokens."""
    if (
        not isinstance(span, list)
        or len(span) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in span
        )
    ):
        raise ValueError("Invalid character span")
    start, end = cast(list[int], span)
    if not 0 <= start < end <= len(text):
        raise ValueError("Character span outside source")
    indices = [
        index
        for index, ((left, right), valid) in enumerate(
            zip(source.offsets, source.selectable, strict=True)
        )
        if valid and right > start and left < end
    ]
    if not indices:
        raise ValueError("Unalignable span")
    first, last = indices[0], indices[-1]
    if source.offsets[first][0] != start or source.offsets[last][1] != end:
        raise ValueError("Gold span splits a token")
    previous = (-1, -1)
    for index in range(first, last + 1):
        left, right = source.offsets[index]
        if (
            not source.selectable[index]
            or not start <= left < right <= end
            or left < previous[0]
            or right < previous[1]
        ):
            raise ValueError("Invalid interior token offset")
        previous = (left, right)
    return first, last + 1


def probabilities(
    values: tuple[object, ...], *, distribution: bool
) -> tuple[float, ...]:
    """Reject nonfinite targets and malformed probability distributions."""
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in values
    ):
        raise ValueError("Invalid target probability")
    result = tuple(float(cast(float, value)) for value in values)
    if distribution and not math.isclose(sum(result), 1, abs_tol=1e-5):
        raise ValueError("Target distribution must sum to one")
    return result


def trim_offsets(
    text: str, offsets: tuple[tuple[int, int], ...], special: tuple[bool, ...]
) -> tuple[tuple[int, int], ...]:
    """Trim text-token edges while preserving whitespace-only token spans."""
    result = []
    for (start, end), is_special in zip(offsets, special, strict=True):
        if not is_special and text[start:end].strip():
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
        result.append((start, end))
    return tuple(result)


def validate(record: Record, *, vocab_size: int) -> None:
    """Admit neutral observations and typed labels before array allocation."""
    if (
        not record.id
        or not record.fields
        or len({field.key for field in record.fields}) != len(record.fields)
    ):
        raise ValueError("Record needs an identity and distinct fields")
    source = record.source
    if (
        not source.ids
        or len(source.ids) != len(source.offsets)
        or len(source.ids) != len(source.special)
    ):
        raise ValueError("Source encoding lengths differ")
    if any(
        not 0 <= start <= end <= len(record.text)
        for start, end in source.offsets
    ):
        raise ValueError("Source offset outside text")
    for field in record.fields:
        if (
            not field.key
            or not isinstance(field.kind, int)
            or isinstance(field.kind, bool)
            or field.kind not in range(4)
            or not field.rows
            or len(field.targets) != len(field.rows)
        ):
            raise ValueError("Invalid field kind or row targets")
        if any(
            isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in field.targets
        ):
            raise ValueError("Invalid field target probability")
        if field.kind in (Kind.CHOICE, Kind.ORDINAL):
            if len(field.candidates) != len(field.rows) or len(
                set(field.candidates)
            ) != len(field.candidates):
                raise ValueError("Invalid candidate inventory")
            if field.supervised and not math.isclose(
                sum(field.targets), 1, abs_tol=1e-5
            ):
                raise ValueError("Target distribution must sum to one")
        elif len(field.rows) != 1:
            raise ValueError("Binary and extraction fields need one row")
        if field.kind != Kind.EXTRACT and field.token_supervised:
            raise ValueError("Token supervision requires an extraction field")
        if field.kind == Kind.EXTRACT and field.supervised:
            present = field.targets[0]
            if (
                present not in (0, 1)
                or (present == 0 and field.span is not None)
                or (
                    present == 1
                    and field.token_supervised
                    and field.span is None
                )
                or (field.span is not None and not field.token_supervised)
            ):
                raise ValueError(
                    "Inconsistent extraction presence or span supervision"
                )
        if field.span is not None:
            start, end = field.span
            if (
                field.kind != Kind.EXTRACT
                or not 0 <= start < end <= len(source.ids)
                or not all(source.selectable[start:end])
            ):
                raise ValueError("Invalid field token span")
    for ids in (
        source.ids,
        *(row for field in record.fields for row in field.rows),
    ):
        if not ids or any(
            not isinstance(token, int)
            or isinstance(token, bool)
            or not 0 <= token < vocab_size
            for token in ids
        ):
            raise ValueError("Token outside configured vocabulary")
