"""Versioned MagicBox requests, schema templates, and exact span admission."""

from collections.abc import Callable
import dataclasses
import math
from typing import cast

FORMAT = "minifield.magicbox/1.0"
TEMPLATE = "magicbox-rows/1"
OFFSET_POLICY = "trim-text-preserve-whitespace/2"
KINDS = ("extract", "choice", "noul", "score")


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


def object_map(value: object) -> dict[str, object]:
    """Admit a JSON object with string keys."""
    if not isinstance(value, dict) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError("Expected a JSON object")
    return cast(dict[str, object], value)


def _text(value: object) -> str:
    """Require a nonempty schema string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty schema text")
    return value


def schema_rows(
    question: dict[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Serialize the pinned templates, preserving candidate insertion order."""
    kind = str(question.get("type"))
    if kind not in KINDS:
        raise ValueError("Unsupported MagicBox question type")
    instructions = _text(question.get("instructions"))
    prefix = f"Type: {kind}\nQuestion: {instructions}"
    if kind == "choice":
        criteria = object_map(question.get("criteria"))
        if not criteria:
            raise ValueError("Choice requires candidates")
        return tuple(criteria), tuple(
            f"{prefix}\nCandidate: {_text(key)}\nDescription: {_text(value)}"
            for key, value in criteria.items()
        )
    if kind == "score":
        levels = question.get("criteria")
        if not isinstance(levels, list) or not levels:
            raise ValueError("Score requires ordered levels")
        return tuple(str(index) for index in range(len(levels))), tuple(
            f"{prefix}\nLevel: {index} of {len(levels)} levels, indexed from 0"
            f"\nDescription: {_text(value)}"
            for index, value in enumerate(levels)
        )
    if kind == "noul" and question.get("criteria") is not None:
        binary = object_map(question["criteria"])
        if set(binary) != {"true", "false"} or any(
            not isinstance(value, str) for value in binary.values()
        ):
            raise ValueError("Invalid boolean criteria")
        true, false = binary["true"], binary["false"]
        prefix += f"\nTrue: {true}\nFalse: {false}"
    return ("",), (prefix,)


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


def _probabilities(
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


def _targets(
    kind: int,
    candidates: tuple[str, ...],
    target: dict[str, object],
    source: Encoding,
    text: str,
) -> tuple[tuple[float, ...], bool, tuple[int, int] | None]:
    """Normalize hard and soft labels without putting them in schema rows."""
    if kind == 0:
        present = target.get("has_answer")
        if not isinstance(present, bool):
            raise ValueError("Extraction requires explicit presence")
        if not present and (
            target.get("span") is not None or target.get("text") is not None
        ):
            raise ValueError("Absent extraction has a span")
        span = (
            aligned_span(source, target["span"], text)
            if present and target.get("span") is not None
            else None
        )
        if (
            span is not None
            and target.get("text")
            != text[source.offsets[span[0]][0] : source.offsets[span[1] - 1][1]]
        ):
            raise ValueError("Gold text differs from original substring")
        return (float(present),), not present or span is not None, span
    if kind == 2:
        return (
            _probabilities((target.get("probability"),), distribution=False),
            False,
            None,
        )
    label_key = "choice" if kind == 1 else "level"
    if set(target) not in ({label_key}, {"probabilities"}):
        raise ValueError("Ambiguous categorical target")
    if "probabilities" in target:
        probabilities = target["probabilities"]
        if kind == 1:
            mapping = object_map(probabilities)
            if set(mapping) != set(candidates):
                raise ValueError("Choice distribution keys differ")
            values = tuple(mapping[key] for key in candidates)
        elif isinstance(probabilities, list) and len(probabilities) == len(
            candidates
        ):
            values = tuple(probabilities)
        else:
            raise ValueError("Score distribution length differs")
        return _probabilities(values, distribution=True), False, None
    label = target.get(label_key)
    if kind == 1 and not isinstance(label, str):
        raise ValueError("Choice targets must be string identifiers")
    if (
        kind == 3 and (not isinstance(label, int) or isinstance(label, bool))
    ) or str(label) not in candidates:
        raise ValueError("Invalid categorical label")
    return tuple(float(key == str(label)) for key in candidates), False, None


def compile_record(
    record_id: str,
    request: object,
    targets: object,
    encode: Callable[[str], Encoding],
) -> Record:
    """Compile public input separately from optional private targets."""
    public, gold = object_map(request), object_map(targets)
    text = public.get("state")
    if not isinstance(text, str):
        raise ValueError("State must be a string")
    questions = object_map(public.get("questions"))
    if not questions or not set(gold) <= set(questions):
        raise ValueError("Invalid question or supervision keys")
    source = encode(text)
    if not source.ids or source.ids[0] != 1:
        raise ValueError("Source must include the native BOS token")
    fields = []
    for key, raw in questions.items():
        _text(key)
        question = object_map(raw)
        candidates, texts = schema_rows(question)
        kind = KINDS.index(str(question["type"]))
        values, token_supervised, span = ((0.0,) * len(texts), False, None)
        if key in gold:
            values, token_supervised, span = _targets(
                kind, candidates, object_map(gold[key]), source, text
            )
        rows = tuple(encode(row).ids for row in texts)
        if any(not row or row[0] != 1 for row in rows):
            raise ValueError(
                "Schema BOS/readout token must be ID 1 at position 0"
            )
        fields.append(
            Field(
                key,
                kind,
                candidates,
                rows,
                values,
                key in gold,
                token_supervised,
                span,
                tuple(cast(list[str], question["criteria"]))
                if kind == 3
                else (),
            )
        )
    return Record(record_id, text, source, tuple(fields))
