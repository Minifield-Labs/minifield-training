"""MagicBox v1 wire admission and pinned product schema wording."""

from collections.abc import Callable
from typing import cast

from minifield_training.core import json_io
from minifield_training.datasets import fields

FORMAT = "minifield.magicbox/1.0"
TEMPLATE = "magicbox-rows/1"
OFFSET_POLICY = "trim-text-preserve-whitespace/2"
KINDS = ("extract", "choice", "noul", "score")


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
        criteria = json_io.object_map(question.get("criteria"))
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
        binary = json_io.object_map(question["criteria"])
        if set(binary) != {"true", "false"} or any(
            not isinstance(value, str) for value in binary.values()
        ):
            raise ValueError("Invalid boolean criteria")
        true, false = binary["true"], binary["false"]
        prefix += f"\nTrue: {true}\nFalse: {false}"
    return ("",), (prefix,)


def _targets(
    kind: int,
    candidates: tuple[str, ...],
    target: dict[str, object],
    source: fields.Encoding,
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
            fields.aligned_span(source, target["span"], text)
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
            fields.probabilities(
                (target.get("probability"),), distribution=False
            ),
            False,
            None,
        )
    label_key = "choice" if kind == 1 else "level"
    if set(target) not in ({label_key}, {"probabilities"}):
        raise ValueError("Ambiguous categorical target")
    if "probabilities" in target:
        probabilities = target["probabilities"]
        if kind == 1:
            mapping = json_io.object_map(probabilities)
            if set(mapping) != set(candidates):
                raise ValueError("Choice distribution keys differ")
            values = tuple(mapping[key] for key in candidates)
        elif isinstance(probabilities, list) and len(probabilities) == len(
            candidates
        ):
            values = tuple(probabilities)
        else:
            raise ValueError("Score distribution length differs")
        return fields.probabilities(values, distribution=True), False, None
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
    encode: Callable[[str], fields.Encoding],
) -> fields.Record:
    """Compile public input separately from optional private targets."""
    public, gold = json_io.object_map(request), json_io.object_map(targets)
    text = public.get("state")
    if not isinstance(text, str):
        raise ValueError("State must be a string")
    questions = json_io.object_map(public.get("questions"))
    if not questions or not set(gold) <= set(questions):
        raise ValueError("Invalid question or supervision keys")
    source = encode(text)
    if not source.ids or source.ids[0] != 1:
        raise ValueError("Source must include the native BOS token")
    compiled_fields = []
    for key, raw in questions.items():
        _text(key)
        question = json_io.object_map(raw)
        candidates, texts = schema_rows(question)
        kind = KINDS.index(str(question["type"]))
        values, token_supervised, span = ((0.0,) * len(texts), False, None)
        if key in gold:
            values, token_supervised, span = _targets(
                kind, candidates, json_io.object_map(gold[key]), source, text
            )
        rows = tuple(encode(row).ids for row in texts)
        if any(not row or row[0] != 1 for row in rows):
            raise ValueError(
                "Schema BOS/readout token must be ID 1 at position 0"
            )
        compiled_fields.append(
            fields.Field(
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
    return fields.Record(record_id, text, source, tuple(compiled_fields))


def format_results(
    record: fields.Record,
    predictions: dict[str, object],
    confidence: Callable[[dict[str, object]], float | None] | None = None,
) -> dict[str, object]:
    """Format the public typed response with optional confidence."""
    result: dict[str, object] = {}
    for field in record.fields:
        prediction = json_io.object_map(predictions[field.key])
        kind = KINDS[field.kind]
        item: dict[str, object] = {"type": kind, kind: prediction["value"]}
        if kind != "noul":
            item["confidence"] = confidence(prediction) if confidence else None
        if kind in ("choice", "score"):
            item["probabilities"] = prediction["probabilities"]
        if kind == "score":
            item["legend"] = {
                str(index): text for index, text in enumerate(field.legend)
            }
        result[field.key] = item
    return result
