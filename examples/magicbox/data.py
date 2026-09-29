"""MagicBox v1 wire admission and pinned product schema wording."""

from collections.abc import Callable
from typing import cast

from minifield_training.core import json_io
from minifield_training.datasets import fields
from minifield_training.datasets import pointer

FORMAT = "minifield.magicbox/1.0"
TEMPLATE = "magicbox-rows/1"
POINTER_TEMPLATE = "magicbox-pointer/1"
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


def labeled_schema_rows(request: object, targets: object) -> tuple[str, ...]:
    """Return the schema row text training encodes, in field order.

    Training keeps only labeled questions, so unlabeled ones add no rows.
    """
    questions = json_io.object_map(json_io.object_map(request).get("questions"))
    gold = json_io.object_map(targets)
    return tuple(
        text
        for key, raw in questions.items()
        if key in gold
        for text in schema_rows(json_io.object_map(raw))[1]
    )


def pointer_texts(
    question: dict[str, object],
) -> tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Serialize one question's query text, labeled options, and legend.

    Each text becomes its own BOS-led token run in the joint sequence; the
    BOS token is the query or option marker the pointer uses.
    """
    kind = str(question.get("type"))
    if kind not in KINDS:
        raise ValueError("Unsupported MagicBox question type")
    instructions = _text(question.get("instructions"))
    query = f"Type: {kind}\nQuestion: {instructions}"
    if kind == "extract":
        return query, (("absent", "Answer: not stated in the text"),), ()
    if kind == "noul":
        criteria = question.get("criteria")
        described = json_io.object_map(criteria) if criteria is not None else {}
        return (
            query,
            tuple(
                (
                    label,
                    f"Answer: {label}"
                    + (
                        f"\nDescription: {described[label]}"
                        if label in described
                        else ""
                    ),
                )
                for label in ("false", "true")
            ),
            (),
        )
    if kind == "choice":
        options = json_io.object_map(question.get("criteria"))
        if not options:
            raise ValueError("Choice requires candidates")
        return (
            query,
            tuple(
                (
                    _text(key),
                    f"Candidate: {key}\nDescription: {_text(value)}",
                )
                for key, value in options.items()
            ),
            (),
        )
    levels = question.get("criteria")
    if not isinstance(levels, list) or not levels:
        raise ValueError("Score requires ordered levels")
    return (
        query,
        tuple(
            (
                str(index),
                f"Level: {index} of {len(levels)} levels, indexed from 0"
                f"\nDescription: {_text(value)}",
            )
            for index, value in enumerate(levels)
        ),
        tuple(_text(value) for value in levels),
    )


def labeled_pointer_texts(request: object, targets: object) -> tuple[str, ...]:
    """Return every labeled question's query and option text, in order."""
    questions = json_io.object_map(json_io.object_map(request).get("questions"))
    gold = json_io.object_map(targets)
    result: list[str] = []
    for key, raw in questions.items():
        if key in gold:
            query, options, _ = pointer_texts(json_io.object_map(raw))
            result.extend((query, *(text for _, text in options)))
    return tuple(result)


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


def _pointer_targets(
    kind: int,
    labels: tuple[str, ...],
    label: dict[str, object],
    source: fields.Encoding,
    text: str,
    score_width: float,
) -> tuple[tuple[float, ...], tuple[int, int] | None] | None:
    """Turn one gold label into option targets, or None if it can't point.

    Extraction targets hold the "absent" probability. Binary targets split
    one probability over false and true. Hard score labels spread over nearby
    levels by ``score_width``; soft labels stay exactly as given.
    """
    values, _, span = _targets(
        kind, labels if kind in (1, 3) else ("",), label, source, text
    )
    if kind == 0:
        # A present answer without an aligned span has nothing to point at.
        return None if values[0] and span is None else ((1 - values[0],), span)
    if kind == 2:
        return (1 - values[0], values[0]), None
    if kind == 3 and "level" in label:
        level = label["level"]
        assert isinstance(level, int)
        return pointer.ordinal_targets(level, len(labels), score_width), None
    return values, None


def compile_pointer_record(
    record_id: str,
    request: object,
    targets: object,
    encode: Callable[[str], fields.Encoding],
    *,
    score_width: float = 0.0,
) -> pointer.Record:
    """Compile the joint pointer layout's questions, options, and targets."""
    public, gold = json_io.object_map(request), json_io.object_map(targets)
    text = public.get("state")
    if not isinstance(text, str):
        raise ValueError("State must be a string")
    questions = json_io.object_map(public.get("questions"))
    if not questions or not set(gold) <= set(questions):
        raise ValueError("Invalid question or supervision keys")
    source = encode(text)
    compiled = []
    for key, raw in questions.items():
        query, options, legend = pointer_texts(json_io.object_map(raw))
        kind = KINDS.index(str(json_io.object_map(raw)["type"]))
        runs = [encode(item).ids for item in (query, *(t for _, t in options))]
        if any(not ids or ids[0] != 1 for ids in (source.ids, *runs)):
            raise ValueError("Query and option markers must be BOS ID 1")
        labels = tuple(label for label, _ in options)
        answer = (
            _pointer_targets(
                kind,
                labels,
                json_io.object_map(gold[_text(key)]),
                source,
                text,
                score_width,
            )
            if key in gold
            else None
        )
        compiled.append(
            pointer.Question(
                key,
                kind,
                runs[0],
                tuple(
                    pointer.Option(label, ids)
                    for label, ids in zip(labels, runs[1:], strict=True)
                ),
                (0.0,) * len(labels) if answer is None else answer[0],
                answer is not None,
                None if answer is None else answer[1],
                legend,
            )
        )
    return pointer.Record(record_id, text, source, tuple(compiled))


def format_results(
    record: fields.Record | pointer.Record,
    predictions: dict[str, object],
    confidence: Callable[[dict[str, object]], float | None] | None = None,
) -> dict[str, object]:
    """Format the public typed response with optional confidence."""
    result: dict[str, object] = {}
    items: tuple[fields.Field | pointer.Question, ...] = (
        record.fields if isinstance(record, fields.Record) else record.questions
    )
    for field in items:
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
