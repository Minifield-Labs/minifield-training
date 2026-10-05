"""Role-marked tool-call records on MagicBox's pointer layout.

Records compile exactly like MagicBox's, then change how the sequence is
marked: one ``<|startoftext|>`` leads the request, as in pretraining, and
each run starts with a marker naming its role (a question of each type, an
option of each kind, or the source) instead of another BOS. The question's
"Type:" line is dropped because its marker says the type. Markers live in
reserved tokenizer slots, are placed by ID and never come from text.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
import dataclasses
import json
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-untyped]

from examples.magicbox import data as magicbox
from examples.magicbox import source
from examples.toolcalls import vocabulary
from minifield_training.core import json_io
from minifield_training.datasets import fields
from minifield_training.datasets import pointer

TEMPLATE = "toolcall-pointer/2"
BOS = 1
MARKERS = (
    "extract_question",
    "choice_question",
    "noul_question",
    "score_question",
    "choice_option",
    "score_option",
    "noul_true",
    "noul_false",
    "extract_absent",
    "source",
)
# The tokenizer's reserved slots 7 to 16 hold IDs 17 to 26.
MARKER_IDS = {name: 17 + index for index, name in enumerate(MARKERS)}
# Readable names for the reserved slots in exported tokenizers.
TOKEN_NAMES = {
    f"<|reserved_{7 + index}|>": f"<|{name}|>"
    for index, name in enumerate(MARKERS)
}


def option_marker(
    kind: str, label: str, markers: Mapping[str, int] = MARKER_IDS
) -> int:
    """The marker that opens one option of a question type."""
    if kind == "extract":
        return markers["extract_absent"]
    if kind == "noul":
        return markers["noul_true" if label == "true" else "noul_false"]
    return markers["choice_option" if kind == "choice" else "score_option"]


def device_markers(tokenizer: Tokenizer) -> dict[str, int]:
    """Marker IDs in an exported tokenizer, found by their readable names.

    Trimming renumbers tokens, so a bundle's markers aren't at 17 to 26.
    """
    if tokenizer.token_to_id("<|startoftext|>") != BOS:
        raise ValueError("The tokenizer must keep BOS at ID 1")
    found = {name: tokenizer.token_to_id(f"<|{name}|>") for name in MARKERS}
    missing = [name for name, index in found.items() if index is None]
    if missing:
        raise ValueError(f"The tokenizer has no markers for {missing}")
    return found


def marked(
    record: pointer.Record,
    request: dict[str, object],
    encode: Callable[[str], fields.Encoding],
    markers: Mapping[str, int] = MARKER_IDS,
) -> pointer.Record:
    """Swap MagicBox's BOS markers for role markers behind one leading BOS.

    ``markers`` defaults to the training IDs; pass ``device_markers`` of a
    bundle's tokenizer to compile for that bundle.
    """
    questions = json_io.object_map(request["questions"])
    compiled = []
    for question in record.questions:
        kind = magicbox.KINDS[question.kind]
        query, _, _ = magicbox.pointer_texts(
            json_io.object_map(questions[question.key])
        )
        # The marker carries the type, so the "Type: ..." line goes.
        query = query.split("\n", 1)[1]
        compiled.append(
            dataclasses.replace(
                question,
                query=(markers[kind + "_question"], *encode(query).ids[1:]),
                options=tuple(
                    pointer.Option(
                        option.label,
                        (
                            option_marker(kind, option.label, markers),
                            *option.ids[1:],
                        ),
                    )
                    for option in question.options
                ),
            )
        )
    text = dataclasses.replace(
        record.source, ids=(markers["source"], *record.source.ids[1:])
    )
    return dataclasses.replace(
        record, source=text, questions=tuple(compiled), prefix=(BOS,)
    )


class Corpus(source.Corpus):
    """MagicBox's verified corpus, read through the model's vocabulary.

    Every record is first checked against the dataset's saved tokens, then
    compiled with the vocabulary and marked with roles. Sizes are measured
    with the vocabulary and MagicBox's layout, which is never shorter: the
    dropped "Type:" line outweighs the one leading BOS, so planned rows fit.
    """

    def __init__(
        self, directory: Path, cache: Path, *, allow_sample: bool = False
    ):
        super().__init__(directory, cache, allow_sample=allow_sample)
        self.vocabulary = vocabulary.Encoder(directory / "tokenizer")

    def compile_pointer(
        self, raw: object, split: str, *, score_width: float = 0.0
    ) -> pointer.Record:
        """Recheck the saved tokens, then compile what the model reads."""
        super().compile_pointer(raw, split, score_width=score_width)
        record_id, request, targets, _ = self._admitted(raw, split)
        record = source.with_accepted(
            magicbox.compile_pointer_record(
                record_id,
                request,
                targets,
                self.vocabulary.encode,
                score_width=score_width,
            ),
            json_io.object_map(raw),
        )
        return marked(record, request, self.vocabulary.encode)

    def _token_counts(
        self,
        splits: Sequence[str],
        texts: Callable[[object, object], tuple[str, ...]],
    ) -> Iterator[tuple[int, int, list[int]]]:
        """Source tokens, labeled questions and text tokens, by vocabulary."""
        for split in splits:
            columns = self.split(split).select_columns(
                ["request_json", "targets_json"]
            )
            for chunk in columns.iter(batch_size=1024):
                requests = [
                    json_io.object_map(json.loads(value))
                    for value in chunk["request_json"]
                ]
                labels = [json.loads(value) for value in chunk["targets_json"]]
                rows = [
                    texts(request, targets)
                    for request, targets in zip(requests, labels, strict=True)
                ]
                states = iter(
                    self.vocabulary.lengths(
                        [str(request["state"]) for request in requests]
                    )
                )
                lengths = iter(
                    self.vocabulary.lengths(
                        [text for items in rows for text in items]
                    )
                )
                for items, targets in zip(rows, labels, strict=True):
                    yield (
                        next(states),
                        len(targets),
                        [next(lengths) for _ in items],
                    )


def stage_directory(root: Path, stage: int) -> Path:
    """Where one curriculum stage sits inside the published dataset."""
    return root / f"stage{stage}"
