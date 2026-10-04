"""Role-marked tool-call records on MagicBox's pointer layout.

Records compile exactly like MagicBox's, then change how the sequence is
marked: one ``<|startoftext|>`` leads the request, as in pretraining, and
each run starts with a marker naming its role (a question of each type, an
option of each kind, or the source) instead of another BOS. The question's
"Type:" line is dropped because its marker says the type. Markers live in
reserved tokenizer slots, are placed by ID and never come from text.
"""

from collections.abc import Callable
import dataclasses
import json
from pathlib import Path

from examples.magicbox import data as magicbox
from examples.magicbox import source
from minifield_training.core import json_io
from minifield_training.datasets import fields
from minifield_training.datasets import pointer

TEMPLATE = "toolcall-pointer/1"
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


def option_marker(kind: str, label: str) -> int:
    """The marker that opens one option of a question type."""
    if kind == "extract":
        return MARKER_IDS["extract_absent"]
    if kind == "noul":
        return MARKER_IDS["noul_true" if label == "true" else "noul_false"]
    return MARKER_IDS["choice_option" if kind == "choice" else "score_option"]


def marked(
    record: pointer.Record,
    request: dict[str, object],
    encode: Callable[[str], fields.Encoding],
) -> pointer.Record:
    """Swap MagicBox's BOS markers for role markers behind one leading BOS."""
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
                query=(MARKER_IDS[kind + "_question"], *encode(query).ids[1:]),
                options=tuple(
                    pointer.Option(
                        option.label,
                        (option_marker(kind, option.label), *option.ids[1:]),
                    )
                    for option in question.options
                ),
            )
        )
    text = dataclasses.replace(
        record.source, ids=(MARKER_IDS["source"], *record.source.ids[1:])
    )
    return dataclasses.replace(
        record, source=text, questions=tuple(compiled), prefix=(BOS,)
    )


class Corpus(source.Corpus):
    """MagicBox's verified corpus, compiled with role markers.

    Sizes are measured with MagicBox's layout, which is never shorter: the
    dropped "Type:" line outweighs the one leading BOS, so planned rows fit.
    """

    def compile_pointer(
        self, raw: object, split: str, *, score_width: float = 0.0
    ) -> pointer.Record:
        """Compile, recheck spans and accepted answers, then mark roles."""
        record = super().compile_pointer(raw, split, score_width=score_width)
        request = json_io.object_map(
            json.loads(str(json_io.object_map(raw)["request_json"]))
        )
        return marked(record, request, self.tokenizer.encode)


def stage_directory(root: Path, stage: int) -> Path:
    """Where one curriculum stage sits inside the published dataset."""
    return root / f"stage{stage}"
