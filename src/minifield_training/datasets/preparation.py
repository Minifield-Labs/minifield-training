"""Group-safe splits and selectable assistant-turn preparation."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from hashlib import sha256

from minifield_training.core.json_io import canonical
from minifield_training.datasets.conversations import Conversation
from minifield_training.datasets.conversations import Message


@dataclass(frozen=True)
class Example:
    """Complete visible prefix and selected assistant message indices."""

    id: str
    source_group: str
    split: str
    messages: tuple[Message, ...]
    tools: tuple[str, ...]
    targets: tuple[int, ...]


def _visible(record: Conversation) -> str:
    """Canonical identity of content that the model may see."""
    return canonical(
        {
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "tool_call_id": message.tool_call_id,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                        for call in message.tool_calls
                    ],
                }
                for message in record.messages
            ],
            "tools": record.tools,
        }
    )


def assign_split(
    source_group: str,
    *,
    seed: str,
    validation_fraction: float = 0.1,
) -> str:
    """Assign a stable group holdout independent of input ordering."""
    if not source_group or not seed:
        raise ValueError("source_group and seed must be nonempty")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    value = (
        int.from_bytes(
            sha256(canonical([seed, source_group]).encode()).digest()[:8], "big"
        )
        / 2**64
    )
    return "validation" if value < validation_fraction else "train"


def prepare(
    records: Iterable[Conversation],
    *,
    mode: str,
    seed: str,
    validation_fraction: float = 0.1,
) -> Iterator[Example]:
    """Yield selected assistant turns after group-safe split admission.

    ``all`` selects every assistant turn in a full conversation. ``turn``
    yields one example per assistant turn, ending exactly at that turn.
    Duplicate visible conversations across source groups fail closed.
    """
    if mode not in {"all", "turn"}:
        raise ValueError("mode must be all or turn")
    seen: dict[str, str] = {}
    ids: set[str] = set()
    for record in records:
        if record.id in ids:
            raise ValueError("duplicate record id")
        ids.add(record.id)
        visible = sha256(_visible(record).encode()).hexdigest()
        owner = seen.setdefault(visible, record.source_group)
        if owner != record.source_group:
            raise ValueError("duplicate visible content across source groups")
        split = assign_split(
            record.source_group,
            seed=seed,
            validation_fraction=validation_fraction,
        )
        targets = tuple(
            index
            for index, item in enumerate(record.messages)
            if item.role == "assistant"
        )
        if mode == "all":
            yield Example(
                record.id,
                record.source_group,
                split,
                record.messages,
                record.tools,
                targets,
            )
        else:
            for index in targets:
                yield Example(
                    f"{record.id}:turn:{index}",
                    record.source_group,
                    split,
                    record.messages[: index + 1],
                    record.tools,
                    (index,),
                )
