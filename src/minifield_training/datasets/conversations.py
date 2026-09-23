"""Validate and stream neutral conversation records from JSONL."""

from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from minifield_training.core.json_io import canonical


@dataclass(frozen=True)
class ToolCall:
    """An assistant's tool request, with canonical JSON arguments."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Message:
    """One ordered message; tool replies refer to a prior call ID."""

    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Conversation:
    """Model-visible conversation and separate audit identity."""

    id: str
    source_group: str
    messages: tuple[Message, ...]
    tools: tuple[str, ...] = ()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys before schema validation."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    """Reject nonfinite JavaScript number extensions."""
    raise ValueError("nonfinite JSON number")


def _nonempty(value: object, label: str) -> str:
    """Require a nonempty string without echoing its contents."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _objects(value: object, label: str) -> list[dict[str, Any]]:
    """Require a list of JSON objects."""
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(f"{label} must be a list of objects")
    return value


def parse_conversation(
    value: object, source_group: str | None = None
) -> Conversation:
    """Admit one neutral envelope, preserving tool-call order and identity."""
    if not isinstance(value, dict):
        raise ValueError("conversation must be an object")
    if set(value) - {"id", "source_group", "messages", "tools"}:
        raise ValueError("unknown conversation field")
    record_id = _nonempty(value.get("id"), "id")
    group = _nonempty(source_group or value.get("source_group"), "source_group")
    if source_group is not None and value.get("source_group", group) != group:
        raise ValueError("source_group conflicts with reader setting")
    messages: list[Message] = []
    pending: set[str] = set()
    used: set[str] = set()
    for index, raw in enumerate(_objects(value.get("messages"), "messages")):
        if set(raw) - {"role", "content", "tool_calls", "tool_call_id"}:
            raise ValueError(f"message {index}: unknown field")
        role = raw.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {index}: invalid role")
        content = raw.get("content", "")
        if not isinstance(content, str):
            raise ValueError(f"message {index}: content must be text")
        calls: list[ToolCall] = []
        for call in _objects(raw.get("tool_calls", []), "tool_calls"):
            if set(call) != {"id", "name", "arguments"}:
                raise ValueError(f"message {index}: invalid tool call")
            call_id = _nonempty(call["id"], "tool call id")
            name = _nonempty(call["name"], "tool name")
            if role != "assistant" or call_id in used:
                raise ValueError(
                    f"message {index}: invalid or repeated tool call"
                )
            arguments = call["arguments"]
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"message {index}: arguments must be an object"
                )
            calls.append(ToolCall(call_id, name, canonical(arguments)))
            pending.add(call_id)
            used.add(call_id)
        reply_id = raw.get("tool_call_id")
        if role == "tool":
            if not isinstance(reply_id, str) or reply_id not in pending:
                raise ValueError(f"message {index}: unmatched tool reply")
            pending.remove(reply_id)
        elif reply_id is not None:
            raise ValueError(
                f"message {index}: tool_call_id requires tool role"
            )
        if pending and role not in {"assistant", "tool"}:
            raise ValueError(f"message {index}: unresolved tool call")
        if role == "assistant" and not content and not calls:
            raise ValueError(f"message {index}: empty assistant message")
        messages.append(Message(role, content, tuple(calls), reply_id))
    if pending:
        raise ValueError("unresolved tool call")
    if not messages or not any(item.role == "assistant" for item in messages):
        raise ValueError("conversation needs an assistant message")
    tools = tuple(
        canonical(item) for item in _objects(value.get("tools", []), "tools")
    )
    return Conversation(record_id, group, tuple(messages), tools)


def read_jsonl(
    path: Path, source_group: str | None = None
) -> Iterator[Conversation]:
    """Stream JSONL with bounded line memory and redacted line errors."""
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_unique_pairs,
                    parse_constant=_reject_constant,
                )
                yield parse_conversation(value, source_group)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path.name}:{number}: {error}") from error
