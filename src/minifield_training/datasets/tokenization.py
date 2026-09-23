"""Verified token supervision using caller-supplied chat templates."""

from dataclasses import dataclass
import json
from typing import Protocol

from minifield_training.datasets.preparation import Example


class ChatTokenizer(Protocol):
    """Minimum behavior required of a pinned chat-template tokenizer."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, object]] | None = None,
    ) -> list[int]:
        """Render complete messages to integer token IDs."""


@dataclass(frozen=True)
class TokenizedExample:
    """Validated dense IDs and selected next-token supervision positions."""

    id: str
    source_group: str
    split: str
    input_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    tokenizer_id: str
    template_id: str


def _messages(example: Example) -> list[dict[str, object]]:
    """Translate neutral messages into template input dictionaries."""
    result: list[dict[str, object]] = []
    for message in example.messages:
        item: dict[str, object] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.loads(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        result.append(item)
    return result


def tokenize_example(
    example: Example,
    tokenizer: ChatTokenizer,
    *,
    tokenizer_id: str,
    template_id: str,
    max_tokens: int,
    overlength: str = "error",
) -> TokenizedExample | None:
    """Encode once and verify every selected span against complete prefixes.

    A template that changes earlier tokens when later messages are added is
    unsupported. The adapter selects complete assistant messages, including
    their role header and end marker, without searching for content text.
    """
    if not tokenizer_id or not template_id or max_tokens < 2:
        raise ValueError(
            "tokenizer/template identities and max_tokens required"
        )
    if overlength not in {"error", "drop"}:
        raise ValueError("overlength must be error or drop")
    messages = _messages(example)
    tools = [json.loads(item) for item in example.tools] or None

    def encode(
        selected: list[dict[str, object]],
        supplied_tools: list[dict[str, object]] | None = tools,
    ) -> tuple[int, ...]:
        """Use the template's tokenization path without extra special tokens."""
        result = tokenizer.apply_chat_template(
            selected,
            tokenize=True,
            add_generation_prompt=False,
            tools=supplied_tools,
        )
        if not isinstance(result, list) or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in result
        ):
            raise ValueError("chat template must return nonnegative token IDs")
        return tuple(result)

    full = encode(messages)
    if tools is not None and encode(messages, None) == full:
        raise ValueError("template ignored supplied tool definitions")
    if any(
        message.tool_calls or message.tool_call_id
        for message in example.messages
    ):
        stripped = [
            {
                key: value
                for key, value in message.items()
                if key not in {"tool_calls", "tool_call_id"}
            }
            for message in messages
        ]
        if encode(stripped) == full:
            raise ValueError("template ignored tool calls or reply identities")
    if len(full) > max_tokens:
        if overlength == "drop":
            return None
        raise ValueError("tokenized example exceeds max_tokens")
    if len(full) < 2:
        raise ValueError("training example needs at least 2 tokens")
    mask = [0] * len(full)
    for target in example.targets:
        if (
            target >= len(messages)
            or example.messages[target].role != "assistant"
        ):
            raise ValueError("target must select an assistant message")
        start = len(encode(messages[:target])) if target else 0
        end = len(encode(messages[: target + 1]))
        if not 0 < start < end <= len(full):
            raise ValueError("assistant span cannot be scored")
        if full[:start] != encode(messages[:target]) or full[:end] != encode(
            messages[: target + 1]
        ):
            raise ValueError("template token prefixes are context-dependent")
        for position in range(start, end):
            if mask[position]:
                raise ValueError("overlapping assistant spans")
            mask[position] = 1
    if not any(mask[1:]):
        raise ValueError("example has no scorable assistant targets")
    return TokenizedExample(
        example.id,
        example.source_group,
        example.split,
        full,
        tuple(mask),
        tokenizer_id,
        template_id,
    )
