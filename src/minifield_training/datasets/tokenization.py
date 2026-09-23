"""Verified token supervision using caller-supplied chat templates."""

from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Protocol, cast

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
    ) -> str | list[int]:
        """Render messages as text or integer token IDs."""


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


def _probes(value: object) -> Iterator[object]:
    """Change one JSON leaf at a time for tool-semantic admission."""
    if isinstance(value, dict):
        if not value:
            yield {"__audit_probe__": "value"}
        for key, child in value.items():
            for changed in _probes(child):
                variant = dict(value)
                variant[key] = changed
                yield variant
    elif isinstance(value, list):
        if not value:
            yield ["__audit_probe__"]
        for index, child in enumerate(value):
            for changed in _probes(child):
                list_variant = list(value)
                list_variant[index] = changed
                yield list_variant
    elif isinstance(value, str):
        yield value + "__audit_probe__"
    elif isinstance(value, bool):
        yield not value
    elif isinstance(value, int | float):
        yield value + 1
    else:
        yield "__audit_probe__"


type _Encoder = Callable[
    [list[dict[str, object]], list[dict[str, object]] | None], tuple[int, ...]
]
type _Renderer = Callable[
    [list[dict[str, object]], list[dict[str, object]] | None], str
]


def _audit_tools(
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
    *,
    encode: _Encoder,
    render: _Renderer,
    full_ids: tuple[int, ...],
    full_text: str,
) -> None:
    """Require each supported tool component to affect text and token IDs."""

    def check(
        candidate: list[dict[str, object]],
        definitions: list[dict[str, object]] | None,
    ) -> None:
        """Reject any component that disappears in text or tokenization."""
        if (
            render(candidate, definitions) == full_text
            or encode(candidate, definitions) == full_ids
        ):
            raise ValueError("template ignored a tool component")

    if tools is not None:
        for index, definition in enumerate(tools):
            for changed in _probes(definition):
                variant_tools = deepcopy(tools)
                variant_tools[index] = cast(dict[str, object], changed)
                check(messages, variant_tools)
    for message_index, message in enumerate(messages):
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call_index, call in enumerate(calls):
                for path in ("id", "name", "arguments"):
                    function = call["function"]
                    original = call["id"] if path == "id" else function[path]
                    for changed in _probes(original):
                        variant_messages = deepcopy(messages)
                        variant_calls = cast(
                            list[dict[str, object]],
                            variant_messages[message_index]["tool_calls"],
                        )
                        variant_call = variant_calls[call_index]
                        if path == "id":
                            variant_call["id"] = changed
                        else:
                            function_fields = cast(
                                dict[str, object], variant_call["function"]
                            )
                            function_fields[path] = changed
                        check(variant_messages, tools)
        reply_id = message.get("tool_call_id")
        if isinstance(reply_id, str):
            variant_messages = deepcopy(messages)
            variant_messages[message_index]["tool_call_id"] = (
                reply_id + "__audit_probe__"
            )
            check(variant_messages, tools)


def tokenize_example(
    example: Example,
    tokenizer: ChatTokenizer,
    *,
    tokenizer_id: str,
    template_id: str,
    max_tokens: int,
    overlength: str = "error",
    audited_tool_template: bool = False,
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
    has_tool_data = tools is not None or any(
        message.tool_calls or message.tool_call_id
        for message in example.messages
    )
    if has_tool_data:
        if not audited_tool_template:
            raise ValueError("tool template needs an explicit audit")

        def render(
            selected: list[dict[str, object]],
            supplied_tools: list[dict[str, object]] | None,
        ) -> str:
            """Inspect the actual template text for semantic differences."""
            result = tokenizer.apply_chat_template(
                selected,
                tokenize=False,
                add_generation_prompt=False,
                tools=supplied_tools,
            )
            if not isinstance(result, str):
                raise ValueError(
                    "chat template must render text for tool audit"
                )
            return result

        _audit_tools(
            messages,
            tools,
            encode=encode,
            render=render,
            full_ids=full,
            full_text=render(messages, tools),
        )
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
