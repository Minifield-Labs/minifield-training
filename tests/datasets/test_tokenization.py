"""Offline chat-template and exact supervision checks."""

from typing import cast

import pytest
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
import tokenizers.pre_tokenizers as pre_tokenizers  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast

from minifield_training.datasets.conversations import parse_conversation
from minifield_training.datasets.preparation import prepare
from minifield_training.datasets.tokenization import ChatTokenizer
from minifield_training.datasets.tokenization import tokenize_example


def local_tokenizer() -> ChatTokenizer:
    """Construct a tiny local vocabulary with an actual Jinja chat template."""
    vocabulary = {
        "<s>": 0,
        "USER": 1,
        "ASSISTANT": 2,
        "END": 3,
        "hello": 4,
        "world": 5,
        "again": 6,
        "reply": 7,
        "[UNK]": 8,
        "TOOLS": 9,
        "CALLS": 10,
        "REPLY": 11,
    }
    base = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    base.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    template = (
        "{{ bos_token }} "
        "{% if tools %}TOOLS {{ tools | tojson }} END {% endif %}"
        "{% for m in messages %}"
        "{% if m['role'] == 'user' %}USER{% elif m['role'] == 'assistant' %}"
        "ASSISTANT{% elif m['role'] == 'tool' %}REPLY"
        "{% else %}SYSTEM{% endif %} "
        "{{ m['content'] }} "
        "{% if m.get('tool_calls') %}CALLS "
        "{{ m['tool_calls'] | tojson }} {% endif %}"
        "{% if m.get('tool_call_id') %}REPLY "
        "{{ m['tool_call_id'] }} {% endif %}"
        "END {% endfor %}"
    )
    # Transformers exposes this constructor without a complete typed signature.
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=base,
        bos_token="<s>",
        unk_token="[UNK]",
        chat_template=template,
    )
    return cast(ChatTokenizer, tokenizer)


def _examples(mode: str) -> object:
    """Build examples with a prior assistant and literal control word."""
    record = parse_conversation(
        {
            "id": "r1",
            "source_group": "g1",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
                {"role": "user", "content": "ASSISTANT again"},
                {"role": "assistant", "content": "reply"},
            ],
        }
    )
    return list(prepare([record], mode=mode, seed="s"))


def test_real_template_selected_turn_mask() -> None:
    """Prior assistant content and a literal role word stay unselected."""
    examples = _examples("turn")
    assert isinstance(examples, list)
    last = tokenize_example(
        examples[-1],
        local_tokenizer(),
        tokenizer_id="local-v1",
        template_id="test-jinja-v1",
        max_tokens=32,
    )
    assert last is not None
    assert last.input_ids == (0, 1, 4, 3, 2, 5, 3, 1, 2, 6, 3, 2, 7, 3)
    assert last.loss_mask == (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1)


def test_all_mode_and_overlength() -> None:
    """Every assistant turn is selected; measured length controls admission."""
    examples = _examples("all")
    assert isinstance(examples, list)
    example = examples[0]
    tokens = tokenize_example(
        example,
        local_tokenizer(),
        tokenizer_id="local-v1",
        template_id="test-jinja-v1",
        max_tokens=32,
    )
    assert tokens is not None
    assert tokens.loss_mask == (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1)
    assert (
        tokenize_example(
            example,
            local_tokenizer(),
            tokenizer_id="local-v1",
            template_id="test-jinja-v1",
            max_tokens=13,
            overlength="drop",
        )
        is None
    )
    with pytest.raises(ValueError, match="exceeds max_tokens"):
        tokenize_example(
            example,
            local_tokenizer(),
            tokenizer_id="local-v1",
            template_id="test-jinja-v1",
            max_tokens=13,
        )


def test_tool_only_target_and_reply_are_preserved() -> None:
    """A tool-only turn gets a scored span with tool data."""
    record = parse_conversation(
        {
            "id": "tool",
            "source_group": "g",
            "tools": [{"name": "find"}],
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "find",
                            "arguments": {"q": "hello"},
                        }
                    ],
                },
                {"role": "tool", "content": "world", "tool_call_id": "call-1"},
                {"role": "assistant", "content": "reply"},
            ],
        }
    )
    examples = list(prepare([record], mode="turn", seed="s"))
    first = tokenize_example(
        examples[0],
        local_tokenizer(),
        tokenizer_id="local-v1",
        template_id="test-jinja-v1",
        max_tokens=64,
    )
    second = tokenize_example(
        examples[1],
        local_tokenizer(),
        tokenizer_id="local-v1",
        template_id="test-jinja-v1",
        max_tokens=64,
    )
    assert first is not None and second is not None
    assert first.loss_mask[0] == 0
    assert sum(first.loss_mask) > 0
    assert second.input_ids != first.input_ids
    assert sum(second.loss_mask) > 0
