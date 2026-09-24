"""Offline chat-template and exact supervision checks."""

import json
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer  # type: ignore[import-untyped]
import tokenizers.decoders as decoders  # type: ignore[import-untyped]
from tokenizers.models import BPE  # type: ignore[import-untyped]
from tokenizers.models import WordLevel
import tokenizers.pre_tokenizers as pre_tokenizers  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast

from minifield_training.batching import contracts
from minifield_training.batching import dense
from minifield_training.batching import sft as sft_batching
from minifield_training.core.json_io import canonical
from minifield_training.core.json_io import digest_file
from minifield_training.datasets.conversations import parse_conversation
from minifield_training.datasets.conversations import read_jsonl
from minifield_training.datasets.preparation import Example
from minifield_training.datasets.preparation import prepare
from minifield_training.datasets.prepared import PreparationSettings
from minifield_training.datasets.prepared import iter_prepared
from minifield_training.datasets.prepared import save_prepared
from minifield_training.datasets.tokenization import ChatTokenizer
from minifield_training.datasets.tokenization import tokenize_example
from minifield_training.models.lfm2_5 import model
from minifield_training.objectives import loss
from minifield_training.optimizers import adamw
from minifield_training.strategies import sft


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


def byte_tokenizer(template: str) -> ChatTokenizer:
    """Use a complete local byte alphabet for tool-content checks."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    base = Tokenizer(
        BPE(
            vocab={char: index for index, char in enumerate(alphabet)},
            merges=[],
        )
    )
    base.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    base.decoder = decoders.ByteLevel()
    base.add_special_tokens(["<s>"])
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=base,
        bos_token="<s>",
        chat_template=template,
    )
    return cast(ChatTokenizer, tokenizer)


_TOOL_TEMPLATE = (
    "{{ bos_token }}"
    "{% if tools %}<TOOLS>{{ tools | tojson }}</TOOLS>{% endif %}"
    "{% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} "
    "{% if m.get('tool_calls') %}<CALLS>"
    "{{ m['tool_calls'] | tojson }}</CALLS>{% endif %}"
    "{% if m.get('tool_call_id') %}<REPLY>"
    "{{ m['tool_call_id'] }}</REPLY>{% endif %}"
    "<END>{% endfor %}"
)


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


def _tool_examples(argument: str | float = "hello") -> list[Example]:
    """Create one tool call, matched reply, and assistant continuation."""
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
                            "arguments": {"q": argument},
                        }
                    ],
                },
                {"role": "tool", "content": "world", "tool_call_id": "call-1"},
                {"role": "assistant", "content": "reply"},
            ],
        }
    )
    return list(prepare([record], mode="turn", seed="s"))


def test_tool_only_target_and_reply_are_preserved() -> None:
    """A tool-only turn gets a scored span with tool data."""
    examples = _tool_examples()
    first = tokenize_example(
        examples[0],
        byte_tokenizer(_TOOL_TEMPLATE),
        tokenizer_id="local-v1",
        template_id="test-jinja-v1",
        max_tokens=256,
        audited_tool_template=True,
    )
    tokenizer = byte_tokenizer(_TOOL_TEMPLATE)
    second = tokenize_example(
        examples[1],
        tokenizer,
        tokenizer_id="local-v1",
        template_id="test-jinja-v1",
        max_tokens=256,
        audited_tool_template=True,
    )
    assert first is not None and second is not None
    assert first.loss_mask[0] == 0
    assert sum(first.loss_mask) > 0
    assert second.input_ids != first.input_ids
    assert sum(second.loss_mask) > 0
    rendered = cast(PreTrainedTokenizerFast, tokenizer).decode(
        second.input_ids, skip_special_tokens=False
    )
    assert rendered == (
        '<s><TOOLS>[{"name": "find"}]</TOOLS><user> hello <END>'
        '<assistant>  <CALLS>[{"id": "call-1", "type": "function", '
        '"function": {"name": "find", "arguments": {"q": "hello"}}}]'
        "</CALLS><END><tool> world <REPLY>call-1</REPLY><END>"
        "<assistant> reply <END>"
    )


@pytest.mark.parametrize(
    ("template", "index"),
    [
        (_TOOL_TEMPLATE.replace("{{ tools | tojson }}", "TOOLS"), 0),
        (
            _TOOL_TEMPLATE.replace(
                "{{ m['tool_calls'] | tojson }}",
                "{{ m['tool_calls'][0]['id'] }} "
                "{{ m['tool_calls'][0]['function']['name'] }}",
            ),
            0,
        ),
        (_TOOL_TEMPLATE.replace("{{ m['tool_call_id'] }}", "REPLY"), 1),
    ],
)
def test_tool_template_cannot_drop_definitions_arguments_or_reply_id(
    template: str, index: int
) -> None:
    """Each tool component must change rendered text and token IDs."""
    with pytest.raises(ValueError, match="ignored a tool component"):
        tokenize_example(
            _tool_examples()[index],
            byte_tokenizer(template),
            tokenizer_id="byte-v1",
            template_id="bad-template",
            max_tokens=256,
            audited_tool_template=True,
        )


def test_large_finite_numeric_tool_argument_is_admitted() -> None:
    """A faithful template accepts a float whose value plus 1 rounds away."""
    tokenizer = byte_tokenizer(_TOOL_TEMPLATE)
    tokens = tokenize_example(
        _tool_examples(1e20)[0],
        tokenizer,
        tokenizer_id="byte-v1",
        template_id="tool-template-v1",
        max_tokens=256,
        audited_tool_template=True,
    )
    assert tokens is not None
    rendered = cast(PreTrainedTokenizerFast, tokenizer).decode(
        tokens.input_ids, skip_special_tokens=False
    )
    assert '"arguments": {"q": 1e+20}' in rendered


def test_tool_template_requires_explicit_audit() -> None:
    """Tool-bearing examples need an audited template contract."""
    with pytest.raises(ValueError, match="explicit audit"):
        tokenize_example(
            _tool_examples()[0],
            byte_tokenizer(_TOOL_TEMPLATE),
            tokenizer_id="byte-v1",
            template_id="template-v1",
            max_tokens=256,
        )


class _ContextDependentTokenizer:
    """A synthetic template whose earlier IDs change with later turns."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, object]] | None = None,
    ) -> str | list[int]:
        """Encode the message count into the leading control tokens."""
        del add_generation_prompt, tools
        if not tokenize:
            return str(len(conversation))
        return [0, len(conversation)] + [1] * len(conversation)


def test_context_dependent_token_prefix_rejected() -> None:
    """A separately encoded prefix can't silently set a loss boundary."""
    examples = _examples("turn")
    assert isinstance(examples, list)
    with pytest.raises(ValueError, match="context-dependent"):
        tokenize_example(
            examples[-1],
            _ContextDependentTokenizer(),
            tokenizer_id="bad",
            template_id="bad",
            max_tokens=32,
        )


def _tokenizer_snapshot(tokenizer: PreTrainedTokenizerFast) -> str:
    """Capture backend bytes and live special-token assignments."""
    return canonical(
        {
            "backend": json.loads(tokenizer.backend_tokenizer.to_str()),
            "special_tokens_map": tokenizer.special_tokens_map,
            "split_special_tokens": tokenizer.split_special_tokens,
        }
    )


def test_full_jsonl_to_replayed_sft_update(tmp_path: Path) -> None:
    """A bounded offline record lowers tiny-model loss after artifact replay."""
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"id":"r","source_group":"g","messages":['
        '{"role":"user","content":"hello"},'
        '{"role":"assistant","content":"world"}]}\n',
        encoding="utf-8",
    )
    tokenizer = cast(PreTrainedTokenizerFast, local_tokenizer())
    tokenizer_asset = tmp_path / "tokenizer.json"
    template_asset = tmp_path / "template.jinja"
    tokenizer_asset.write_text(_tokenizer_snapshot(tokenizer), encoding="utf-8")
    assert isinstance(tokenizer.chat_template, str)
    template_asset.write_text(tokenizer.chat_template, encoding="utf-8")
    settings = PreparationSettings("all", "seed", 0.0, 16, "error")
    source_examples = list(
        prepare(
            read_jsonl(source),
            mode=settings.mode,
            seed=settings.seed,
            validation_fraction=settings.validation_fraction,
        )
    )
    examples = []
    for example in source_examples:
        encoded = tokenize_example(
            example,
            cast(ChatTokenizer, tokenizer),
            tokenizer_id=digest_file(tokenizer_asset),
            template_id=digest_file(template_asset),
            max_tokens=settings.max_tokens,
        )
        assert encoded is not None
        examples.append(encoded)
    assert len(examples) == 1
    artifact = tmp_path / "prepared"
    save_prepared(
        artifact,
        examples,
        source=source,
        tokenizer_asset=tokenizer_asset,
        template_asset=template_asset,
        settings=settings,
    )
    replayed = list(
        iter_prepared(
            artifact,
            source=source,
            tokenizer_asset=tokenizer_asset,
            template_asset=template_asset,
            settings=settings,
        )
    )
    assert replayed == examples
    batch = next(
        dense.DenseBatchStrategy(
            contracts.BatchShape(2, 1, 16, 8, 12), sft_batching.TokenTargets()
        ).iter_updates(replayed, seed=0)
    )
    assert int(batch.microbatches["loss_mask"].sum()) == 3
    assert np.asarray(batch.active).tolist() == [True, False]
    cfg = model.Config(
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=12,
        layer_types=("conv",),
        conv_kernel=3,
    )
    rng = np.random.default_rng(7)
    params = {
        name: jnp.asarray(rng.normal(0, 0.1, shape), dtype=jnp.float32)
        for name, shape in model.expected_shapes(cfg).items()
    }
    inventory = model.parameter_inventory(cfg)
    current = adamw.initialize_state(params, inventory)

    def measured_loss() -> float:
        """Measure the real public model's selected next-token NLL."""
        tensors = {key: value[0] for key, value in batch.microbatches.items()}
        logits = model.forward(
            current["params"],
            tensors["input_ids"],
            tensors["attention_mask"],
            cfg,
            dtype=jnp.float32,
        )
        total, count = loss.causal_loss_terms(
            logits,
            tensors["input_ids"],
            tensors["loss_mask"],
            tensors["attention_mask"],
        )
        return float(total / count)

    before = measured_loss()
    update = sft.make_lfm2_5_step(
        cfg,
        inventory,
        adamw.AdamWConfig(0.03),
        dtype=jnp.float32,
    )
    for _ in range(6):
        result = update(current, batch.microbatches, jnp.asarray(batch.active))
        assert bool(result.committed)
        current = result.state
    assert measured_loss() < before - 0.05

    old_backend = tokenizer.backend_tokenizer.to_str()
    old_template = tokenizer.chat_template
    old_digest = digest_file(tokenizer_asset)
    tokenizer.bos_token = "[UNK]"
    assert tokenizer.backend_tokenizer.to_str() == old_backend
    assert tokenizer.chat_template == old_template
    changed = tokenize_example(
        source_examples[0],
        cast(ChatTokenizer, tokenizer),
        tokenizer_id="changed",
        template_id=digest_file(template_asset),
        max_tokens=settings.max_tokens,
    )
    assert changed is not None
    assert changed.input_ids != examples[0].input_ids
    tokenizer_asset.write_text(_tokenizer_snapshot(tokenizer), encoding="utf-8")
    assert digest_file(tokenizer_asset) != old_digest
    with pytest.raises(ValueError, match="stale"):
        list(
            iter_prepared(
                artifact,
                source=source,
                tokenizer_asset=tokenizer_asset,
                template_asset=template_asset,
                settings=settings,
            )
        )
