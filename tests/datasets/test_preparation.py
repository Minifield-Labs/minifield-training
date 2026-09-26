"""Split and conversation preparation behavior."""

import pytest

from minifield_training.datasets.conversations import Conversation
from minifield_training.datasets.conversations import parse_conversation
from minifield_training.datasets.preparation import assign_split
from minifield_training.datasets.preparation import prepare


def _record(record_id: str, group: str, answer: str) -> Conversation:
    """Build a small valid envelope with 2 assistant turns."""
    return parse_conversation(
        {
            "id": record_id,
            "source_group": group,
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "One"},
                {"role": "user", "content": "Second"},
                {"role": "assistant", "content": answer},
            ],
        }
    )


def test_turn_mode_retains_context_without_future_messages() -> None:
    """Target indices refer to complete assistant turns in each prefix."""
    record = _record("r1", "g1", "Two")
    examples = list(prepare([record], mode="turn", seed="x"))
    assert [item.targets for item in examples] == [(2,), (4,)]
    assert [len(item.messages) for item in examples] == [3, 5]
    assert examples[0].messages[-1].content == "One"
    assert all(item.source_group == "g1" for item in examples)
    assert len({item.split for item in examples}) == 1


def test_all_mode_and_order_independent_groups() -> None:
    """Group assignment survives input order changes and expansion."""
    first = _record("r1", "g1", "Two")
    second = _record("r2", "g2", "Different")
    forward = list(prepare([first, second], mode="all", seed="fixed"))
    reverse = list(prepare([second, first], mode="all", seed="fixed"))
    assert forward[0].targets == (2, 4)
    assert {item.id: item.split for item in forward} == {
        item.id: item.split for item in reverse
    }
    assert assign_split("g1", seed="fixed") == forward[0].split


def test_duplicate_visible_content_across_groups_rejected() -> None:
    """Identical prompts and targets can't leak across source groups."""
    first = _record("r1", "g1", "Two")
    second = _record("r2", "g2", "Two")
    with pytest.raises(ValueError, match="duplicate visible content"):
        list(prepare([first, second], mode="turn", seed="fixed"))


def test_shared_turn_prefix_across_splits_rejected() -> None:
    """A shared first turn can't train in one split and validate in another."""
    assert (
        assign_split("g1", seed="fixed", validation_fraction=0.5)
        == "validation"
    )
    assert assign_split("g2", seed="fixed", validation_fraction=0.5) == "train"
    first = _record("r1", "g1", "Two")
    second = _record("r2", "g2", "Different")
    with pytest.raises(ValueError, match="duplicate visible content"):
        list(
            prepare(
                [first, second],
                mode="turn",
                seed="fixed",
                validation_fraction=0.5,
            )
        )
    assert (
        len(
            list(
                prepare(
                    [first, second],
                    mode="all",
                    seed="fixed",
                    validation_fraction=0.5,
                )
            )
        )
        == 2
    )


def test_shared_turn_prefix_within_one_group_is_allowed() -> None:
    """A group can contain related conversations with identical early turns."""
    first = _record("r1", "g1", "Two")
    second = _record("r2", "g1", "Different")
    examples = list(prepare([first, second], mode="turn", seed="fixed"))
    assert len(examples) == 4
    assert examples[0].messages == examples[2].messages
    assert len({item.split for item in examples}) == 1
