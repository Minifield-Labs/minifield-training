"""Independent JSONL admission fixtures."""

import json
from pathlib import Path

import pytest

from minifield_training.datasets.conversations import read_jsonl


def test_stream_preserves_tool_semantics(tmp_path: Path) -> None:
    """Tool calls and replies retain IDs while audit IDs stay separate."""
    path = tmp_path / "sample.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "r1",
                "source_group": "source",
                "messages": [
                    {"role": "user", "content": "Search"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "name": "find",
                                "arguments": {"q": "a"},
                            }
                        ],
                    },
                    {"role": "tool", "content": "hit", "tool_call_id": "c1"},
                    {"role": "assistant", "content": "Done"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = list(read_jsonl(path))
    assert len(records) == 1
    assert records[0].source_group == "source"
    assert records[0].messages[1].tool_calls[0].arguments == '{"q":"a"}'
    assert records[0].messages[2].tool_call_id == "c1"


@pytest.mark.parametrize(
    "line",
    [
        '{"id":"x","id":"y"}',
        '{"id":"x","source_group":"g","messages":[],"x":NaN}',
        (
            '{"id":"x","source_group":"g","messages":['
            '{"role":"tool","content":"private","tool_call_id":"missing"}]}'
        ),
    ],
)
def test_bad_lines_are_rejected_without_payload(
    tmp_path: Path, line: str
) -> None:
    """Source locations are useful without echoing sensitive content."""
    path = tmp_path / "bad.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1") as error:
        list(read_jsonl(path))
    assert "private" not in str(error.value)
