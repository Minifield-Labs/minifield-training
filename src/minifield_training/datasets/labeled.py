"""Product-neutral hard-label sequence records."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LabeledSequence:
    """One complete tokenized observation and its decision label."""

    id: str
    group_id: str
    input_ids: tuple[int, ...]
    label: int
