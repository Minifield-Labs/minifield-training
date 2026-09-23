"""Host-side admission contracts for score plans and two-stage batches."""

import numpy as np
import numpy.typing as npt
import pytest

from minifield_training.objectives import validation

_VOCAB = 16


def _valid_inputs() -> tuple[
    npt.NDArray[np.generic],
    npt.NDArray[np.generic],
    npt.NDArray[np.generic],
    npt.NDArray[np.generic],
]:
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=np.int32)
    positions = np.array([[0, 1], [1, 0]], dtype=np.int32)
    targets = np.array([3, 6], dtype=np.int32)
    return ids, mask, positions, targets


def test_accepts_valid_score_plan() -> None:
    """Pass a well-formed position and target plan."""
    ids, mask, positions, targets = _valid_inputs()
    validation.validate_target_positions(
        ids, mask, positions, targets, vocab_size=_VOCAB
    )
    validation.validate_teacher_forced_targets(
        ids, mask, positions, targets, vocab_size=_VOCAB
    )


def test_empty_positions_still_validate_batch() -> None:
    """Validate batch structure with no queried positions."""
    ids, mask, _, _ = _valid_inputs()
    validation.validate_target_positions(
        ids,
        mask,
        np.empty((0, 2), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        vocab_size=_VOCAB,
    )


@pytest.mark.parametrize(
    ("ids", "mask", "positions", "targets", "match"),
    [
        (
            np.array([1, 2, 3]),
            np.array([[1, 1, 1]]),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            "rank-two",
        ),
        (
            np.array([[1.0, 2.0, 3.0, 4.0]]),
            np.ones((1, 4), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            "integer",
        ),
        (
            np.array([[1, 2, 3, 16]]),
            np.ones((1, 4), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            "vocabulary",
        ),
        (
            np.array([[1, 2, 3, -1]]),
            np.ones((1, 4), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            "vocabulary",
        ),
        (
            np.array([[1, 2, 3, 4]]),
            np.array([[1, 1, 0.5, 0]]),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            "binary",
        ),
        (
            np.array([[1, 2, 3, 4]]),
            np.array([[1, 1, np.nan, 0]]),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            "binary",
        ),
    ],
)
def test_rejects_malformed_batch(
    ids: npt.NDArray[np.generic],
    mask: npt.NDArray[np.generic],
    positions: npt.NDArray[np.generic],
    targets: npt.NDArray[np.generic],
    match: str,
) -> None:
    """Reject malformed ID and mask inputs."""
    with pytest.raises(ValueError, match=match):
        validation.validate_target_positions(
            ids, mask, positions, targets, vocab_size=_VOCAB
        )


@pytest.mark.parametrize(
    ("positions", "targets", "match"),
    [
        (np.array([0, 1]), np.array([2]), "shape \\[N, 2\\]"),
        (np.zeros((1, 3), dtype=np.int64), np.array([2]), "shape \\[N, 2\\]"),
        (
            np.array([[0, 1], [1, 0]]),
            np.array([2]),
            "one entry for every position",
        ),
        (
            np.array([[0.0, 1.0]]),
            np.array([2]),
            "integer arrays",
        ),
        (
            np.array([[0, 1]]),
            np.array([2.0]),
            "integer arrays",
        ),
        (
            np.array([[0, -1]]),
            np.array([2]),
            "outside the causal input range",
        ),
        (
            np.array([[2, 0]]),
            np.array([2]),
            "outside the causal input range",
        ),
        (
            np.array([[0, 3]]),
            np.array([2]),
            "outside the causal input range",
        ),
        (
            np.array([[0, 1]]),
            np.array([-1]),
            "vocabulary",
        ),
        (
            np.array([[0, 1]]),
            np.array([16]),
            "vocabulary",
        ),
    ],
)
def test_rejects_malformed_positions(
    positions: npt.NDArray[np.generic],
    targets: npt.NDArray[np.generic],
    match: str,
) -> None:
    """Reject malformed position and target plans."""
    ids, mask, _, _ = _valid_inputs()
    with pytest.raises(ValueError, match=match):
        validation.validate_target_positions(
            ids, mask, positions, targets, vocab_size=_VOCAB
        )


def test_rejects_positions_over_padding() -> None:
    """Reject positions selecting masked columns or successors."""
    ids, mask, _, _ = _valid_inputs()
    with pytest.raises(ValueError, match="padding"):
        validation.validate_target_positions(
            ids,
            mask,
            np.array([[1, 1]]),
            np.array([7]),
            vocab_size=_VOCAB,
        )
    with pytest.raises(ValueError, match="padding"):
        validation.validate_target_positions(
            ids,
            mask,
            np.array([[1, 2]]),
            np.array([7]),
            vocab_size=_VOCAB,
        )


def test_teacher_forced_requires_following_token() -> None:
    """Reject targets that differ from the following input token."""
    ids, mask, positions, _ = _valid_inputs()
    with pytest.raises(ValueError, match="following input token"):
        validation.validate_teacher_forced_targets(
            ids, mask, positions, np.array([4, 6]), vocab_size=_VOCAB
        )


def test_two_stage_requires_suffix_padding() -> None:
    """Reject masks with interior gaps or left padding."""
    ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
    mask = np.array([[1, 1, 0, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="suffix-only right padding"):
        validation.validate_two_stage_batch(ids, mask, vocab_size=_VOCAB)


def test_two_stage_requires_even_length() -> None:
    """Reject odd physical sequence lengths."""
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    mask = np.array([[1, 1, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="even physical sequence length"):
        validation.validate_two_stage_batch(ids, mask, vocab_size=_VOCAB)


def test_two_stage_accepts_suffix_padded_even_batch() -> None:
    """Pass a suffix-padded even-length batch."""
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    mask = np.array([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=np.int32)
    validation.validate_two_stage_batch(ids, mask, vocab_size=_VOCAB)


def test_two_stage_score_inputs_composes_both_checks() -> None:
    """Run both score-plan and batch admission checks."""
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    mask = np.array([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=np.int32)
    positions = np.array([[0, 0], [1, 1]], dtype=np.int32)
    targets = np.array([2, 7], dtype=np.int32)
    validation.validate_two_stage_score_inputs(
        ids, mask, positions, targets, vocab_size=_VOCAB
    )
    with pytest.raises(ValueError, match="following input token"):
        validation.validate_two_stage_score_inputs(
            ids,
            mask,
            positions,
            np.array([9, 7]),
            vocab_size=_VOCAB,
        )
