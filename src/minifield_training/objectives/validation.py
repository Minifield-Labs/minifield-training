"""Host-side validation for score plans and supervision batch admission.

These checks intentionally run outside jax.jit and gradient transforms:
preparation must reject malformed rows before pure JAX math starts.
"""

import numpy as np


def validate_target_positions(
    ids: object,
    attention_mask: object,
    positions: object,
    target_ids: object,
    *,
    vocab_size: int,
) -> None:
    """Validate host-side next-token positions before a traced projection.

    Positions have shape [N, 2]. Every row is (batch_index,
    causal_input_index) and selects logits that predict the following token.
    """
    ids_array = np.asarray(ids)
    mask_array = np.asarray(attention_mask)
    position_array = np.asarray(positions)
    target_array = np.asarray(target_ids)
    if ids_array.ndim != 2 or mask_array.shape != ids_array.shape:
        raise ValueError(
            "Input IDs and attention mask must be rank-two matches"
        )
    if ids_array.dtype.kind not in "iu":
        raise ValueError("Input IDs must be an integer array")
    if np.any(ids_array < 0) or np.any(ids_array >= vocab_size):
        raise ValueError("Input token ID is outside the model vocabulary")
    if mask_array.dtype.kind not in "biuf":
        raise ValueError("Attention mask must contain finite binary values")
    if not np.all(np.isfinite(mask_array)) or not np.all(
        (mask_array == 0) | (mask_array == 1)
    ):
        raise ValueError("Attention mask must contain finite binary values")
    if position_array.ndim != 2 or position_array.shape[1:] != (2,):
        raise ValueError("Target positions must have shape [N, 2]")
    if (
        target_array.ndim != 1
        or target_array.shape[0] != position_array.shape[0]
    ):
        raise ValueError("Target IDs must have one entry for every position")
    if (
        position_array.dtype.kind not in "iu"
        or target_array.dtype.kind not in "iu"
    ):
        raise ValueError("Target positions and IDs must be integer arrays")
    if (
        np.any(position_array < 0)
        or np.any(position_array[:, 0] >= ids_array.shape[0])
        or np.any(position_array[:, 1] >= ids_array.shape[1] - 1)
    ):
        raise ValueError("Target position is outside the causal input range")
    if np.any(target_array < 0) or np.any(target_array >= vocab_size):
        raise ValueError("Target token ID is outside the model vocabulary")
    if np.any(
        mask_array[position_array[:, 0], position_array[:, 1]] == 0
    ) or np.any(
        mask_array[position_array[:, 0], position_array[:, 1] + 1] == 0
    ):
        raise ValueError("Target position points at padding")


def validate_teacher_forced_targets(
    ids: object,
    attention_mask: object,
    positions: object,
    target_ids: object,
    *,
    vocab_size: int,
) -> None:
    """Validate a complete teacher-forced continuation score plan.

    Every position must be below sequence length minus one because its
    following token has to be present in the branch input. In addition to
    generic queried target validation, this checks that each supplied target
    is that following input token. Apply it to every token of multi-token
    candidate continuations.
    """
    validate_target_positions(
        ids, attention_mask, positions, target_ids, vocab_size=vocab_size
    )
    ids_array = np.asarray(ids)
    position_array = np.asarray(positions)
    target_array = np.asarray(target_ids)
    following_ids = ids_array[position_array[:, 0], position_array[:, 1] + 1]
    if not np.array_equal(following_ids, target_array):
        raise ValueError(
            "Teacher-forced target ID must equal its following input token"
        )


def validate_two_stage_batch(
    ids: object,
    attention_mask: object,
    *,
    vocab_size: int,
) -> None:
    """Require complete logical branches followed only by suffix padding.

    A two-stage path with a per-row sequence-length mask requires its caller
    to compact every logical context, suffix, and candidate before one final
    right-padding operation. It cannot reuse a state produced after padding,
    because convolution and RoPE use physical positions. A dense forward
    keeps its previous arbitrary binary-mask behavior.
    """
    empty_positions = np.empty((0, 2), dtype=np.int64)
    empty_targets = np.empty((0,), dtype=np.int64)
    validate_target_positions(
        ids,
        attention_mask,
        empty_positions,
        empty_targets,
        vocab_size=vocab_size,
    )
    ids_array = np.asarray(ids)
    mask_array = np.asarray(attention_mask)
    lengths = np.asarray(np.sum(mask_array, axis=1, dtype=np.int64))
    suffix_mask = np.arange(ids_array.shape[1])[None, :] < lengths[:, None]
    if not np.array_equal(mask_array, suffix_mask):
        raise ValueError(
            "Two-stage attention requires suffix-only right padding"
        )
    if ids_array.shape[1] % 2:
        raise ValueError(
            "Two-stage path requires an even physical sequence length"
        )


def validate_two_stage_score_inputs(
    ids: object,
    attention_mask: object,
    positions: object,
    target_ids: object,
    *,
    vocab_size: int,
) -> None:
    """Validate host inputs before a compiled two-stage selected-head score."""
    validate_teacher_forced_targets(
        ids, attention_mask, positions, target_ids, vocab_size=vocab_size
    )
    validate_two_stage_batch(ids, attention_mask, vocab_size=vocab_size)
