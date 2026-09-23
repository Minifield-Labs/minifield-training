"""Build fixed-shape dense logical updates from tokenized examples."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np

from minifield_training.datasets.tokenization import TokenizedExample
from minifield_training.kernels.types import DeviceBatch


@dataclass(frozen=True)
class PhysicalUpdate:
    """One logical update with traceable real example IDs."""

    microbatches: DeviceBatch
    active: jax.Array
    example_ids: tuple[str, ...]


def _validate_example(
    example: TokenizedExample, *, sequence_length: int, vocab_size: int
) -> None:
    """Reject malformed supervision before device transfer."""
    ids = example.input_ids
    mask = example.loss_mask
    if (
        len(ids) < 2
        or len(ids) > sequence_length
        or len(ids) != len(mask)
        or any(
            not isinstance(token, int)
            or isinstance(token, bool)
            or not 0 <= token < vocab_size
            for token in ids
        )
        or any(value not in (0, 1) for value in mask)
        or mask[0] != 0
        or not any(mask[1:])
    ):
        raise ValueError("invalid tokenized example for dense SFT")


def iter_updates(
    examples: Sequence[TokenizedExample],
    *,
    microbatches: int,
    rows_per_microbatch: int,
    sequence_length: int,
    pad_token_id: int,
    vocab_size: int,
    seed: int,
    start_update: int = 0,
    shuffle: bool = True,
) -> Iterator[PhysicalUpdate]:
    """Yield stable padded updates, each real example exactly once.

    ``start_update`` skips earlier complete updates in the same seeded order.
    Partial rows and microbatches are finite, masked, and inactive.
    """
    if (
        min(microbatches, rows_per_microbatch, sequence_length, vocab_size) < 1
        or sequence_length < 2
        or not 0 <= pad_token_id < vocab_size
        or seed < 0
        or start_update < 0
    ):
        raise ValueError("invalid batch shape, vocabulary, seed, or cursor")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("duplicate example id")
    for example in examples:
        _validate_example(
            example, sequence_length=sequence_length, vocab_size=vocab_size
        )
    order = (
        np.random.default_rng(seed).permutation(len(examples))
        if shuffle
        else np.arange(len(examples))
    )
    capacity = microbatches * rows_per_microbatch
    for update_index in range(start_update, math.ceil(len(order) / capacity)):
        chosen = order[update_index * capacity : (update_index + 1) * capacity]
        ids = np.full(
            (microbatches, rows_per_microbatch, sequence_length),
            pad_token_id,
            dtype=np.int32,
        )
        attention = np.zeros_like(ids)
        loss = np.zeros_like(ids)
        active = np.zeros(microbatches, dtype=np.bool_)
        names: list[str] = []
        for slot, source_index in enumerate(chosen):
            example = examples[int(source_index)]
            microbatch, row = divmod(slot, rows_per_microbatch)
            length = len(example.input_ids)
            ids[microbatch, row, :length] = example.input_ids
            attention[microbatch, row, :length] = 1
            loss[microbatch, row, :length] = example.loss_mask
            active[microbatch] = True
            names.append(example.id)
        yield PhysicalUpdate(
            {
                "input_ids": jnp.asarray(ids),
                "attention_mask": jnp.asarray(attention),
                "loss_mask": jnp.asarray(loss),
            },
            jnp.asarray(active),
            tuple(names),
        )
