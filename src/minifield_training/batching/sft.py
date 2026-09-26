"""Dense next-token supervision for the shared batching strategy."""

import numpy as np

from minifield_training.batching import contracts
from minifield_training.datasets.tokenization import TokenizedExample


class TokenTargets:
    """Encode the selected next-token positions of a tokenized example."""

    def validate(self, example: TokenizedExample) -> None:
        """Reject missing or unscorable next-token supervision."""
        mask = example.loss_mask
        if (
            len(example.input_ids) < 2
            or len(mask) != len(example.input_ids)
            or any(value not in (0, 1) for value in mask)
            or mask[0] != 0
            or not any(mask[1:])
        ):
            raise ValueError("Invalid tokenized supervision for dense SFT")

    def allocate(self, shape: contracts.BatchShape) -> contracts.HostBatch:
        """Pad all target positions as unscored."""
        if shape.sequence_length < 2:
            raise ValueError("Dense SFT requires at least 2 token positions")
        return {
            "loss_mask": np.zeros(
                (
                    shape.microbatches,
                    shape.rows_per_microbatch,
                    shape.sequence_length,
                ),
                dtype=np.int32,
            )
        }

    def write(
        self,
        targets: contracts.HostBatch,
        slot: tuple[int, int],
        example: TokenizedExample,
    ) -> None:
        """Copy the row's explicit token supervision mask."""
        targets["loss_mask"][slot][: len(example.loss_mask)] = example.loss_mask
