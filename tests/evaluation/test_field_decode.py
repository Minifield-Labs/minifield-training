"""Independent exhaustive oracle for neutral span decoding."""

import random

from minifield_training.evaluation import field_decode


def test_linear_span_decoder_matches_exhaustive_oracle() -> None:
    """Compare masks, ties, and negative gaps against enumeration."""
    rng = random.Random(7)
    for _ in range(600):
        size = rng.randint(0, 12)
        values = [rng.choice([-3.0, -1.0, 0.0, 1.0, 3.0]) for _ in range(size)]
        mask = [rng.random() > 0.2 for _ in range(size)]
        candidates = [
            (sum(values[start:end]), -(end - start), -start, start, end)
            for start in range(size)
            for end in range(start + 1, size + 1)
            if all(mask[start:end]) and sum(values[start:end]) > 0
        ]
        best = max(candidates) if candidates else None
        expected = (best[3], best[4]) if best else None
        assert field_decode.best_span(values, mask) == expected
    assert field_decode.best_span([2, -1, 2], [True] * 3) == (0, 3)
