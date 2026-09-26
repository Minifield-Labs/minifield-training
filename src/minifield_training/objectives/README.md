# Supervision and loss mathematics

SFT, distillation, and policy objective mathematics expressed against declared inputs and model contracts. Own token weighting, reduction, and normalization explicitly. Do not acquire datasets, restore checkpoints, or run worker lifecycle code here. Different loss semantics remain named and independently tested instead of being hidden behind mode flags.

Status: causal loss and host admission checks implemented. `loss.py` owns
the shifted next-token terms and the zero-count-safe average over a caller's
`loss_mask` and the batch `attention_mask`. `validation.py` owns the
position, teacher-forced, and suffix-padding admission checks that run on
the host before traced math. Model callers pass `vocab_size` explicitly;
nothing here knows a model family.

`classification.hard_label_terms` returns FP32 summed hard-label NLL and a
valid-decision count. `allowed` excludes classes from the softmax; padded rows
use a safe class for gathers and contribute zero. Valid disallowed labels cause
a nonfinite loss and rejection by the shared update transaction. Host batch
admission rejects them earlier.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
