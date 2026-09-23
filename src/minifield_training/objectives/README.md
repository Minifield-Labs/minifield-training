# Supervision and loss mathematics

SFT, distillation, and policy objective mathematics expressed against declared inputs and model contracts. Own token weighting, reduction, and normalization explicitly. Do not acquire datasets, restore checkpoints, or run worker lifecycle code here. Different loss semantics remain named and independently tested instead of being hidden behind mode flags.

Status: causal loss and host admission checks implemented. `loss.py` owns
the shifted next-token terms and the zero-count-safe average over a caller's
`loss_mask` and the batch `attention_mask`. `validation.py` owns the
position, teacher-forced, and suffix-padding admission checks that run on
the host before traced math. Model callers pass `vocab_size` explicitly;
nothing here knows a model family.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
