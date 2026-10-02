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

`schema_fields.losses` implements grouped categorical cross entropy, Bernoulli
binary/presence losses, and mean selectable-token BCE per extraction field.
Absent extraction labels supervise all selectable tokens as zero;
presence-only labels omit token loss. `terms` returns preweighted loss sum
and weight mass to the existing step engine. All reductions use FP32, and
missing labels and padding have zero weight. Group sizes don't increase a
field's loss weight. Tests use analytical zero-logit expectations.

`schema_fields.balance_types` computes one mean per active task, followed by a
weighted mean across tasks. The caller passes it into the batch strategy so its
host weights cover the whole logical update before device/microbatch slicing.
There are no model-family or product-template imports in these objectives.

`pointer.losses` is one masked soft-target cross-entropy for every question
type: a softmax over each question's allowed tokens, averaged over the start
and end pointers. Padding questions have zero targets and zero loss.
`pointer.terms` returns the type-weighted sum and weight mass. Tests check
hand-derived losses and gradients, including masked and padding positions.
`pointer.distillation` is the temperature-softened KL from a teacher's
pointer distributions to a student's over each question's allowed tokens,
times T², with no gradient to the teacher. `pointer.distilled_terms` trains
shared weights as a dense parent and a quantized student at once: dense
cross-entropy plus weighted student cross-entropy and distillation.
