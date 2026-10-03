# Evaluation contracts and metrics

Reusable evaluation receives model execution and admitted records through
explicit callbacks. Product response formatting and judge rules stay with the
consuming example or experiment.

`field_decode` owns stable categorical probabilities, expected ordinal scores,
binary probabilities, and linear-time maximum-positive-sum extraction with
shortest/earliest ties. Selectability masks form span barriers; original offsets
copy exact source substrings. It consumes neutral `datasets.fields.Record`.

`schema_fields.Predictor` shares packing, JIT execution, and decoding between
inference and evaluation, with an injected forward and schema batch strategy.
Inference skips objective computation with `include_losses=False`.
`Evaluator` accepts a record iterator factory and aggregates per-field metrics
before averaging, so unequal field counts retain correct denominators.
Extraction exactness, false presence/absence, categorical accuracy, binary Brier
score, and ordinal MAE remain distinct. Callers may supply metric display names.

Independent tests cover an exhaustive span oracle, a separate zero-logit model,
unequal counts, partial labels, presence-only extraction, and soft targets.

The executable dependency policy is [architecture.toml](../../../architecture.toml).

`pointer.decode` turns start/end logits into typed answers. Options take the
average of the start and end softmaxes; choice returns the top label, ordinal
the expected level, and binary the probability of "true". Extraction reports
presence as one minus the "not stated" probability and the best span inside
one selectable run. `pointer.Predictor` and `pointer.Metrics` plug into the
generic `Evaluator` (pass `metrics=pointer.Metrics`), which accepts any scorer
and recorder for its record type. `SingleRequest` shares single-request jitted
execution between both predictors. Tests decode hand-built logits for every
type and check each metric.

Beyond those means, `pointer.Metrics` reports:

| Type | Metrics |
| --- | --- |
| Choice | `nll` of the gold option, 10-bin `ece` on top-1 confidence, top-2 `margin` |
| Binary | `auroc` (gold ≥ 0.5 is positive), soft-label `nll`, 10-bin `ece` |
| Ordinal | `spearman` between predicted and gold expected levels, `within_1` (rounded prediction within 1 of the gold level) |
| Extraction | lowercase whitespace `token_f1` (two nulls score 1), character `span_iou` on answerable questions, `null_f1` with "not stated" as positive, `accepted` (the gold or any `Question.accepted` alternative, such as another mention of the same entity) |
| Choice, binary, ordinal | `kl` from the gold distribution to the predicted one: the loss above the labels' own entropy, which matters for vote-share labels |

`<type>/error_reduction` is `1 - error / baseline` against a trivial answer:
always null (exact-or-null error), a uniform guess (choice error), 0.5
(Brier), or the middle level (MAE). `error_reduction` weights the types
equally. `pointer.degradation(in_domain, shifted)` gives each shared metric's
change, signed so positive is worse: relative for loss, NLL, MAE and KL, and
an absolute difference for rates and scores in [0, 1]. AUROC, Spearman and ECE need at
least 2 questions; tests check each against hand-derived values.
