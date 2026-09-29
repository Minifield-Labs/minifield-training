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
