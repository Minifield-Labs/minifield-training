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
