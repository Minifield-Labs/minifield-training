# Evaluation contracts and metrics

Generic evaluation orchestration and metric aggregation using model execution contracts and admitted datasets. Keep parsing acceptance, task success, scope rejection, clarification, and permission limits distinct. Reject ambiguous structured outputs. Product-specific judge rules and customer fixtures stay in registered experiment inputs rather than becoming reusable library defaults.

`magicbox` implements stable grouped probabilities, expected ordinal scores,
and linear-time maximum-positive-sum extraction with shortest/earliest ties.
Masked tokens are barriers; source offsets copy the exact substring. Presence
below 0.5, no selectable source, or no positive interval returns null.
`decode` exposes raw diagnostics. `format_results` emits the four public typed
response shapes and optional externally calibrated confidence. Confidence is
null by default; score legends retain the original rubric. Randomized tests
compare the span algorithm with an independent exhaustive oracle.

The executable dependency policy is [architecture.toml](../../../architecture.toml).
Document each added public contract, consumer, example, and test here.
