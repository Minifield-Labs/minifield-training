# Batch strategies

`contracts.BatchStrategy[RecordT]` defines resumable epoch iteration.
`dense.DenseBatchStrategy` implements one record per row: token validation,
seeded order, padding, active slots, and device transfer have one owner.
`contracts.PhysicalUpdate` is the shared result type for every batch consumer.

The dense strategy receives a `TargetEncoder[RecordT]`. `sft.TokenTargets`
encodes next-token loss masks; `classification.ClassTargets` validates class
IDs and encodes labels plus valid-row masks. Target encoders own supervision
admission and arrays, and cannot replace observation arrays.

```python
from minifield_training.batching import classification
from minifield_training.batching import contracts
from minifield_training.batching import dense
from minifield_training.batching import sft

shape = contracts.BatchShape(
    microbatches=4, rows_per_microbatch=2, sequence_length=512,
    pad_token_id=0, vocab_size=65536,
)
token_batches = dense.DenseBatchStrategy(shape, sft.TokenTargets())
decision_batches = dense.DenseBatchStrategy(
    shape, classification.ClassTargets((True, True, False), padding_label=2)
)
# token_records contain datasets.tokenization.TokenizedExample values.
updates = token_batches.iter_updates(token_records, seed=17, start_update=0)
# decision_records contain datasets.labeled.LabeledSequence values.
decisions = decision_batches.iter_updates(decision_records, seed=17)
```

Model inputs and attention masks have shape `[M, B, T]`. Token targets have
the same shape; class labels and valid-row masks have shape `[M, B]`.
`PhysicalUpdate.active` is a NumPy boolean `[M]` vector for host slot selection.
All arrays in `microbatches` are JAX arrays. The scanned JIT step places `active`
at its call boundary; direct eager calls should use `jnp.asarray(active)`.
The streaming step consumes host flags directly.

`start_update` skips complete logical updates in the same seeded epoch order.
Each real example appears once; partial updates receive finite inert rows.
`example_ids` lists only real records, in row order. Duplicate IDs, overlength
inputs, out-of-range tokens and invalid supervision fail admission.
`shuffle=False` preserves the supplied record order.

A new dense supervision mode implements `TargetEncoder`. A different packing
algorithm implements `BatchStrategy` and yields the shared `PhysicalUpdate`.
Its `update_count(examples)` declares a seed-independent epoch length; the
runner uses that count to resume without assuming dense packing.
`BatchSource` supplies a replayable stream from a global update cursor and an
optional absolute deadline. The shared runner consumes either interface.

Sequence packing isn't implemented. Concatenating records needs segment-aware
attention and an objective that preserves supervision boundaries.
