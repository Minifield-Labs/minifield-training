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
Dense batches hold JAX arrays in `microbatches`. Schema batches hold host arrays
for transfer by the streaming engine. The scanned JIT step places `active`
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

`packing.pack(sequences, rows, length, pad_token_id=...)` places whole token
sequences into fixed `[rows, length]` arrays by first-fit decreasing; equal
lengths keep input order. It returns token IDs, segment IDs (0 for padding,
`index + 1` for input sequence `index`), positions that restart at 0 in each
sequence, and every sequence's placement. Empty, overlong, and over-capacity
inputs raise instead of truncating. `packing.rows_required(lengths, length)`
counts the rows `pack` would fill. `packing.gather_index(placements,
row_length, width)` maps each sequence back to a `[sequences, width]` view of
the flattened rows, with a mask for columns past its length.

```python
from minifield_training.batching import packing

packed = packing.pack([[1, 5, 6], [1, 7], [1, 8, 9, 4]], 2, 5, pad_token_id=0)
index, mask = packing.gather_index(packed.placements, 5, 4)
rows = packed.input_ids.reshape(-1)[index] * mask
# [[1, 5, 6, 0], [1, 7, 0, 0], [1, 8, 9, 4]]
```

Packed consumers need segment-aware kernels: `kernels.bidirectional` for
encoders and the packed causal kernels for decoders. Schema batches pack their
encoder inputs; dense record batches still pad one record per row.

`packing.plan_rows(sizes, capacity)` groups whole items into rows by online
first-fit in the given order. Each item has one size per capacity dimension,
such as tokens and questions, and joins the first open row with room in every
dimension. A row closes once any dimension has less than `close_below` (5%) of
its capacity free. At `open_limit` (64) open rows, the fullest one closes
before another opens, which bounds how far an item lands from its position.
`packing.plan_updates(sizes, capacity, rows_per_update, seed=...)` shuffles by
`seed`, plans rows in that order, and groups them into updates of
`rows_per_update` rows. Every item appears once, and the plan depends only on
the inputs, so a run replays it from `seed + epoch` instead of saving it.
`packing.thin(labels, weights, seed=...)` keeps each item with its label's
weight as the probability, so an epoch can use a fresh sample of an
over-represented source. Weights are in (0, 1]; unlisted labels keep every
item.

```python
updates = packing.plan_updates(
    [(900, 2), (1000, 4), (500, 1), (1700, 3)], (1792, 32), 2, seed=17
)
# [[[0, 2], [1]], [[3]]]: requests 0 and 2 share a row; 2 rows per update.
```

## Schema-conditioned batches

`schema_fields.SchemaBatchStrategy` implements the same `BatchStrategy`
contract for `[microbatches, requests, schema_rows, schema_tokens]` plus one
source sequence per request. `contracts.CapacityShape` exposes only logical
capacity, allowing dense and multi-sequence layouts to share the interface.

`schema_fields.Shape` requires explicit vocabulary and pad token IDs. The model
adapter owns context limits. `bucket` chooses power-of-two dimensions within
caller caps; overflow raises without truncation. Set `fixed_shape=True` on
`SchemaBatchStrategy` to bypass bucketing and pad every update to `shape`,
including partial final updates. Source tokens, schema tokens, and schema row
count then stay constant across batches; masks and field weights make padding
inert. MagicBox training selects this policy. The default bucketed behavior
remains available for other consumers. Candidate groups stay on one
request/device and replay seeds bind update, record, field, and candidate IDs.

Schema batches also carry packed encoder inputs. `Shape.schema_sequences`
sets how many `schema_tokens`-long encoder rows hold one request's schema rows.
The default, one encoder row per schema row, always fits. Arrays
`packed_schema_ids`, `packed_schema_segments`, and `packed_schema_positions`
have shape `[M, R, sequences, schema_tokens]`. `schema_token_index`, shaped like
`schema_mask`, gathers each schema row from the flattened packed axis. The
row-layout arrays stay unchanged for row-level consumers. A request that
doesn't fit raises at packing time. Bucketing picks a power-of-two sequence
count within the configured cap.

The caller injects a weighting function over labeled field kinds. It runs once
for the complete logical update before physical slicing; batching only assigns
its returned weights to the first row of each field. Missing labels and further
candidate rows carry zero weight. Objective mathematics stays in `objectives`.
Inference packing explicitly permits an unsupervised request.

`stream.EpochStream` owns bounded chunk compilation, partial final updates,
global cursor and epoch recovery, and deadlines. Callers provide the ordered
per-epoch reader, record compiler, and pack function. Both training examples use
it while retaining their own Arrow or NumPy shuffle policy. Independent tests
verify exact replay and an unrelated schema consumer with different padding,
vocabulary, context length, and loss weights.

`stream.PlannedStream` replays a per-epoch plan of packed rows, such as
`plan_updates` output. `plan(epoch)` must return the same plan on every call.
Updates per epoch can vary, so `total_updates` sums every epoch's plan, and
the global update cursor resumes inside whichever epoch it lands in. Only
planned records are read and compiled, one update at a time. `pack(rows,
update)` receives compiled records grouped by row.

Both streams take `prefetch`, the number of updates to prepare ahead on a
background thread (`stream.Prefetch`). Order and resume are unchanged; source
errors re-raise on the consuming thread, and `close()` stops the producer.

## Joint pointer batches

`pointer.PointerBatchStrategy` lays each request out as one sequence: every
question's query, then its options, then the source (`pointer.layout`). Shape
`[M, R, sequence_tokens]` holds `input_ids`, `input_mask`, `segment_ids`, and
`positions`; per question (`[M, R, questions]`) it holds `query_index`, `kind`,
and `field_weight`; and `[M, R, questions, sequence_tokens]` holds `allowed`,
`start_target`, and `end_target`. Option questions may point only at their
option markers. Extraction may point at selectable source tokens or its "not
stated" marker. Every shape is fixed, so the training step compiles once.
Overflow raises.

A row can hold several whole requests back to back (`pointer.build_rows`, or
`pack_rows` for a planned stream). Each request gets its own segment ID, from
1, and positions restarting at 0, and its questions take consecutive question
slots. With a segment-aware encoder, a packed request's losses match the same
request alone in a row. `pack` and `build` keep one request per row.
`schema_fields.weighted_update` applies the caller's type weighting for both
schema and pointer batches.
