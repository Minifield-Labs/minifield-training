# MagicBox architecture and training

The notebook is `examples/kaggle_magicbox_lfm350m_tpu_v5e_8.ipynb`. Setup clones
`https://github.com/Minifield-Labs/minifield-training.git` and checks out the full `SOURCE_REVISION` recorded in its settings cell before
installing locked dependencies into the Python 3.12 or 3.13 notebook kernel.
Select a single-host TPU runtime, enable internet, and run the cells. It detects
the visible TPU devices and downloads the pinned dataset from Hugging Face.
Training executes directly in the kernel. Separate cells expose weight loading,
optimizer initialization, training-step lowering, compilation, training, validation,
and export. The accelerator-free host-memory monitor prints periodic samples
and flushes them to `OUTPUT/diagnostics/`. A kernel killed by the OS still has no
Python traceback; the last active stage and flushed samples identify where it stopped.

## Joint pointer model

Training now uses one joint encoder pass per request and answers every
question by pointing. The request becomes one sequence: each question's query
text, then its options, then the source text. Every query and option starts
with a BOS marker token (template `magicbox-pointer/1`):

| Type | Options the question may point at |
| --- | --- |
| choice | one `Candidate: ...` marker per candidate |
| score | one `Level: i of N ...` marker per level |
| noul | `Answer: false` and `Answer: true` markers |
| extract | a source span, or one `Answer: not stated in the text` marker |

After the encoder, 2 query/key projections give start and end logits for each
question over its request's tokens. One masked soft-target cross-entropy
trains every type. Option answers put the same target on start and end;
extraction targets the span's first and last tokens, or the "not stated"
marker. Soft teacher labels pass through unchanged. Hard score labels spread
over nearby levels (`SCORE_WIDTH`, default 0.15 of the scale's range);
evaluation compiles its records without spreading, so metrics compare against
the original labels. Decoding averages the start and end probabilities for
options, returns the expected level for scores, and picks the best span inside
one selectable run for extraction.

The notebook measures the longest joint sequence and the most questions in
any split, rounding tokens up to a multiple of 128. Exported bundles use
`minifield.magicbox.model/3`. The fusion architecture below still describes
v1 and v2 bundles, which `bundle.load` continues to admit.

Training packs several whole requests into each row of that length. Each
request keeps its own segment ID and positions from 0, and attention and
convolutions stay inside a segment, so a packed request gets the same losses
and gradients as it would alone. Each epoch plans rows by online first-fit
over a seeded shuffle, within the row's tokens and `QUESTIONS_PER_ROW` (32)
questions. The plan derives from the seed and epoch, so a resumed run replays
it exactly. `PACK = False` keeps 1 request per row.

On the pinned TPU compiler (compile-only v5e, see the workspace experiment
`magicbox-2026-09-29-tpu-compile-memory`), the joint gradient at 2,048 tokens
and 8 questions peaked at 1.68, 1.76, 2.51, 2.82, and 3.17 GiB of host RAM for
1, 2, 4, 8, and all 16 encoder layers. The former fusion gradient exceeded
7 GiB at 1 layer. The single training program including the AdamW commit
peaked at 1.96 GiB at 1 layer and 6.74 GiB at 4; the commit's finiteness checks
dominate that difference. Full-depth step memory, TPU execution, throughput,
and model quality remain unmeasured.

## Architecture

The backbone is `LiquidAI/LFM2.5-Encoder-350M`, revision
`b886781f7c6f10ca9b7096e21b83e30a073c2f39`. The native JAX implementation
matches the published configuration and all 148 checkpoint tensors. Both
attention and short convolutions are bidirectional. The convolution uses
centered padding, as the publisher's custom implementation specifies.

The source is encoded once per request. Every schema row uses the same
encoder parameters in an independent call. A shared projection maps both
paths to width 256. Schema rows pass through 2 distinct fusion layers with
4 attention heads, pre-LayerNorm self attention, source cross attention, and
a width-512 GELU FFN. Source memory stays unchanged. Dropout is 0.1 during
training and disabled during evaluation.

The native BOS token, ID 1 at position 0, is the schema readout. A shared
scalar head scores choice candidates and score levels. Separate scalar
heads predict binary probability and extraction presence. Extraction uses
a width-128 query/source match and one logit per source token.

| Parameter group | Count | Master dtype |
| --- | ---: | --- |
| Shared encoder | 354,483,968 | FP32 |
| Projection, norms, fusion | 1,840,640 | FP32 |
| Typed heads and span matching | 66,308 | FP32 |
| Total | 356,390,916 | FP32 |

Masters occupy 1,425,563,664 bytes before array/container overhead. Adam's
two FP32 moment trees add twice that amount. Activations, gradients, optimizer
temporaries, executable memory, and data buffers are separate allocations.
No vocabulary-output matrix or second encoder parameter copy is created.
The 67,108,864 token-embedding parameters are frozen, leaving 289,282,052
trainable parameters. Frozen embeddings receive neither gradients nor decay.
Their zero Adam moment slots remain allocated for checkpoint compatibility.

### Packed schema rows

The fixed training shape allows 256 schema rows of 512 tokens per request, but
real requests are much smaller. On the one local source available offline
(6,658 expanded requests, counted with an LFM2.5 tokenizer whose file differs
from the pinned one, so treat these as estimates), a request has 7 schema rows
at the median, 14 at p90, 21 at p99, and 34 at most. Rows run 47 tokens at the
median and 77 at most, so a request holds 346 schema tokens at the median and
1,558 at most. Encoding each row in its own padded 512-token pass spends over
99% of schema encoder work on padding.

The fusion model therefore packs a request's schema rows end to end into
`Shape.schema_sequences` encoder rows of 512 tokens. Segment IDs keep attention and
convolution inside each schema row, and rotary positions restart per row, so
every row encodes exactly as it would alone. Fusion still sees one
`[rows, 512]` view per request, gathered from the packed encoder output.
`Corpus.packed_sequences` measures the most any record in any split needs.
The joint pointer model above replaces this path for training; its sequence
holds each option once, without repeating the question. First-fit packing of the local
source above needs 1 sequence at the median, 2 at p99, and 4 at most: 2,048
schema tokens instead of 131,072. The shape stays fixed, so
the training step still compiles once. The chosen count enters the checkpoint
identity. Throughput on TPU is unmeasured.

## Dataset contract

The dataset's consumer snapshot is
[the format document](../src/minifield_training/datasets/magicbox/format-v1.md),
with its producer revision and checksum in
[the snapshot identity](../src/minifield_training/datasets/magicbox/format-v1.json).
The supported format is `minifield.magicbox/1.0`, schema template
`magicbox-rows/1`, and offset policy `trim-text-preserve-whitespace/2`.

The loader verifies manifest completion, full-build mode, identity digest,
tokenizer bytes/contract, shard paths, byte counts, hashes, and row counts.
It memory-maps Arrow caches and preserves all upstream split assignments.
Native IDs retain BOS; offset trimming changes text-token edges while
preserving whitespace-only token spans. Gold character spans must match the
saved token intervals exactly. No truncation, boundary snapping, or text
normalization occurs.

The model receives only source and schema tokens, masks, and row metadata.
Targets, provenance, source names, record IDs, and split names never enter the
encoder. The training reader omits unlabeled questions because independent
rows can't contribute a gradient to other questions. This also handles
questions whose labels the dataset builder dropped for schema overflow.
The reusable compiler supports partially labeled requests and inference with
no labels.

## Run defaults and recovery

The notebook starts in `RUN_MODE = 'smoke'`. It runs up to 10 total updates of
the full model, using 4 rows per device, 1 microbatch, and 8 validation
records. Its first 2 updates save a checkpoint and verify its parameters, moments, step,
and cursor against live state before continuing to the 10-update target. Rerunning
an already completed smoke run adds no training updates. The inference cell
reloads the exported bundle in the notebook kernel.

`DEVICES = None` detects the runtime's visible TPU devices. An explicit count
requires exactly that many devices on the same host. This supports a Colab
v5e-1 smoke run and an 8-device full run without changing model architecture.
Rows per microbatch are `DEVICES * ROWS_PER_DEVICE`, with a default of 4 rows
per device; with packing, each row holds 1 or more whole requests. Both modes
fix the row length from the dataset's longest joint sequence. Padding retains
these shapes for every update, including the final partial batch. Explicit
overflow fails admission without truncation.

Set `RUN_MODE = 'full'` for these full-training defaults:

- 1 epoch, frozen pretrained token embeddings; the encoder trunk and pointer projections train.
  Raise `EPOCHS` and rerun to resume into more epochs.
- All visible TPU devices, 4 packed rows per device, 1 microbatch. On 8
  devices this is 32 rows, about 115 requests, per logical update.
- BF16 activations, FP32 losses, parameters, and optimizer state.
- AdamW: learning rate 0.00002, betas 0.9/0.95, epsilon 1e-8,
  weight decay 0.01 for matrices, gradient clipping 1.0.
- Gradient checkpointing for encoder blocks.
- A fixed row length from the dataset's longest joint sequence and 32
  question slots per row; shorter rows are padded.
- Checkpoint every 250 updates; keep 2 complete states after evaluation.
- 256 seeded validation records at each checkpoint, plus gold and predicted
  answers for 3 fixed validation requests (`SAMPLE_RECORDS`) in the progress
  log. After all epochs, save the bundle, then evaluate up to 2,000 records
  from each source in each of validation, calibration, test, and OOD
  (`FINAL_RECORDS = 0` evaluates all). `final-<split>.json` holds each
  source's metrics and an `all` entry pooled by count.
- Each epoch uses a fresh 10% sample of the Nemotron-PII and
  PubMedAbstractsNER training records (`SOURCE_WEIGHTS`), and every record
  from the other sources. Those 2 NER sources hold most extraction fields,
  and their "earliest mention of a type" targets taught the model to answer
  with the first value-looking token.
- Each session runs up to 8 hours. Rerunning resumes the next unread update.

The recipe uses a constant learning rate. There is no warmup, scheduler,
quantization or automatic best-checkpoint selection. Token embeddings remain
frozen throughout the run.
Validation reports contain per-type loss and field counts, extraction exact
match/false-positive/false-null rates, choice accuracy, binary Brier score,
and ordinal MAE. Exact extraction compares the dataset's canonical gold span.
Alternative acceptable spans in provenance aren't included in that metric.

Losses first average selectable token BCE within each extraction field, then
average fields within each type, then average active types. The whole logical
update sets these weights before sharding, including uneven final batches.
Missing labels contribute zero. An entirely unsupervised training batch raises
`no_supervision` before the optimizer can commit.

Checkpoints contain parameters, Adam moments, update count, immutable source
and dataset identities, and the next data cursor. Shuffle and dropout keys
derive from that cursor and the seed. Changing batch topology, precision,
encoder, architecture, seed, optimizer settings, dataset, fixed-shape
batching policy, or packing settings prevents resume.
Epoch bounds and session time can increase without resetting the run.

Default output directories include mode and device count, for example
`magicbox-smoke-1dev` and `magicbox-full-8dev`. The full run starts from the
pretrained encoder; the single-device smoke checkpoint remains a separate
test artifact. Data parallelism replicates weights and optimizer state on
each device. The smaller smoke batch reduces activation memory; actual TPU memory and throughput still require the hardware run.

The default dataset is `protodotdesign/magicbox-v1`, revision
`f074bb549f16ea091fd8ece12e79652b8082871f`. It contains 693,376 records across
141 Parquet shards, including 446,751 training records. The notebook downloads
the processed shards, tokenizer, and metadata. `DATASET` can instead point
to a local completed dataset.

Kaggle's output directory contains checkpoints, metrics, `run.json`, a JSONL
progress log, and step-numbered inference bundles. For another session, attach
the previous output and point `RESUME_FROM` at its run folder.
Scratch dependencies, source weights, and Arrow caches live outside output.
Each exported session bundle is retained; users can archive earlier bundles
once they have copied the desired inference artifact.

## Decoding and export

`minifield.magicbox.model/2` bundles include weights, original encoder config,
fusion settings, tokenizer/offset contract, file hashes, and decode policy.
`examples.magicbox.predict` reloads the bundle and emits typed JSON.
It supports an empty questions map without running the model. The loader also
accepts v1 bundles with their original all-trainable inventory. New checkpoints
bind the frozen-embedding recipe and reject old all-trainable optimizer resumes.

Choice uses a softmax within that field's candidates. Score returns the
expected zero-based level and original rubric legend. Binary uses sigmoid.
Extraction requires presence at least 0.5 and a positive maximum-sum token
interval; masked tokens are barriers, and ties prefer shorter then earlier
intervals. The selected original substring is copied exactly. Public confidence
is null until a confidence adapter is supplied; span sum and presence are
separate diagnostics.

An inference-only schema cache stores complete pre-projection encoder tokens.
It binds tokenizer/template/precision revisions, schema IDs/masks, BOS position,
and immutable encoder parameter identities. Stale caches and training use
raise errors. Training never uses detached schema caches.

## Verification and remaining hardware evidence

The local environment uses Python 3.12 and JAX 0.7.2 on CPU. Tests cover
independent attention/convolution outputs and gradients, the publisher's tensor
inventory, FP32/BF16 encoder padding, shared encoder gradients, dynamic
candidate counts, row/request isolation, cache admission, partial labels,
equal-type reduction, Unicode offsets, and bundle round trips. A randomized
span test compares 600 short arrays with exhaustive enumeration.

The tiny all-four-types example reduced loss from 0.896994 to 0.00106434 in
80 steps. Its saved checkpoint reproduced the next full optimizer state
exactly. An eight-device CPU simulation matched the unsharded loss, mass,
and gradients with maximum absolute difference 5.96e-8.

A synthetic Parquet round trip with the actual pinned native tokenizer
verified Unicode gold alignment, omission of an over-length unlabeled
question, bucket construction, and identical record order after resume.

No full pretrained training run or TPU execution was launched locally.
The notebook contains a 2-update full-model TPU startup check before the
remaining smoke or full run. TPU peak memory, throughput, pretrained
PyTorch/JAX activation parity, and held-out semantic accuracy remain unmeasured. The default dense
XLA attention path is correct under the CPU checks; hardware measurements
will determine whether a fused TPU attention kernel is worth adding.

The reference implementation was inspected at the pinned
[LiquidAI source revision](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M/blob/b886781f7c6f10ca9b7096e21b83e30a073c2f39/modeling_lfm2_bidirectional.py).
Its source SHA-256 is
`f171f518be2a07da48b17fdea5655cad0a2452ab548e90e8ae903143686647e2`.
The small committed inventory fixture came from the checkpoint's HTTP range
header; no model weights or generated training records are committed.

### Compiler memory investigation

The September 29 single-TPU attempt was killed during XLA compilation after
reaching 44.72 GiB of host RAM on a 47 GiB host. Its first physical batch had
1 request, 4 schema rows, 512 source tokens, and 256 schema tokens. Lowering
compiler effort produced the same failure before the first optimizer update.
The exact TPU compiler pass responsible hasn't been identified.

The encoder now scans its layers, retaining one block body per operator kind
instead of expanding every layer into the compiled program. Weight selection
preserves the configured order and distinct convolution/attention tensors.
The model equations, FP32 master inventory, row checkpointing, and attention
implementation are unchanged.

A CPU abstract compile of the full 356,390,916-parameter gradient at the failed
batch shape, using JAX 0.7.2 and row chunk 1, measured:

| Measurement | Unrolled encoder | Scanned encoder |
| --- | ---: | ---: |
| Lowered HLO text | 2.30 MB | 1.20 MB |
| Lowering plus compilation | 30.86 s | 7.23 s |
| Process peak host RAM | 1.00 GiB | 0.58 GiB |
| Compiled temporary buffer estimate | 3.95 GB | 5.20 GB |

This reduces the local compiler workload at the cost of extra packed-weight
buffers. It hasn't yet been qualified on TPU. The CPU measurements don't
establish TPU peak memory or throughput. Repeated/interleaved operator tests
compare every master gradient and output with the prior unrolled schedule;
their BF16 comparisons enforce declared rounding in both graphs by disabling
CPU excess precision. The existing default-compiler masking, shared-gradient,
row-chunking, and tiny training checks also remain required.

## Reusable ownership

The model family retains the architecture and parameter names. Product wire
admission and formatting, tokenizer pins, and encoder/head selection live in the
example. Shared schema records, batching, weighting, evaluation, replay, artifact
verification, and bundle I/O have task-level owners. See the
[architecture audit and compatibility evidence](schema-components-audit.md).

The September 29 phase-marked preflight reused the ahead-of-time gradient,
normalized it, and then reached 45 GB of host RAM inside the first call to the
separately compiled AdamW commit. The commit had no ahead-of-time compile cell.
On CPU the commit's StableHLO is about 1.9 MB and compiles within 1.2 GiB, so
this doesn't identify the TPU pass either.

Training now compiles gradients, accumulation, and the commit as one donated
program (`engine.step.make_jit_step`). The notebook lowers and compiles that
exact call before training under the host-memory monitor and prints the
executable's device memory analysis. The first update reuses the executable.
Set `XLA_DUMP` in the settings cell to keep the compiled HLO for inspection.
One program may need more compiler memory than either former program; its
TPU peak remains unmeasured.

The scanned encoder still received SIGKILL in the user's subsequent TPU attempt.
The direct-kernel notebook exposes the failure stages; it does not establish
that the TPU compiler memory problem is resolved. Python 3.13 passed 11 selected
CPU tests covering notebook syntax, tiny optimization, exact checkpoint resume,
bundle reload, the schema-field strategy, and host-memory monitoring.

The [Polyomino comparison](../examples/magicbox/training-audit.md) records the
frozen-embedding correction, validation transfers, and remaining compiler limits.
