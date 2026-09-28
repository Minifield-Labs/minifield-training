# MagicBox architecture and training

The notebook is `examples/kaggle_magicbox_lfm350m_tpu_v5e_8.ipynb`. Setup clones
`https://github.com/Minifield-Labs/minifield-training.git` and checks out commit
`588a9e278ed2592e342fe8e56446e005e4842841` before installing its locked dependencies.
Select a single-host TPU runtime, enable internet, and run the cells. It detects
the visible TPU devices and downloads the pinned dataset from Hugging Face.

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
the full model, using 1 request per device, 1 microbatch, schema chunks of 1,
and 8 validation records. Its first 2 updates save a checkpoint, which the
next invocation reloads before continuing to the 10-update target. Rerunning
an already completed smoke run adds no training updates. The inference cell
reloads the exported bundle in a fresh process.

`DEVICES = None` detects the runtime's visible TPU devices. An explicit count
requires exactly that many devices on the same host. This supports a Colab
v5e-1 smoke run and an 8-device full run without changing model architecture.
Requests per microbatch are `DEVICES * REQUESTS_PER_DEVICE`, with a default
of 1 request per device. Both modes retain source and schema token limits.

Set `RUN_MODE = 'full'` for these full-training defaults:

- 3 epochs, all parameters trainable, including embeddings.
- All visible TPU devices, 1 request per device, 4 accumulated microbatches.
  On 8 devices this is 32 requests per logical update.
- BF16 activations, FP32 losses, parameters, and optimizer state.
- AdamW: learning rate 0.00002, betas 0.9/0.95, epsilon 1e-8,
  weight decay 0.01 for matrices, gradient clipping 1.0.
- Gradient checkpointing for encoder and schema-row computation.
- Schema processing in chunks of 4. Power-of-two buckets for source length,
  schema length, and row count; default operational maximum 256 rows.
- Checkpoint every 250 updates; keep 2 complete states after evaluation.
- 256 seeded validation records at each checkpoint. After all epochs,
  evaluate all available validation, calibration, test, and OOD records.
- Each session runs up to 8 hours. Rerunning resumes the next unread update.

The recipe uses a constant learning rate. There is no warmup, scheduler,
quantization, freezing stage, or automatic best-checkpoint selection.
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
encoder, architecture, seed, optimizer settings, or dataset prevents resume.
Epoch bounds and session time can increase without resetting the run.

Default output directories include mode and device count, for example
`magicbox-smoke-1dev` and `magicbox-full-8dev`. The full run starts from the
pretrained encoder; the single-device smoke checkpoint remains a separate
test artifact. Data parallelism replicates weights and optimizer state on
each device. The smaller smoke batch and schema chunks reduce activation
memory; actual TPU memory and throughput still require the hardware run.

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

`minifield.magicbox.model/1` bundles include weights, original encoder config,
fusion settings, tokenizer/offset contract, file hashes, and decode policy.
`examples.magicbox.predict` reloads the bundle and emits typed JSON.
It supports an empty questions map without running the model.

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

The tiny all-four-types example reduced loss from 0.896994 to 0.00092325 in
80 steps. Its saved checkpoint reproduced the next full optimizer state
exactly. An eight-device CPU simulation matched the unsharded loss, mass,
and gradients with maximum absolute difference 3.58e-7.

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

## Reusable ownership

The model family retains the architecture and parameter names. Product wire
admission and formatting, tokenizer pins, and encoder/head selection live in the
example. Shared schema records, batching, weighting, evaluation, replay, artifact
verification, and bundle I/O have task-level owners. See the
[architecture audit and compatibility evidence](schema-components-audit.md).
