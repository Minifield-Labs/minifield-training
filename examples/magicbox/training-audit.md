# Training comparison with Polyomino

This audit compares the dense Polyomino v5e-8 recipe with the MagicBox
single-device smoke and full recipes. Paths below identify the implementation,
rather than inferring behavior from notebook descriptions. The Polyomino
notebook pins `79d9bcfc302b743efbb439d151385e08236c33b4`; its classifier,
streaming-step, and optimizer implementations match the pre-fix versions
examined here. The existing example runner changes were checkpoint-discovery
deduplication.

## Confirmed omissions and corrections

1. **Token embeddings were trainable.** Polyomino's
   [`classification.parameter_inventory`](../../src/minifield_training/strategies/classification.py)
   freezes its input embeddings. MagicBox's [composition](composition.py)
   previously trained all 356,390,916 parameters. It now freezes exactly
   `lfm2.embed_tokens.weight`, excluding its 67,108,864 values from gradients,
   decay, and updates. The remaining 289,282,052 parameters train. A real tiny
   update verifies unchanged embedding bits and zero moments while the trunk
   and task parameters change. Frozen moment slots remain allocated, matching
   the existing shared checkpoint contract.
2. **The real preflight lacked a checkpoint comparison.** Polyomino restored
   its full checkpoint and compared parameters, both moment trees, step, and
   cursor with live state. MagicBox only covered exact continuation on a tiny
   CPU model. Both now call the shared
   [`training_state.verify_roundtrip`](../../src/minifield_training/checkpoints/training_state.py).
   The notebook checks the full checkpoint after its preflight updates.
3. **State validation retained full host copies.** The shared eager AdamW
   validator converted all 3 state trees to NumPy before the first update.
   JAX 0.7.2 caches device-to-host copies on the arrays. MagicBox's tensor
   payload alone is 4,276,690,992 bytes across these trees. The validator now
   reduces each leaf on device and transfers 3 booleans. This defect also
   affected Polyomino. Pretrained admission and checkpoint serialization are
   separate boundaries that still read full tensors.

Training identity changes to `magicbox-jax/2-frozen-token-embeddings`, so old
all-trainable optimizer checkpoints cannot silently resume under the new
recipe. Inference exports use `minifield.magicbox.model/2`; the loader continues
to admit v1 bundles using their original inventory.

## End-to-end comparison

| Area | Working Polyomino recipe | MagicBox recipe / finding |
| --- | --- | --- |
| Backbone | LFM2.5 Base, causal | LFM2.5 Encoder 350M, bidirectional; intentionally different architecture |
| Encoder work | 1 encoding per decision row | 1 source encoding per request plus 1 encoding per schema row, then fusion |
| Physical batch | 16 decisions across 8 devices | 8 requests across 8 devices; request work varies with schema rows |
| Accumulation | 4 microbatches | 4 full-run microbatches; smoke uses 1; both use the same streaming update |
| Sequence shapes | Fixed 512 tokens | Source 128/256/512/1024, schema 128/256/512, rows 4/8/16/32/64/128/256: up to 84 gradient shapes |
| Precision | BF16 computation, FP32 masters, gradients, moments | Same; fusion norms and objective reductions use FP32 where required |
| Rematerialization | Encoder blocks checkpointed | Encoder scan body and schema rows checkpointed; extra schema work remains |
| Attention | Dense causal attention | Dense bidirectional attention through JAX's XLA implementation; CPU oracle tests cover its outputs and gradients |
| Gradient/optimizer boundary | Separate compiled programs | Same shared implementation; compiling the gradient alone doesn't compile normalization or AdamW |
| Accumulation fusion | Opt-in, default off | Off, same default; no full-state optimizer wrapped around the model gradient |
| AdamW | Betas 0.9/0.95, epsilon 1e-8, decay 0.01, clip 1 | Same; task-specific learning rate 2e-5 instead of 1e-4 |
| Learning-rate schedule | Constant | Constant; no missing scheduler or warmup relative to this baseline |
| Placement/donation | Replicated state, donated AdamW | Same shared runner and transaction; more devices don't shard the model state |
| Frozen moments | Zero slots retained | Same after embedding correction; freezing doesn't remove 512MiB of moment storage |
| Weight/source admission | Pinned source, exact shape/dtype/hash checks | Same shared loader with encoder-specific mapping |
| Data reduction | Sum decision loss/count, normalize once | Type-balanced field weights established across the logical batch before sharding; shared gradient normalization |
| Checkpoint cadence | Every 5000 full-run updates | Every 250; more serialization and evaluation overhead, unrelated to a failure before update 1 |
| Resume | Full state, immutable identities, next-batch cursor | Same shared persistence and runner; seeded epoch/dropout replay |
| Evaluation | Task-specific gameplay | Task-specific field metrics; CPU correctness doesn't establish pretrained semantic accuracy |

The row count means 1 request is not a hardware-equivalent unit to 1 Polyomino
decision. The observed failed batch had 512 source tokens and 4 schema rows of
256 tokens. Row chunk 1 limits concurrent schema work but still processes all
4 rows and differentiates their shared encoder use.

Power-of-two bucketing avoids padding every request to 256 rows, but it also
creates additional compilations and retained executables. A successful first
gradient compile covers only that shape. The 10-update smoke doesn't establish
capacity for every dataset bucket or the full-run row chunk. No arbitrary
truncation, removed supervision, or global cache-clearing policy was introduced
as a speculative fix.

## Failure trace and evidence

The user's gradient compile completed in about 78 seconds with roughly
34.6GiB peak host RSS. The next cell reached roughly 41GiB and its kernel died.
The missing `first_update_seconds` event places the failure before the first
completed transaction. The call chain is:

`training_run.run` → state validation → batch retrieval → gradient execution
→ normalization → donated AdamW → commit check → first-update event.

Checkpoint saving and validation evaluation occur after this event. They
cannot explain this particular first-update failure. A later undefined
`diagnostics` variable reflects the restarted kernel losing its namespace.

Preflight now reports synchronized gradient, normalization, and optimizer
boundaries. Normal training leaves this diagnostic callback disabled. Local
CPU execution confirmed that explicit gradient compilation can be reused by
the streaming update; that alone doesn't establish TPU cache/layout reuse.

Local Docker ran the actual pinned JAX 0.7.2 / libtpu 0.0.23 compiler against a
compile-only `v5e:1x1` topology without TPU hardware. The full-size gradient and
optimizer probes were killed under the deliberately bounded 5GiB process
limit. They didn't produce full-model TPU memory estimates or qualify a fix.

A reduced optimizer probe retained the real 65,536 × 1,024 embedding table,
the first 2 convolution layers, and the full task-head shapes:

| TPU AOT optimizer measurement | Embeddings trainable | Embeddings frozen |
| --- | ---: | ---: |
| Compiler host peak | 2.852GiB | 2.604GiB |
| Compiled argument bytes | 1,691,645,440 | 1,423,209,984 |
| Compiled temporary bytes | 1,739,544,576 | 665,189,888 |

These are reduced-model compiler measurements, not full-model training peaks.
The reduced gradient still exceeded the local 5GiB bound. No TPU training was
executed locally, and the exact operation that killed the user's kernel
remains unproven. The confirmed fixes must not be described as a successful
full-size TPU run.

A full-shape CPU gradient compile at the failed 1 × 512 source / 4 × 256 schema
shape completed in both recipes. Frozen embeddings reduced gradient output
storage by 268,435,464 bytes (including tuple metadata). CPU temporary-buffer
estimates remained about 5.20GB in both cases, with compilation near 6.6 seconds.
These results don't support claiming that embedding freezing alone resolves
compiler host memory.
