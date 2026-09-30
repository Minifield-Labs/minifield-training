# MagicBox training entrypoints

Open [the TPU notebook](../kaggle_magicbox_lfm350m_tpu_v5e_8.ipynb)
for dataset admission, dependency setup, encoder download, startup checks,
full training, and inference reload. Setup clones the training repository from
GitHub and verifies `SOURCE_REVISION` before installing dependencies and running
Python directly in the notebook kernel. Weight loading, optimizer initialization,
training-step lowering, compilation, training, and evaluation have separate cells.
Set `XLA_DUMP` to a directory to save text HLO for every compiled program.
Host-memory samples print during compilation and persist under `diagnostics/`.
The kernel must run Python 3.12 or 3.13; dependencies install into that kernel.
Packages the kernel imported before the first cell, such as Kaggle's preloaded
NumPy, keep their versions when pip accepts them, so Run All needs no restart.
Only an incompatible preloaded version is replaced, with one restart requested. Edit the notebook directly. When training source changes,
publish its commit and update `SOURCE_REVISION` to that full commit SHA.

The notebook defaults to `RUN_MODE = 'smoke'`: 10 updates of the full pretrained
model, checkpoint saving and full-state reload verification, 8-record validation,
and inference export/reload. It
detects all TPU devices on one host. `ROWS_PER_DEVICE = 4` sets the rows each
device takes per update. Set `DEVICES = 1` to require a single device.

On the larger runtime, set `RUN_MODE = 'full'`, which trains 1 epoch and
prints 3 fixed validation requests' answers at every checkpoint. Set `DEVICES = 8` to require 8 visible devices. Smoke and full modes use separate output folders; full mode
starts from pretrained weights. Exact optimizer resume requires the same device
count, batch settings, and packing settings.

With `PACK = True`, each row holds as many whole requests as fit in
`SEQUENCE_TOKENS` tokens and `QUESTIONS_PER_ROW` questions. Every epoch gets a
fresh seeded plan, built before training starts, so a resumed run replays the
same updates. Updates per epoch vary with the plan; the notebook prints the
count and how full the rows are. `PACK = False` restores 1 request per row.
`PREFETCH = 2` prepares 2 updates ahead on a background thread, and progress
reports include `batch_wait_seconds`.
Final evaluation reports every source separately.

By default, both modes download `protodotdesign/magicbox-v1` at revision
`f074bb549f16ea091fd8ece12e79652b8082871f`. Set `DATASET` to a completed local
directory to use an attached dataset. Both modes train the joint pointer model
(see [the MagicBox guide](../../docs/magicbox.md#joint-pointer-model)). With
`SEQUENCE_TOKENS = None` and `QUESTIONS = None`, the notebook measures the
longest joint question-and-text sequence and the most questions in any split
before fixing the shape. `SCORE_WIDTH` spreads hard score labels over nearby
levels. Both modes freeze the pretrained token embeddings. `train.py` accepts
the same settings as `--sequence-tokens`, `--questions`, and `--score-width`,
and `predict.py` loads v3 pointer bundles. `OPTIMIZER = 'optax'` selects the
optax commit; `'transactional'` restores the older checked commit.
`KEEP_CHECKPOINTS = 2` bounds output to about 2 checkpoints of 4.3 GB each
plus the bundle, and the final cell prints the output folder's size.

Local offline optimization and checkpoint check:

```sh
uv run --no-sync python -m examples.magicbox.smoke
```

Eight-device CPU gradient comparison:

```sh
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run --no-sync python -m examples.magicbox.parallel_smoke
```

The TPU entrypoint accepts completed data and pinned encoder files:

```sh
python -m examples.magicbox.train \
  --dataset /data/magicbox --model-dir /data/encoder \
  --output /output/magicbox-run --cache /scratch/arrow \
  --devices 8 --platform tpu --epochs 3 --max-hours 8
```

For a bounded single-device run, use a separate output directory with
`--devices 1 --rows 4 --microbatches 1 --max-steps 10
--validation-records 8 --final-records 8`. The CLI's `--max-steps` bounds new
updates per invocation; the notebook caps smoke mode at 10 total updates
across repeated invocations.

Rerun the same command to resume. Use `--max-steps 2` for a bounded startup
check; those updates remain part of the same run. `--evaluate-only` requires
an existing checkpoint. `--allow-sample` explicitly admits a sample dataset
for engineering checks. Dataset/model identities and batch settings remain
bound to checkpoints.

```sh
python -m examples.magicbox.predict \
  --bundle /output/magicbox-run/bundle-00010000 \
  --request /data/request.json
```

`data.py` owns the published request/target format, schema wording, and public
response formatting. `tokenizer.py` binds the pinned native tokenizer to shared
offset normalization. `source.py` admits the published manifest and Arrow rows,
then supplies ordering and compilation to the shared epoch stream.

`composition.py` selects the LFM encoder and MagicBox heads; `bundle.py` interprets
the product bundle metadata through shared checkpoint I/O. The CLIs bind these
adapters to reusable batching, objectives, evaluation, and training lifecycle.
`smoke.py` and `parallel_smoke.py` are bounded architecture diagnostics.
No sibling source tree is imported.

Architecture, recipe, metrics, and verification details are in
[the training guide](../../docs/magicbox.md).

The [training comparison](training-audit.md) records the Polyomino recipe audit,
confirmed omissions, and local compiler evidence.
