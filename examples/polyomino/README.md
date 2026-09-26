# Polyomino decision classifier

This example trains `LiquidAI/LFM2.5-230M-Base` to choose actions in an
11-column, 22-row falling-piece game. The pretrained backbone and frozen input
embeddings are retained. Only the output head has 8 logits: 7 playable actions
and PAD. PAD is masked from loss and prediction.

The input is the published
[`protodotdesign/polyomino-decisions-v1`](https://huggingface.co/datasets/protodotdesign/polyomino-decisions-v1)
train split, pinned to a commit in `source.py`. Its 4,737,585 rows contain a
resumable game state and an 8-slot one-hot expert action. The source reads
Parquet through Hugging Face Datasets' disk-backed Arrow cache. It derives
ChatML prompts and token IDs one physical update at a time, so it doesn't keep
4.7M Python records in memory. A seeded permutation visits each decision once;
the checkpoint's update cursor resumes at the next row. There is no local data
generator in this example.

Run the [Colab notebook](../colab_polyomino_classifier.ipynb) on one v5e TPU.
It downloads pinned Base weights and the pinned dataset, runs a 2-update
smoke, then offers a bounded 3-hour training run with persistent checkpoints.
A smoke checkpoint is reloaded and every parameter, optimizer moment, step,
and cursor is compared with the live state before gameplay starts. A mismatch
stops the run with the affected tensor name.
A full FP32 checkpoint is about 2.75 GB. Keep the Hugging Face cache and
checkpoints outside Git.

From a clone, `python -m examples.polyomino.train --help` and
`python -m examples.polyomino.evaluate --help` describe the command-line
paths. Training uses `--dataset-cache` for the Arrow files. Evaluation plays
new games with held-out seeds using only classifier logits and writes a readable replay. The game
rules in `engine.py` match the published dataset's state transition rules;
the expert policy and its generation settings remain part of the dataset,
not the trainer.

The previous v5e profile used 4 microbatches of 2 rows at 512 tokens and
peaked at 7.75 of 15.75 GiB device memory. `--microbatches 2 --rows 4`,
`--no-remat`, and `--fuse-accumulation` are separate performance trials; their
v5e speed and memory have not been measured. Each trial needs its own run ID
and checkpoint root. The final `end_to_end_updates_per_second` includes
prompt preparation, checkpoints, gameplay, and profiling export.

For a single-host v5e-8, train with `--platform tpu --devices 8 --rows 16
--microbatches 4`. Rows are global: this gives 2 rows per device per physical
microbatch and 64 decisions per logical optimizer update. The shared engine
replicates the FP32 model/Adam state and combines gradients over all 8 devices.
The learning rate stays at 0.0001; the larger logical batch is a new recipe
whose convergence and TPU throughput need measurement.

Use a separate run ID and checkpoint root. Device count and global batch shape
are part of the resume identity, so single-device checkpoints won't silently
resume with a different recipe. Pass the same `--devices`, `--rows`, and
`--microbatches` to the evaluation CLI for checkpoint admission. Gameplay
scores on one device, including when invoked after a distributed checkpoint.
The 8-device CPU tests cover gradient reduction and checkpoint restore;
the full pretrained v5e-8 run remains unqualified until its hardware smoke.

The separate [v5e-8 notebook](../colab_polyomino_classifier_tpu_v5e_8.ipynb)
uses those settings, verifies 8 local TPUs, and keeps the original notebook
unchanged. It pins the trainer source, verifies a 2-update checkpoint, then
offers resumed training bounded by 3 hours and the remaining dataset updates.
Changing its recipe requires a fresh checkpoint directory. Evaluation receives
the same training settings when admitting a saved checkpoint.

The pinned commit must be available on GitHub or in a source bundle. To run
from a local branch, create a bundle from this checkout and upload it to
`/content/minifield-training-tpu-v5e-8.bundle` in the notebook runtime:

```sh
git bundle create /tmp/minifield-training-tpu-v5e-8.bundle HEAD
```

The notebook automatically clones the uploaded bundle when present, verifies
its exact source revision, and still downloads pinned model and dataset files
from Hugging Face. Bundles and training outputs stay outside Git.
