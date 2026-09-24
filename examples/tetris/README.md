# Tetris decision example

This example fine-tunes the pinned `LiquidAI/LFM2.5-230M-Base` model on tick
decisions from a scripted Tetris expert. It uses the existing shared classifier,
optimizer, runner and checkpoint code. The 22-by-10 game, expert, prompt and
gameplay callback live here. Actions 0 through 6 are playable; output 7 is PAD
and is excluded from loss and prediction. The pretrained input embeddings stay
frozen.

Run the [Colab notebook](../colab_tetris_classifier.ipynb) on one v5e TPU. It
uses Python 3.12 in a local environment, downloads an immutable Base revision,
checks the complete pretrained weights, generates expert decisions, and runs a
2-update smoke with a short learned-policy game. A second call restores the
saved state for 2 more updates. The longer run is optional and requires an
explicit persistent storage path.

From a clone, use `python -m examples.tetris.prepare`,
`python -m examples.tetris.train`, and `python -m examples.tetris.evaluate`
from the repository root. Each command exposes `--help`. Generation preserves
the supplied ChatML framing, encodes it once with the pinned native tokenizer,
and rejects any observation longer than the configured sequence length. Its
manifest binds the data, model revision, tokenizer and generator source. The
evaluator chooses actions from learned logits and writes a readable board
replay. No model weights, generated data, checkpoints or logs belong in Git.

A full FP32 training checkpoint is about 2.75 GB. The example leaves saved
states in the configured checkpoint directory so a run can resume.

Training reports first-update time separately from warm update throughput.
For an accelerator trace, add `--profile-dir /absolute/output/path
--profile-updates 30 --max-steps 33 --checkpoint-every 1000 --eval-games 0` to
a resumed run. The first 3 updates warm the process, then JAX writes a device
trace covering 30 annotated updates. Choose a new explicit profile directory
for each run and keep traces outside Git.
