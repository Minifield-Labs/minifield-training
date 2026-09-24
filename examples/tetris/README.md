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
The CLI defaults to saving every 5,000 updates and runs no in-loop games;
evaluate a saved checkpoint with `examples.tetris.evaluate` after training.

The v5e profile used 4 microbatches of 2 rows at 512 tokens and peaked at
7.75 of 15.75 GiB device memory. For a separate performance comparison,
`--microbatches 2 --rows 4` keeps 8 decisions per update while cutting the
number of gradient calls. `--no-remat` keeps block activations for backward
instead of recomputing them. Use a new run ID and a separate checkpoint root
for each trial because checkpoints are named by step within that root. Neither
choice has a measured v5e speedup yet.
`--fuse-accumulation` combines each later microbatch's gradient with the
accumulated gradient in one compiled program, removing the separate add
program. Give this trial its own run ID and checkpoint root too. The v5e
profile spent about 8.6 ms per update in separate add programs, but fused
training's actual speed and memory use remain unmeasured.

Training reports first-update time separately from warm update throughput.
The final `end_to_end_updates_per_second` includes batch preparation,
checkpoints, gameplay, and profiling export when requested.
The Colab setup installs `tensorflow-cpu==2.20.0` in the training environment
and checks its Python profiling hook before training. For an existing environment,
install that version with `uv pip install --python /path/to/venv/bin/python
tensorflow-cpu==2.20.0` before capturing a trace.
For an accelerator trace, run a separate 4-50 update session with
`--profile-dir /absolute/output/path --max-steps 30 --checkpoint-every 1000
--eval-games 0`. JAX starts tracing after the first compiled update and exports
the native XPlane after the final checkpoint. Every update carries a numbered
`train` annotation. No profile export interrupts an ongoing training run, and
the hours-long run uses no profiling flags. Choose a new explicit profile
directory for each diagnostic run and keep traces outside Git.
