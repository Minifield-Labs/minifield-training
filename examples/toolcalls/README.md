# Tool-call model

The tool-call model picks the next tool for a conversation and fills that
tool's arguments. It's MagicBox's joint pointer encoder with its own input
marking. Each request asks a choice question for the next tool (the options
are tool names with descriptions, plus `none`). Each argument is then an
extract, choice or noul question over the same text.

Open [the TPU notebook](../colab_toolcalls_lfm350m_tpu.ipynb) to train it.
It clones the pinned `SOURCE_REVISION`, downloads
`protodotdesign/toolcalls-v1` at revision
`9d63f7b434c26c2dadcead30a3152b55f9c298b1` and trains the 4 curriculum
stages in order:

| Stage | Records |
| --- | --- |
| 0 | Single easy calls, with confusers added gradually |
| 1 | Single calls among harder confusers and renamed tools |
| 2 | Turns with several independent calls |
| 3 | Calls that need an earlier call's result |

Every stage is its own run, with its own folder (`OUTPUT/stage{N}`),
checkpoints, learning-rate schedule and optimizer state. A stage starts from
the previous stage's final weights. One `SESSION_HOURS` deadline covers the
whole curriculum; rerun the training cell to resume the unfinished stage.
Each finished stage saves a bundle and runs its final evaluation, and the
last stage also exports trimmed FP32 and NF4 device bundles. `RUN_MODE =
'smoke'` caps every stage at 10 updates.

## Input marking

`data.py` compiles records exactly like MagicBox, then re-marks the
sequence. One `<|startoftext|>` leads the request, as in pretraining. Every
question, option and the source then opens with a marker naming its role,
in place of MagicBox's per-run BOS. The question's `Type:` line is dropped
because the marker carries the type.

The 10 markers sit in the tokenizer's reserved slots 7 to 16 (IDs 17 to 26).
They're placed by ID, so marker strings typed into a request stay plain
text. Exported tokenizers rename the slots to readable names, such as
`<|choice_question|>`, at the same IDs.

## Trainable markers

`composition.py` keeps the token embedding table frozen and trains 1
vector per marker (`toolcalls.markers`, undecayed and never quantized).
They start as copies of the BOS vector. Each forward writes them into
their embedding rows, so gradients reach only the markers. Bundles store
the folded table, so a runtime runs them as an ordinary pointer model with
template `toolcall-pointer/1`.

## Shared pieces

`composition.MODEL` plugs into MagicBox's `train.py` through
`train.Model`: corpus, extra parameters, forward, initialization, folding
and token renames. `train.py` here runs the curriculum on top of
MagicBox's run composition. The dataset itself is built outside this
repository, in the workspace experiment folder.

```sh
uv run --no-sync python -m pytest tests/examples/test_toolcalls.py tests/examples/test_toolcalls_notebook.py
```
