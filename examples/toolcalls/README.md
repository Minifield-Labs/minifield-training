# Tool-call model

The tool-call model picks the next tool for a conversation and fills that
tool's arguments. It's MagicBox's joint pointer encoder with its own input
marking. Each request asks a choice question for the next tool (the options
are tool names with descriptions, plus `none`). Each argument is then an
extract, choice or noul question over the same text.

Open [the TPU notebook](../colab_toolcalls_lfm350m_tpu.ipynb) to train it.
It clones the pinned `SOURCE_REVISION`, downloads
`protodotdesign/toolcalls-v1` at revision
`f4d88a672bfddda94e2bec7b5b8bfa15d33e586b` and trains the 4 curriculum
stages in order:

| Stage | Records |
| --- | --- |
| 0 | Single easy calls, with confusers added gradually |
| 1 | Single calls among harder confusers and renamed tools |
| 2 | Turns with several independent calls |
| 3 | Calls that need an earlier call's result, including audited synthetic conversations |

Every stage is its own run, with its own folder (`OUTPUT/stage{N}`),
checkpoints, learning-rate schedule and optimizer state. A stage starts from
the previous stage's final weights. One `SESSION_HOURS` deadline covers the
whole curriculum; rerun the training cell to resume the unfinished stage.
Each finished stage saves a bundle and runs its final evaluation, and the
last stage also exports FP32 and quantized device bundles. `RUN_MODE = 'smoke'`
caps every stage at 10 updates.

`ENCODER` picks the LFM2.5 encoder: `230m` (the default; 14 layers, a
narrower feed-forward) or `350m`. Both share the tokenizer and embedding
width, so the dataset and vocabulary serve either.

## Quantization warm-up

With `QUANTIZER` set (`ternary` by default), the notebook first warms the
student up on generic text, then runs the curriculum from those weights.
`warmup.py` trains the encoder's projections under the quantizer to match
the frozen dense encoder on FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, one
pinned `sample/10BT` shard): the KL between their masked-LM predictions at
25% masked tokens (the head is the tied embedding table, over the model's
vocabulary) plus the cosine distance between their final hidden states.
Rows are 512-token document chunks read through the model's vocabulary.
`WARMUP_TOKENS` sets the budget (130M by default) and `WARMUP_HOURS` the
deadline; rerunning the cell resumes from its last checkpoint. The result
is one encoder file whose digest joins the curriculum's run identity.

## Vocabulary

The model reads with 12,000 tokens instead of the encoder's 65,536.
`vocabulary-v1.json` pins them: the dataset's most frequent ASCII tokens
(each seen at least 239 times), plus every byte, added and
merge-intermediate token, so any text still encodes; rare words take a
few more pieces. The experiment's `select_vocab.py` picked them from
dataset revision `f4d88a6`. Curly quotes and en and em dashes become
their ASCII forms in the tokenizer's normalizer, one character for one,
so offsets still point into the original text.

`vocabulary.py` builds that tokenizer from the dataset's. Training keeps
the full frozen embedding table and maps encodings back to original IDs;
records are first checked against the dataset's saved tokens, then
compiled and sized with the vocabulary. Bundles keep only the 12,000
rows and ship the trimmed tokenizer, whose marker tokens have readable
names but new IDs: `data.device_markers` finds them by name.

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
template `toolcall-pointer/2`.

## Shared pieces

`composition.MODEL` plugs into MagicBox's `train.py` through
`train.Model`: corpus, extra parameters, forward, initialization, folding
and token renames. `train.py` here runs the curriculum on top of
MagicBox's run composition. The dataset itself is built outside this
repository, in the workspace experiment folder.

```sh
uv run --no-sync python -m pytest tests/examples/test_toolcalls.py tests/examples/test_toolcalls_notebook.py
```
