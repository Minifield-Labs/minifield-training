# Conversation admission

`labeled.LabeledSequence` is a product-neutral record for one complete
tokenized observation, hard label, stable record ID, and group ID. The
classification batcher validates lengths, token IDs, and allowed labels before
device transfer. Product-specific serialization and tokenizer calls remain in
the consuming experiment.

`read_jsonl(path)` streams one neutral JSON envelope per line. Each envelope has
an `id`, a caller-assigned `source_group`, ordered `messages`, and optional
`tools` definitions. A message has `role` (`system`, `user`, `assistant`, or
`tool`) and string `content`. Assistant `tool_calls` contain unique `id`, `name`,
and JSON object `arguments`; tool replies carry the matching `tool_call_id`.
The caller may set a fixed `source_group` for a plain export.

```json
{"id":"demo-1","source_group":"document-1","messages":[{"role":"user","content":"Find records"},{"role":"assistant","content":"Found 2."}]}
```

Admission rejects malformed JSON, duplicate keys, nonfinite numbers, malformed
roles, unmatched tool replies, and unknown envelope fields. Errors include the
source filename and line number without quoting payloads. IDs and groups stay
separate from model-visible messages. The reader holds one line at a time;
callers supply a suitably bounded stream or file.

## Group-safe preparation

`prepare(records, mode="all" | "turn", seed="run-1")` assigns train or
validation from the caller's source group before expanding assistant turns.
`all` selects every assistant message in a complete conversation. `turn`
yields a separate example for each assistant message with the complete prior
context and no later messages. Both preserve tool definitions. The returned
`Example.targets` are message indices, which a template adapter maps to token
spans. Exact duplicate model-visible conversations from different groups fail
closed. Holdout assignment is stable under input reordering. The default
validation fraction is 0.1; callers can set it explicitly.

## Template tokenization

`tokenize_example` accepts a caller-supplied tokenizer implementing
`apply_chat_template`, plus explicit tokenizer and template identity strings.
A pinned Hugging Face tokenizer can be supplied through the optional `text`
extra. Training examples use `tokenize=True` and
`add_generation_prompt=False`. The adapter masks complete selected assistant
messages, including template control and end tokens. It verifies each prefix
against the full token sequence and rejects templates whose tokenization
changes earlier tokens when another message is appended. Tool-bearing examples
require `audited_tool_template=True`. Admission probes each
tool-definition leaf, call ID, name, argument leaf, and reply ID; every change
must alter both rendered text and token IDs. These probes catch dropped fields,
but cannot prove arbitrary Jinja semantics. Callers must pin the template and
maintain exact golden tests for role, tool, and end-marker behavior.

Examples exceeding `max_tokens` either raise or return `None` with
`overlength="drop"`; no sequence is cut.

## Verified local reuse

`save_prepared(directory, examples, source=..., tokenizer_asset=...,
template_asset=..., settings=...)` writes content-named JSONL and publishes a
versioned manifest only after validation. `iter_prepared` recomputes SHA-256
from the actual source JSONL, tokenizer serialization, and template file,
checks the requested preparation settings, then verifies payload bytes, schema,
IDs, source-group split assignments, lengths, and masks before yielding rows.
Use absolute, symlink-free local paths. Store artifacts outside Git. Asset
identity is a byte digest; labels alone don't authorize cache reuse. Callers
must serialize the exact tokenizer and template used for tokenization into the
supplied files and use their digests as `tokenizer_id` and `template_id`. For
Hugging Face fast tokenizers, the offline fixture captures backend tokenizer
JSON, live `special_tokens_map`, and `split_special_tokens` in canonical JSON.
Capture any further tokenizer configuration that the chosen template reads.
No remote artifact registry or in-place mutation protection is included.

The bounded synthetic acceptance path is executable with
`uv run --no-sync pytest tests/datasets/test_tokenization.py::test_full_jsonl_to_replayed_sft_update`.
It reads JSONL, prepares and tokenizes with an offline local vocabulary,
replays a verified artifact, builds a padded update, and decreases selected
next-token loss on the public tiny LFM2.5 model. This is CPU evidence only.

## Schema-conditioned records

`fields.Encoding`, `Field`, and `Record` describe token offsets, independently
encoded schema rows, and separately stored supervision. `Kind` identifies
extraction, categorical, binary, and ordinal tasks. `validate` checks configured
vocabulary bounds, field grouping, normalized targets, and consistent extraction
presence/span supervision. It imposes no BOS ID, tokenizer, template, product
wire format, or context limit.

`aligned_span` preserves exact character boundaries and overlapping Unicode
byte tokens. `trim_offsets` provides the explicit whitespace-edge policy;
`probabilities` admits finite hard/soft target distributions. Product adapters
own request keys, schema wording, release pins, and their special-token rules.
The package stays CPU-only and imports no optional tokenizer/Arrow libraries.

## Dataset contract snapshots

Keep each dataset's format documentation and snapshot identity in its own
directory here. The MagicBox dataset's [v1 format](magicbox/format-v1.md) and
[producer revision/checksum](magicbox/format-v1.json) define its published
record contract.

## Pointer questions

`pointer.Record` holds one source and the questions asked about it. Each
`pointer.Question` has a query token run and the `Option`s it may select;
`ids[0]` of a query or option is its marker token. Choice, ordinal, and binary
questions carry a target distribution over their options. An extraction
question has one option meaning "not stated": `targets[0]` is its probability,
and `span` marks the answer's `[start, end)` source tokens otherwise.
Unsupervised questions carry zero targets. `pointer.validate` admits token
IDs, option inventories, distributions, and selectable spans.

`pointer.ordinal_targets(level, count, width)` spreads a hard ordinal label
over nearby levels with a Gaussian whose standard deviation is `width` times
the scale's range. Zero width or one level stays one-hot. Tests check
hand-computed weights and admission failures.
