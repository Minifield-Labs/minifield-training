# MagicBox training records, version 1.0

One logical record contains one source string and a dynamic set of field
definitions. The model reads only the `request` object. Supervision, provenance,
split membership, and token alignment are separate.

```json
{
  "format": "minifield.magicbox/1.0",
  "id": "sha256-record-identity",
  "group_id": "sha256-connected-document-group",
  "split": "train",
  "request": {
    "state": "Email Alice at alice@example.org.",
    "questions": {
      "contact": {
        "type": "extract",
        "instructions": "Extract the email address."
      }
    }
  },
  "targets": {
    "contact": {
      "has_answer": true,
      "span": [15, 32],
      "text": "alice@example.org"
    }
  },
  "encoding": {
    "source_tokens": 9,
    "max_schema_tokens": 13,
    "token_spans": {"contact": [4, 8]}
  },
  "provenance": {
    "source": {"dataset": "synthetic/example", "revision": "example"},
    "conversion": {"window": [0, 33], "window_index": 0, "field_chunk": 0}
  }
}
```

The builder computes and verifies token counts and intervals for every output row.
Character positions use Python-string code points and half-open intervals.
The source slice at `[start:end]` must equal `target.text` byte for byte when
encoded as UTF-8. Repeated substrings remain distinct occurrences.

## Schema and targets

| Type | Public field | Private target |
| --- | --- | --- |
| `extract` | `type`, `instructions` | Present: `{"has_answer":true,"span":[start,end],"text":"exact slice"}`; absent: `{"has_answer":false,"span":null,"text":null}` |
| `choice` | `type`, `instructions`, `criteria:{id:description}` | `{"choice":"id"}` or `{"probabilities":{"id":0.8,"other":0.2}}` |
| `noul` | `type`, `instructions`, optional `criteria:{true:description,false:description}` | `{"probability":0.0}` through `1.0` |
| `score` | `type`, `instructions`, ordered `criteria:[description,...]` | `{"level":1}` or `{"probabilities":[0.0,0.1,0.9]}` |

Missing supervision stays missing. The builder currently removes unsupervised
questions from exported training records. Unknown binary values are never
converted to false. Soft distributions retain every option and must sum to 1;
source rounding within 1e-5 is normalized and recorded. A fractional expected
score alone cannot supply a training distribution.

Each choice selects from its actual supplied inventory. Local nullable enum
labels add a documented `__not_specified__` option. This is part of that
converted schema. No implicit abstention class is bolted onto other sources.

Confidence fields and teacher factors are excluded from targets and model
inputs. The trainer computes score expectation from the level distribution.

## Physical Parquet schema

All columns are UTF-8 strings:

```text
format, id, group_id, split,
request_json, targets_json, provenance_json, encoding_json
```

The 4 JSON columns contain canonical JSON, preserving arbitrary field and
candidate keys without creating new Parquet columns for every schema. This
avoids dependence on an accelerator framework or a fixed field inventory.

```python
import json
import pyarrow.parquet as pq

for batch in pq.ParquetFile(shard_path).iter_batches(batch_size=256):
    for row in batch.to_pylist():
        request = json.loads(row["request_json"])
        targets = json.loads(row["targets_json"])
        # Only request goes into the source/schema encoders.
```

## Tokenizer and compiler contract

The tokenizer comes from `LiquidAI/LFM2.5-Encoder-350M` revision
`b886781f7c6f10ca9b7096e21b83e30a073c2f39`. The notebook downloads only
`tokenizer.json`. It doesn't import that model's remote code.

Offset policy `trim-text-preserve-whitespace/2` reads the tokenizer's native
untrimmed offsets. It removes whitespace from the margins of text-bearing
tokens and preserves original ranges for whitespace-only tokens. Token IDs and
embeddings retain their pretrained meaning. This makes `ĠAlice` align to the
exact name while keeping interior spaces in amounts and phone numbers
selectable. `magicbox_data.tokenization.encode_with_offsets` owns this policy.
Training and runtime must reuse it together with the saved tokenizer JSON and
`contract.json`; loading the tokenizer JSON alone doesn't apply the adapter.

No gold boundaries are snapped or strings normalized to pass admission. Spans
inside indivisible tokens, spans crossing zero-length offsets, and
ambiguous offsets are rejected and counted. All byte tokens for a complete
Unicode character must be included. Extraction copies original source slices.

Native BOS token ID 1 is the source sentinel and schema readout. The saved
tokenizer inserts it once per sequence. Source attention includes it;
extraction excludes it. Source and schema limits include all native special
tokens. The operational cap is at most 8,192, matching the encoder's published
trained context rather than its larger positional configuration.

`magicbox_data.contract.schema_rows` owns `magicbox-rows/1`. It serializes the
spec's type/question text, one choice candidate or score level per row. Score
rows include their semantic zero-based level and level count. Choice rows
contain no arbitrary ordinal or option count. These exact templates must be
reused or intentionally versioned by the JAX compiler.

## Conversion and evaluation rules

NER annotations provide extraction labels. For sparse labels, instructions ask
for a mention of the category. The earliest annotated valid mention supplies
the deterministic training target; known alternatives stay in private
`conversion.acceptable_spans` metadata for evaluation. Missing annotations
don't create negative labels. Unsupported discontinuous spans are omitted.

NER windows preserve literal substrings and record their original character
range. A cut entity isn't turned into a null answer. Direct classification
fields require their full context; biomedical relation propositions require
their entire annotated passage. No partial document is assigned a whole-document
decision label.

Source groups are joined through upstream document identity and normalized
text fingerprints. Native held-out partitions are preserved, with conflicts
resolved conservatively: OOD, test, calibration, validation, then train.
Train-only groups receive deterministic 98/1/1 train/validation/test allocation.
Final window text is grouped again before export so shared text under different
questions can't cross splits. Exact request/target duplicates collapse;
identical requests with conflicting labels are quarantined.

This grouping detects normalized exact duplication and supplied document
lineage. Semantic near-duplicates and shared generation templates without
lineage remain evaluation concerns. The source report retains native split
and workflow provenance for stricter future holdouts.

Source mixes aren't padded with invented labels. Read `report.json` to choose
per-source and per-type training sampling weights. Preserve separate metrics
for all 4 types, extraction presence, known alternative spans, and native OOD
partitions. CPU preparation doesn't establish model accuracy.
