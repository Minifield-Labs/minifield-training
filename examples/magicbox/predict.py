"""Load a trained bundle and turn public JSON requests into typed results."""

import argparse
import json
from pathlib import Path

from examples.magicbox import bundle as magicbox_bundle
from examples.magicbox import composition as magicbox
from examples.magicbox import data
from examples.magicbox import tokenizer
from minifield_training.batching import schema_fields as batching
from minifield_training.core import json_io
from minifield_training.evaluation import schema_fields as evaluation
from minifield_training.models.lfm2_5 import encoder
from minifield_training.objectives import schema_fields as objective


def main() -> None:
    """Run one request file through restored encoder, heads, and decoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    request = json_io.object_map(json.loads(args.request.read_text()))
    if request.get("questions") == {}:
        print("{}")
        return
    cfg, fusion, parameters, decode_config = magicbox_bundle.load(args.bundle)
    adapter = tokenizer.Adapter(args.bundle / "tokenizer")
    record = data.compile_record("inference", request, {}, adapter.encode)
    maximum = batching.Shape(
        1,
        1,
        encoder.MAX_SEQUENCE_LENGTH,
        encoder.MAX_SEQUENCE_LENGTH,
        max(sum(len(field.rows) for field in record.fields), 4),
        cfg.vocab_size,
        0,
    )
    predictor = evaluation.Predictor(
        magicbox.bind(cfg, fusion, bf16=not args.fp32),
        batching.SchemaBatchStrategy(
            maximum, objective.balance_types, min_tokens=128, min_rows=4
        ),
        presence_threshold=float(str(decode_config["presence_threshold"])),
    )
    predictions, _ = predictor.score(parameters, record, include_losses=False)
    print(
        json.dumps(
            data.format_results(record, predictions),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
