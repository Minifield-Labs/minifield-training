"""Load a trained pointer bundle and turn public JSON requests into answers."""

import argparse
import json
from pathlib import Path

from examples.magicbox import bundle as magicbox_bundle
from examples.magicbox import composition as magicbox
from examples.magicbox import data
from examples.magicbox import tokenizer
from minifield_training.batching import pointer as batching
from minifield_training.core import json_io
from minifield_training.evaluation import pointer as evaluation
from minifield_training.objectives import schema_fields as objective


def main() -> None:
    """Run one request file through the restored encoder and pointers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    request = json_io.object_map(json.loads(args.request.read_text()))
    if request.get("questions") == {}:
        print("{}")
        return
    cfg, head, parameters, decode_config = magicbox_bundle.load_pointer(
        args.bundle
    )
    adapter = tokenizer.Adapter(args.bundle / "tokenizer")
    record = data.compile_pointer_record(
        "inference", request, {}, adapter.encode
    )
    predictor = evaluation.Predictor(
        magicbox.bind_pointer(cfg, head, bf16=not args.fp32),
        batching.PointerBatchStrategy(
            batching.Shape(
                1,
                1,
                record.sequence_tokens,
                len(record.questions),
                cfg.vocab_size,
                0,
            ),
            objective.balance_types,
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
