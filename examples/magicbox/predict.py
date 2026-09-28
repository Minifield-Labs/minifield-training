"""Load a trained bundle and turn public JSON requests into typed results."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from examples.magicbox import source
from examples.magicbox import tokenizer
from minifield_training.batching import magicbox as batching
from minifield_training.datasets import magicbox as data
from minifield_training.evaluation import magicbox as decoding
from minifield_training.strategies import magicbox
from minifield_training.strategies import magicbox_bundle


def main() -> None:
    """Run one request file through restored encoder, heads, and decoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    request = data.object_map(json.loads(args.request.read_text()))
    if request.get("questions") == {}:
        print("{}")
        return
    cfg, fusion, parameters, decode_config = magicbox_bundle.load(args.bundle)
    adapter = tokenizer.Adapter(args.bundle / "tokenizer")
    record = data.compile_record("inference", request, {}, adapter.encode)
    maximum = batching.Shape(
        1,
        1,
        8192,
        8192,
        max(sum(len(field.rows) for field in record.fields), 4),
    )
    packed = batching.build(
        [record],
        source.bucket([record], maximum),
        seed=0,
        update=0,
        allow_unsupervised=True,
    )
    batch = {
        key: jnp.asarray(value[0]) for key, value in packed.microbatches.items()
    }
    outputs = magicbox.forward(
        parameters, cfg, fusion, batch, bf16=not args.fp32
    )
    arrays = {key: np.asarray(value)[0] for key, value in outputs.items()}
    predictions = decoding.decode(
        record,
        {
            key: value.tolist()
            for key, value in arrays.items()
            if key != "tokens"
        },
        arrays["tokens"].tolist(),
        presence_threshold=float(str(decode_config["presence_threshold"])),
    )
    print(
        json.dumps(
            decoding.format_results(record, predictions),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
