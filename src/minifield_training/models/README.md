# Model families

Family configurations, checkpoint parameter mapping, block-stack assembly,
and cache adapters. Families compose `layers` and `kernels`; checkpoint
names and model configs never cross into them. A model module is mostly a
mapping from saved parameter names to shared weight packs plus the code that
decides which layer goes where.

`contracts.PretrainedModel[ConfigT]` defines config parsing, expected tensor
shapes, source dtype, and master validation. Each family implements that
interface beside its model code. `contracts.PretrainedSource` records immutable
file identities; the shared loader verifies them before admitting tensors.

`lfm2_5.pretrained` owns `Adapter` and the pinned `BASE` release:
`LiquidAI/LFM2.5-230M-Base` revision
`9d2be5519834990d30996f878b6771cccbd24f2c`. Obtain `config.json`,
`tokenizer.json`, and `model.safetensors` from that revision into one directory,
then explicitly compose the loader:

```python
from minifield_training.models.lfm2_5 import pretrained as lfm_source
from minifield_training.strategies import pretrained

cfg, parameters = pretrained.load_verified(
    directory, lfm_source.BASE, lfm_source.Adapter()
)
```

The pinned SHA-256 values verify each file
before the 132 BF16 backbone tensors become FP32 masters. The source is covered
by Liquid AI's LFM Open License v1.0. This repository doesn't redistribute its
weights or tokenizer. Forward numerical parity against the released weights is
still unverified here.
