# Model families

Family configurations, checkpoint parameter mapping, block-stack assembly,
and cache adapters. Families compose `layers` and `kernels`; checkpoint
names and model configs never cross into them. A model module is mostly a
mapping from saved parameter names to shared weight packs plus the code that
decides which layer goes where.

`strategies.pretrained.load_verified` composes this family's exact expected
shapes with the shared tensor reader. Its default source is the pinned
`LiquidAI/LFM2.5-230M-Base` revision
`9d2be5519834990d30996f878b6771cccbd24f2c`. Obtain `config.json`,
`tokenizer.json`, and `model.safetensors` from that revision into one directory,
then call `load_verified(directory)`. The pinned SHA-256 values verify each file
before the 132 BF16 backbone tensors become FP32 masters. The source is covered
by Liquid AI's LFM Open License v1.0. This repository doesn't redistribute its
weights or tokenizer. Forward numerical parity against the released weights is
still unverified here.
