# Model families

Family configurations, checkpoint parameter mapping, block-stack assembly,
and cache adapters. Families compose `layers` and `kernels`; checkpoint
names and model configs never cross into them. A model module is mostly a
mapping from saved parameter names to shared weight packs plus the code that
decides which layer goes where.
