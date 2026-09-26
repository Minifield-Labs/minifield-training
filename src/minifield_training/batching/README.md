# Dense SFT batch construction

`iter_updates` takes validated `TokenizedExample` values and explicit
`microbatches`, `rows_per_microbatch`, `sequence_length`, `pad_token_id`,
`vocab_size`, and `seed`. It shuffles deterministically, emits one real example
per row, and pads partial updates with finite inert rows. The returned arrays
have shape `[M, B, T]` and match `engine.step.make_step`'s keys and boolean
`active[M]` vector. `start_update` resumes at a complete update boundary in
the same seeded order. The real `example_ids` are returned for audit.

No sequence packing is implemented. The current dense causal objective doesn't
carry segment boundaries; concatenating examples into one row would allow
cross-example attention and corrupt supervision. Use a segment-aware forward
and objective before adding packing.

`classification.iter_updates` accepts neutral `datasets.labeled.LabeledSequence`
records with stable ID, group ID, complete token IDs, and one hard label. It
checks token range, length, allowed label, and duplicate IDs before yielding
fixed `[M, B, T]` input/mask arrays plus `[M, B]` labels and valid-row mask.
Partial rows can carry a masked padding label safely. The seeded shuffle and
`start_update` identify a repeatable update boundary within one epoch.
The boolean `active[M]` vector stays on the host for the streaming step's slot
selection; the model inputs, masks, labels, and valid-row mask are JAX arrays.
