# Reusable model layers

Composition of kernels into reusable parameterized layers, with explicit parameter layout, state transitions, masks, and precision. Shared prefill and decode behavior belongs here when the semantics match. Model families should call these owners so an accepted kernel or layer improvement reaches every declared consumer.
