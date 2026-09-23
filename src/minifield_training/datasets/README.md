# Conversation admission

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
