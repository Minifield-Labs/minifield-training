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
