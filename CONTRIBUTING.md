# Contributing

Read [AGENTS.md](AGENTS.md) before changing implementation. Work on one bounded
responsibility. Name its contract, existing consumers and destination owner
before writing code.

## Environment and commands

Use Python 3.12 and uv 0.11.30. The committed lockfile pins development tools.
Add a dependency only when a real component needs it.

```sh
uv sync --locked
git config --local core.hooksPath .githooks
uv run --no-sync python -m pyink --workers=1 src scripts tests
uv run --no-sync python -m ruff check src scripts tests
uv run --no-sync python scripts/check_structure.py
uv run --no-sync python -m pytest tests/governance
uv run --no-sync python scripts/check_quality.py
```

The complete gate checks lock consistency, structure, formatting, lint, strict
types, tests, source/wheel builds, installed file completeness, and clean-process
host imports from an independent wheel environment. Build dependency downloads
may need network access on the first run. CUDA qualification is a separate
explicit task.

The Google Pylint configuration retains its upstream attribution header. Its
defaults disable refactor messages, including duplicate-code. The complete gate
explicitly enables that message in a separate production-only pass with an
8-line similarity threshold. The [AST duplicate gate](docs/duplication.md)
supplies complementary coverage.

## Review

Update the destination README with the public contract, concrete example,
consumer list and acceptance tests. Preserve numerical and serialization
behavior unless a change deliberately versions it.

## Completion and enforcement limits

The [full procedure](docs/procedure.md) defines completion. Run meaningful tests,
then the full gate, review the diff, and commit conventional commits on `main`.
Preserve unrelated work. Document pending device checks without calling the
component a supported training route.

The pre-commit hook checks current worktree structure. It doesn't inspect the
Git index in isolation, so review partially staged changes carefully. The full
quality command and CI validate the complete checked-out tree.

Local hooks can be bypassed. When the repository gets a remote, require the
`CPU quality and package` CI job and review for changes to architecture,
duplication policy and quality scripts. Repository files cannot enable
server-side protection before a remote exists.
