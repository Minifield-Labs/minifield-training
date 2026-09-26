# Duplication and size policy

Remove duplicated ownership before adding a new consumer of shared behavior.
Run `uv run --no-sync python scripts/check_duplicates.py`; the full quality
command runs this gate too. The gate has no baseline, exclusion list, inline
suppression or automatic acceptance mode.

## What the gate checks

The gate parses every `.py` file under `src/minifield_training/`, including
package initializers. It compares ordinary functions, methods, asynchronous
functions and nested functions within and across files. Every matching location
appears in the diagnostic with its fingerprint.

Two substantial functions match when their Python ASTs are identical after
removing their own function name, leading docstring, and source positions.
Formatting and comments have no effect. Constants, attributes, argument/local
names, annotations, defaults, decorators and synchronous/asynchronous behavior
remain significant. Local variables aren't renamed: Python reflection, keyword
arguments, closures and shadowing make careless name normalization unsound.

Named record contracts are checked independently of function size. Classes
with the same name and the same nonempty set of directly annotated fields fail
even when their annotations, defaults, field order or methods differ. This
catches copied contracts such as `PhysicalUpdate` before their types drift.
It covers dataclasses, named tuples and typed dictionaries written as classes,
without depending on decorator spelling. Distinct class names or field sets
still require review for semantic overlap.

`duplication.toml` contains the versioned size limits:

| Setting | Current limit | Meaning |
| --- | --- | --- |
| `min_clone_statements` | 12 | Minimum body size for duplicate comparison |
| `max_function_statements` | 70 | Maximum descendant statements per function |
| `max_module_statements` | 350 | Maximum statements per module |

A statement is an `ast.stmt` node. This includes imports, definitions, compound
statements and the statements inside them. Leading docstrings in all scopes
are excluded. A function excludes its own `def` statement from its count;
the containing module includes it. Nested function bodies count toward their
outer function's size because nesting mustn't bypass the size limit. Blank
lines, comments and multiline expression formatting don't change the count.

Unknown/missing policy fields, unsupported versions, boolean limits, invalid
limit ordering, missing policy/source, invalid encoding and syntax errors fail
the gate. Exactly meeting a size limit passes. Source files must stay inside
the repository and cannot be symlinked.

## What still needs a reviewer

Function detection uses conservative exact structural matching. Renamed locals, changed
signatures, partial function copies, equivalent algorithms and giant expressions
can escape it. Pylint's textual similarity check provides another signal. Neither
check establishes a useful abstraction or replaces review of the dependency
rules. Do not rename variables, insert dead statements or split arbitrary
fragments to silence a failure.

When a clone fails, name the shared responsibility and its lowest valid owner.
Move the behavior into that owner, test its contract, then make every consumer
call it. Keep model equations, objective mathematics and execution
lifecycle in their separate owners. Similar code with different contracts may
need separate implementations; establish the behavioral difference explicitly.

Independent numerical oracles belong under `tests/`, outside this production
scan. They should use independently specified mathematics and acceptance bounds.
Sharing the optimized implementation with its oracle would destroy the check.

There are currently **no duplication exemptions**. If an independently justified
production oracle eventually requires one, first propose a separately reviewed
policy change and decision record. Any future exemption mechanism must require
the exact fingerprint, exactly 2 repository-relative function locations, a
nonblank reason and an existing checked-in decision document. It must reject
stale entries, wildcards and broad path exclusions. Don't raise thresholds or
extend the checker during a component change to get that component through.
