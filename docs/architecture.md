# Architecture and dependency ownership

Owner: repository governance.
Executable authority: [architecture.toml](../architecture.toml).
Verification: `uv run --no-sync python scripts/check_architecture.py` and
`uv run --no-sync pytest tests/governance/test_architecture.py`.

Directories reserve ownership; a README in a reserved directory promises no
implemented capability.

## Put each responsibility in one place

| Owner | Responsibility | Boundary |
| --- | --- | --- |
| `core` | Immutable metadata, identifiers, JSON contracts, CPU protocols | Standard library only; no array aliases or numerical imports |
| `kernels` | Numerical primitives, precision/mask contracts, backend selection, array aliases in `types.py` | No model, objective, optimizer or engine dependencies |
| `layers` | Shared parameterized layer composition | Core and kernels; no model-family policy |
| `models` | Family configurations, parameter mapping, block composition and cache adapters | Shared numerical model interface in `contracts.py`; each family owns its differences |
| `optimizers` | Update mathematics and numerical optimizer state | Core and kernels; no model inventories or strategy state |
| `objectives` | Supervision, loss normalization, sufficient statistics | Core, kernels and neutral model contracts; no concrete model family |
| `artifacts` | CPU manifests, hashes, admission, atomic file lifecycle | Core; accelerator-free inspection |
| `datasets` | Admission, source/split identity, serializers, tokenizer protocols, verified token storage | Core and artifacts; no model or objective implementation |
| `batching` | Packing/window mechanics, physical batches, deterministic cursor | Core, datasets and kernels |
| `checkpoints` | Numerical state writing/restoring, explicit compatibility mappings | Core, kernels, optimizer state and artifacts |
| `evaluation` | Evaluation orchestration and outcome contracts | Neutral model/objective/data interfaces; no family implementation |
| `engine` | Host training lifecycle, update transactions, cadence, cancellation and resume | Shared interfaces; receives concrete implementations as arguments |
| `strategies` | Composition of model, objective, optimizer, data, evaluation and engine | Concrete model selection belongs here; avoid copied loops and helpers |
| `execution` | Wire normalization, local CLI and worker dispatch | CPU-safe by default; exact lazy dispatch edges require policy entries |
| `supervisor` | Worker claim/child coordination and reporting | Core, CPU artifacts and execution; Backend owns durable job state |

The complete permitted dependency lists live in `architecture.toml`; this table
explains their purpose. Adding a directory never implicitly grants dependencies.
Checkpoint modules use the optional safetensors package to serialize FP32 state;
the base wheel still has no mandatory dependencies. Model families own
pretrained release metadata and adapters implementing `models.contracts.PretrainedModel`.
The generic pretrained loader receives an explicit source and adapter.
The checker rejects cycles in those owner rules, unknown owners and source files
at the package root (apart from its docstring-only initializer).

Concrete model families cannot import one another. Generic engines, objectives
and evaluators import `models.contracts`, and receive model implementations from
strategy composition. Shared model contracts cannot import a concrete family.
Use `kernels/types.py` for shared numerical aliases and `optimizers/state.py` for
optimizer-specific state. Keep the host-only `core` useful without JAX installed.

Batching contracts live in `batching.contracts`: one `PhysicalUpdate`, a
`BatchStrategy` for epoch iteration, and a `BatchSource` for replayable streams.
The dense strategy shares observation admission, ordering and padding, while
`TargetEncoder` implementations supply task-specific supervision. The engine
receives these interfaces and has no concrete batching or dataset-record import.

All package initializers contain only a docstring. Import concrete modules;
re-export layers obscure ownership and trigger import-time work. Add a
docstring-only `__init__.py` when a reserved directory gets its first Python
implementation. Setuptools intentionally excludes bare namespace directories
and keeps ownership READMEs in the source archive, outside the installed wheel.

## Enforced import rules

The AST checker reads every Python file in `src/minifield_training` without
importing it. Python files in other `src` packages and symlinks under `src` are
rejected. Absolute imports, relative imports, function-local imports, conditional
imports and `TYPE_CHECKING` imports receive the same ownership checks.

`core`, `datasets`, `artifacts`, `execution` and `supervisor` are host-safe. Every
source edge from those owners must stay in host-safe owners. Their external
imports default to standard-library modules only. A reviewed owner-specific
`external` list can admit an installed CPU dependency, with its import behavior
tested independently. NumPy may be admitted for CPU token storage when needed;
accelerator frameworks such as JAX, Torch and Triton can't enter that allowlist.
Checking every edge prevents a numerical dependency from hiding one module away.

The checker rejects import-loader APIs, dynamic Python execution, package-root
or wildcard internal imports, source-path wiring outside the checkout,
`PYTHONPATH` access, and direct/aliased `sys.path` or import-cache access.
Runtime discovery must use concrete modules and an explicit registry.

One narrow exception is available for a worker dispatch adapter. Add an exact
`execution.<module>` to `strategies.<module>` pair to `lazy_imports`, and use a
literal import inside a function in that exact source module. The destination
must resolve to a real module. An eager import or unused allowance fails.
No such exceptions exist yet. Dispatch functions execute only after
metadata/discovery has finished and the worker deliberately starts computation.
Test CPU discovery in a clean process before accepting an adapter.

Missing or malformed policy, unknown policy keys, duplicate names, owner-rule
cycles, unresolved internal imports and unused lazy allowances fail closed.
Policy edits receive the same review as code. Don't widen a rule simply because
a new implementation wants a forbidden dependency. Extract or inject the
missing interface first.

## Run and extend the guard

```sh
uv run --no-sync python scripts/check_architecture.py
uv run --no-sync python scripts/check_architecture.py --root /path/to/checkout
uv run --no-sync pytest tests/governance/test_architecture.py
```

The script exports `inspect(root: pathlib.Path) -> list[Diagnostic]`. It performs
read-only inspection, returns sorted diagnostics, and never loads the source
package. `main()` prints `path:line:message` violations and exits with status 1
when any exist. The repository quality gate invokes this same implementation.

When changing an import rule, add fixtures for its intended allowed use and a
real forbidden form. Keep relative, aliased, function-local and type-only forms
covered. Run the complete quality gate after the targeted tests.

## What still needs human and runtime review

Static ownership checks don't establish numerical equivalence, performance,
artifact compatibility or useful abstractions. Shared kernels need independent
output/gradient/state tests and target-device evidence for each semantic mode.
Source deduplication has its own [gate and limits](duplication.md).

The checker follows repository import syntax, not arbitrary Python data flow or
installed third-party internals. It is a development guard, not a sandbox against
hostile code. An allowed lazy adapter can still be called eagerly by another
function; review call sites and test import/discovery in a fresh CPU process.
Reflective or generated imports, native loaders and source-path tricks violate
the architecture even when a novel spelling escapes static detection. Extend a
specific regression fixture when such a pattern appears.

Array math inside the wrong module can satisfy import checks. Reviews must also
confirm that a strategy composes shared services, an optimizer accepts neutral
state, and a new model reuses qualified kernel/layer contracts. Preserve distinct
mathematics explicitly when their precision or masking semantics differ.
