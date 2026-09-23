# Documentation ownership

Each document has one purpose. Keep operational contracts close to the component
that implements them, and record dated evidence against the revision tested.

| Document | Owns |
| --- | --- |
| [Repository README](../README.md) | Setup and entry points |
| [Repository rules](../AGENTS.md) | Instructions for engineers and coding agents |
| [Architecture](architecture.md) | Responsibilities and enforced import boundaries |
| [Duplication policy](duplication.md) | Clone detection, size limits and review limits |
| [Procedure](procedure.md) | Required checks and completion criteria |

Each directory under `src/minifield_training/` has a README naming its
owner. With its first implementation, update that README with public contracts,
working examples, consumers, tests, compatibility and qualification status.
Avoid copying implementation details into several competing guides.

Use dated reports under `docs/validation/` when numerical or device checks
produce evidence. Reports must identify code revisions, environment, method,
results, limits and unrun checks. No report exists merely because a directory is
reserved.

The repository guard checks local Markdown link destinations, module ownership
READMEs, and prohibited future imports. It doesn't infer whether prose matches
the implementation. Review examples and capability claims alongside every
changed contract.
