"""Check source ownership and dependency boundaries without importing source."""

import argparse
import ast
import dataclasses
import pathlib
import re
import sys
import tomllib

_PACKAGE = "minifield_training"
_SOURCE = "src/minifield_training"
_ACCELERATOR = frozenset(
    {
        "jax",
        "jaxlib",
        "torch",
        "tensorflow",
        "flax",
        "optax",
        "qwix",
        "triton",
        "cupy",
    }
)
_LOADERS = frozenset(
    {"importlib", "runpy", "pkgutil", "zipimport", "site", "builtins", "ctypes"}
)
_DYNAMIC = frozenset({"__import__", "eval", "exec", "compile", "__builtins__"})


@dataclasses.dataclass(frozen=True, order=True)
class Diagnostic:
    """One actionable static-policy violation."""

    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


@dataclasses.dataclass(frozen=True)
class _Rule:
    depends: frozenset[str]
    host_safe: bool
    external: frozenset[str]


@dataclasses.dataclass(frozen=True)
class _Policy:
    rules: dict[str, _Rule]
    shared: frozenset[str]
    consumers: frozenset[str]
    lazy: frozenset[tuple[str, str]]


@dataclasses.dataclass(frozen=True)
class _Source:
    path: pathlib.Path
    relative: str
    module: str
    tree: ast.Module

    @property
    def owner(self) -> str:
        """Return the top-level owner, or the empty package root."""
        return self.module.partition(".")[0]


@dataclasses.dataclass(frozen=True)
class _Import:
    source: _Source
    target: str
    line: int
    lazy: bool


def _strings(value: object) -> frozenset[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and re.fullmatch(r"[a-z][a-z0-9_]*", item)
        for item in value
    ):
        raise ValueError("expected a list of simple module names")
    if len(value) != len(set(value)):
        raise ValueError("duplicate module name")
    return frozenset(value)


def _load_policy(root: pathlib.Path) -> _Policy:
    with (root / "architecture.toml").open("rb") as stream:
        raw = tomllib.load(stream)
    if set(raw) != {
        "version",
        "package",
        "source_root",
        "model_shared",
        "model_consumers",
        "lazy_imports",
        "modules",
    }:
        raise ValueError("missing or unknown architecture policy keys")
    if (
        not isinstance(raw["version"], int)
        or isinstance(raw["version"], bool)
        or raw["version"] != 1
    ):
        raise ValueError("unsupported architecture policy version")
    if raw["package"] != _PACKAGE or raw["source_root"] != _SOURCE:
        raise ValueError(
            "package and source_root must name src/minifield_training"
        )
    modules = raw["modules"]
    if not isinstance(modules, dict) or not modules:
        raise ValueError("modules must contain explicit owner rules")
    rules = {}
    for name, item in modules.items():
        _strings([name])
        if not isinstance(item, dict) or set(item) != {
            "depends",
            "host_safe",
            "external",
        }:
            raise ValueError(f"invalid rule for {name}")
        if not isinstance(item["host_safe"], bool):
            raise ValueError(f"host_safe must be boolean for {name}")
        rules[name] = _Rule(
            _strings(item["depends"]),
            item["host_safe"],
            _strings(item["external"]),
        )
    for name, rule in rules.items():
        if not rule.depends <= rules.keys() or name in rule.depends:
            raise ValueError(f"unknown or self dependency for {name}")
        if rule.external & (_ACCELERATOR | _LOADERS):
            raise ValueError(f"forbidden external allowlist for {name}")
    if "core" not in rules or rules["core"] != _Rule(
        frozenset(), True, frozenset()
    ):
        raise ValueError(
            "core must be host-safe, dependency-free and stdlib-only"
        )
    for name, rule in rules.items():
        pending = list(rule.depends)
        visited = set()
        while pending:
            target = pending.pop()
            if target == name:
                raise ValueError(f"dependency cycle involving {name}")
            if target not in visited:
                visited.add(target)
                pending.extend(rules[target].depends)
    consumers = _strings(raw["model_consumers"])
    if not consumers <= rules.keys():
        raise ValueError("unknown model consumer")
    lazy = _lazy_policy(raw["lazy_imports"], rules)
    return _Policy(rules, _strings(raw["model_shared"]), consumers, lazy)


def _lazy_policy(
    value: object, rules: dict[str, _Rule]
) -> frozenset[tuple[str, str]]:
    if not isinstance(value, list):
        raise ValueError("lazy_imports must be a list")
    result = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"source", "target"}:
            raise ValueError("lazy import needs exact source and target")
        source, target = item["source"], item["target"]
        if not all(
            isinstance(name, str)
            and re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+", name)
            for name in (source, target)
        ):
            raise ValueError(
                "lazy imports require exact qualified module names"
            )
        if (
            source.partition(".")[0] != "execution"
            or target.partition(".")[0] != "strategies"
        ):
            raise ValueError(
                "lazy edges are reserved for execution to strategies"
            )
        if "execution" not in rules or "strategies" not in rules:
            raise ValueError("lazy edge references missing owner")
        pair = (source, target)
        if pair in result:
            raise ValueError("duplicate lazy edge")
        result.add(pair)
    return frozenset(result)


def _sources(root: pathlib.Path) -> tuple[list[_Source], list[Diagnostic]]:
    base = root / _SOURCE
    sources = []
    errors = []
    if (root / "src").is_symlink():
        return [], [Diagnostic("src", 1, "symlinked source is forbidden")]
    if not base.is_dir():
        return [], [Diagnostic(_SOURCE, 1, "source package is missing")]
    for path in sorted((root / "src").rglob("*")):
        if path.is_symlink():
            errors.append(
                Diagnostic(
                    path.relative_to(root).as_posix(),
                    1,
                    "symlinked source is forbidden",
                )
            )
        elif path.suffix == ".py" and not path.is_relative_to(base):
            errors.append(
                Diagnostic(
                    path.relative_to(root).as_posix(),
                    1,
                    "source outside the owned package is forbidden",
                )
            )
    for path in sorted(base.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or base.is_symlink():
            continue
        module = (
            path.relative_to(base).with_suffix("").as_posix().replace("/", ".")
        )
        module = module.removesuffix(".__init__")
        if module == "__init__":
            module = ""
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"), filename=relative
            )
        except (SyntaxError, UnicodeError) as error:
            errors.append(
                Diagnostic(relative, 1, f"cannot parse source: {error}")
            )
            continue
        sources.append(_Source(path, relative, module, tree))
    if not (base / "__init__.py").is_file():
        errors.append(Diagnostic(_SOURCE, 1, "package __init__.py is required"))
    return sources, errors


def _imports(source: _Source) -> list[_Import]:
    result: list[_Import] = []
    parents = {
        child: node
        for node in ast.walk(source.tree)
        for child in ast.iter_child_nodes(node)
    }
    for node in ast.walk(source.tree):
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        ancestor: ast.AST = node
        lazy = False
        while ancestor in parents:
            ancestor = parents[ancestor]
            lazy = lazy or isinstance(
                ancestor, ast.FunctionDef | ast.AsyncFunctionDef
            )
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        else:
            prefix = node.module or ""
            if node.level:
                package = [_PACKAGE, *source.module.split(".")]
                if source.path.name != "__init__.py":
                    package.pop()
                if node.level > len(package):
                    prefix = "<invalid-relative>"
                else:
                    prefix = ".".join(
                        package[: len(package) - node.level + 1]
                        + ([prefix] if prefix else [])
                    )
            targets = [f"{prefix}.{alias.name}" for alias in node.names]
        result.extend(
            _Import(source, target, node.lineno, lazy) for target in targets
        )
    return result


def _internal(target: str) -> str | None:
    if target == _PACKAGE:
        return ""
    if target.startswith(f"{_PACKAGE}."):
        return target[len(_PACKAGE) + 1 :]
    return None


def _resolve(target: str, modules: set[str], packages: set[str]) -> str | None:
    original = target
    while target:
        if target in modules:
            if target in packages and target != original:
                return None
            return target
        target = target.rpartition(".")[0]
    return None


def _model_violation(source: str, target: str, policy: _Policy) -> bool:
    parts = target.split(".")
    if parts[0] != "models" or len(parts) < 2:
        return False
    family = parts[1]
    if family in policy.shared:
        return False
    origin = source.split(".")
    if origin[0] == "models":
        return len(origin) < 2 or origin[1] != family
    return origin[0] not in policy.consumers


def _edge_error(
    edge: _Import, policy: _Policy, modules: set[str], packages: set[str]
) -> str | None:
    target = _internal(edge.target)
    rule = policy.rules[edge.source.owner]
    if target is None:
        package = edge.target.partition(".")[0]
        if package in _LOADERS:
            return f"forbidden loader import: {edge.target}"
        if edge.target.startswith(
            ("sys.path", "sys.meta_path", "sys.modules", "sys.path_hooks")
        ):
            return f"import-resolution access is forbidden: {edge.target}"
        if package == "<invalid-relative>":
            return "relative import escapes the package"
        if (
            rule.host_safe
            and package not in sys.stdlib_module_names
            and package not in rule.external
        ):
            return (
                f"host-safe {edge.source.owner} cannot import external "
                f"{edge.target}"
            )
        return None
    if not target or target.endswith(".*"):
        return "opaque package-root or wildcard imports are forbidden"
    owner = target.partition(".")[0]
    if owner not in policy.rules:
        return f"import has no owner rule: {edge.target}"
    resolved = _resolve(target, modules, packages)
    if resolved is None:
        return f"internal import does not resolve to source: {edge.target}"
    if _model_violation(edge.source.module, target, policy):
        return f"concrete model import crosses ownership: {edge.target}"
    pair = (edge.source.module, resolved)
    if pair in policy.lazy:
        if not edge.lazy:
            return f"dispatch edge must be function-local: {edge.target}"
        return None
    if owner != edge.source.owner and owner not in rule.depends:
        return f"{edge.source.owner} cannot depend on {owner}: {edge.target}"
    if rule.host_safe and not policy.rules[owner].host_safe:
        return (
            f"host-safe {edge.source.owner} cannot reach numerical "
            f"owner {owner}"
        )
    return None


def _bypasses(source: _Source) -> list[Diagnostic]:
    errors = []
    aliases = {
        alias.asname or alias.name.partition(".")[0]: alias.name
        for node in ast.walk(source.tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for node in ast.walk(source.tree):
        message = None
        if isinstance(node, ast.Name) and node.id in _DYNAMIC:
            message = (
                f"dynamic execution/import mechanism is forbidden: {node.id}"
            )
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and aliases.get(node.value.id, node.value.id) == "sys"
            and node.attr
            in {
                "path",
                "meta_path",
                "modules",
                "path_hooks",
                "path_importer_cache",
                "__dict__",
            }
        ):
            message = f"import-resolution access is forbidden: sys.{node.attr}"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "vars"}
        ) and (
            node.args
            and isinstance(node.args[0], ast.Name)
            and aliases.get(node.args[0].id, node.args[0].id) == "sys"
        ):
            message = "reflection on sys can bypass import ownership"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.replace("\\", "/")
            if value == "PYTHONPATH" or any(
                marker in value
                for marker in (
                    "../",
                    "/src/minifield_training",
                )
            ):
                message = "external source/PYTHONPATH path wiring is forbidden"
        if message and isinstance(node, ast.expr):
            errors.append(Diagnostic(source.relative, node.lineno, message))
    return errors


def inspect(root: pathlib.Path) -> list[Diagnostic]:
    """Return deterministic violations without modifying or importing source."""
    try:
        policy = _load_policy(root)
    except (OSError, ValueError) as error:
        return [Diagnostic("architecture.toml", 1, str(error))]
    sources, errors = _sources(root)
    modules = {source.module for source in sources}
    packages = {
        source.module for source in sources if source.path.name == "__init__.py"
    }
    used_lazy = set()
    for source in sources:
        if source.path.name == "__init__.py" and any(
            not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
            for node in source.tree.body
        ):
            errors.append(
                Diagnostic(
                    source.relative,
                    1,
                    "package initializers must contain only a docstring",
                )
            )
        if not source.module:
            continue
        if source.owner not in policy.rules or (
            "." not in source.module and source.path.name != "__init__.py"
        ):
            errors.append(
                Diagnostic(
                    source.relative,
                    1,
                    "source module has no directory owner rule",
                )
            )
            continue
        errors.extend(_bypasses(source))
        for edge in _imports(source):
            message = _edge_error(edge, policy, modules, packages)
            if message:
                errors.append(Diagnostic(source.relative, edge.line, message))
            target = _internal(edge.target)
            resolved = _resolve(target, modules, packages) if target else None
            if resolved:
                used_lazy.add((source.module, resolved))
    for pair in sorted(policy.lazy - used_lazy):
        errors.append(
            Diagnostic(
                "architecture.toml", 1, f"unused lazy import edge: {pair}"
            )
        )
    return sorted(set(errors))


def main() -> int:
    """Run the gate against this checkout or an explicit fixture root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    errors = inspect(args.root)
    for error in errors:
        print(error)
    if not errors:
        print("Architecture boundaries passed.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
