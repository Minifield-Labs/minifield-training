"""Compose named QAT selection with a numerical fake quantizer."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import jax

from minifield_training.core import parameters
from minifield_training.core import quantization as core_quantization
from minifield_training.kernels import quantization as kernels
from minifield_training.kernels import types

_ELIGIBLE = frozenset({"projection"})


def _check_selection(
    inventory: parameters.FullParameterInventory,
    roles: Mapping[str, str],
    selected: frozenset[str],
) -> None:
    """Enforce protected roles and shape bounds for any injected strategy."""
    specs = {spec.name: spec for spec in inventory.specs}
    if set(roles) != set(specs):
        raise ValueError("Quantization roles must cover exact inventory")
    if selected.difference(specs):
        raise ValueError("Quantization selects unknown tensor")
    for name in selected:
        spec = specs[name]
        if (
            roles[name] not in _ELIGIBLE
            or not spec.trainable
            or len(spec.shape) != 2
            or 0 in spec.shape
        ):
            raise ValueError(f"Ineligible quantized tensor: {name}")


class QuantizationPlan(core_quantization.QuantizationStrategy, Protocol):
    """Inject exact selection and numerical effective-weight behavior."""

    @property
    def names(self) -> frozenset[str]:
        """Return the exact selected tensor names."""

    def effective(self, weight: jax.Array) -> jax.Array:
        """Return effective forward values with the chosen gradient."""


@dataclass(frozen=True)
class NamedQuantization:
    """Select only explicitly named projection roles before JIT."""

    quantizer: kernels.Quantizer
    names: frozenset[str]

    @property
    def identity(self) -> str:
        """Identify quantizer and exact resolved names in resume metadata."""
        return self.quantizer.identity

    def select(
        self,
        inventory: parameters.FullParameterInventory,
        roles: Mapping[str, str],
    ) -> frozenset[str]:
        """Reject typos, protected roles, frozen leaves, and invalid shapes."""
        _check_selection(inventory, roles, self.names)
        shapes = {spec.name: spec.shape for spec in inventory.specs}
        for name in self.names:
            self.quantizer.validate_shape(shapes[name])
        return self.names

    def effective(self, weight: jax.Array) -> jax.Array:
        """Delegate numerical behavior to the injected quantizer."""
        return self.quantizer.effective(weight)


def select(
    inventory: parameters.FullParameterInventory,
    strategy: QuantizationPlan,
    roles: Mapping[str, str],
) -> frozenset[str]:
    """Resolve and enforce exact eligible names before building metadata."""
    selected = strategy.select(inventory, roles)
    if selected != strategy.names:
        raise ValueError("Quantization selection differs from declared names")
    _check_selection(inventory, roles, selected)
    return selected


def validate_plan(
    inventory: parameters.FullParameterInventory,
    strategy: QuantizationPlan | None,
    roles: Mapping[str, str] | None = None,
) -> None:
    """Reject a missing or changed plan before any forward computation."""
    selected = frozenset(
        spec.name for spec in inventory.specs if spec.quantized
    )
    if strategy is None:
        if selected or inventory.quantization_profile is not None:
            raise ValueError("QAT inventory requires quantization plan")
        return
    if (
        inventory.quantization_profile != strategy.identity
        or selected != strategy.names
    ):
        raise ValueError("Quantization plan differs from inventory")
    if roles is not None and select(inventory, strategy, roles) != selected:
        raise ValueError("Quantization selection differs from inventory")


def apply(
    masters: types.Parameters,
    inventory: parameters.FullParameterInventory,
    strategy: QuantizationPlan | None,
) -> types.Parameters:
    """Apply one plan to training, evaluation, and final output weights."""
    validate_plan(inventory, strategy)
    if strategy is None:
        return masters
    return {
        name: strategy.effective(value) if name in strategy.names else value
        for name, value in masters.items()
    }
