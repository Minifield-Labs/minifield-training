"""Numerical optimizer state shared by update transactions."""

from collections.abc import Callable
from typing import TypedDict

import jax

from minifield_training.kernels import types


class State(TypedDict):
    """Float32 master parameters, Adam moments, and the optimizer step."""

    params: types.Parameters
    m: types.Parameters
    v: types.Parameters
    step: jax.Array


type Step = Callable[
    [State, types.DeviceBatch], tuple[State, dict[str, jax.Array]]
]
