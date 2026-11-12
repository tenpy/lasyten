"""The spaces, i.e. the legs of a tensor."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

import abc

from .._core import (  # noqa: F401
    AbelianLegPipe,
    DirectSumSpace,
    ElementarySpace,
    LegPipe,
    Space,
    TensorProduct,
    _flat_leg_permutation,
    _parse_inputs_drop_symmetry,
    _sort_sectors,
    _twist_gate_diag,
    _unique_sorted_sectors,
    swap_gate,
    twist_gate,
)


class Leg(abc.ABC):
    """Marker ABC for a single tensor axis.

    C++ ``Leg`` is a value-type handle over ``ElementarySpace | LegPipe``. Python keeps
    ``isinstance(x, Leg)`` via virtual subclass registration of the concrete types.
    ``TensorProduct`` is intentionally *not* registered.
    """


Leg.register(ElementarySpace)
Leg.register(LegPipe)


def is_leg_pipe(leg) -> bool:
    """True for plain `LegPipe` and for `AbelianLegPipe` (composed pipe structure).

    Prefer this over ``isinstance(leg, LegPipe)`` when abelian combined legs should
    count as pipes; after the spaces MI refactor, ``AbelianLegPipe`` is an
    `ElementarySpace` and is *not* a subclass of `LegPipe`.
    """
    ip = getattr(leg, 'is_pipe', False)
    return bool(ip() if callable(ip) else ip)


# auto-maintained by scripts/generate_reexport_all.py; do not edit by hand
__all__ = [
    'AbelianLegPipe',
    'DirectSumSpace',
    'ElementarySpace',
    'LegPipe',
    'Space',
    'TensorProduct',
    'swap_gate',
    'twist_gate',
]
