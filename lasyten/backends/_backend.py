"""Tensor backends: abstract base and shared helpers.

C++ implementations via nanobind (``lasyten._core``).
"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from typing import Protocol, TypeVar

from .._core import TensorBackend, conventional_leg_order, get_same_backend  # noqa: F401

# placeholder for a backend-specific type that holds all data of a tensor
#  (except the symmetry data stored in its legs)
Data = TypeVar('Data')
DiagonalData = TypeVar('DiagonalData')
MaskData = TypeVar('MaskData')


class HasBackend(Protocol):  # noqa D101
    @property
    def backend(self) -> TensorBackend: ...


__all__ = [
    'Data',
    'DiagonalData',
    'MaskData',
    'TensorBackend',
    'conventional_leg_order',
    'get_same_backend',
    'HasBackend',
]
