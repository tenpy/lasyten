"""Utility functions to access backend instances.

C++ ``get_backend`` via nanobind (``lasyten._core``).
"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from .._core import get_backend  # noqa: F401

# Re-exports kept for call sites that look up backend classes on this module.
from ._backend import TensorBackend  # noqa: F401
from .abelian import AbelianBackend  # noqa: F401
from .fusion_tree_backend import FusionTreeBackend  # noqa: F401
from .no_symmetry import NoSymmetryBackend  # noqa: F401

__all__ = [
    'get_backend',
    'TensorBackend',
    'NoSymmetryBackend',
    'AbelianBackend',
    'FusionTreeBackend',
]
