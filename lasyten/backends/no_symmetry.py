"""Implements a 'dummy' tensor backend that does not exploit symmetries.

C++ implementation via nanobind (``lasyten._core``).
"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from .._core import NoSymmetryBackend  # noqa: F401

__all__ = ['NoSymmetryBackend']
