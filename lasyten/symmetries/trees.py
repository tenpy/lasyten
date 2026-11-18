"""TODO module docstring"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from .._core import FusionTree, fusion_trees  # noqa: F401

# C++ implementations via nanobind
from ._symmetries import SectorArray

# auto-maintained by scripts/generate_reexport_all.py; do not edit by hand
__all__ = [
    'FusionTree',
    'SectorArray',
    'fusion_trees',
]


def _concat_sector_arrays(*arrays: SectorArray) -> SectorArray:
    res = arrays[0]
    for array in arrays[1:]:
        res = res.concat(array)
    return res
