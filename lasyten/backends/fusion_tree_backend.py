"""Backend based on fusion trees.

C++ implementations via nanobind (``lasyten._core``).
"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._core import (  # noqa: F401
    BendInstruction,
    BraidInstruction,
    FactorizedTreeMapping,
    FusionTreeBackend,
    FusionTreeData,
    PermuteLegsInstructionEngine,
    TreePairMapping,
    TwistInstruction,
)
from ..symmetries import fusion_trees

if TYPE_CHECKING:
    from ..tensors import SymmetricTensor

# Typing alias matching the old ABC; concrete instruction types are the three dataclasses above.
Instruction = BraidInstruction | BendInstruction | TwistInstruction


def _tree_block_iter(a: SymmetricTensor):
    """Iterate over tree-blocks of a fusion-tree tensor (used by tests / cross-checks)."""
    sym = a.symmetry
    domain_are_dual = [sp.is_dual for sp in a.domain.flat_legs]
    codomain_are_dual = [sp.is_dual for sp in a.codomain.flat_legs]
    for (bi, _), block in zip(a.data.block_inds, a.data.blocks):
        coupled = a.codomain.sector_decomposition[bi]
        i1_forest = 0  # start row index of the current forest block
        i2_forest = 0  # start column index of the current forest block
        for b_sectors, b_mults in a.domain.iter_uncoupled():
            tree_block_width = np.prod(b_mults)
            forest_block_width = 0
            for a_sectors, a_mults in a.codomain.iter_uncoupled():
                tree_block_height = np.prod(a_mults)
                i1 = i1_forest  # start row index of the current tree block
                i2 = i2_forest  # start column index of the current tree block
                for alpha_tree in fusion_trees(sym, a_sectors, coupled, codomain_are_dual):
                    i2 = i2_forest  # reset to the left of the current forest block
                    for beta_tree in fusion_trees(sym, b_sectors, coupled, domain_are_dual):
                        idx1 = slice(i1, i1 + tree_block_height)
                        idx2 = slice(i2, i2 + tree_block_width)
                        entries = block[idx1, idx2]
                        yield alpha_tree, beta_tree, entries
                        i2 += tree_block_width  # move right by one tree block
                    i1 += tree_block_height  # move down by one tree block
                forest_block_height = i1 - i1_forest
                forest_block_width = max(forest_block_width, i2 - i2_forest)
                i1_forest += forest_block_height
            i1_forest = 0  # reset to the top of the block
            i2_forest += forest_block_width


__all__ = [
    'FusionTreeData',
    'FusionTreeBackend',
    'Instruction',
    'BraidInstruction',
    'BendInstruction',
    'TwistInstruction',
    'PermuteLegsInstructionEngine',
    'TreePairMapping',
    'FactorizedTreeMapping',
    '_tree_block_iter',
]
