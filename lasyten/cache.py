"""RAM caches for data read from the SU(N) HDF5 files and for fusion-tree computations.

Three process-wide least-recently-used caches speed up symmetric tensor operations with SU(N)
symmetry; their size is capped by config options (see :mod:`lasyten.config`):

- the **symbol cache** (F symbols, R symbols, fusion outcomes and multiplicities read from the
  HDF5 files), limited by ``su_n_symbol_cache_mb``;
- the **tree-plan cache** (mappings between fusion trees built when legs are permuted), limited by
  ``tree_plan_cache_mb``;
- the memo of the number of fusion trees, limited by ``tree_size_memo_mb``.

The budgets count *estimated* entry sizes (including a per-allocation overhead); the heap actually
used can differ from the estimate by some tens of percent, so the total ceiling is roughly the sum
of the three budgets plus the memory of open packed symbol files (``su_n_packed_ram_mb``, index and
block data) and of symmetries that cached tree plans keep alive (see below).

A limit of ``0`` disables the respective cache. The limits are read whenever an entry is inserted,
so they can be changed at runtime, e.g. with :func:`lasyten.temporary_options`; lowering a limit
evicts least-recently-used entries.

The fusion trees of a cached tree plan own their symmetry, and a symmetry owns its open data files
(and, for symbol files in the packed layout, their index in RAM). A symmetry that is no longer used
anywhere else is therefore released only when its plans are purged: this happens regularly while
plans are inserted and whenever :func:`cache_info` is called, when the plans are evicted, and when
:func:`clear_caches` is called, which also runs automatically at interpreter exit.
"""

import atexit

from ._core import (
    clear_symbol_cache,
    clear_tree_plan_cache,
    clear_tree_size_cache,
    packed_symbol_info,
    symbol_cache_info,
    tree_plan_cache_info,
    tree_size_cache_info,
)

__all__ = ['cache_info', 'clear_caches']


def cache_info() -> dict[str, dict[str, int]]:
    """Statistics of all caches.

    Returns
    -------
    dict
        Keys ``'symbols'``, ``'tree_plans'`` and ``'tree_sizes'``; each maps to a dict with the
        entries ``hits``, ``misses``, ``evictions``, ``rejected`` (insertions that were not cached),
        ``entries``, ``bytes`` (estimated size of the stored entries) and ``capacity_bytes``.
        The key ``'packed_files'`` maps to the memory held by open SU(N) symbol files in the packed
        layout, which is not part of the symbol cache budget: ``files``, ``index_bytes`` and
        ``data_bytes_in_ram``.

    """
    return {
        'symbols': symbol_cache_info(),
        'tree_plans': tree_plan_cache_info(),
        'tree_sizes': tree_size_cache_info(),
        'packed_files': packed_symbol_info(),
    }


def clear_caches() -> None:
    """Remove all entries from all caches (the hit/miss counters are kept).

    This also releases the symmetries (and their open data files) that cached tree plans still held.
    It is registered to run at interpreter exit, so that the symmetries are destroyed while Python
    is still fully functional.
    """
    clear_symbol_cache()
    clear_tree_plan_cache()
    clear_tree_size_cache()


atexit.register(clear_caches)
