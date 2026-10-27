"""RAM cache for the data read from the SU(N) HDF5 files.

A process-wide least-recently-used cache holds the F symbols, R symbols, fusion outcomes and
multiplicities read from the HDF5 files of the ``SUN`` symmetry. Its size is capped by the config
option ``su_n_symbol_cache_mb`` (see :mod:`lasyten.config`); ``0`` disables the cache. The limit is
read whenever an entry is inserted, so it can be changed at runtime, e.g. with
:func:`lasyten.temporary_options`; lowering it evicts least-recently-used entries.
"""

from ._core import clear_symbol_cache, symbol_cache_info

__all__ = ['cache_info', 'clear_caches']


def cache_info() -> dict[str, dict[str, int]]:
    """Statistics of the caches.

    Returns
    -------
    dict
        Key ``'symbols'``, mapping to a dict with the entries ``hits``, ``misses``, ``evictions``,
        ``rejected`` (insertions that were not cached), ``entries``, ``bytes`` (estimated size of the
        stored entries) and ``capacity_bytes``.
    """
    return {'symbols': symbol_cache_info()}


def clear_caches() -> None:
    """Remove all entries from all caches (the hit/miss counters are kept)."""
    clear_symbol_cache()
