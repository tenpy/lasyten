cache
=====

- full name: lasyten.cache
- parent module: :mod:`lasyten`
- type: module

.. py:module:: lasyten.cache

Declared in ``lasyten/cache.py``; the caches are implemented in C++
(``include/lasyten/tools/lru_cache.h``). Their sizes are set by the config options
``su_n_symbol_cache_mb``, ``tree_plan_cache_mb`` and ``tree_size_memo_mb`` (see :mod:`lasyten.config`).
The block data of open packed symbol files (``su_n_packed_ram_mb``) is held outside these budgets.

Functions
---------

.. autofunction:: lasyten.cache.cache_info

.. autofunction:: lasyten.cache.clear_caches
