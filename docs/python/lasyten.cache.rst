cache
=====

- full name: lasyten.cache
- parent module: :mod:`lasyten`
- type: module

.. py:module:: lasyten.cache

Declared in ``lasyten/cache.py``; the caches are implemented in C++
(``include/lasyten/tools/lru_cache.h``). Their sizes are set by the config options
``su_n_symbol_cache_mb`` and ``tree_plan_cache_mb`` (see :mod:`lasyten.config`).

Functions
---------

.. autofunction:: lasyten.cache.cache_info

.. autofunction:: lasyten.cache.clear_caches
