"""Block-backends implement matrix and array algebra on dense blocks, similar to e.g. numpy"""
# Copyright (C) TeNPy Developers, Apache license

# Note: order matters to avoid circular imports!
# pyright: ignore
from .._core import Dtype  # noqa
from .._core import (
    WITH_NUMPY_BACKEND,
    BlockBackend,
    EigenBlockBackend,
    TorchBlockBackend,
)
from . import dtypes

# NumpyBlockBackend / ArrayApiBlockBackend are transitional Python-dispatch backends. They are only
# available if lasyten was built with ``-DLASYTEN_WITH_NUMPY_BACKEND=ON`` (the default).
# EigenBlockBackend is the preferred CPU backend.
if WITH_NUMPY_BACKEND:
    from .._core import ArrayApiBlockBackend, NumpyBlockBackend


# auto-maintained by scripts/generate_reexport_all.py; do not edit by hand
__all__ = [
    'BlockBackend',
    'Dtype',
    'EigenBlockBackend',
    'TorchBlockBackend',
    'WITH_NUMPY_BACKEND',
    'dtypes',
]

if WITH_NUMPY_BACKEND:
    __all__ += ['ArrayApiBlockBackend', 'NumpyBlockBackend']

Block = BlockBackend.BlockCls
Scalar = BlockBackend.Scalar
