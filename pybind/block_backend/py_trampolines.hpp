#pragma once

// nanobind port (docs/cpp_conversion/convert_nanobind.md): the pybind11 trampoline classes that
// used to live here were removed. Only ``LinearOperator`` / ``PlanarLinearOperator`` keep a
// trampoline (defined next to their bindings in tensors/py_sparse.cpp and tensors/py_planar.cpp),
// because TeNPy subclasses them from Python. Python subclassing of the other bound C++ types
// (Tensor, BlockBackend, Symmetry, Site, Space, TensorBackend, ...) is unsupported until
// trampolines are restored (convert_nanobind.md, leftover #3).
#include "tools/hdf5_py_bridge.h"
#include <lasyten/block_backend/block_backend.h>
#if LASYTEN_WITH_NUMPY_BACKEND
#include <lasyten/block_backend/array_api.h>
#include <lasyten/block_backend/numpy.h>
#endif
#include <memory>
#include <span>
