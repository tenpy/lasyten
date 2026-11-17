#pragma once

// nanobind port (docs/cpp_conversion/convert_nanobind.md): the pybind11 trampoline classes that
// used to live here were removed. Only ``LinearOperator`` / ``PlanarLinearOperator`` keep a
// trampoline (defined next to their bindings in tensors/py_sparse.cpp and tensors/py_planar.cpp),
// because TeNPy subclasses them from Python. Python subclassing of the other bound C++ types
// (Tensor, BlockBackend, Symmetry, Site, Space, TensorBackend, ...) is unsupported until
// trampolines are restored (convert_nanobind.md, leftover #3).
#include <lasyten/symmetries/abelian_group.h>
#include <lasyten/symmetries/base_symmetry.h>
#include <lasyten/symmetries/fusion_symbol.h>
#include <lasyten/symmetries/group.h>
#include <lasyten/symmetries/sector_numpy.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/symmetry_factor.h>
