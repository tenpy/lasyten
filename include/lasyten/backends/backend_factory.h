#pragma once

#include <lasyten/backends/tensor_backend.h>
#include <lasyten/symmetries/symmetry.h>

#include <optional>
#include <string>

namespace lasyten {

/// Get an instance of an appropriate tensor backend (cached).
///
/// Returns a Python object wrapping a C++ `NoSymmetryBackend`,
/// `AbelianBackend`, or `FusionTreeBackend`.
///
/// Parameters mirror `get_backend`.
/// Get an instance of an appropriate backend.
///
/// Backends are instantiated only once and then cached. If a suitable backend instance is in
/// the cache, that same instance is returned.
///
/// @param symmetry Specifies which subclass of `TensorBackend` to use.
/// If ``None``, use the ``default_tensor_backend`` config option (``'abelian'`` by default).
/// If one of ``'fusion_tree'``, ``'abelian'`` or ``'no_symmetry'``, the respective backend.
/// If a symmetry, the simplest backend that supports it.
/// @param block_backend Specify which block backend to use. Preferred: ``'eigen'`` (native C++
/// Eigen/LAPACK, CPU) and ``'torch'`` / ``'gpu'`` / ``'apple_silicon'`` (libtorch). Transitional:
/// ``'numpy'`` and ``'cpu'`` (currently still the NumPy backend, which dispatches into Python;
/// compiled only with ``LASYTEN_WITH_NUMPY_BACKEND``, otherwise ``'cpu'`` is ``'eigen'``).
#if LASYTEN_HAS_PYTHON
py::object get_backend(py::object symmetry = py::none(), py::object block_backend = py::none());
#endif

/// Typed overload: pick a backend that supports `symmetry`. Python-free.
TensorBackend::Ptr get_backend(Symmetry::Ptr symmetry,
                               std::optional<std::string> block_backend = std::nullopt);

} // namespace lasyten
