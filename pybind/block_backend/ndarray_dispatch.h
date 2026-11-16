#pragma once

/// Dispatch ``nb::ndarray`` import/export across ``BlockBackend`` implementations.
///
/// Eigen: zero-copy export / copying import via ``eigen_ndarray.h``.
/// Other backends: fall back to ``block_from_numpy`` / ``to_numpy`` (owning NumPy copy).

#include "eigen_ndarray.h"

#include <lasyten/block_backend/block_backend.h>
#include <lasyten/block_backend/eigen.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace lasyten::ndarray_dispatch {

namespace nb = nanobind;

using ExportArray =
  std::variant<eigen_ndarray::NumpyArray, eigen_ndarray::NumpyArrayRo, nb::object>;

[[nodiscard]] inline BlockBackend::BlockPtr
block_from_ndarray(BlockBackend& backend,
                   eigen_ndarray::AnyCpuArray const& arr,
                   std::optional<Dtype> dtype = std::nullopt,
                   std::optional<std::string> device = std::nullopt)
{
    if (auto* eigen = dynamic_cast<EigenBlockBackend*>(&backend)) {
        (void)device; // Eigen blocks are always CPU; ignore device for the Eigen fast path.
        return eigen_ndarray::block_from_ndarray(*eigen, arr, dtype);
    }
    // Fall back: materialize a Python array object, then use the classic py::array path.
    nb::object py_obj = nb::cast(arr);
    return backend.block_from_numpy(py::array::ensure(py_obj), dtype, std::move(device));
}

[[nodiscard]] inline ExportArray
block_to_ndarray(BlockBackend& /*backend*/,
                 BlockBackend::Block& block,
                 nb::handle owner,
                 bool writable = true)
{
    if (auto* eigen = dynamic_cast<EigenBlockBackend::Block*>(&block)) {
        if (writable) {
            return eigen_ndarray::block_to_ndarray(*eigen, owner);
        }
        return eigen_ndarray::block_to_ndarray_readonly(*eigen, owner);
    }
    // Owning NumPy copy for non-Eigen backends (never aliases device memory).
    return nb::cast(block.to_numpy());
}

[[nodiscard]] inline nb::object
export_to_python(ExportArray&& exp)
{
    return std::visit([](auto&& v) -> nb::object { return nb::cast(std::move(v)); },
                      std::move(exp));
}

} // namespace lasyten::ndarray_dispatch
