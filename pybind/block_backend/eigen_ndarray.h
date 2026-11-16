#pragma once

/// ``nb::ndarray`` interchange for ``EigenBlockBackend`` blocks (convert_nanobind.md, leftover
/// #10).
///
/// ``nb::ndarray`` is a *view/interchange* type only (NumPy, PyTorch, JAX, anything DLPack /
/// buffer-protocol); it does not provide compute. The Eigen backend owns a contiguous C-order
/// ``std::vector<std::byte>``; this header exposes it across the Python boundary without going
/// through the ``py::array`` adapter:
///
/// - export: ``block_to_ndarray`` returns a *zero-copy* NumPy view kept alive by the block.
///   Default is **writable** (aliases block memory). Pass ``writable=false`` for a read-only
///   view (TeNPy-safe default for consumers that must not mutate storage).
/// - import: ``block_from_ndarray`` **copies** into owned Eigen storage from any CPU array that
///   supports DLPack / the buffer protocol.
///
/// Dual path vs ``to_numpy()`` / ``block_from_numpy()`` (``py::array`` via ``detail/py_array.h``):
/// those allocate an owning NumPy copy (never alias Eigen storage). Prefer ``to_ndarray`` /
/// ``block_from_ndarray`` at new Python call sites; keep ``to_numpy`` for code that needs an
/// independent buffer or a non-Eigen backend.
///

#include <lasyten/block_backend/dtypes.h>
#include <lasyten/block_backend/eigen.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <string>
#include <vector>

namespace lasyten::eigen_ndarray {

namespace nb = nanobind;

/// Any CPU array, C-contiguous (non-contiguous inputs are made contiguous by nanobind).
using AnyCpuArray = nb::ndarray<nb::c_contig, nb::device::cpu>;
/// Writable NumPy array handed back to Python.
using NumpyArray = nb::ndarray<nb::numpy>;
/// Read-only NumPy view (``flags.writeable == False``).
using NumpyArrayRo = nb::ndarray<nb::numpy, nb::ro>;

[[nodiscard]] inline nb::dlpack::dtype
to_dlpack_dtype(Dtype dt)
{
    using nb::dlpack::dtype_code;
    switch (dt) {
        case Dtype::Bool:
            return { static_cast<std::uint8_t>(dtype_code::Bool), 8, 1 };
        case Dtype::Int64:
            return { static_cast<std::uint8_t>(dtype_code::Int), 64, 1 };
        case Dtype::Float32:
            return { static_cast<std::uint8_t>(dtype_code::Float), 32, 1 };
        case Dtype::Float64:
            return { static_cast<std::uint8_t>(dtype_code::Float), 64, 1 };
        case Dtype::Complex64:
            return { static_cast<std::uint8_t>(dtype_code::Complex), 64, 1 };
        case Dtype::Complex128:
            return { static_cast<std::uint8_t>(dtype_code::Complex), 128, 1 };
    }
    throw std::invalid_argument("unsupported Dtype");
}

[[nodiscard]] inline Dtype
from_dlpack_dtype(nb::dlpack::dtype const& dt)
{
    using nb::dlpack::dtype_code;
    if (dt.lanes == 1) {
        auto const code = static_cast<dtype_code>(dt.code);
        if (code == dtype_code::Bool && dt.bits == 8) {
            return Dtype::Bool;
        }
        if (code == dtype_code::Int && dt.bits == 64) {
            return Dtype::Int64;
        }
        if (code == dtype_code::Float && dt.bits == 32) {
            return Dtype::Float32;
        }
        if (code == dtype_code::Float && dt.bits == 64) {
            return Dtype::Float64;
        }
        if (code == dtype_code::Complex && dt.bits == 64) {
            return Dtype::Complex64;
        }
        if (code == dtype_code::Complex && dt.bits == 128) {
            return Dtype::Complex128;
        }
    }
    throw nb::type_error(
      std::format("EigenBlockBackend: unsupported array dtype (code={}, bits={}); supported are "
                  "bool, int64, float32/64 and complex64/128.",
                  static_cast<int>(dt.code),
                  static_cast<int>(dt.bits))
        .c_str());
}

/// Zero-copy NumPy view of the block's buffer. ``owner`` must be the Python object wrapping
/// ``block``; the view keeps it alive. Writable alias of the block's memory.
[[nodiscard]] inline NumpyArray
block_to_ndarray(EigenBlockBackend::Block& block, nb::handle owner)
{
    auto const shape_i = block.shape();
    std::vector<std::size_t> shape(shape_i.begin(), shape_i.end());
    return NumpyArray(block.bytes().data(),
                      shape.size(),
                      shape.data(),
                      owner,
                      nullptr,
                      to_dlpack_dtype(block.dtype()));
}

/// Zero-copy **read-only** NumPy view of the block's buffer (same lifetime rules as
/// ``block_to_ndarray``).
[[nodiscard]] inline NumpyArrayRo
block_to_ndarray_readonly(EigenBlockBackend::Block const& block, nb::handle owner)
{
    auto const shape_i = block.shape();
    std::vector<std::size_t> shape(shape_i.begin(), shape_i.end());
    return NumpyArrayRo(block.bytes().data(),
                        shape.size(),
                        shape.data(),
                        owner,
                        nullptr,
                        to_dlpack_dtype(block.dtype()));
}

/// Copy any CPU array into a new Eigen block, optionally converting to ``dtype``.
[[nodiscard]] inline EigenBlockBackend::BlockPtr
block_from_ndarray(EigenBlockBackend& backend,
                   AnyCpuArray const& arr,
                   std::optional<Dtype> dtype = std::nullopt)
{
    Dtype const src_dtype = from_dlpack_dtype(arr.dtype());
    std::vector<int64> shape(arr.ndim());
    std::size_t numel = 1;
    for (std::size_t i = 0; i < arr.ndim(); ++i) {
        shape[i] = static_cast<int64>(arr.shape(i));
        numel *= arr.shape(i);
    }
    std::vector<std::byte> bytes(numel * dtype::itemsize(src_dtype));
    if (!bytes.empty()) {
        std::memcpy(bytes.data(), arr.data(), bytes.size());
    }
    EigenBlockBackend::BlockPtr block =
      std::make_shared<EigenBlockBackend::Block>(src_dtype, std::move(shape), std::move(bytes));
    if (dtype && *dtype != src_dtype) {
        block = backend.to_dtype(block, *dtype);
    }
    return block;
}

} // namespace lasyten::eigen_ndarray
