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
/// C-contiguous array on any DLPack device (CPU, CUDA, …).
using AnyDeviceArray = nb::ndarray<nb::c_contig>;
/// Writable NumPy array handed back to Python.
using NumpyArray = nb::ndarray<nb::numpy>;
/// Read-only NumPy view (``flags.writeable == False``).
using NumpyArrayRo = nb::ndarray<nb::numpy, nb::ro>;

[[nodiscard]] inline bool
is_cpu_device(int device_type)
{
    return device_type == nb::device::cpu::value;
}

/// Eigen / Numpy imports only support host memory; Torch may keep CUDA pointers.
inline void
require_cpu_device(int device_type, char const* backend_name)
{
    if (!is_cpu_device(device_type)) {
        throw nb::type_error(std::format("{}: block_from_ndarray requires a CPU array "
                                         "(got DLPack device_type={}); move to host first "
                                         "(e.g. tensor.cpu() / cupy.asnumpy).",
                                         backend_name,
                                         device_type)
                               .c_str());
    }
}

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

/// Target framework tag for ``to_ndarray`` exports (nanobind DLPack consumer).
enum class NdFramework
{
    Numpy,
    Pytorch,
    Jax,
    Tensorflow,
};

[[nodiscard]] inline NdFramework
parse_nd_framework(std::optional<std::string> const& name, NdFramework fallback)
{
    if (!name.has_value()) {
        return fallback;
    }
    auto const& s = *name;
    if (s == "numpy" || s == "np") {
        return NdFramework::Numpy;
    }
    if (s == "torch" || s == "pytorch") {
        return NdFramework::Pytorch;
    }
    if (s == "jax") {
        return NdFramework::Jax;
    }
    if (s == "tensorflow" || s == "tf") {
        return NdFramework::Tensorflow;
    }
    throw nb::type_error("framework must be one of: 'numpy', 'torch', 'jax', 'tensorflow' "
                         "(aliases: np, pytorch, tf)");
}

/// Zero-copy export of a host buffer with the requested framework tag.
[[nodiscard]] inline nb::object
export_cpu_buffer(void* data,
                  std::vector<std::size_t> const& shape,
                  nb::handle owner,
                  nb::dlpack::dtype dt,
                  bool writable,
                  NdFramework framework,
                  int device_type = nb::device::cpu::value,
                  int device_id = 0)
{
    auto pack = [&](auto writable_tensor, auto readonly_tensor) -> nb::object {
        if (writable) {
            return nb::cast(writable_tensor);
        }
        return nb::cast(readonly_tensor);
    };
    switch (framework) {
        case NdFramework::Numpy:
            return pack(
              nb::ndarray<nb::numpy>(
                data, shape.size(), shape.data(), owner, nullptr, dt, device_type, device_id),
              nb::ndarray<nb::numpy, nb::ro>(
                data, shape.size(), shape.data(), owner, nullptr, dt, device_type, device_id));
        case NdFramework::Pytorch:
            return pack(
              nb::ndarray<nb::pytorch>(
                data, shape.size(), shape.data(), owner, nullptr, dt, device_type, device_id),
              nb::ndarray<nb::pytorch, nb::ro>(
                data, shape.size(), shape.data(), owner, nullptr, dt, device_type, device_id));
        case NdFramework::Jax:
            return pack(
              nb::ndarray<nb::jax>(
                data, shape.size(), shape.data(), owner, nullptr, dt, device_type, device_id),
              nb::ndarray<nb::jax, nb::ro>(
                data, shape.size(), shape.data(), owner, nullptr, dt, device_type, device_id));
        case NdFramework::Tensorflow:
            return pack(
              nb::ndarray<nb::tensorflow>(
                data, shape.size(), shape.data(), owner, nullptr, dt, device_type, device_id),
              nb::ndarray<nb::tensorflow, nb::ro>(
                data, shape.size(), shape.data(), owner, nullptr, dt, device_type, device_id));
    }
    throw nb::type_error("unknown ndarray framework");
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

/// Zero-copy export with an explicit framework tag (numpy / torch / jax / tensorflow).
[[nodiscard]] inline nb::object
block_to_ndarray_framework(EigenBlockBackend::Block& block,
                           nb::handle owner,
                           bool writable,
                           NdFramework framework)
{
    auto const shape_i = block.shape();
    std::vector<std::size_t> shape(shape_i.begin(), shape_i.end());
    return export_cpu_buffer(
      block.bytes().data(), shape, owner, to_dlpack_dtype(block.dtype()), writable, framework);
}

/// Copy any CPU array into a new Eigen block, optionally converting to ``dtype``.
/// Non-CPU DLPack devices raise ``TypeError`` (Eigen storage is host-only).
[[nodiscard]] inline EigenBlockBackend::BlockPtr
block_from_ndarray(EigenBlockBackend& backend,
                   AnyDeviceArray const& arr,
                   std::optional<Dtype> dtype = std::nullopt)
{
    require_cpu_device(arr.device_type(), "EigenBlockBackend");
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
