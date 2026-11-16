#pragma once

/// Dispatch ``nb::ndarray`` import/export across ``BlockBackend`` implementations.
///
/// - Eigen: zero-copy export / copying import via ``eigen_ndarray.h``.
/// - Numpy: zero-copy export of the stored ``py::array``; import builds a Numpy block.
/// - Torch: zero-copy ``nb::ndarray`` over the tensor buffer (CPU or CUDA); import copies
///   into a new contiguous torch tensor on the requested device.
/// - Other backends: fall back to ``block_from_numpy`` / ``to_numpy``.

#include "eigen_ndarray.h"

#include <lasyten/block_backend/block_backend.h>
#include <lasyten/block_backend/eigen.h>
#if defined(LASYTEN_WITH_NUMPY_BACKEND) && LASYTEN_WITH_NUMPY_BACKEND
#include <lasyten/block_backend/numpy.h>
#endif
#if defined(LASYTEN_WITH_TORCH_BACKEND) && LASYTEN_WITH_TORCH_BACKEND
#include <lasyten/block_backend/torch.h>
#endif

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lasyten::ndarray_dispatch {

namespace nb = nanobind;

[[nodiscard]] inline nb::object
numpy_array_as_ndarray(py::array arr, bool writable)
{
    nb::object view = nb::borrow(arr.ptr());
    if (!writable) {
        // Match Eigen read-only export: consumers must not mutate through the view.
        // Call Python with a real keyword — ``nb::arg`` is only for ``.def()`` signatures.
        nb::dict kwargs;
        kwargs["write"] = false;
        view.attr("setflags")(**kwargs);
    }
    return view;
}

#if defined(LASYTEN_WITH_TORCH_BACKEND) && LASYTEN_WITH_TORCH_BACKEND
[[nodiscard]] inline nb::object
torch_tensor_as_ndarray(torch::Tensor t,
                        nb::handle owner,
                        bool writable,
                        eigen_ndarray::NdFramework framework = eigen_ndarray::NdFramework::Pytorch)
{
    t = t.detach().contiguous().resolve_conj().resolve_neg();
    std::vector<std::size_t> shape(static_cast<std::size_t>(t.dim()));
    for (int64_t i = 0; i < t.dim(); ++i) {
        shape[static_cast<std::size_t>(i)] = static_cast<std::size_t>(t.size(i));
    }
    auto const dt = eigen_ndarray::to_dlpack_dtype(dtype::from_torch_dtype(t.scalar_type()));
    int const device_type = t.is_cuda() ? static_cast<int>(nb::device::cuda::value)
                                        : static_cast<int>(nb::device::cpu::value);
    int const device_id = t.is_cuda() ? static_cast<int>(t.device().index()) : 0;
    return eigen_ndarray::export_cpu_buffer(
      t.data_ptr(), shape, owner, dt, writable, framework, device_type, device_id);
}
#endif

[[nodiscard]] inline BlockBackend::BlockPtr
block_from_ndarray(BlockBackend& backend,
                   eigen_ndarray::AnyDeviceArray const& arr,
                   std::optional<Dtype> dtype = std::nullopt,
                   std::optional<std::string> device = std::nullopt)
{
    if (auto* eigen = dynamic_cast<EigenBlockBackend*>(&backend)) {
        (void)device;
        return eigen_ndarray::block_from_ndarray(*eigen, arr, dtype);
    }
#if defined(LASYTEN_WITH_NUMPY_BACKEND) && LASYTEN_WITH_NUMPY_BACKEND
    if (dynamic_cast<NumpyBlockBackend*>(&backend) != nullptr) {
        eigen_ndarray::require_cpu_device(arr.device_type(), "NumpyBlockBackend");
        nb::object py_obj = nb::cast(arr);
        py::array a = py::array::ensure(py_obj);
        if (!a) {
            throw nb::type_error("block_from_ndarray: expected array-like");
        }
        if (dtype.has_value()) {
            // ``nb::arg`` is only for ``.def()`` signatures — use a real kwargs dict.
            nb::dict kwargs;
            kwargs["copy"] = false;
            a = py::reinterpret_steal<py::array>(
              a.attr("astype")(dtype::to_numpy_dtype(*dtype), **kwargs).ptr());
        }
        return backend.block_from_numpy(a, dtype, std::move(device));
    }
#endif
#if defined(LASYTEN_WITH_TORCH_BACKEND) && LASYTEN_WITH_TORCH_BACKEND
    if (dynamic_cast<TorchBlockBackend*>(&backend) != nullptr) {
        // Copy into a new torch tensor. CPU sources use host memcpy then optional
        // ``.to(device)``; CUDA (or other) sources use ``from_blob`` on that device
        // so non-CPU buffers never bounce through host memory.
        Dtype const src_dtype = eigen_ndarray::from_dlpack_dtype(arr.dtype());
        Dtype const out_dtype = dtype.value_or(src_dtype);
        std::vector<int64_t> sizes(arr.ndim());
        std::size_t numel = 1;
        for (std::size_t i = 0; i < arr.ndim(); ++i) {
            sizes[i] = static_cast<int64_t>(arr.shape(i));
            numel *= arr.shape(i);
        }
        std::string const dev = backend.as_device(device);
        torch::Tensor t;
        if (eigen_ndarray::is_cpu_device(arr.device_type())) {
            auto cpu_opts =
              torch::TensorOptions().dtype(dtype::to_torch_dtype(src_dtype)).device(torch::kCPU);
            t = torch::empty(sizes, cpu_opts);
            if (numel > 0) {
                std::memcpy(t.data_ptr(), arr.data(), numel * dtype::itemsize(src_dtype));
            }
        } else {
            torch::Device src_dev =
              (arr.device_type() == nb::device::cuda::value)
                ? torch::Device(torch::kCUDA, static_cast<torch::DeviceIndex>(arr.device_id()))
                : torch::Device(dev);
            auto opts =
              torch::TensorOptions().dtype(dtype::to_torch_dtype(src_dtype)).device(src_dev);
            // Clone so the block owns the storage (from_blob aliases the ndarray).
            t = torch::from_blob(const_cast<void*>(arr.data()), sizes, opts).clone();
        }
        if (out_dtype != src_dtype) {
            t = t.to(dtype::to_torch_dtype(out_dtype));
        }
        t = t.to(torch::Device(dev));
        return std::make_shared<TorchBlockBackend::Block>(std::move(t));
    }
#endif
    eigen_ndarray::require_cpu_device(arr.device_type(), backend.get_backend_name().c_str());
    nb::object py_obj = nb::cast(arr);
    return backend.block_from_numpy(py::array::ensure(py_obj), dtype, std::move(device));
}

[[nodiscard]] inline nb::object
block_to_ndarray(BlockBackend& /*backend*/,
                 BlockBackend::Block& block,
                 nb::handle owner,
                 bool writable = true,
                 std::optional<std::string> framework = std::nullopt)
{
    // Prefer nanobind casts of ``owner`` over ``dynamic_cast`` on ``Block&``: with
    // ``-fvisibility=hidden``, Block RTTI can fail for Numpy/Torch even when the
    // Python type is correct (Eigen happens to work). ``nb::try_cast`` uses the
    // bound type registry instead.
    {
        EigenBlockBackend::Block* eigen = nullptr;
        if (nb::try_cast(owner, eigen) && eigen != nullptr) {
            auto const fw =
              eigen_ndarray::parse_nd_framework(framework, eigen_ndarray::NdFramework::Numpy);
            return eigen_ndarray::block_to_ndarray_framework(*eigen, owner, writable, fw);
        }
    }
#if defined(LASYTEN_WITH_NUMPY_BACKEND) && LASYTEN_WITH_NUMPY_BACKEND
    {
        NumpyBlockBackend::Block* np = nullptr;
        if (nb::try_cast(owner, np) && np != nullptr) {
            auto const fw =
              eigen_ndarray::parse_nd_framework(framework, eigen_ndarray::NdFramework::Numpy);
            if (fw == eigen_ndarray::NdFramework::Numpy) {
                return numpy_array_as_ndarray(np->to_numpy(), writable);
            }
            // Retag the stored buffer for torch/jax/tensorflow consumers.
            py::array a = np->to_numpy();
            if (!a) {
                throw nb::type_error("Numpy block has no array");
            }
            std::vector<std::size_t> shape(static_cast<std::size_t>(a.ndim()));
            for (py::ssize_t i = 0; i < a.ndim(); ++i) {
                shape[static_cast<std::size_t>(i)] = static_cast<std::size_t>(a.shape(i));
            }
            return eigen_ndarray::export_cpu_buffer(
              const_cast<void*>(a.data()),
              shape,
              owner,
              eigen_ndarray::to_dlpack_dtype(dtype::from_numpy_dtype(a.dtype())),
              writable,
              fw);
        }
    }
#endif
#if defined(LASYTEN_WITH_TORCH_BACKEND) && LASYTEN_WITH_TORCH_BACKEND
    {
        TorchBlockBackend::Block* tb = nullptr;
        if (nb::try_cast(owner, tb) && tb != nullptr) {
            auto const fw =
              eigen_ndarray::parse_nd_framework(framework, eigen_ndarray::NdFramework::Pytorch);
            return torch_tensor_as_ndarray(tb->tensor(), owner, writable, fw);
        }
    }
#endif
    // Fallbacks: C++ RTTI (Eigen) then owning NumPy copy.
    if (auto* eigen = dynamic_cast<EigenBlockBackend::Block*>(&block)) {
        auto const fw =
          eigen_ndarray::parse_nd_framework(framework, eigen_ndarray::NdFramework::Numpy);
        return eigen_ndarray::block_to_ndarray_framework(*eigen, owner, writable, fw);
    }
    return nb::cast(block.to_numpy());
}

[[nodiscard]] inline nb::object
export_to_python(nb::object exp)
{
    return exp;
}

} // namespace lasyten::ndarray_dispatch
