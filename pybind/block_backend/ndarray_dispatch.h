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
torch_tensor_as_ndarray(torch::Tensor t, nb::handle owner, bool writable)
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
    if (writable) {
        return nb::cast(nb::ndarray<nb::pytorch>(
          t.data_ptr(), shape.size(), shape.data(), owner, nullptr, dt, device_type, device_id));
    }
    return nb::cast(nb::ndarray<nb::pytorch, nb::ro>(
      t.data_ptr(), shape.size(), shape.data(), owner, nullptr, dt, device_type, device_id));
}
#endif

[[nodiscard]] inline BlockBackend::BlockPtr
block_from_ndarray(BlockBackend& backend,
                   eigen_ndarray::AnyCpuArray const& arr,
                   std::optional<Dtype> dtype = std::nullopt,
                   std::optional<std::string> device = std::nullopt)
{
    if (auto* eigen = dynamic_cast<EigenBlockBackend*>(&backend)) {
        (void)device;
        return eigen_ndarray::block_from_ndarray(*eigen, arr, dtype);
    }
#if defined(LASYTEN_WITH_NUMPY_BACKEND) && LASYTEN_WITH_NUMPY_BACKEND
    if (dynamic_cast<NumpyBlockBackend*>(&backend) != nullptr) {
        nb::object py_obj = nb::cast(arr);
        py::array a = py::array::ensure(py_obj);
        if (!a) {
            throw nb::type_error("block_from_ndarray: expected array-like");
        }
        if (dtype.has_value()) {
            a = py::reinterpret_steal<py::array>(
              a.attr("astype")(dtype::to_numpy_dtype(*dtype), nb::arg("copy") = false).ptr());
        }
        return backend.block_from_numpy(a, dtype, std::move(device));
    }
#endif
#if defined(LASYTEN_WITH_TORCH_BACKEND) && LASYTEN_WITH_TORCH_BACKEND
    if (dynamic_cast<TorchBlockBackend*>(&backend) != nullptr) {
        // Copy CPU ndarray bytes into a new torch tensor on the backend device.
        Dtype const src_dtype = eigen_ndarray::from_dlpack_dtype(arr.dtype());
        Dtype const out_dtype = dtype.value_or(src_dtype);
        std::vector<int64_t> sizes(arr.ndim());
        std::size_t numel = 1;
        for (std::size_t i = 0; i < arr.ndim(); ++i) {
            sizes[i] = static_cast<int64_t>(arr.shape(i));
            numel *= arr.shape(i);
        }
        std::string const dev = backend.as_device(device);
        auto cpu_opts =
          torch::TensorOptions().dtype(dtype::to_torch_dtype(src_dtype)).device(torch::kCPU);
        torch::Tensor t = torch::empty(sizes, cpu_opts);
        if (numel > 0) {
            std::memcpy(t.data_ptr(), arr.data(), numel * dtype::itemsize(src_dtype));
        }
        if (out_dtype != src_dtype) {
            t = t.to(dtype::to_torch_dtype(out_dtype));
        }
        t = t.to(torch::Device(dev));
        return std::make_shared<TorchBlockBackend::Block>(std::move(t));
    }
#endif
    nb::object py_obj = nb::cast(arr);
    return backend.block_from_numpy(py::array::ensure(py_obj), dtype, std::move(device));
}

[[nodiscard]] inline nb::object
block_to_ndarray(BlockBackend& /*backend*/,
                 BlockBackend::Block& block,
                 nb::handle owner,
                 bool writable = true)
{
    // Prefer nanobind casts of ``owner`` over ``dynamic_cast`` on ``Block&``: with
    // ``-fvisibility=hidden``, Block RTTI can fail for Numpy/Torch even when the
    // Python type is correct (Eigen happens to work). ``nb::try_cast`` uses the
    // bound type registry instead.
    {
        EigenBlockBackend::Block* eigen = nullptr;
        if (nb::try_cast(owner, eigen) && eigen != nullptr) {
            if (writable) {
                return nb::cast(eigen_ndarray::block_to_ndarray(*eigen, owner));
            }
            return nb::cast(eigen_ndarray::block_to_ndarray_readonly(*eigen, owner));
        }
    }
#if defined(LASYTEN_WITH_NUMPY_BACKEND) && LASYTEN_WITH_NUMPY_BACKEND
    {
        NumpyBlockBackend::Block* np = nullptr;
        if (nb::try_cast(owner, np) && np != nullptr) {
            return numpy_array_as_ndarray(np->to_numpy(), writable);
        }
    }
#endif
#if defined(LASYTEN_WITH_TORCH_BACKEND) && LASYTEN_WITH_TORCH_BACKEND
    {
        TorchBlockBackend::Block* tb = nullptr;
        if (nb::try_cast(owner, tb) && tb != nullptr) {
            return torch_tensor_as_ndarray(tb->tensor(), owner, writable);
        }
    }
#endif
    // Fallbacks: C++ RTTI (Eigen) then owning NumPy copy.
    if (auto* eigen = dynamic_cast<EigenBlockBackend::Block*>(&block)) {
        if (writable) {
            return nb::cast(eigen_ndarray::block_to_ndarray(*eigen, owner));
        }
        return nb::cast(eigen_ndarray::block_to_ndarray_readonly(*eigen, owner));
    }
    return nb::cast(block.to_numpy());
}

[[nodiscard]] inline nb::object
export_to_python(nb::object exp)
{
    return exp;
}

} // namespace lasyten::ndarray_dispatch
