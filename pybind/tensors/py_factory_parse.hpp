#pragma once

/// Python-flexible parsing for tensor factory bindings.
///
/// Sequence-of-spaces (co)domains, nested labels, numpy blocks, and optional tensors stay at
/// the binding layer. Typed C++ factories receive Ptrs / BlockPtrs / LegLabelList.

#include <lasyten/tensors/charged_tensor.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/symmetric_tensor.h>
#include <lasyten/tensors/tensor.h>

#include "../py_lasyten_pybind11.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace lasyten {

inline Space::Ptr
py_as_space_leg(py::object leg)
{
    if (leg.is_none()) {
        return nullptr;
    }
    if (py::isinstance<LegPipe>(leg)) {
        throw std::invalid_argument("DiagonalTensor / Mask is not defined on LegPipes.");
    }
    return leg.cast<Space::Ptr>();
}

inline Tensor::InitParsed
py_parse_diag(py::object leg, TensorBackend::Ptr backend, py::object labels)
{
    return parse_tensor_init(py::make_tuple(leg),
                             py::make_tuple(leg),
                             std::move(backend),
                             labels,
                             /*is_endomorphism=*/true);
}

inline TensorCPtr
py_optional_tensor(py::object obj)
{
    if (obj.is_none()) {
        return nullptr;
    }
    return obj.cast<TensorCPtr>();
}

inline SymmetricTensorCPtr
py_as_symmetric_cptr(py::object obj)
{
    return obj.cast<SymmetricTensorCPtr>();
}

inline DiagonalTensorCPtr
py_as_diagonal_cptr(py::object obj)
{
    return obj.cast<DiagonalTensorCPtr>();
}

inline BlockBackend::BlockPtr
py_optional_block(py::object obj,
                  TensorBackend::Ptr const& backend,
                  std::optional<Dtype> dtype = std::nullopt,
                  std::optional<std::string> device = std::nullopt)
{
    if (obj.is_none()) {
        return nullptr;
    }
    try {
        auto block = obj.cast<BlockBackend::BlockPtr>();
        if (dtype.has_value() || device.has_value()) {
            return backend->block_backend->as_block(obj, dtype, device);
        }
        return block;
    } catch (py::cast_error const&) {
        return backend->block_backend->as_block(obj, dtype, device);
    }
}

inline BlockBackend::BlockPtr
py_as_block(py::object obj,
            TensorBackend::Ptr const& backend,
            std::optional<Dtype> dtype = std::nullopt,
            std::optional<std::string> device = std::nullopt)
{
    return backend->block_backend->as_block(obj, dtype, device);
}

inline std::variant<ElementarySpace::Ptr, Sector>
py_as_charge(py::object charge)
{
    if (py::isinstance<ElementarySpace>(charge)) {
        return charge.cast<ElementarySpace::Ptr>();
    }
    return charge.cast<Sector>();
}

inline std::optional<std::variant<ElementarySpace::Ptr, Sector>>
py_optional_charge(py::object charge)
{
    if (charge.is_none()) {
        return std::nullopt;
    }
    return py_as_charge(charge);
}

inline py::object
py_from_charged_or_scalar(std::variant<ChargedTensor::Ptr, BlockBackend::Scalar> result)
{
    return std::visit([](auto const& v) -> py::object { return py::cast(v); }, result);
}

} // namespace lasyten
