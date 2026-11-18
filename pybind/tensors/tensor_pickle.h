#pragma once

/// Shared pieces of the pickle state of the tensor classes.
///
/// The keys mirror the HDF5 sub-group names of ``SymmetricTensor::save_hdf5`` (``domain``,
/// ``codomain``, ``backend``, ``symmetry``, ``dtype``, ``device``, ``shape``, ``labels``; the
/// class-specific ``data`` etc. are added by the individual classes). See ``pickle_bind.h``.

#include "../py_lasyten_nanobind.h"
#include "../tools/pickle_bind.h"

#include <lasyten/tensors/symmetric_tensor.h>
#include <lasyten/tensors/tensor.h>

#include <string>
#include <vector>

namespace lasyten::pickle {

/// The fields every tensor class shares (the HDF5 sub-groups of ``Tensor``).
struct TensorCommonState
{
    TensorProduct::Ptr codomain;
    TensorProduct::Ptr domain;
    TensorBackend::Ptr backend;
    Symmetry::Ptr symmetry;
    Dtype dtype = Dtype::Float64;
    std::string device;
    std::vector<float64> shape;
    LegLabelList labels;
};

inline void
tensor_common_to_dict(Tensor const& t, nb::dict& d)
{
    d["domain"] = t.domain;
    d["codomain"] = t.codomain;
    d["backend"] = t.backend;
    d["symmetry"] = t.symmetry;
    d["dtype"] = t.dtype;
    d["device"] = t.device;
    d["shape"] = nb::cast(t.shape);
    d["labels"] = nb::cast(t.labels.as_list());
}

inline TensorCommonState
tensor_common_from_dict(nb::dict const& state)
{
    TensorCommonState c;
    c.domain = get_item<TensorProduct::Ptr>(state, "domain");
    c.codomain = get_item<TensorProduct::Ptr>(state, "codomain");
    c.backend = get_item<TensorBackend::Ptr>(state, "backend");
    c.symmetry = get_item<Symmetry::Ptr>(state, "symmetry");
    c.dtype = get_item<Dtype>(state, "dtype");
    c.device = get_item<std::string>(state, "device");
    c.shape = get_item<std::vector<float64>>(state, "shape");
    c.labels = get_item<LegLabelList>(state, "labels");
    // like ``from_hdf5``: empty labels mean "unlabelled"
    auto const nlegs = c.codomain->num_factors + c.domain->num_factors;
    if (c.labels.empty() && nlegs > 0) {
        c.labels.assign(static_cast<std::size_t>(nlegs), "");
    }
    return c;
}

/// State of a ``SymmetricTensor`` (also used for its subclasses): the common fields plus
/// ``data`` (and ``allow_charge_leg_label``, which HDF5 does not persist).
inline nb::dict
symmetric_tensor_to_dict(SymmetricTensor const& t)
{
    nb::dict d;
    tensor_common_to_dict(t, d);
    d["data"] = t.data;
    d["allow_charge_leg_label"] = t.allow_charge_leg_label;
    return d;
}

/// Rebuild a ``SymmetricTensor`` like ``SymmetricTensor::from_hdf5`` does: via the constructor,
/// then restoring ``dtype`` and ``shape``.
inline SymmetricTensor::Ptr
symmetric_tensor_from_dict(nb::dict const& state)
{
    auto c = tensor_common_from_dict(state);
    auto data = get_item<TensorBackend::DataPtr>(state, "data");
    auto obj = std::make_shared<SymmetricTensor>(
      std::move(data), c.codomain, c.domain, c.backend, c.symmetry, std::move(c.labels));
    obj->dtype = c.dtype;
    obj->device = c.device;
    obj->shape = std::move(c.shape);
    if (state.contains("allow_charge_leg_label")) {
        obj->allow_charge_leg_label = get_item<bool>(state, "allow_charge_leg_label");
    }
    return obj;
}

} // namespace lasyten::pickle
