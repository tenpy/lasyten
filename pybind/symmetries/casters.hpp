#pragma once

/// Optional conversion helpers: accept NumPy arrays / sequences as Sector arguments.
///
/// ``py::class_<Sector>`` / ``SectorArray`` own the Python types. These casters only
/// extend *load* so call sites can still pass ndarrays during migration; *cast*
/// always produces bound Sector / SectorArray instances (never bare ndarrays).
///
/// ``Leg`` value-type caster lives in ``include/lasyten/symmetries/spaces.h`` so library
/// TUs that ``.cast<Leg>()`` see the same specialization.

#include <lasyten/symmetries/fusion_symbol.h>
#include <lasyten/symmetries/sector.h>
#include <lasyten/symmetries/sector_numpy.h>
#include <lasyten/symmetries/spaces.h>

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

namespace lasyten {

inline py::array
vector_i64_to_numpy(std::vector<int64> const& v)
{
    py::array_t<int64> arr(static_cast<py::ssize_t>(v.size()));
    auto* ptr = arr.mutable_data();
    for (std::size_t i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return arr;
}

inline py::array
vector_f64_to_numpy(std::vector<float64> const& v)
{
    py::array_t<float64> arr(static_cast<py::ssize_t>(v.size()));
    auto* ptr = arr.mutable_data();
    for (std::size_t i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return arr;
}

inline std::vector<int64>
vector_i64_from_numpy(py::object obj)
{
    auto arr = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw py::type_error("expected 1D int array");
    }
    auto info = arr.request();
    if (info.ndim != 1) {
        throw py::type_error("expected 1D int array");
    }
    auto const* ptr = static_cast<int64 const*>(info.ptr);
    return std::vector<int64>(ptr, ptr + info.shape[0]);
}

inline std::vector<float64>
vector_f64_from_numpy(py::object obj)
{
    auto arr = py::array_t<float64, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw py::type_error("expected 1D float array");
    }
    auto info = arr.request();
    if (info.ndim != 1) {
        throw py::type_error("expected 1D float array");
    }
    auto const* ptr = static_cast<float64 const*>(info.ptr);
    return std::vector<float64>(ptr, ptr + info.shape[0]);
}

} // namespace lasyten

namespace pybind11::detail {

/// Transparent FusionSymbol ↔ numpy.ndarray for bindings / trampolines.
template<>
struct type_caster<lasyten::FusionSymbol>
{
    PYBIND11_TYPE_CASTER(lasyten::FusionSymbol, const_name("numpy.ndarray"));

    bool load(handle src, bool convert)
    {
        if (!convert && !isinstance<array>(src)) {
            return false;
        }
        try {
            value = lasyten::fusion_symbol_from_numpy(array::ensure(src));
            return true;
        } catch (...) {
            return false;
        }
    }

    static handle cast(lasyten::FusionSymbol const& src,
                       return_value_policy /*policy*/,
                       handle /*parent*/)
    {
        return lasyten::fusion_symbol_to_numpy(src).release();
    }
};

template<>
struct type_caster<lasyten::Sector> : type_caster_base<lasyten::Sector>
{
    using base = type_caster_base<lasyten::Sector>;

    bool load(handle src, bool convert)
    {
        if (base::load(src, convert)) {
            return true;
        }
        if (!convert) {
            return false;
        }
        try {
            lasyten::Sector tmp = lasyten::sector_from_numpy(src);
            value = new lasyten::Sector(std::move(tmp));
            return true;
        } catch (...) {
            return false;
        }
    }
};

template<>
struct type_caster<lasyten::SectorArray> : type_caster_base<lasyten::SectorArray>
{
    using base = type_caster_base<lasyten::SectorArray>;

    bool load(handle src, bool convert)
    {
        if (base::load(src, convert)) {
            return true;
        }
        if (!convert) {
            return false;
        }
        try {
            lasyten::SectorArray tmp = lasyten::sector_array_from_numpy(src);
            value = new lasyten::SectorArray(std::move(tmp));
            return true;
        } catch (...) {
            return false;
        }
    }
};

} // namespace pybind11::detail
