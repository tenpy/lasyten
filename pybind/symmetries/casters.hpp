#pragma once

/// nanobind casters / conversion helpers for the symmetries layer.
///
/// ``nb::class_<Sector>`` / ``SectorArray`` own the Python types. NumPy arrays and sequences are
/// accepted wherever a ``Sector`` / ``SectorArray`` is expected through
/// ``nb::implicitly_convertible`` (registered in ``py_sector.cpp``), which calls the
/// ``Sector(values)`` / ``SectorArray(values)`` constructors. *Casting to Python* always produces
/// bound Sector / SectorArray instances (never bare ndarrays).
///
/// The pybind11-side counterparts used by liblasyten itself live in
/// ``include/lasyten/py_nanobind_bridge.h`` (and, for ``Leg``,
/// ``include/lasyten/symmetries/spaces.h``).

#include <lasyten/symmetries/fusion_symbol.h>
#include <lasyten/symmetries/sector.h>
#include <lasyten/symmetries/sector_numpy.h>
#include <lasyten/symmetries/spaces.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <pybind11/numpy.h>

#include <vector>

#include "../py_pybind11_bridge.h"

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

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

/// Transparent FusionSymbol <-> numpy.ndarray for bindings.
template<>
struct type_caster<lasyten::FusionSymbol>
{
    NB_TYPE_CASTER(lasyten::FusionSymbol, const_name("numpy.ndarray"))

    bool from_python(handle src, uint32_t flags, cleanup_list*) noexcept
    {
        pybind11::handle h(src.ptr());
        bool const convert = (flags & (uint32_t)cast_flags::convert) != 0;
        if (!convert && !pybind11::isinstance<pybind11::array>(h)) {
            return false;
        }
        try {
            value = lasyten::fusion_symbol_from_numpy(pybind11::array::ensure(h));
            return true;
        } catch (...) {
            PyErr_Clear();
            return false;
        }
    }

    static handle from_cpp(lasyten::FusionSymbol const& src, rv_policy, cleanup_list*) noexcept
    {
        try {
            return handle(lasyten::fusion_symbol_to_numpy(src).release().ptr());
        } catch (...) {
            PyErr_Clear();
            return handle();
        }
    }
};

/// ``Leg`` is a value-type handle (not a bound class). Accept ``ElementarySpace`` / ``LegPipe``
/// from Python (and ``None`` as the empty leg); unwrap to the held object on cast. Python exposes
/// ``Leg`` as an ABC that registers the concrete types.
template<>
struct type_caster<lasyten::Leg>
{
    NB_TYPE_CASTER(lasyten::Leg, const_name("Leg"))

    bool from_python(handle src, uint32_t flags, cleanup_list* cleanup) noexcept
    {
        if (src.is_none()) {
            value = lasyten::Leg(nullptr);
            return true;
        }
        flags &= ~(uint32_t)cast_flags::convert;
        {
            make_caster<std::shared_ptr<lasyten::ElementarySpace>> c;
            if (c.from_python(src, flags, cleanup)) {
                value = lasyten::Leg(c.operator std::shared_ptr<lasyten::ElementarySpace>&());
                return true;
            }
        }
        {
            make_caster<std::shared_ptr<lasyten::LegPipe>> c;
            if (c.from_python(src, flags, cleanup)) {
                value = lasyten::Leg(c.operator std::shared_ptr<lasyten::LegPipe>&());
                return true;
            }
        }
        return false;
    }

    static handle from_cpp(lasyten::Leg const& src,
                           rv_policy policy,
                           cleanup_list* cleanup) noexcept
    {
        if (!src) {
            return none().release();
        }
        if (auto es = src.as_elementary_space()) {
            return make_caster<std::shared_ptr<lasyten::ElementarySpace>>::from_cpp(
              es, policy, cleanup);
        }
        if (auto pipe = src.as_leg_pipe()) {
            return make_caster<std::shared_ptr<lasyten::LegPipe>>::from_cpp(pipe, policy, cleanup);
        }
        return none().release();
    }
};

/// ``None`` is the empty ``Leg``.
template<>
struct has_arg_defaults<lasyten::Leg> : std::true_type
{};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
