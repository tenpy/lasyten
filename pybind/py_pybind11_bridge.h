#pragma once

/// nanobind casters for pybind11 object types (``py::object``, ``py::array``, ``py::dict``, ...).
///
/// The public headers under ``include/lasyten/`` still declare functions taking / returning
/// pybind11 types (docs/cpp_conversion/convert_nanobind.md, Phase 4 removes them). Bindings
/// registered through nanobind can still use those types as lambda parameter / return types:
/// both libraries wrap a ``PyObject*``, so the conversion is a (borrowed) reinterpretation.
///
/// The opposite direction (bound C++ classes reached from pybind11 code, i.e. liblasyten) is
/// handled by ``lasyten/py_nanobind_bridge.h``.

#include <nanobind/nanobind.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <type_traits>

namespace lasyten::py_bridge {

template<typename T>
struct is_pybind_array : std::false_type
{};
template<>
struct is_pybind_array<pybind11::array> : std::true_type
{};
template<typename T, int Flags>
struct is_pybind_array<pybind11::array_t<T, Flags>> : std::true_type
{};

} // namespace lasyten::py_bridge

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

/// ``pybind11::handle`` / ``pybind11::object`` and all subclasses (array, dict, str, ...).
template<typename T>
struct type_caster<T, enable_if_t<std::is_base_of_v<pybind11::handle, T>>>
{
    static constexpr auto Name = const_name("typing.Any");
    template<typename T_>
    using Cast = movable_cast_t<T_>;
    template<typename T_>
    static constexpr bool can_cast()
    {
        return true;
    }
    using Value = T;

    explicit operator Value*() { return &value; }
    explicit operator Value&() { return value; }
    explicit operator Value&&() { return std::move(value); }

    Value value;

    bool from_python(handle src, uint32_t flags, cleanup_list*) noexcept
    {
        pybind11::handle h(src.ptr());
        try {
            if constexpr (std::is_same_v<T, pybind11::handle>) {
                value = h;
                return true;
            } else if constexpr (std::is_same_v<T, pybind11::object>) {
                value = pybind11::reinterpret_borrow<pybind11::object>(h);
                return true;
            } else if constexpr (lasyten::py_bridge::is_pybind_array<T>::value) {
                bool const convert = (flags & (uint32_t)cast_flags::convert) != 0;
                if (!convert && !T::check_(h)) {
                    return false;
                }
                value = T::ensure(h);
                if (!value) {
                    PyErr_Clear();
                    return false;
                }
                return true;
            } else {
                if (!T::check_(h)) {
                    return false;
                }
                value = pybind11::reinterpret_borrow<T>(h);
                return true;
            }
        } catch (...) {
            PyErr_Clear();
            return false;
        }
    }

    static handle from_cpp(const Value& src, rv_policy, cleanup_list*) noexcept
    {
        if (!src.ptr()) {
            return handle();
        }
        Py_INCREF(src.ptr());
        return handle(src.ptr());
    }

    static handle from_cpp(Value&& src, rv_policy, cleanup_list*) noexcept
    {
        if constexpr (std::is_base_of_v<pybind11::object, T>) {
            return handle(src.release().ptr());
        } else {
            if (!src.ptr()) {
                return handle();
            }
            Py_INCREF(src.ptr());
            return handle(src.ptr());
        }
    }
};

// nanobind rejects ``None`` for every parameter unless ``nb::arg(...).none()`` is given or the
// parameter type "has arg defaults" (``std::optional``, ...). The pybind11 bindings accepted
// ``None`` for pybind object types (the caster type-checks where it matters) and for
// ``std::shared_ptr`` parameters (a null pointer); keep that behavior without having to annotate
// every single parameter.
#define LASYTEN_NB_ACCEPT_NONE(...)                                                               \
    template<>                                                                                    \
    struct has_arg_defaults<__VA_ARGS__> : std::true_type                                         \
    {};

LASYTEN_NB_ACCEPT_NONE(pybind11::handle)
LASYTEN_NB_ACCEPT_NONE(pybind11::object)
LASYTEN_NB_ACCEPT_NONE(pybind11::str)
LASYTEN_NB_ACCEPT_NONE(pybind11::bytes)
LASYTEN_NB_ACCEPT_NONE(pybind11::none)
LASYTEN_NB_ACCEPT_NONE(pybind11::int_)
LASYTEN_NB_ACCEPT_NONE(pybind11::float_)
LASYTEN_NB_ACCEPT_NONE(pybind11::bool_)
LASYTEN_NB_ACCEPT_NONE(pybind11::tuple)
LASYTEN_NB_ACCEPT_NONE(pybind11::list)
LASYTEN_NB_ACCEPT_NONE(pybind11::dict)
LASYTEN_NB_ACCEPT_NONE(pybind11::set)
LASYTEN_NB_ACCEPT_NONE(pybind11::iterable)
LASYTEN_NB_ACCEPT_NONE(pybind11::iterator)
LASYTEN_NB_ACCEPT_NONE(pybind11::sequence)
LASYTEN_NB_ACCEPT_NONE(pybind11::slice)
LASYTEN_NB_ACCEPT_NONE(pybind11::type)
LASYTEN_NB_ACCEPT_NONE(pybind11::function)
LASYTEN_NB_ACCEPT_NONE(pybind11::module_)
LASYTEN_NB_ACCEPT_NONE(pybind11::capsule)
LASYTEN_NB_ACCEPT_NONE(pybind11::array)
LASYTEN_NB_ACCEPT_NONE(pybind11::args)
LASYTEN_NB_ACCEPT_NONE(pybind11::kwargs)

#undef LASYTEN_NB_ACCEPT_NONE

template<typename T, int Flags>
struct has_arg_defaults<pybind11::array_t<T, Flags>> : std::true_type
{};

template<typename T>
struct has_arg_defaults<std::shared_ptr<T>> : std::true_type
{};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
