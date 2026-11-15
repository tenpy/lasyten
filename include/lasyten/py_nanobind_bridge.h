#pragma once

/// pybind11 <-> nanobind bridge for liblasyten (docs/cpp_conversion/convert_nanobind.md, Phase 3).
///
/// The Python extension ``lasyten._core`` is built with nanobind, but the public C++ API of
/// liblasyten still passes ``py::object`` / ``py::array`` around (Phase 4 removes that). Library
/// code also converts between Python objects and bound C++ types via ``obj.cast<Tensor::Ptr>()``,
/// ``py::cast(std::shared_ptr<Space>)``, ... Those conversions go through pybind11's type
/// registry, which knows nothing about nanobind-registered classes.
///
/// This header closes the gap: for every bound lasyten class ``T`` it specializes the pybind11
/// type casters of ``T``, ``std::shared_ptr<T>`` and ``std::shared_ptr<T const>`` so that they
/// delegate to nanobind. Both frameworks speak ``PyObject*``, so no further glue is needed.
///
/// Usage: nothing; this is included at the end of ``lasyten.h``. Use ``py_isinstance<T>(obj)`` in
/// place of ``py::isinstance<T>(obj)`` for bound lasyten classes (the pybind11 version always
/// answers ``false`` now).
///
/// Also: nanobind exceptions are translated to their pybind11 counterparts so that library code
/// that catches ``py::error_already_set`` / ``py::cast_error`` keeps working.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <utility>

namespace lasyten {

// Forward declarations of all bound lasyten classes / enums (see py_nanobind_bridge_types.inc).
#define LASYTEN_PYB_CLASS(key, name) key name;
#define LASYTEN_PYB_ENUM(name, underlying) enum class name : underlying;
#include <lasyten/py_nanobind_bridge_types.inc>
#undef LASYTEN_PYB_CLASS
#undef LASYTEN_PYB_ENUM

namespace py_bridge {

namespace nb = ::nanobind;

inline nb::handle
to_nb(pybind11::handle h)
{
    return nb::handle(h.ptr());
}

inline nb::rv_policy
to_nb_policy(pybind11::return_value_policy p)
{
    using P = pybind11::return_value_policy;
    switch (p) {
        case P::automatic:
            return nb::rv_policy::automatic;
        case P::automatic_reference:
            return nb::rv_policy::automatic_reference;
        case P::take_ownership:
            return nb::rv_policy::take_ownership;
        case P::copy:
            return nb::rv_policy::copy;
        case P::move:
            return nb::rv_policy::move;
        case P::reference:
            return nb::rv_policy::reference;
        case P::reference_internal:
            return nb::rv_policy::reference_internal;
    }
    return nb::rv_policy::automatic;
}

/// Run ``f`` and convert nanobind exceptions to the pybind11 ones.
template<typename F>
decltype(auto)
translate_nb_errors(F&& f)
{
    try {
        return std::forward<F>(f)();
    } catch (nb::python_error& e) {
        e.restore();
        throw pybind11::error_already_set();
    } catch (nb::cast_error const& e) {
        throw pybind11::cast_error(e.what());
    }
}

/// ``isinstance(obj, T)`` for a nanobind-bound class ``T``.
template<typename T>
[[nodiscard]] bool
isinstance_bound(pybind11::handle obj)
{
    return obj.ptr() != nullptr && nb::isinstance<T>(to_nb(obj));
}

inline constexpr std::uint32_t kConvertFlag =
  static_cast<std::uint32_t>(nb::detail::cast_flags::convert);

/// pybind11 caster for a nanobind-bound class ``T`` (plain value / pointer / reference).
template<typename T>
struct bound_caster : pybind11::detail::type_caster_generic
{
    // Deriving from type_caster_generic (without a pybind11 type_info) is what tells pybind11 that
    // ``obj.cast<T*>()`` / ``obj.cast<T&>()`` point into the Python object, not into the caster.
    static constexpr auto name = pybind11::detail::const_name("lasyten.bound");
    template<typename U>
    using cast_op_type = pybind11::detail::cast_op_type<U>;

    bound_caster()
      : pybind11::detail::type_caster_generic(
          static_cast<pybind11::detail::type_info const*>(nullptr))
    {
    }

    bool load(pybind11::handle src, bool convert)
    {
        if (!src) {
            return false;
        }
        if (src.is_none()) {
            ptr_ = nullptr;
            return true;
        }
        nb::detail::make_caster<T> c;
        // The cleanup list owns temporaries created by implicit conversions; keep it alive
        // together with this caster (i.e. for the duration of the call).
        std::uint32_t flags = convert ? kConvertFlag : 0u;
        cl_.reset(new nb::detail::cleanup_list(src.ptr()));
        if (!c.from_python(to_nb(src), flags, cl_.get())) {
            return false;
        }
        ptr_ = c.operator T*();
        return true;
    }

    operator T*() { return ptr_; }
    operator T&()
    {
        if (ptr_ == nullptr) {
            throw pybind11::reference_cast_error();
        }
        return *ptr_;
    }
    operator T&&() &&
    {
        if (ptr_ == nullptr) {
            throw pybind11::reference_cast_error();
        }
        return std::move(*ptr_);
    }

    /// pybind11 returns the already-existing Python wrapper of a registered C++ instance whatever
    /// the policy is; nanobind only does so for non-copy policies. Emulate pybind11 (and avoid
    /// copying non-copyable instances).
    static pybind11::handle find_existing(T const* p)
    {
        nb::object o = nb::find(*p);
        if (o.is_valid()) {
            return pybind11::handle(o.release().ptr());
        }
        return pybind11::handle();
    }

    static pybind11::handle cast(T const& v,
                                 pybind11::return_value_policy p,
                                 pybind11::handle parent)
    {
        if (auto h = find_existing(&v)) {
            return h;
        }
        if (p == pybind11::return_value_policy::automatic ||
            p == pybind11::return_value_policy::automatic_reference ||
            p == pybind11::return_value_policy::take_ownership) {
            p = pybind11::return_value_policy::copy;
        }
        return translate_nb_errors([&] {
            return pybind11::handle(nb::cast(v, to_nb_policy(p), to_nb(parent)).release().ptr());
        });
    }

    static pybind11::handle cast(T&& v, pybind11::return_value_policy, pybind11::handle)
    {
        return translate_nb_errors([&] {
            return pybind11::handle(nb::cast(std::move(v), nb::rv_policy::move).release().ptr());
        });
    }

    static pybind11::handle cast(T const* p,
                                 pybind11::return_value_policy policy,
                                 pybind11::handle parent)
    {
        if (p == nullptr) {
            return pybind11::none().release();
        }
        if (auto h = find_existing(p)) {
            return h;
        }
        if (policy == pybind11::return_value_policy::automatic) {
            policy = pybind11::return_value_policy::take_ownership;
        } else if (policy == pybind11::return_value_policy::automatic_reference) {
            policy = pybind11::return_value_policy::reference;
        }
        return translate_nb_errors([&] {
            return pybind11::handle(
              nb::cast(const_cast<T*>(p), to_nb_policy(policy), to_nb(parent)).release().ptr());
        });
    }

  private:
    struct cleanup_deleter
    {
        void operator()(nb::detail::cleanup_list* p) const noexcept
        {
            p->release();
            delete p;
        }
    };
    // Heap allocated so that the caster stays movable (pybind11 returns casters by value).
    std::unique_ptr<nb::detail::cleanup_list, cleanup_deleter> cl_;
    T* ptr_ = nullptr;
};

/// pybind11 caster for ``std::shared_ptr<T>`` of a nanobind-bound class ``T``.
template<typename T>
struct bound_sp_caster
{
    static constexpr auto name = pybind11::detail::const_name("lasyten.bound");
    template<typename U>
    using cast_op_type = pybind11::detail::movable_cast_op_type<U>;

    bool load(pybind11::handle src, bool /*convert*/)
    {
        if (!src) {
            return false;
        }
        if (src.is_none()) {
            value_.reset();
            return true;
        }
        nb::detail::cleanup_list cl(src.ptr());
        nb::detail::make_caster<std::shared_ptr<T>> c;
        bool ok = c.from_python(to_nb(src), 0u, &cl);
        if (ok) {
            value_ = c.operator std::shared_ptr<T>&();
        }
        cl.release();
        return ok;
    }

    operator std::shared_ptr<T>&() { return value_; }
    operator std::shared_ptr<T>&&() && { return std::move(value_); }
    operator std::shared_ptr<T>*() { return &value_; }
    operator T*() { return value_.get(); }
    operator T&()
    {
        if (!value_) {
            throw pybind11::reference_cast_error();
        }
        return *value_;
    }

    static pybind11::handle cast(std::shared_ptr<T> const& sp,
                                 pybind11::return_value_policy /*policy*/,
                                 pybind11::handle /*parent*/)
    {
        if (!sp) {
            return pybind11::none().release();
        }
        return translate_nb_errors([&] { return pybind11::handle(nb::cast(sp).release().ptr()); });
    }

  private:
    std::shared_ptr<T> value_;
};

/// pybind11 caster for an enum bound with ``nb::enum_``.
template<typename T>
struct bound_enum_caster
{
    static constexpr auto name = pybind11::detail::const_name("lasyten.enum");
    template<typename U>
    using cast_op_type = pybind11::detail::cast_op_type<U>;

    bool load(pybind11::handle src, bool convert)
    {
        if (!src) {
            return false;
        }
        nb::detail::cleanup_list cl(src.ptr());
        nb::detail::make_caster<T> c;
        bool ok = c.from_python(to_nb(src), convert ? kConvertFlag : 0u, &cl);
        if (ok) {
            value_ = c.operator T&();
        }
        cl.release();
        return ok;
    }

    operator T*() { return &value_; }
    operator T&() { return value_; }

    static pybind11::handle cast(T v, pybind11::return_value_policy, pybind11::handle)
    {
        return translate_nb_errors([&] { return pybind11::handle(nb::cast(v).release().ptr()); });
    }

  private:
    T value_{};
};

} // namespace py_bridge

/// ``isinstance(obj, T)`` for bound lasyten classes (replacement for ``py::isinstance<T>``).
template<typename T>
[[nodiscard]] bool
py_isinstance(pybind11::handle obj)
{
    return py_bridge::isinstance_bound<T>(obj);
}

/// For programs that embed Python themselves (``py::scoped_interpreter``) and link liblasyten:
/// the program carries its own copy of nanobind, which must be attached to the nanobind internals
/// shared with the ``lasyten._core`` extension. Call once, after the interpreter is up.
inline void
init_embedded_nanobind()
{
    auto core = pybind11::module_::import("lasyten._core");
    if (!::nanobind::register_module(::nanobind::handle(core.ptr()))) {
        throw pybind11::error_already_set();
    }
}

/// The Python class object of a bound lasyten class (replacement for ``py::type::of<T>()``).
template<typename T>
[[nodiscard]] pybind11::object
py_type_of()
{
    return pybind11::reinterpret_borrow<pybind11::object>(
      pybind11::handle(::nanobind::type<T>().ptr()));
}

} // namespace lasyten

#define LASYTEN_PY_BRIDGE_CLASS(...)                                                              \
    namespace pybind11::detail {                                                                  \
    template<>                                                                                    \
    struct type_caster<__VA_ARGS__> : ::lasyten::py_bridge::bound_caster<__VA_ARGS__>             \
    {};                                                                                           \
    template<>                                                                                    \
    struct type_caster<std::shared_ptr<__VA_ARGS__>>                                              \
      : ::lasyten::py_bridge::bound_sp_caster<__VA_ARGS__>                                        \
    {};                                                                                           \
    template<>                                                                                    \
    struct type_caster<std::shared_ptr<__VA_ARGS__ const>>                                        \
      : ::lasyten::py_bridge::bound_sp_caster<__VA_ARGS__ const>                                  \
    {};                                                                                           \
    }

#define LASYTEN_PY_BRIDGE_ENUM(...)                                                               \
    namespace pybind11::detail {                                                                  \
    template<>                                                                                    \
    struct type_caster<__VA_ARGS__> : ::lasyten::py_bridge::bound_enum_caster<__VA_ARGS__>        \
    {};                                                                                           \
    }

#define LASYTEN_PYB_CLASS(key, name) LASYTEN_PY_BRIDGE_CLASS(::lasyten::name)
#define LASYTEN_PYB_ENUM(name, underlying) LASYTEN_PY_BRIDGE_ENUM(::lasyten::name)
#include <lasyten/py_nanobind_bridge_types.inc>
#undef LASYTEN_PYB_CLASS
#undef LASYTEN_PYB_ENUM
