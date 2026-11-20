#pragma once

/// The single place in ``include/lasyten`` that pulls in the Python binding library (nanobind).
///
/// Every public header gets its ``py::`` / ``nb::`` names through this file (via
/// ``<lasyten/lasyten.h>``); nothing else may include ``<nanobind/...>`` or ``<Python.h>``
/// directly. pybind11 is **not** used anywhere (it cannot be compiled with ``Py_LIMITED_API``,
/// which nanobind's Stable ABI mode — ``LASYTEN_STABLE_ABI`` — requires).
///
/// ``LASYTEN_HAS_PYTHON`` (set by CMake from ``LASYTEN_WITH_PYTHON``; defaults to 1 for builds
/// that do not go through CMake) selects one of two modes:
///
/// * ``1`` — nanobind. ``lasyten::nb`` is ``nanobind``; ``lasyten::py`` is a *thin compatibility
///   namespace* over it: it re-exports the nanobind object types (``py::object``, ``py::handle``,
///   ``py::dict``, ``py::list``, ...) and keeps the familiar pybind11 spellings of a few helpers
///   (``py::reinterpret_borrow``, ``py::module_::import``, ``py::error_already_set``,
///   ``py::value_error(std::string)``, ...). ``py::array`` / ``py::array_t<T>`` / ``py::dtype``
///   are a small ``numpy.ndarray`` adapter (``detail/py_array.h``) that only uses the Stable-ABI
///   subset of the C-API. New code should prefer spelling ``nb::`` directly.
///
/// * ``0`` — *Python-free stub mode*. ``lasyten::py`` is a small namespace of **opaque** stand-ins
///   (``py::object``, ``py::handle``, ``py::array``, ``py::array_t<T>``, ``py::dict``, ...) that
///   have no members. Declarations that merely mention them in signatures (and trivial
///   definitions that never touch them) compile, but any real use — ``py::cast<T>(...)``,
///   ``.attr(...)``, ``py::cast(...)``, ``py::isinstance<...>`` — fails at *compile time* with
///   "no member / not declared". (Exception: ``py::value_error`` / ``type_error`` / ``key_error``
///   / ``index_error`` are ordinary ``std::runtime_error`` subclasses so that error paths keep
///   working.) ``LASYTEN_WITH_NUMPY_BACKEND``, ``LASYTEN_WITH_TORCH_BACKEND`` and the Python
///   model classes are unavailable in this mode (their headers ``#error``).
///
/// Use ``#if LASYTEN_HAS_PYTHON`` around the Python-boundary parts of a header or source file, and
/// in headers that are inherently Python-only put::
///
///     #include <lasyten/detail/python_api.h>
///     #if !LASYTEN_HAS_PYTHON
///     #error "<this header> needs LASYTEN_WITH_PYTHON=ON"
///     #endif
///
/// Rules of the road: new code in ``include/lasyten`` must not add new ``py::`` mentions; move
/// Python-facing overloads into ``pybind/`` instead.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef LASYTEN_HAS_PYTHON
#define LASYTEN_HAS_PYTHON 1
#endif

#if LASYTEN_HAS_PYTHON

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <optional>
#include <string_view>

/// Symbol visibility for classes that appear in bound signatures.
#define LASYTEN_EXPORT __attribute__((visibility("default")))

namespace lasyten {
namespace nb = ::nanobind;
} // namespace lasyten

// Parameters of class type (``std::shared_ptr<T>``, ``const T&``) reject ``None``, unless the
// binding says ``nb::arg("x") = nb::none()`` or ``nb::arg("x").none()``. Plain ``object`` /
// ``handle`` parameters accept ``None`` (the binding inspects the object itself, see also
// ``py::cast_non_null``).
NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)
template<>
struct has_arg_defaults<::nanobind::object> : std::true_type
{};
template<>
struct has_arg_defaults<::nanobind::handle> : std::true_type
{};
NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

#include <lasyten/detail/py_array.h>

namespace lasyten {

/// Compatibility namespace: pybind11-style spellings of nanobind (+ the numpy adapter types of
/// ``detail/py_array.h``, which already live in ``lasyten::py``).
namespace py {

using ::nanobind::arg;
using ::nanobind::arg_v;
using ::nanobind::args;
using ::nanobind::bool_;
using ::nanobind::borrow;
using ::nanobind::builtin_exception;
using ::nanobind::bytes;
using ::nanobind::callable;
using ::nanobind::cast;
using ::nanobind::cast_error;
using ::nanobind::ellipsis;
using ::nanobind::float_;
using ::nanobind::getattr;
using ::nanobind::handle;
using ::nanobind::hasattr;
using ::nanobind::int_;
using ::nanobind::isinstance;
using ::nanobind::iter;
using ::nanobind::iterator;
using ::nanobind::kwargs;
using ::nanobind::make_tuple;
using ::nanobind::none;
using ::nanobind::object;
using ::nanobind::python_error;
using ::nanobind::repr;
using ::nanobind::set;
using ::nanobind::steal;

/// ``py::function`` (any callable).
using function = ::nanobind::callable;
/// pybind11 spelling of ``nb::python_error``.
using error_already_set = ::nanobind::python_error;

template<typename T>
[[nodiscard]] inline T
reinterpret_borrow(::nanobind::handle h)
{
    return ::nanobind::borrow<T>(h);
}

template<typename T>
[[nodiscard]] inline T
reinterpret_steal(::nanobind::handle h)
{
    return ::nanobind::steal<T>(h);
}

/// ``py::cast<T>(h)`` for a ``std::shared_ptr`` that a Python argument must provide: ``None``
/// raises ``TypeError`` instead of becoming an empty pointer. ``what`` names the argument.
template<typename T>
[[nodiscard]] inline T
cast_non_null(::nanobind::handle h, char const* what = "argument")
{
    if (h.is_none()) {
        throw ::nanobind::type_error((std::string(what) + " must not be None").c_str());
    }
    return ::nanobind::cast<T>(h);
}

/// ``py::module_::import("numpy")`` (nanobind spells it ``import_``).
class module_ : public ::nanobind::module_
{
  public:
    using ::nanobind::module_::module_;
    module_(::nanobind::module_ m)
      : ::nanobind::module_(std::move(m))
    {
    }
    static module_ import(char const* name) { return module_(::nanobind::module_::import_(name)); }
};

/// ``py::str``: nanobind's ``str`` plus the ``std::string`` / ``std::string_view`` constructors.
class str : public ::nanobind::str
{
  public:
    using ::nanobind::str::str;
    str() = default;
    explicit str(std::string const& s)
      : ::nanobind::str(s.c_str(), s.size())
    {
    }
    explicit str(std::string_view s)
      : ::nanobind::str(s.data(), s.size())
    {
    }
    str(::nanobind::str s)
      : ::nanobind::str(std::move(s))
    {
    }
    /// ``py::str s = some_object;`` converts via ``str(obj)`` (as pybind11 did).
    /// (A template over ``api<T>`` so that accessors like ``obj.attr("x")`` are not first
    /// converted
    /// -- unchecked -- to a ``nb::str``.)
    template<typename T>
    str(::nanobind::detail::api<T> const& o)
      : ::nanobind::str(::nanobind::handle(o.derived().ptr()))
    {
    }
    /// pybind11's ``py::str`` converts to ``std::string`` implicitly.
    operator std::string() const { return std::string(c_str()); }
};

/// ``py::tuple`` with pybind11 semantics: ``tuple(n)`` creates a tuple of ``n`` unset items that
/// are then assigned with ``t[i] = value``; converts implicitly from any sequence.
class tuple : public ::nanobind::tuple
{
  public:
    using ::nanobind::tuple::tuple;
    tuple() = default;
    explicit tuple(ssize_t n)
      : ::nanobind::tuple(::nanobind::steal<::nanobind::tuple>(PyTuple_New(n)))
    {
        if (!m_ptr) {
            throw ::nanobind::python_error();
        }
    }
    tuple(::nanobind::tuple t)
      : ::nanobind::tuple(std::move(t))
    {
    }
    tuple(::nanobind::object const& o)
      : ::nanobind::tuple(::nanobind::handle(o))
    {
    }

    /// Element accessor policy: like nanobind's tuple accessor, but assignable
    /// (``PyTuple_SetItem``; only valid while the tuple is under construction).
    struct item_policy
    {
        static constexpr bool cache_dec_ref = false;
        using key_type = ssize_t;
        static PyObject* get(PyObject* o, ssize_t i)
        {
            PyObject* r = PyTuple_GetItem(o, i);
            if (r == nullptr) {
                throw ::nanobind::python_error();
            }
            return r;
        }
        static void set(PyObject* o, ssize_t i, PyObject* v)
        {
            Py_INCREF(v);
            if (PyTuple_SetItem(o, i, v) != 0) {
                throw ::nanobind::python_error();
            }
        }
    };
    [[nodiscard]] ::nanobind::detail::accessor<item_policy> operator[](ssize_t i) const
    {
        return { ::nanobind::handle(m_ptr), i };
    }
};

/// ``py::slice::compute`` and the pybind11 spelling of ``slice`` construction are below; this is
/// ``py::dict`` (converts implicitly from any mapping via ``dict(obj)``).
class dict : public ::nanobind::dict
{
  public:
    using ::nanobind::dict::dict;
    dict() = default;
    dict(::nanobind::dict d)
      : ::nanobind::dict(std::move(d))
    {
    }
    dict(::nanobind::object const& o)
      : ::nanobind::dict(::nanobind::borrow<::nanobind::dict>(
          ::nanobind::isinstance<::nanobind::dict>(o)
            ? ::nanobind::object(o)
            : ::nanobind::module_::import_("builtins").attr("dict")(o)))
    {
    }
};

/// ``py::iterable``: converts implicitly from any iterable object (as pybind11 did).
class iterable : public ::nanobind::iterable
{
  public:
    using ::nanobind::iterable::iterable;
    iterable() = default;
    iterable(::nanobind::iterable i)
      : ::nanobind::iterable(std::move(i))
    {
    }
    iterable(::nanobind::object const& o)
      : ::nanobind::iterable(::nanobind::handle(o), ::nanobind::detail::borrow_t{})
    {
        if (!::nanobind::iterable::check_(o)) {
            throw ::nanobind::type_error("object is not iterable");
        }
    }
};

/// ``py::sequence`` (+ ``size()`` / ``operator[]``).
class sequence : public ::nanobind::sequence
{
  public:
    using ::nanobind::sequence::sequence;
    sequence() = default;
    sequence(::nanobind::sequence q)
      : ::nanobind::sequence(std::move(q))
    {
    }
    sequence(::nanobind::object const& o)
      : ::nanobind::sequence(::nanobind::handle(o), ::nanobind::detail::borrow_t{})
    {
        if (!::nanobind::sequence::check_(o)) {
            throw ::nanobind::type_error("object is not a sequence");
        }
    }
    [[nodiscard]] ssize_t size() const
    {
        ssize_t const n = PySequence_Size(m_ptr);
        if (n < 0) {
            throw ::nanobind::python_error();
        }
        return n;
    }
};

/// ``py::list``: ``list(n)`` creates a list of ``n`` unset items (assign with ``l[i] = v``),
/// converts implicitly from any sequence.
class list : public ::nanobind::list
{
  public:
    using ::nanobind::list::list;
    list() = default;
    explicit list(ssize_t n)
      : ::nanobind::list(::nanobind::steal<::nanobind::list>(PyList_New(n)))
    {
        if (!m_ptr) {
            throw ::nanobind::python_error();
        }
    }
    list(::nanobind::list l)
      : ::nanobind::list(std::move(l))
    {
    }
    list(::nanobind::object const& o)
      : ::nanobind::list(::nanobind::handle(o))
    {
    }
};

/// ``py::len(obj)``.
[[nodiscard]] inline ssize_t
len(::nanobind::handle h)
{
    return static_cast<ssize_t>(::nanobind::len(h));
}

/// ``py::hash(obj)``.
[[nodiscard]] inline ssize_t
hash(::nanobind::handle h)
{
    ssize_t const r = PyObject_Hash(h.ptr());
    if (r == -1 && PyErr_Occurred()) {
        throw ::nanobind::python_error();
    }
    return r;
}

/// ``py::slice`` with pybind11 semantics: ``slice()`` is ``slice(None, None, None)``; integers or
/// ``std::optional`` integers (``std::nullopt`` for ``None``) are accepted.
class slice : public ::nanobind::slice
{
  public:
    using ::nanobind::slice::slice;
    slice()
      : ::nanobind::slice(::nanobind::none(), ::nanobind::none(), ::nanobind::none())
    {
    }
    slice(std::optional<ssize_t> start, std::optional<ssize_t> stop, std::optional<ssize_t> step)
      : ::nanobind::slice(opt_(start), opt_(stop), opt_(step))
    {
    }
    template<typename A,
             typename B,
             typename C,
             std::enable_if_t<std::is_base_of_v<::nanobind::handle, A> &&
                                std::is_base_of_v<::nanobind::handle, B> &&
                                std::is_base_of_v<::nanobind::handle, C>,
                              int> = 0>
    slice(A const& start, B const& stop, C const& step)
      : ::nanobind::slice(::nanobind::handle(start),
                          ::nanobind::handle(stop),
                          ::nanobind::handle(step))
    {
    }

    /// pybind11-style: resolve the slice against a sequence of length ``length``.
    bool compute(std::size_t length,
                 std::size_t* start,
                 std::size_t* stop,
                 std::size_t* step,
                 std::size_t* slicelength) const
    {
        ssize_t a = 0, b = 0, c = 0;
        if (PySlice_Unpack(m_ptr, &a, &b, &c) != 0) {
            return false;
        }
        *slicelength =
          static_cast<std::size_t>(PySlice_AdjustIndices(static_cast<ssize_t>(length), &a, &b, c));
        *start = static_cast<std::size_t>(a);
        *stop = static_cast<std::size_t>(b);
        *step = static_cast<std::size_t>(c);
        return true;
    }
    bool compute(ssize_t length,
                 ssize_t* start,
                 ssize_t* stop,
                 ssize_t* step,
                 ssize_t* slicelength) const
    {
        if (PySlice_Unpack(m_ptr, start, stop, step) != 0) {
            return false;
        }
        *slicelength = PySlice_AdjustIndices(length, start, stop, *step);
        return true;
    }

  private:
    static ::nanobind::object opt_(std::optional<ssize_t> v)
    {
        return v ? ::nanobind::object(::nanobind::int_(*v))
                 : ::nanobind::object(::nanobind::none());
    }
};

/// ``py::type::of<T>()`` (the Python class object of a bound C++ class).
struct type
{
    template<typename T>
    [[nodiscard]] static ::nanobind::object of()
    {
        return ::nanobind::borrow(::nanobind::type<T>());
    }
    /// ``py::type::of(obj)``: the Python type of an object.
    [[nodiscard]] static ::nanobind::object of(::nanobind::handle h)
    {
        return ::nanobind::borrow(h.type());
    }
};

// nanobind's builtin exceptions are factory functions taking ``const char*``; these add the
// ``std::string`` spellings used throughout the library (``throw py::value_error(msg)``).
#define LASYTEN_PY_EXCEPTION(name)                                                                \
    [[nodiscard]] inline ::nanobind::builtin_exception name(std::string const& what)              \
    {                                                                                             \
        return ::nanobind::name(what.c_str());                                                    \
    }                                                                                             \
    [[nodiscard]] inline ::nanobind::builtin_exception name(char const* what = nullptr)           \
    {                                                                                             \
        return ::nanobind::name(what);                                                            \
    }
LASYTEN_PY_EXCEPTION(value_error)
LASYTEN_PY_EXCEPTION(type_error)
LASYTEN_PY_EXCEPTION(key_error)
LASYTEN_PY_EXCEPTION(index_error)
#undef LASYTEN_PY_EXCEPTION

/// ``py::cast(lvalue)`` for a bound class: pybind11 first looks up the Python object that already
/// wraps this very C++ instance and only copies the value if there is none. nanobind copies right
/// away for lvalues (which aborts for non-copyable types like tensors), so look up first.
/// (An lvalue-reference overload: more specialized than ``nb::cast(T&&)``, which stays visible
/// through the ``using`` above, and not viable for ``py::`` object types, so nanobind's own
/// unqualified ``cast(...)`` calls (found by ADL) are not ambiguous.)
template<typename T,
         std::enable_if_t<::nanobind::detail::is_base_caster_v<::nanobind::detail::make_caster<T>>,
                          int> = 0>
[[nodiscard]] inline ::nanobind::object
cast(T& value, ::nanobind::rv_policy policy = ::nanobind::rv_policy::automatic_reference)
{
    using Caster = ::nanobind::detail::make_caster<T>;
    if (policy == ::nanobind::rv_policy::automatic ||
        policy == ::nanobind::rv_policy::automatic_reference) {
        ::nanobind::handle existing =
          Caster::from_cpp(value, ::nanobind::rv_policy::none, nullptr);
        if (existing.is_valid()) {
            return ::nanobind::steal(existing);
        }
    }
    return ::nanobind::cast(value, policy);
}

/// pybind11's ``cast<py::tuple>(obj)`` converted other types (``tuple(obj)``); nanobind's
/// ``nb::cast`` only type-checks. These helpers (``py::to_tuple`` etc.) keep the converting
/// behavior.
[[nodiscard]] inline tuple
to_tuple(::nanobind::handle h)
{
    return ::nanobind::isinstance<tuple>(h) ? ::nanobind::borrow<tuple>(h)
                                            : tuple(::nanobind::borrow<::nanobind::object>(h));
}
/// ``cast<bool>(obj)`` in pybind11 accepted anything with ``__bool__`` (e.g. ``numpy.bool_``);
/// nanobind's bool caster only accepts ``True`` / ``False``.
[[nodiscard]] inline bool
to_bool(::nanobind::handle h)
{
    int r = PyObject_IsTrue(h.ptr());
    if (r < 0) {
        throw ::nanobind::python_error();
    }
    return r != 0;
}
[[nodiscard]] inline list
to_list(::nanobind::handle h)
{
    return ::nanobind::isinstance<list>(h) ? ::nanobind::borrow<list>(h)
                                           : list(::nanobind::borrow<::nanobind::object>(h));
}
[[nodiscard]] inline dict
to_dict(::nanobind::handle h)
{
    return ::nanobind::isinstance<dict>(h) ? ::nanobind::borrow<dict>(h)
                                           : dict(::nanobind::borrow<::nanobind::object>(h));
}

} // namespace py

} // namespace lasyten

#else // !LASYTEN_HAS_PYTHON

#define LASYTEN_EXPORT

namespace lasyten {

/// Opaque, member-less stand-ins for the nanobind types that still occur in public signatures.
///
/// They are complete (so that declarations, defaulted ``py::none()`` arguments and trivial
/// bodies compile) but offer *no* operations: any attempt to actually use a Python object is a
/// compile error. Never add members here.
namespace py {

class handle
{};

class object : public handle
{};

class none : public object
{};

class str : public object
{};

class dict : public object
{};

class iterable : public object
{};

class array : public object
{};

template<typename T, int ExtraFlags = 0>
class array_t : public array
{};

/// The one *functional* part of the stub namespace: the builtin-exception types are ordinary
/// ``std::runtime_error`` subclasses, so that error paths of otherwise Python-free code
/// (``throw py::value_error(...)``) keep working without an interpreter.
#define LASYTEN_PY_STUB_EXCEPTION(name)                                                           \
    class name : public std::runtime_error                                                        \
    {                                                                                             \
      public:                                                                                     \
        using std::runtime_error::runtime_error;                                                  \
    }
LASYTEN_PY_STUB_EXCEPTION(value_error);
LASYTEN_PY_STUB_EXCEPTION(type_error);
LASYTEN_PY_STUB_EXCEPTION(key_error);
LASYTEN_PY_STUB_EXCEPTION(index_error);
#undef LASYTEN_PY_STUB_EXCEPTION

} // namespace py

} // namespace lasyten

#endif // LASYTEN_HAS_PYTHON

namespace lasyten {

/// Heap-allocate a Python handle and never destroy it.
///
/// Function-local or global ``py::object`` destructors run from ``atexit`` *after*
/// ``Py_Finalize``, and ``Py_DECREF`` then segfaults (Python 3.14: ``tstate == NULL``).
template<typename T>
[[nodiscard]] T&
leak_py_object(T obj)
{
    return *new T(std::move(obj));
}

} // namespace lasyten
