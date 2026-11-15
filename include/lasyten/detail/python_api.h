#pragma once

/// The single place in ``include/lasyten`` that pulls in the Python binding libraries.
///
/// (docs/cpp_conversion/convert_nanobind.md, Phase 5.) Every public header gets its ``py::``
/// types through this file (via ``<lasyten/lasyten.h>``); nothing else may include
/// ``<pybind11/...>``, ``<nanobind/...>`` or ``<Python.h>`` directly.
///
/// ``LASYTEN_HAS_PYTHON`` (set by CMake from ``LASYTEN_WITH_PYTHON``; defaults to 1 for builds
/// that do not go through CMake) selects one of two modes:
///
/// * ``1`` — pybind11 (+ the pybind11 <-> nanobind bridge): ``lasyten::py`` is ``pybind11``,
///   ``lasyten::nb`` is ``nanobind``. This is what ``lasyten._core`` needs today.
///
/// * ``0`` — *Python-free stub mode*. ``lasyten::py`` is a small namespace of **opaque** stand-ins
///   (``py::object``, ``py::handle``, ``py::array``, ``py::array_t<T>``, ``py::dict``, ...) that
///   have no members. Declarations that merely mention them in signatures (and trivial
///   definitions that never touch them) compile, but any real use — ``.cast<T>()``,
///   ``.attr(...)``,
///   ``py::cast(...)``, ``py::isinstance<...>`` — fails at *compile time* with "no member /
///   not declared". (Exception: ``py::value_error`` / ``type_error`` / ``key_error`` /
///   ``index_error`` are ordinary ``std::runtime_error`` subclasses so that error paths keep
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
/// Rules of the road for the remaining port: new code in ``include/lasyten`` must not add new
/// ``py::`` mentions; move Python-facing overloads into ``pybind/`` instead.

#include <cstddef>
#include <stdexcept>
#include <utility>

#ifndef LASYTEN_HAS_PYTHON
#define LASYTEN_HAS_PYTHON 1
#endif

#if LASYTEN_HAS_PYTHON

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/warnings.h>

/// Symbol visibility for classes that appear in bound signatures.
#define LASYTEN_EXPORT PYBIND11_EXPORT

namespace lasyten {

namespace py = ::pybind11;

} // namespace lasyten

// pybind11 <-> nanobind bridge: ``lasyten._core`` is a nanobind module, but the C++ API still
// uses pybind11 object types (convert_nanobind.md, Phase 4). It forward-declares all bound
// lasyten classes and specializes the pybind11 type casters for them, and defines
// ``LASYTEN_PY_BRIDGE_CLASS`` used at the end of several headers.
#include <lasyten/py_nanobind_bridge.h>

namespace lasyten {
namespace nb = ::nanobind;
} // namespace lasyten

#else // !LASYTEN_HAS_PYTHON

#define LASYTEN_EXPORT
#define LASYTEN_PY_BRIDGE_CLASS(...)
#define LASYTEN_PY_BRIDGE_ENUM(...)

namespace lasyten {

/// Opaque, member-less stand-ins for the pybind11 types that still occur in public signatures.
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
/// ``std::runtime_error`` subclasses (as in pybind11, where they derive from
/// ``std::runtime_error`` too), so that error paths of otherwise Python-free code (``throw
/// py::value_error(...)``) keep working without an interpreter.
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

/// Heap-allocate a pybind handle and never destroy it.
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
