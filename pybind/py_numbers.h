#pragma once

// What counts as "a number" for the Python bindings, in one place (tenpy/lasyten#330).
//
// Besides the Python numbers (``numbers.Number``), these are numpy scalars (including
// ``np.bool_``, which is not a ``numbers.Number``) and 0-d numpy arrays / torch tensors with a
// bool, integer, float or complex dtype. The number is ``x.item()``, i.e. a Python ``bool``,
// ``int``, ``float`` or ``complex``, so that every function treats them exactly like that Python
// number.

#include "py_lasyten_nanobind.h"

#include <optional>
#include <string>

namespace lasyten {

namespace detail_py_numbers {

/// The attribute ``name`` of the module ``module_name``, if that module was already imported
/// (an object of that module can only exist if so), else a null object.
inline py::object
attr_of_imported_module(const char* module_name, const char* name)
{
    PyObject* module = PyDict_GetItemString(PyImport_GetModuleDict(), module_name); // borrowed
    if (module == nullptr)
        return py::object();
    PyObject* attr = PyObject_GetAttrString(module, name);
    if (attr == nullptr) {
        PyErr_Clear();
        return py::object();
    }
    return py::reinterpret_steal<py::object>(attr);
}

inline bool
is_instance_of(PyObject* obj, const py::object& cls)
{
    if (!cls.is_valid())
        return false;
    int res = PyObject_IsInstance(obj, cls.ptr());
    if (res < 0) {
        PyErr_Clear();
        return false;
    }
    return res == 1;
}

} // namespace detail_py_numbers

/// ``obj.item()`` if ``obj`` is a numpy scalar or a 0-d numpy array / torch tensor with a bool,
/// integer, float or complex dtype, else ``std::nullopt``.
///
/// The result is always an *exact* Python ``bool``, ``int``, ``float`` or ``complex``. For
/// ``np.longdouble`` and ``np.clongdouble``, whose ``item()`` is the numpy scalar itself, it is
/// the ``float`` / ``complex`` (rounded to double precision).
inline std::optional<py::object>
py_number_from_zero_dim(const py::handle& obj)
{
    using detail_py_numbers::attr_of_imported_module;
    using detail_py_numbers::is_instance_of;
    PyObject* o = obj.ptr();
    PyTypeObject* type = Py_TYPE(o);
    // fast exit for the common builtin types: they are not zero-dimensional arrays
    if (o == Py_None || type == &PyFloat_Type || type == &PyLong_Type || type == &PyBool_Type ||
        type == &PyComplex_Type || type == &PyUnicode_Type)
        return std::nullopt;
    try {
        bool zero_dim_candidate = false;
        py::object numpy_generic = attr_of_imported_module("numpy", "generic");
        if (is_instance_of(o, numpy_generic)) {
            zero_dim_candidate = true;
        } else if (is_instance_of(o, attr_of_imported_module("numpy", "ndarray"))) {
            zero_dim_candidate = true;
        } else if (is_instance_of(o, attr_of_imported_module("torch", "Tensor"))) {
            zero_dim_candidate = true;
        }
        if (!zero_dim_candidate)
            return std::nullopt;
        py::object ndim = py::getattr(obj, "ndim", py::none());
        if (!py::isinstance<py::int_>(ndim) || py::cast<int64>(ndim) != 0)
            return std::nullopt;
        py::object dtype = py::getattr(obj, "dtype", py::none());
        py::object kind = py::getattr(dtype, "kind", py::none()); // numpy only, not torch
        char kind_char = 0;
        if (py::isinstance<py::str>(kind)) {
            // numpy: 'b' bool, 'i' / 'u' integer, 'f' float, 'c' complex (not datetime, str, ...)
            std::string k = py::cast<std::string>(kind);
            if (k.size() != 1 || std::string("biufc").find(k[0]) == std::string::npos)
                return std::nullopt;
            kind_char = k[0];
        }
        py::object item = obj.attr("item")();
        PyTypeObject* item_type = Py_TYPE(item.ptr());
        if (item_type == &PyBool_Type || item_type == &PyLong_Type || item_type == &PyFloat_Type ||
            item_type == &PyComplex_Type)
            return item;
        // np.longdouble, np.clongdouble: item() is the numpy scalar itself; use float / complex
        PyTypeObject* target = kind_char == 'c'   ? &PyComplex_Type
                               : kind_char == 'f' ? &PyFloat_Type
                                                  : nullptr;
        if (target == nullptr)
            return std::nullopt;
        PyObject* converted =
          PyObject_CallFunctionObjArgs(reinterpret_cast<PyObject*>(target), item.ptr(), nullptr);
        if (converted == nullptr)
            throw py::python_error();
        return py::reinterpret_steal<py::object>(converted);
    } catch (const py::python_error&) {
        return std::nullopt;
    } catch (const py::cast_error&) {
        return std::nullopt;
    }
}

/// If ``obj`` is a number: a Python number (``numbers.Number``), or one of the numbers that
/// ``py_number_from_zero_dim`` handles (e.g. ``np.bool_``, 0-d torch tensor).
inline bool
is_py_number(const py::handle& obj)
{
    if (PyFloat_Check(obj.ptr()) || PyLong_Check(obj.ptr()) || PyComplex_Check(obj.ptr()))
        return true;
    if (py::isinstance(obj, py::module_::import("numbers").attr("Number")))
        return true;
    return py_number_from_zero_dim(obj).has_value();
}

/// If ``obj`` is a numpy array or a torch tensor, of any dimension (including 0-d ones).
inline bool
is_py_array_or_tensor(const py::handle& obj)
{
    using detail_py_numbers::attr_of_imported_module;
    using detail_py_numbers::is_instance_of;
    return is_instance_of(obj.ptr(), attr_of_imported_module("numpy", "ndarray")) ||
           is_instance_of(obj.ptr(), attr_of_imported_module("torch", "Tensor"));
}

/// ``obj.item()`` for those numbers that are not ``numbers.Number`` (``np.bool_``, 0-d numpy
/// arrays, 0-d torch tensors); other objects are returned unchanged. For functions that operate on
/// the number itself, to give them the same result as for the Python number.
inline py::object
py_number_as_builtin(const py::object& obj)
{
    if (py::isinstance(obj, py::module_::import("numbers").attr("Number")))
        return obj;
    if (auto item = py_number_from_zero_dim(obj))
        return *item;
    return obj;
}

} // namespace lasyten
