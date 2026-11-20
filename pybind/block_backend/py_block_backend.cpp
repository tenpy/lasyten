#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "../py_numbers.h"
#include "docstrings/block_backend/block_backend.h"
#include "ndarray_dispatch.h"
#if LASYTEN_WITH_NUMPY_BACKEND
#include "py_array_api.cpp"
#endif
#include "py_dtypes.cpp"
#include "py_eigen.cpp"
#if LASYTEN_WITH_NUMPY_BACKEND
#include "py_numpy.cpp"
#endif
#include "py_torch.cpp"
#include "py_trampolines.hpp"

#include "tools/hdf5_bind.h"
#include "tools/pickle_bind.h"
#include <lasyten/block_backend/block_backend.h>
#if LASYTEN_WITH_NUMPY_BACKEND
#include <lasyten/block_backend/array_api.h>
#include <lasyten/block_backend/numpy.h>
#endif
#include <lasyten/block_backend/torch.h>
#include <lasyten/config.h>
#include <lasyten/symmetries/spaces.h>
#include <optional>
#include <span>
#include <sstream>
#include <string>

namespace lasyten {

namespace {

void
require_implicit_scalar_conversion(const char* dest)
{
    if (!get_config().implicit_scalar_conversion) {
        throw py::type_error(
          std::string("Implicit conversion of Scalar to ") + dest +
          " is disabled. Use as_float64() / as_complex128() / to_numpy(), or set "
          "lasyten.set_option('implicit_scalar_conversion', True).");
    }
}

bool
is_py_integer_scalar(const py::handle& obj)
{
    if (obj.is_none())
        return false;
    // Numpy arrays are sequences and may expose __index__, but are not scalar indices.
    if (py::isinstance<py::array>(obj))
        return false;
    if (py::isinstance<py::slice>(obj))
        return false;
    try {
        (void)py::cast<int64>(obj);
        return true;
    } catch (const py::cast_error&) {
        return false;
    }
}

/// A numpy complex scalar (``np.complex128``, ``np.complex64``, ...) or 0-d complex array, as
/// complex128.
///
/// Used as the *last* overload of every function that has a ``float64`` and a ``complex128``
/// overload. nanobind first tries all overloads without implicit conversions, where the casters
/// accept only an exact ``float`` / ``complex``, so numpy complex scalars match nothing. In the
/// second pass (with conversions), the ``float64`` overload then takes them through ``__float__``,
/// which drops the imaginary part and only warns. An overload using this helper is reached in the
/// first pass, and converts losslessly via ``__complex__``. Everything else (real numpy scalars,
/// ``np.bool_``, non-numbers, ...) is passed on to the next overload by ``nb::next_overload()``,
/// i.e. is handled exactly as without that overload.
complex128
complex_like_or_next_overload(const py::handle& obj)
{
    try {
        if (!PyComplex_Check(obj.ptr())) {
            // dtype.kind == 'c' and ndim == 0, e.g. np.complex64 or a 0-d complex array
            py::object dtype = py::getattr(obj, "dtype", py::none());
            py::object kind =
              dtype.is_none() ? py::none() : py::getattr(dtype, "kind", py::none());
            if (!py::isinstance<py::str>(kind) || py::cast<std::string>(kind) != "c")
                throw nb::next_overload();
            py::object ndim = py::getattr(obj, "ndim", py::none());
            if (!py::isinstance<py::int_>(ndim) || py::cast<int64>(ndim) != 0)
                throw nb::next_overload();
        }
        return py::cast<complex128>(obj);
    } catch (const py::python_error&) {
        throw nb::next_overload();
    } catch (const py::cast_error&) {
        throw nb::next_overload();
    }
}

/// ``BlockBackend::as_scalar`` for numpy scalars and 0-d numpy arrays / torch tensors, with the
/// dtype determined by the kind of the dtype (bool, integer, float, complex) via ``value.item()``.
///
/// Used as the *last* ``as_scalar`` overload. nanobind first tries all overloads without implicit
/// conversions, where the casters accept only an exact ``bool`` / ``int`` / ``float`` /
/// ``complex``, so none of these types matches. In the second pass (with conversions), the
/// ``float64`` overload would take them through ``__float__``, which drops the imaginary part of
/// a numpy complex scalar and only warns, and turns ``np.bool_`` into ``float64``. This overload
/// is reached in the first pass instead. Everything else (non-numbers, ``int`` too large for
/// ``int64``, ...) is passed on to the next overload by ``nb::next_overload()``, i.e. is handled
/// exactly as without this overload.
BlockBackend::Scalar
as_scalar_from_zero_dim_number_or_next_overload(BlockBackend& backend, const py::object& value)
{
    auto number = py_number_from_zero_dim(value);
    if (!number) {
        // a subclass of complex, which is not a numpy scalar
        if (PyComplex_Check(value.ptr()))
            return backend.as_scalar(py::cast<complex128>(value));
        throw nb::next_overload();
    }
    PyObject* n = number->ptr();
    if (PyBool_Check(n))
        return backend.as_scalar(n == Py_True);
    if (PyLong_Check(n)) {
        long long v = PyLong_AsLongLong(n);
        if (v == -1 && PyErr_Occurred()) { // does not fit into int64
            PyErr_Clear();
            throw nb::next_overload();
        }
        return backend.as_scalar(static_cast<int64>(v));
    }
    if (PyFloat_Check(n))
        return backend.as_scalar(static_cast<float64>(PyFloat_AsDouble(n)));
    return backend.as_scalar(py::cast<complex128>(*number));
}

/// Parse ``key`` as an integer or tuple/list of integer scalars (not slices, strings, or arrays).
bool
try_parse_int_index_sequence(const py::object& key, std::vector<int64>& indices)
{
    if (is_py_integer_scalar(key)) {
        indices = { py::cast<int64>(key) };
        return true;
    }
    if (!py::isinstance<py::tuple>(key) && !py::isinstance<py::list>(key))
        return false;
    const py::sequence seq = key;
    indices.clear();
    indices.reserve(static_cast<std::size_t>(seq.size()));
    for (py::ssize_t i = 0; i < seq.size(); ++i) {
        const py::handle item = seq[i];
        if (!is_py_integer_scalar(item))
            return false;
        indices.push_back(py::cast<int64>(item));
    }
    return true;
}

bool
is_scalar_element_index(const BlockBackend::Block& block,
                        const py::object& key,
                        const std::vector<int64>& indices)
{
    if (indices.size() == 1 && block.ndim() == 1 && is_py_integer_scalar(key))
        return true;
    return indices.size() == static_cast<std::size_t>(block.ndim());
}

} // namespace

void
bind_block_backend(nb::module_& m)
{
    bind_block_backend_dtypes(m);

    nb::class_<BlockBackend> block_backend(m, "BlockBackend");
    block_backend.doc() = "Abstract base class that defines the operation on dense blocks.";

    // Declare nested shells before methods so signatures resolve Python type names (stubgen).
    nb::class_<BlockBackend::Block> block_cls(
      block_backend, "BlockCls", "Abstract base for dense blocks.");
    nb::class_<BlockBackend::Scalar> scalar(
      block_backend,
      "Scalar",
      "Scalar value with a Dtype, stored as a 0-d Block (may live on GPU).\n\n"
      "Explicit conversion: as_float64(), as_complex128(), as_int64(), as_bool(), to_numpy().\n"
      "Implicit Python conversion (float(), complex(), np.asarray()) is gated by the config\n"
      "option implicit_scalar_conversion (default True; tests force False).");

    // Define Scalar methods before BlockCls ops that take Scalar.
    scalar
      .def(nb::init<std::shared_ptr<BlockBackend::Block>>(),
           nb::arg("block"),
           "Construct from a 0-d block (ndim == 0). Raises if block is null or ndim != 0.")
      .def_prop_ro("dtype", &BlockBackend::Scalar::dtype)
      .def(
        "__str__",
        [](const BlockBackend::Scalar& self) {
            std::ostringstream oss;
            oss << self;
            return oss.str();
        },
        "String representation of the scalar.")
      .def(
        "__repr__",
        [](const BlockBackend::Scalar& self) {
            std::ostringstream oss;
            oss << self;
            return oss.str();
        },
        "String representation of the scalar.")
      .def("as_float64",
           &BlockBackend::Scalar::as_float64,
           "As float; raises if dtype is not Float32/Float64.")
      .def("as_complex128",
           &BlockBackend::Scalar::as_complex128,
           "As complex (real/bool have zero imaginary part).")
      .def("as_int64", &BlockBackend::Scalar::as_int64, "As int64; raises if dtype is not Int64.")
      .def("as_bool", &BlockBackend::Scalar::as_bool, "As bool; raises if dtype is not Bool.")
      .def(
        "to_numpy", &BlockBackend::Scalar::to_numpy, DOC(lasyten, BlockBackend, Scalar, to_numpy))
      .def(
        "__float__",
        [](const BlockBackend::Scalar& self) {
            require_implicit_scalar_conversion("float");
            return self.as_float64();
        },
        "Convert to Python float. Gated by implicit_scalar_conversion; raises if dtype is "
        "complex or bool.")
      .def(
        "__complex__",
        [](const BlockBackend::Scalar& self) {
            require_implicit_scalar_conversion("complex");
            return self.as_complex128();
        },
        "Convert to Python complex. Gated by implicit_scalar_conversion.")
      .def(
        "__array__",
        [](const BlockBackend::Scalar& self, py::object dtype, py::object /*copy*/) {
            require_implicit_scalar_conversion("numpy");
            py::object val = self.to_numpy();
            auto np = py::module_::import("numpy");
            if (dtype.is_none())
                return np.attr("asarray")(val);
            return np.attr("asarray")(val, nb::arg("dtype") = dtype);
        },
        nb::arg("dtype").none() = nb::none(),
        nb::arg("copy").none() = nb::none(),
        "NumPy array protocol. Gated by implicit_scalar_conversion.")
      .def(
        "__bool__",
        [](const BlockBackend::Scalar& self) {
            return self.as_bool(); // throws if dtype is not Bool!
        },
        "Return value of boolean scalar. Raises if dtype != bool.")
      .def(
        "__neg__", [](const BlockBackend::Scalar& self) { return -self; }, "Unary negation.")
      .def("real", &BlockBackend::Scalar::real, "Real part as a Scalar (valid for any dtype).")
      .def(
        "imag", &BlockBackend::Scalar::imag, "Imaginary part as a Scalar (valid for any dtype).")
      .def("conj",
           &BlockBackend::Scalar::conj,
           "Complex conjugate as a Scalar. Real dtypes are unchanged.")
      .def("real_if_close",
           &BlockBackend::Scalar::real_if_close,
           nb::arg("tol") = 100.,
           "If close to real, return the real part; otherwise return this Scalar. "
           "`tol` is in multiples of machine epsilon (numpy.real_if_close convention).")
      .def("__abs__", &BlockBackend::Scalar::abs, "Absolute value.")
      .def("sqrt", &BlockBackend::Scalar::sqrt, "Square root.")
      .def("exp", &BlockBackend::Scalar::exp, "Elementwise / scalar exponential.")
      .def("log", &BlockBackend::Scalar::log, "Natural logarithm.")
      .def("pow", &BlockBackend::Scalar::pow, nb::arg("exponent"), "Raise to a scalar power.")
      .def(
        "__pow__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& exponent) {
            return self.pow(exponent);
        },
        nb::arg("exponent"),
        "Raise to a scalar power.")
      .def(
        "__pow__",
        [](const BlockBackend::Scalar& self, float64 exponent) {
            return self.pow(self._block()->get_backend()->as_scalar(exponent));
        },
        nb::arg("exponent"))
      .def(
        "__pow__",
        [](const BlockBackend::Scalar& self, complex128 exponent) {
            return self.pow(self._block()->get_backend()->as_scalar(exponent));
        },
        nb::arg("exponent"))
      .def(
        "__pow__",
        [](const BlockBackend::Scalar& self, py::object exponent) {
            return self.pow(
              self._block()->get_backend()->as_scalar(complex_like_or_next_overload(exponent)));
        },
        nb::arg("exponent"))
      .def(
        "__add__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self + other;
        },
        nb::arg("other"),
        "Addition with another scalar.")
      .def(
        "__add__",
        [](const BlockBackend::Scalar& self, float64 other) { return self + other; },
        nb::arg("other"))
      .def(
        "__add__",
        [](const BlockBackend::Scalar& self, complex128 other) { return self + other; },
        nb::arg("other"))
      .def(
        "__add__",
        [](const BlockBackend::Scalar& self, py::object other) {
            return self + complex_like_or_next_overload(other);
        },
        nb::arg("other"))
      .def(
        "__radd__",
        [](const BlockBackend::Scalar& self, float64 other) { return other + self; },
        nb::arg("other"))
      .def(
        "__radd__",
        [](const BlockBackend::Scalar& self, complex128 other) { return other + self; },
        nb::arg("other"))
      .def(
        "__radd__",
        [](const BlockBackend::Scalar& self, py::object other) {
            return complex_like_or_next_overload(other) + self;
        },
        nb::arg("other"))
      .def(
        "__sub__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self - other;
        },
        nb::arg("other"),
        "Subtraction with another scalar.")
      .def(
        "__sub__",
        [](const BlockBackend::Scalar& self, float64 other) { return self - other; },
        nb::arg("other"))
      .def(
        "__sub__",
        [](const BlockBackend::Scalar& self, complex128 other) { return self - other; },
        nb::arg("other"))
      .def(
        "__sub__",
        [](const BlockBackend::Scalar& self, py::object other) {
            return self - complex_like_or_next_overload(other);
        },
        nb::arg("other"))
      .def(
        "__rsub__",
        [](const BlockBackend::Scalar& self, float64 other) { return other - self; }, // reversed!
        nb::arg("other"))
      .def(
        "__rsub__",
        [](const BlockBackend::Scalar& self, complex128 other) {
            return other - self;
        }, // reversed!
        nb::arg("other"))
      .def(
        "__rsub__",
        [](const BlockBackend::Scalar& self, py::object other) {
            return complex_like_or_next_overload(other) - self;
        },
        nb::arg("other"))
      .def(
        "__mul__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self * other;
        },
        nb::arg("other"),
        "Multiplication with another scalar.")
      .def(
        "__mul__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Block& other) {
            return other * self;
        },
        nb::arg("other"),
        "Multiplication with another scalar.")
      .def(
        "__mul__",
        [](const BlockBackend::Scalar& self, float64 other) { return self * other; },
        nb::arg("other"))
      .def(
        "__mul__",
        [](const BlockBackend::Scalar& self, complex128 other) { return self * other; },
        nb::arg("other"))
      .def(
        "__mul__",
        [](const BlockBackend::Scalar&, py::object) -> py::object {
            return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        },
        nb::arg("other"))
      .def(
        "__rmul__",
        [](const BlockBackend::Scalar& self, float64 other) { return other * self; },
        nb::arg("other"))
      .def(
        "__rmul__",
        [](const BlockBackend::Scalar& self, complex128 other) { return other * self; },
        nb::arg("other"))
      .def(
        "__rmul__",
        [](const BlockBackend::Scalar& self, py::object other) {
            return complex_like_or_next_overload(other) * self;
        },
        nb::arg("other"))
      .def(
        "__truediv__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self / other;
        },
        nb::arg("other"))
      .def(
        "__truediv__",
        [](const BlockBackend::Scalar& self, float64 other) { return self / other; },
        nb::arg("other"))
      .def(
        "__truediv__",
        [](const BlockBackend::Scalar& self, complex128 other) { return self / other; },
        nb::arg("other"))
      .def(
        "__truediv__",
        [](const BlockBackend::Scalar& self, py::object other) {
            return self / complex_like_or_next_overload(other);
        },
        nb::arg("other"))
      .def(
        "__rtruediv__",
        [](const BlockBackend::Scalar& self, float64 other) {
            return other / self; // reversed!
        },
        nb::arg("other"))
      .def(
        "__rtruediv__",
        [](const BlockBackend::Scalar& self, complex128 other) {
            return other / self; // reversed!
        },
        nb::arg("other"))
      .def(
        "__rtruediv__",
        [](const BlockBackend::Scalar& self, py::object other) {
            return complex_like_or_next_overload(other) / self;
        },
        nb::arg("other"))
      .def(
        "__lt__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self < other;
        },
        nb::arg("other"))
      .def(
        "__lt__",
        [](const BlockBackend::Scalar& self, float64 other) { return self < other; },
        nb::arg("other"))
      .def(
        "__lt__",
        [](const BlockBackend::Scalar& self, py::object other) {
            return self < real_number_or_complex_not_ordered(other);
        },
        nb::arg("other"))
      .def(
        "__gt__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self > other;
        },
        nb::arg("other"))
      .def(
        "__gt__",
        [](const BlockBackend::Scalar& self, float64 other) { return self > other; },
        nb::arg("other"))
      .def(
        "__gt__",
        [](const BlockBackend::Scalar& self, py::object other) {
            return self > real_number_or_complex_not_ordered(other);
        },
        nb::arg("other"))
      .def(
        "__le__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self <= other;
        },
        nb::arg("other"))
      .def(
        "__le__",
        [](const BlockBackend::Scalar& self, float64 other) { return self <= other; },
        nb::arg("other"))
      .def(
        "__le__",
        [](const BlockBackend::Scalar& self, py::object other) {
            return self <= real_number_or_complex_not_ordered(other);
        },
        nb::arg("other"))
      .def(
        "__ge__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self >= other;
        },
        nb::arg("other"))
      .def(
        "__ge__",
        [](const BlockBackend::Scalar& self, float64 other) { return self >= other; },
        nb::arg("other"))
      .def(
        "__ge__",
        [](const BlockBackend::Scalar& self, py::object other) {
            return self >= real_number_or_complex_not_ordered(other);
        },
        nb::arg("other"))
      .def("inverse", &BlockBackend::Scalar::inverse, "The inverse of the scalar, 1./self")
      .def_prop_ro("_block", &BlockBackend::Scalar::_block, "Return the underlying block.")
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5<BlockBackend::Scalar>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"),
           "Save scalar to HDF5.")
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<BlockBackend::Scalar>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"),
                  "Load scalar from HDF5.");

    block_cls
      .def_prop_ro(
        "shape",
        [&](const BlockBackend::Block& self) { return py::to_tuple(py::cast(self.shape())); },
        "The shape of the block.")
      .def_prop_ro("dtype", &BlockBackend::Block::dtype)
      .def_prop_ro("device", &BlockBackend::Block::device)
      .def("get_backend",
           &BlockBackend::Block::get_backend,
           nb::rv_policy::reference,
           "Return the backend for this block's device.")
      .def(
        "__add__",
        [](const BlockBackend::Block& self, const BlockCPtr& other) { return self + (*other); },
        nb::arg("other"),
        "Elementwise addition with another block.")
      .def(
        "__sub__",
        [](const BlockBackend::Block& self, const BlockCPtr& other) { return self - (*other); },
        nb::arg("other"),
        "Elementwise subtraction with another block.")
      .def(
        "__mul__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& s) { return self * s; },
        nb::arg("other"),
        "Multiplication by a scalar.")
      .def(
        "__mul__",
        [](const BlockBackend::Block& self, float64 s) {
            return self * self.get_backend()->as_scalar(s);
        },
        nb::arg("other"))
      .def(
        "__mul__",
        [](const BlockBackend::Block& self, complex128 s) {
            return self * self.get_backend()->as_scalar(s);
        },
        nb::arg("other"))
      .def(
        "__rmul__",
        [](const BlockBackend::Block& self, BlockBackend::Scalar s) { return self * s; },
        nb::arg("other"))
      .def(
        "__rmul__",
        [](const BlockBackend::Block& self, float64 s) {
            return self.get_backend()->as_scalar(s) * self;
        },
        nb::arg("other"))
      .def(
        "__rmul__",
        [](const BlockBackend::Block& self, complex128 s) {
            return self.get_backend()->as_scalar(s) * self;
        },
        nb::arg("other"))
      .def(
        "__rmul__",
        [](const BlockBackend::Block& self, py::object other) {
            return self.get_backend()->as_scalar(complex_like_or_next_overload(other)) * self;
        },
        nb::arg("other"))
      .def(
        "__mul__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& s) { return self * s; },
        nb::arg("other"),
        "Elementwise multiplication with another block.")
      .def(
        "__mul__",
        [](const BlockBackend::Block& self, py::object other) {
            return self * self.get_backend()->as_scalar(complex_like_or_next_overload(other));
        },
        nb::arg("other"))
      .def(
        "__truediv__",
        [](const BlockBackend::Block& self, const BlockCPtr& other) {
            return self.operator/(*other);
        },
        nb::arg("other"),
        "Elementwise division with another block.")
      .def(
        "__truediv__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& s) { return self / s; },
        nb::arg("other"),
        "Division by a scalar.")
      .def(
        "__truediv__",
        [](const BlockBackend::Block& self, float64 s) {
            return self / self.get_backend()->as_scalar(s);
        },
        nb::arg("other"))
      .def(
        "__truediv__",
        [](const BlockBackend::Block& self, complex128 s) {
            return self / self.get_backend()->as_scalar(s);
        },
        nb::arg("other"))
      .def(
        "__truediv__",
        [](const BlockBackend::Block& self, py::object other) {
            return self / self.get_backend()->as_scalar(complex_like_or_next_overload(other));
        },
        nb::arg("other"))
      .def("__abs__", &BlockBackend::Block::abs, "Elementwise absolute value.")
      .def(
        "__pow__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& exponent) {
            return self.pow(exponent);
        },
        nb::arg("exponent"),
        "Elementwise power with scalar exponent.")
      .def(
        "__pow__",
        [](const BlockBackend::Block& self, float64 exponent) {
            return self.pow(self.get_backend()->as_scalar(exponent));
        },
        nb::arg("exponent"))
      .def(
        "__pow__",
        [](const BlockBackend::Block& self, complex128 exponent) {
            return self.pow(self.get_backend()->as_scalar(exponent));
        },
        nb::arg("exponent"))
      .def(
        "__pow__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& exponent) {
            return self.pow(exponent);
        },
        nb::arg("exponent"),
        "Elementwise power with block exponent.")
      .def(
        "__pow__",
        [](const BlockBackend::Block& self, py::object exponent) {
            return self.pow(
              self.get_backend()->as_scalar(complex_like_or_next_overload(exponent)));
        },
        nb::arg("exponent"))
      .def(
        "__lt__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& other) {
            return self < other;
        },
        nb::arg("other"),
        "Less than with another block.")
      .def(
        "__lt__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& other) {
            return self < other;
        },
        nb::arg("other"))
      .def(
        "__lt__",
        [](const BlockBackend::Block& self, float64 other) { return self < other; },
        nb::arg("other"))
      .def(
        "__lt__",
        [](const BlockBackend::Block& self, py::object other) {
            return self < real_number_or_complex_not_ordered(other);
        },
        nb::arg("other"))
      .def(
        "__le__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& other) {
            return self <= other;
        },
        nb::arg("other"),
        "Less than or equal to with another block.")
      .def(
        "__le__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& other) {
            return self <= other;
        },
        nb::arg("other"))
      .def(
        "__le__",
        [](const BlockBackend::Block& self, float64 other) { return self <= other; },
        nb::arg("other"))
      .def(
        "__le__",
        [](const BlockBackend::Block& self, py::object other) {
            return self <= real_number_or_complex_not_ordered(other);
        },
        nb::arg("other"))
      .def(
        "__gt__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& other) {
            return self > other;
        },
        nb::arg("other"),
        "Greater than with another block.")
      .def(
        "__gt__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& other) {
            return self > other;
        },
        nb::arg("other"))
      .def(
        "__gt__",
        [](const BlockBackend::Block& self, float64 other) { return self > other; },
        nb::arg("other"))
      .def(
        "__gt__",
        [](const BlockBackend::Block& self, py::object other) {
            return self > real_number_or_complex_not_ordered(other);
        },
        nb::arg("other"))
      .def(
        "__ge__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& other) {
            return self >= other;
        },
        nb::arg("other"),
        "Greater than or equal to with another block.")
      .def(
        "__ge__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& other) {
            return self >= other;
        },
        nb::arg("other"))
      .def(
        "__ge__",
        [](const BlockBackend::Block& self, float64 other) { return self >= other; },
        nb::arg("other"))
      .def(
        "__ge__",
        [](const BlockBackend::Block& self, py::object other) {
            return self >= real_number_or_complex_not_ordered(other);
        },
        nb::arg("other"))
      .def(
        "__eq__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& other) {
            return self == other;
        },
        nb::arg("other"),
        "Equality comparison with another block.")
      .def(
        "__ne__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& other) {
            return self != other;
        },
        nb::arg("other"),
        "Inequality comparison with another block.")
      .def(
        "__getitem__",
        [](BlockBackend::Block& self, py::object key) -> py::object {
            std::vector<int64> indices;
            if (try_parse_int_index_sequence(key, indices) &&
                is_scalar_element_index(self, key, indices)) {
                if (indices.size() == 1 && self.ndim() == 1 && is_py_integer_scalar(key))
                    return py::cast(self.get_item(indices[0]));
                return py::cast(self.get_item(indices));
            }
            if (auto idcs = BlockBackend::try_py_key_to_block_indices(key))
                return py::cast(self.get_item(std::span<const BlockBackend::BlockIndex>(*idcs)));
            return py::cast(self.get_item(key));
        },
        nb::arg("key"))
      .def(
        "__setitem__",
        [](BlockBackend::Block& self, py::object key, py::object value) {
            if (py::isinstance<BlockBackend::Scalar>(value)) {
                std::vector<int64> indices;
                if (try_parse_int_index_sequence(key, indices) &&
                    is_scalar_element_index(self, key, indices)) {
                    const auto& scalar = py::cast<BlockBackend::Scalar>(value);
                    if (indices.size() == 1 && self.ndim() == 1 && is_py_integer_scalar(key))
                        self.set_item(indices[0], scalar);
                    else
                        self.set_item(indices, scalar);
                    return;
                }
            }
            if (auto idcs = BlockBackend::try_py_key_to_block_indices(key)) {
                if (py::isinstance<BlockBackend::Block>(value)) {
                    self.set_item(std::span<const BlockBackend::BlockIndex>(*idcs),
                                  py::cast<BlockBackend::Block&>(value));
                    return;
                }
                if (py::isinstance<BlockBackend::Scalar>(value)) {
                    self.set_item(std::span<const BlockBackend::BlockIndex>(*idcs),
                                  py::cast<BlockBackend::Scalar&>(value));
                    return;
                }
            }
            self.set_item(key, value);
        },
        nb::arg("key"),
        nb::arg("value"))
      .def("to_numpy",
           nb::overload_cast<Dtype>(&BlockBackend::Block::to_numpy, nb::const_),
           nb::arg("dtype"),
           "Owning NumPy copy converted to dtype (never aliases block storage).")
      .def(
        "to_ndarray",
        [](nb::object self, bool writable, std::optional<std::string> framework) {
            auto& b = nb::cast<BlockBackend::Block&>(self);
            return ndarray_dispatch::export_to_python(ndarray_dispatch::block_to_ndarray(
              *b.get_backend(), b, self, writable, std::move(framework)));
        },
        nb::arg("writable") = true,
        nb::arg("framework") = nb::none(),
        "Array interchange view of the block. framework=None|'numpy'|'torch'|'jax'|'tensorflow' "
        "selects the DLPack consumer tag (defaults: numpy for Eigen/Numpy, torch for Torch). "
        "writable=False is read-only where supported.")
      .def("_item_as_complex128",
           &BlockBackend::Block::_item_as_complex128,
           "Return the element of a zero-dimensional block as a complex128.")
      .def("_item_as_int64",
           &BlockBackend::Block::_item_as_int64,
           "Return the element of a zero-dimensional block as a int64.")
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5<BlockBackend::Block>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"),
           "Save block state to HDF5.")
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<BlockBackend::Block>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"),
                  "Load block from HDF5 (subclass must implement).")
      .def(
        "__str__",
        [](const BlockBackend::Block& self) {
            std::ostringstream oss;
            oss << self;
            return oss.str();
        },
        "String representation of the block.")
      .def(
        "__repr__",
        [](const BlockBackend::Block& self) {
            std::ostringstream oss;
            oss << self;
            return oss.str();
        },
        "String representation of the block.");

    block_backend // init and attributes
      .def(lasyten::ctor<decltype(block_backend), std::string>(), nb::arg("device") = "cpu")
      .def_ro("default_device", &BlockBackend::default_device)
      .def("get_backend_name",
           &BlockBackend::get_backend_name,
           "Name of the backend class (e.g. \"EigenBlockBackend\", \"NumpyBlockBackend\").");

    block_backend //  methods
      .def("__repr__",
           [](const BlockBackend& self) { return self.get_backend_name() + std::string("()"); })
      .def("__str__",
           [](const BlockBackend& self) { return self.get_backend_name() + std::string("()"); })
      .def("__eq__",
           [](BlockBackend const& self, py::object other) {
               if (!py::isinstance<BlockBackend>(other)) {
                   return false;
               }
               return self == py::cast<BlockBackend const&>(other);
           })
      .def(
        "as_scalar",
        [](BlockBackend& self, py::bool_ value) { return self.as_scalar(py::to_bool(value)); },
        nb::arg("value"),
        "Convert a Python bool to a scalar block.")
      .def("as_scalar",
           nb::overload_cast<int64>(&BlockBackend::as_scalar),
           nb::arg("value"),
           "Convert an int64 to a scalar block.")
      .def("as_scalar",
           nb::overload_cast<float64>(&BlockBackend::as_scalar),
           nb::arg("value"),
           "Convert a float64 to a scalar block.")
      .def("as_scalar",
           nb::overload_cast<complex128>(&BlockBackend::as_scalar),
           nb::arg("value"),
           "Convert a complex128 to a scalar block.")
      .def("as_scalar",
           nb::overload_cast<py::object, Dtype>(&BlockBackend::as_scalar),
           nb::arg("value"),
           nb::arg("dtype"),
           "Convert a Python object to a scalar block with the given Dtype.")
      .def("as_scalar",
           nb::overload_cast<const BlockBackend::Scalar&>(&BlockBackend::as_scalar),
           nb::arg("value"),
           "Return a Scalar unchanged.")
      .def(
        "as_scalar",
        [](BlockBackend& self, py::object value) {
            return as_scalar_from_zero_dim_number_or_next_overload(self, value);
        },
        nb::arg("value"),
        "Convert a numpy scalar, or a 0-d numpy array or torch tensor, to a scalar block. "
        "The dtype of the scalar block is determined by the dtype kind (bool, int, float, "
        "complex), as for ``value.item()``.")
      .def("abs",
           &BlockBackend::abs,
           nb::arg("a"),
           "The absolute value of a complex number, elementwise.")
      .def(
        "apply_basis_perm",
        [](BlockBackend& self, BlockBackend::BlockCPtr block, py::iterable legs_obj, bool inv) {
            // Accept sequences or generators (e.g. conventional_leg_order(...)).
            std::vector<BlockBackend::LegCPtr> legs;
            for (py::handle item : legs_obj) {
                // Legs may be ElementarySpace / LegPipe / AbelianLegPipe; all register as Leg.
                legs.push_back(py::cast<Leg::Ptr>(item));
            }
            return self.apply_basis_perm(block, legs, inv);
        },
        nb::arg("block"),
        nb::arg("legs"),
        nb::arg("inv") = false,
        DOC(lasyten, BlockBackend, apply_basis_perm))
      .def("apply_leg_permutations",
           &BlockBackend::apply_leg_permutations,
           nb::arg("block"),
           nb::arg("perms"),
           "Apply permutations to every axis of a dense block")
      .def("as_block",
           &BlockBackend::as_block,
           nb::arg("a"),
           nb::arg("dtype").none() = nb::none(),
           nb::arg("device").none() = nb::none(),
           DOC(lasyten, BlockBackend, as_block))
      .def("as_device",
           &BlockBackend::as_device,
           nb::arg("device"),
           DOC(lasyten, BlockBackend, as_device))
      .def("abs_argmax",
           &BlockBackend::abs_argmax,
           nb::arg("block"),
           "Return the indices (one per axis) of the largest entry (by magnitude) of the block")
      .def("argmin",
           &BlockBackend::argmin,
           nb::arg("block"),
           "Return the indices (one per axis) of the smallest entry of the block")
      .def("add_axis", &BlockBackend::add_axis, nb::arg("a"), nb::arg("pos"))
      .def("all",
           &BlockBackend::all,
           nb::arg("a"),
           "Require a boolean block. If all of its entries are True")
      .def("allclose",
           &BlockBackend::allclose,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("rtol") = 1e-05,
           nb::arg("atol") = 1e-08)
      .def("angle",
           &BlockBackend::angle,
           nb::arg("a"),
           "The angle of a complex number such that ``a == exp(1.j * angle)``. Elementwise.")
      .def("any",
           &BlockBackend::any,
           nb::arg("a"),
           "Require a boolean block. If any of its entries are True")
      .def("apply_mask",
           &BlockBackend::apply_mask,
           nb::arg("block"),
           nb::arg("mask"),
           nb::arg("ax"),
           "Apply a mask (1D boolean block) to a block, slicing/projecting that axis")
      .def("argsort",
           &BlockBackend::argsort,
           nb::arg("block"),
           nb::arg("sort").none() = nb::none(),
           nb::arg("axis") = 0,
           DOC(lasyten, BlockBackend, argsort))
      .def("_argsort",
           &BlockBackend::_argsort,
           nb::arg("block"),
           nb::arg("axis"),
           "Like :meth:`block_argsort` but can assume real valued block, and sort ascending")
      .def("combine_legs",
           nb::overload_cast<const BlockCPtr&,
                             const std::vector<std::vector<int64>>&,
                             const std::vector<bool>&>(&BlockBackend::combine_legs),
           nb::arg("a"),
           nb::arg("leg_idcs_combine"),
           nb::arg("cstyles"),
           DOC(lasyten, BlockBackend, combine_legs))
      .def("combine_legs",
           nb::overload_cast<const BlockCPtr&, const std::vector<std::vector<int64>>&, bool>(
             &BlockBackend::combine_legs),
           nb::arg("a"),
           nb::arg("leg_idcs_combine"),
           nb::arg("cstyles") = true)
      .def("conj", &BlockBackend::conj, nb::arg("a"), "Complex conjugate of a block")
      .def("copy_block",
           &BlockBackend::copy_block,
           nb::arg("a"),
           nb::arg("device").none() = nb::none(),
           DOC(lasyten, BlockBackend, copy_block))
      .def(
        "cutoff_inverse",
        &BlockBackend::cutoff_inverse,
        nb::arg("a"),
        nb::arg("cutoff"),
        "The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``.")
      .def("dagger",
           &BlockBackend::dagger,
           nb::arg("a"),
           "Permute axes to reverse order and elementwise conj.")
      .def("get_dtype", &BlockBackend::get_dtype, nb::arg("a"))
      .def("eigh",
           &BlockBackend::eigh,
           nb::arg("block"),
           nb::arg("sort").none() = nb::none(),
           DOC(lasyten, BlockBackend, eigh))
      .def("eig",
           &BlockBackend::eig,
           nb::arg("block"),
           nb::arg("sort").none() = nb::none(),
           DOC(lasyten, BlockBackend, eig))
      .def("eigvalsh",
           &BlockBackend::eigvalsh,
           nb::arg("block"),
           nb::arg("sort").none() = nb::none(),
           DOC(lasyten, BlockBackend, eigvalsh))
      .def("eigvals",
           &BlockBackend::eigvals,
           nb::arg("block"),
           nb::arg("sort").none() = nb::none(),
           DOC(lasyten, BlockBackend, eigvals))
      .def("enlarge_leg",
           &BlockBackend::enlarge_leg,
           nb::arg("block"),
           nb::arg("mask"),
           nb::arg("axis"))
      .def("exp", &BlockBackend::exp, nb::arg("a"), DOC(lasyten, BlockBackend, exp))
      .def("block_from_diagonal",
           &BlockBackend::block_from_diagonal,
           nb::arg("diag"),
           "Return a 2D square block that has the 1D ``diag`` on the diagonal")
      .def("block_from_mask",
           &BlockBackend::block_from_mask,
           nb::arg("mask"),
           nb::arg("dtype"),
           DOC(lasyten, BlockBackend, block_from_mask))
      .def("block_from_numpy",
           &BlockBackend::block_from_numpy,
           nb::arg("a"),
           nb::arg("dtype").none() = nb::none(),
           nb::arg("device").none() = nb::none(),
           "Copy a NumPy array into a new block. Prefer block_from_ndarray for "
           "Torch/DLPack/buffer inputs (Eigen fast path; other backends fall back).")
      .def(
        "block_from_ndarray",
        [](BlockBackend& self,
           eigen_ndarray::AnyDeviceArray arr,
           std::optional<Dtype> dtype,
           std::optional<std::string> device) {
            return ndarray_dispatch::block_from_ndarray(self, arr, dtype, std::move(device));
        },
        nb::arg("arr"),
        nb::arg("dtype") = nb::none(),
        nb::arg("device") = nb::none(),
        "Import a CPU array (NumPy, PyTorch, DLPack, buffer protocol) into a new block. "
        "Eigen/Numpy/Torch use native paths; others route through block_from_numpy.")
      .def(
        "to_ndarray",
        [](BlockBackend& self,
           nb::object block,
           bool writable,
           std::optional<std::string> framework) {
            auto& b = nb::cast<BlockBackend::Block&>(block);
            return ndarray_dispatch::export_to_python(
              ndarray_dispatch::block_to_ndarray(self, b, block, writable, std::move(framework)));
        },
        nb::arg("block"),
        nb::arg("writable") = true,
        nb::arg("framework") = nb::none(),
        "Export a block as an array. framework selects numpy/torch/jax/tensorflow tagging. "
        "Contrast with to_numpy(), which always returns an owning NumPy copy.")
      .def("get_device", &BlockBackend::get_device, nb::arg("a"))
      .def("get_diagonal",
           &BlockBackend::get_diagonal,
           nb::arg("a"),
           nb::arg("tol"),
           "Get the diagonal of a 2D block as a 1D block")
      .def("imag",
           &BlockBackend::imag,
           nb::arg("a"),
           "The imaginary part of a complex number, elementwise.")
      .def("inner",
           &BlockBackend::inner,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("do_dagger"),
           DOC(lasyten, BlockBackend, inner))
      .def("is_real", &BlockBackend::is_real, nb::arg("a"), DOC(lasyten, BlockBackend, is_real))
      .def("item",
           &BlockBackend::item,
           nb::arg("a"),
           "Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as "
           "python float or complex")
      .def(
        "kron", &BlockBackend::kron, nb::arg("a"), nb::arg("b"), DOC(lasyten, BlockBackend, kron))
      .def("linear_combination",
           &BlockBackend::linear_combination,
           nb::arg("a"),
           nb::arg("v"),
           nb::arg("b"),
           nb::arg("w"))
      .def("log", &BlockBackend::log, nb::arg("a"), DOC(lasyten, BlockBackend, log))
      .def("max", &BlockBackend::max, nb::arg("a"))
      .def("max_abs", &BlockBackend::max_abs, nb::arg("a"))
      .def("min", &BlockBackend::min, nb::arg("a"))
      .def("mul",
           nb::overload_cast<const BlockBackend::Scalar&, const BlockCPtr&>(&BlockBackend::mul),
           nb::arg("a"),
           nb::arg("b"))
      .def("mul",
           nb::overload_cast<float64, const BlockCPtr&>(&BlockBackend::mul),
           nb::arg("a"),
           nb::arg("b"))
      .def("mul",
           nb::overload_cast<complex128, const BlockCPtr&>(&BlockBackend::mul),
           nb::arg("a"),
           nb::arg("b"))
      .def(
        "mul",
        [](BlockBackend& self, py::object a, const BlockCPtr& b) {
            return self.mul(complex_like_or_next_overload(a), b);
        },
        nb::arg("a"),
        nb::arg("b"))
      .def("norm",
           &BlockBackend::norm,
           nb::arg("a"),
           nb::arg("order") = 2,
           nb::arg("axis").none() = nb::none(),
           DOC(lasyten, BlockBackend, norm))
      .def("outer",
           &BlockBackend::outer,
           nb::arg("a"),
           nb::arg("b"),
           DOC(lasyten, BlockBackend, outer))
      .def("permute_axes", &BlockBackend::permute_axes, nb::arg("a"), nb::arg("permutation"))
      .def("permute_combined_matrix",
           &BlockBackend::permute_combined_matrix,
           nb::arg("block"),
           nb::arg("dims1"),
           nb::arg("idcs1"),
           nb::arg("dims2"),
           nb::arg("idcs2"),
           DOC(lasyten, BlockBackend, permute_combined_matrix))
      .def("permute_combined_idx",
           &BlockBackend::permute_combined_idx,
           nb::arg("block"),
           nb::arg("axis"),
           nb::arg("dims"),
           nb::arg("idcs"),
           DOC(lasyten, BlockBackend, permute_combined_idx))
      .def("random_normal",
           &BlockBackend::random_normal,
           nb::arg("dims"),
           nb::arg("dtype"),
           nb::arg("sigma"),
           nb::arg("device").none() = nb::none())
      .def("random_uniform",
           &BlockBackend::random_uniform,
           nb::arg("dims"),
           nb::arg("dtype"),
           nb::arg("device").none() = nb::none())
      .def("real",
           &BlockBackend::real,
           nb::arg("a"),
           "The real part of a complex number, elementwise.")
      .def("real_if_close",
           &BlockBackend::real_if_close,
           nb::arg("a"),
           nb::arg("tol"),
           DOC(lasyten, BlockBackend, real_if_close))
      .def("tile",
           &BlockBackend::tile,
           nb::arg("a"),
           nb::arg("repeats"),
           "Repeat a (1d) block multiple times. Similar to numpy.tile and torch.Tensor.repeat.")
      .def("_block_repr_lines",
           &BlockBackend::_block_repr_lines,
           nb::arg("a"),
           nb::arg("indent"),
           nb::arg("max_width"),
           nb::arg("max_lines"))
      .def("reshape", &BlockBackend::reshape, nb::arg("a"), nb::arg("shape"))
      .def("scale_axis",
           &BlockBackend::scale_axis,
           nb::arg("block"),
           nb::arg("factors"),
           nb::arg("axis"),
           DOC(lasyten, BlockBackend, scale_axis))
      .def(
        "get_shape",
        [](BlockBackend& self, const BlockCPtr& a) {
            return py::to_tuple(py::cast(self.get_shape(a)));
        },
        nb::arg("a"))
      .def("split_legs",
           nb::overload_cast<const BlockCPtr&,
                             const std::vector<int64>&,
                             const std::vector<std::vector<int64>>&,
                             const std::vector<bool>&>(&BlockBackend::split_legs),
           nb::arg("a"),
           nb::arg("idcs"),
           nb::arg("dims"),
           nb::arg("cstyles"),
           DOC(lasyten, BlockBackend, split_legs))
      .def("split_legs",
           nb::overload_cast<const BlockCPtr&,
                             const std::vector<int64>&,
                             const std::vector<std::vector<int64>>&,
                             bool>(&BlockBackend::split_legs),
           nb::arg("a"),
           nb::arg("idcs"),
           nb::arg("dims"),
           nb::arg("cstyles") = true)
      .def("sqrt", &BlockBackend::sqrt, nb::arg("a"), "The elementwise square root")
      .def("squeeze_axes", &BlockBackend::squeeze_axes, nb::arg("a"), nb::arg("idcs"))
      .def("stable_log",
           &BlockBackend::stable_log,
           nb::arg("block"),
           nb::arg("cutoff"),
           "Elementwise stable log. For entries > cutoff, yield their natural log. Otherwise 0.")
      .def("sum", &BlockBackend::sum, nb::arg("a"), nb::arg("ax"), "The sum over a single axis.")
      .def("sum_all", &BlockBackend::sum_all, nb::arg("a"), DOC(lasyten, BlockBackend, sum_all))
      .def("tdot",
           &BlockBackend::tdot,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("idcs_a"),
           nb::arg("idcs_b"))
      .def("tensor_outer",
           &BlockBackend::tensor_outer,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("K"),
           DOC(lasyten, BlockBackend, tensor_outer))
      .def("to_dtype", &BlockBackend::to_dtype, nb::arg("a"), nb::arg("dtype"))
      .def("to_numpy",
           &BlockBackend::to_numpy,
           nb::arg("a"),
           nb::arg("numpy_dtype").none() = nb::none())
      .def("trace_full", &BlockBackend::trace_full, nb::arg("a"))
      .def("trace_partial",
           &BlockBackend::trace_partial,
           nb::arg("a"),
           nb::arg("idcs1"),
           nb::arg("idcs2"),
           nb::arg("remaining_idcs"))
      .def("eye_block",
           &BlockBackend::eye_block,
           nb::arg("legs"),
           nb::arg("dtype"),
           nb::arg("device").none() = nb::none(),
           DOC(lasyten, BlockBackend, eye_block))
      .def("eye_matrix",
           &BlockBackend::eye_matrix,
           nb::arg("dim"),
           nb::arg("dtype"),
           nb::arg("device").none() = nb::none(),
           "The ``dim x dim`` identity matrix")
      .def("get_block_element", &BlockBackend::get_block_element, nb::arg("a"), nb::arg("idcs"))
      .def("get_block_mask_element",
           &BlockBackend::get_block_mask_element,
           nb::arg("a"),
           nb::arg("large_leg_idx"),
           nb::arg("small_leg_idx"),
           nb::arg("sum_block") = 0,
           DOC(lasyten, BlockBackend, get_block_mask_element))
      .def("matrix_dot",
           &BlockBackend::matrix_dot,
           nb::arg("a"),
           nb::arg("b"),
           "As in numpy.dot, both a and b might be matrix or vector.")
      .def("matrix_exp", &BlockBackend::matrix_exp, nb::arg("matrix"))
      .def("matrix_lq", &BlockBackend::matrix_lq, nb::arg("a"), nb::arg("full"))
      .def("matrix_qr",
           &BlockBackend::matrix_qr,
           nb::arg("a"),
           nb::arg("full"),
           "QR decomposition of a 2D block")
      .def("matrix_svd",
           &BlockBackend::matrix_svd,
           nb::arg("a"),
           nb::arg("algorithm"),
           "Perform a SVD decomposition of a matrix.")
      .def("possible_svd_algorithms",
           &BlockBackend::possible_svd_algorithms,
           "Possible algorithms for :meth:`matrix_svd`.")
      .def("ones_block",
           &BlockBackend::ones_block,
           nb::arg("shape"),
           nb::arg("dtype"),
           nb::arg("device").none() = nb::none())
      .def("synchronize",
           &BlockBackend::synchronize,
           "Wait for asynchronous processes (if any) to finish")
      .def("test_block_sanity",
           &BlockBackend::test_block_sanity,
           nb::arg("block"),
           nb::arg("expect_shape").none() = nb::none(),
           nb::arg("expect_dtype").none() = nb::none(),
           nb::arg("expect_device").none() = nb::none())
      .def("zeros",
           &BlockBackend::zeros,
           nb::arg("shape"),
           nb::arg("dtype"),
           nb::arg("device").none() = nb::none())
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5<BlockBackend>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<BlockBackend>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath")); // completed block_backend methods

    // --- pickle support (state mirrors save_hdf5 / from_hdf5; no HDF5 round trip) ---

    // BlockBackend: backends are per-device singletons, so unpickling calls
    // ``type(self).from_factory(default_device)`` (what HDF5 ``from_hdf5`` does as well).
    block_backend.def(
      "__reduce__",
      [](nb::object self) {
          auto const& bb = nb::cast<BlockBackend const&>(self);
          return lasyten::pickle::reduce_via_method(
            lasyten::pickle::type_of(self), "from_factory", bb.default_device);
      },
      "Pickle support: ``type(self).from_factory(default_device)``.");

    // Block: state = owning NumPy copy (dtype + shape are encoded in the array), rebuilt with
    // ``backend.block_from_numpy(arr, dtype, device)``. The backend itself pickles as above.
    block_cls.def(
      "__reduce__",
      [](nb::object self) {
          auto const& b = nb::cast<BlockBackend::Block const&>(self);
          nb::object arr = self.attr("to_numpy")();
          nb::object backend = nb::cast(b.get_backend(), nb::rv_policy::reference);
          return lasyten::pickle::reduce_via_method(
            backend, "block_from_numpy", arr, nb::cast(b.dtype()), b.device());
      },
      "Pickle support: ``backend.block_from_numpy(self.to_numpy(), dtype, device)``.");

    // Scalar: state = the underlying 0-d block (``Scalar(block)``).
    lasyten::pickle::def_reduce_ctor(
      scalar, [](BlockBackend::Scalar const& self) { return nb::make_tuple(self._block()); });

    // Transitional Python-dispatch backends (LASYTEN_WITH_NUMPY_BACKEND, convert_nanobind.md
    // leftover #5); EigenBlockBackend is the preferred CPU backend.
    m.attr("WITH_NUMPY_BACKEND") = bool(LASYTEN_WITH_NUMPY_BACKEND);
#if LASYTEN_WITH_NUMPY_BACKEND
    bind_block_backend_numpy(m);
#endif
    bind_block_backend_eigen(m);
    bind_block_backend_torch(m);
#if LASYTEN_WITH_NUMPY_BACKEND
    bind_block_backend_array_api(m);
#endif
}

} // namespace lasyten
