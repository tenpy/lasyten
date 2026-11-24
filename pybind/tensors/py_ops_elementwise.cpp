#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/ops_elementwise.h>
#include <lasyten/tensors/tensor.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "../py_numbers.h"

#include "docstrings/tensors/ops_elementwise.h"

#include <format>

namespace lasyten {

namespace {

py::object
numpy()
{
    return py::module_::import("numpy");
}

bool
is_number_or_scalar(py::object x)
{
    return is_py_number(x) ||
           py::isinstance(x, py::module_::import("lasyten.block_backends").attr("Scalar")) ||
           (py::isinstance<Tensor>(x) &&
            py::to_bool(py::module_::import("lasyten.tensors._tensors").attr("is_scalar")(x)));
}

[[noreturn]] void
throw_elementwise_type_error(py::object x)
{
    throw py::type_error(std::format("Expected DiagonalTensor or scalar. Got {}",
                                     std::string(py::str(py::type::of(x)))));
}

template<typename TensorFn, typename ScalarPyFn>
py::object
dispatch_elementwise(py::object x, TensorFn tensor_fn, ScalarPyFn scalar_py_fn)
{
    x = py_number_as_builtin(x); // np.bool_, 0-d array / tensor: same as the Python number
    if (py::isinstance<DiagonalTensor>(x)) {
        return py::cast(tensor_fn(py::cast<DiagonalTensorCPtr>(x)));
    }
    if (is_number_or_scalar(x)) {
        return scalar_py_fn(x);
    }
    throw_elementwise_type_error(x);
}

template<typename TensorTensorFn, typename TensorScalarFn, typename ScalarTensorFn>
py::object
dispatch_pairwise(py::object x,
                  py::object y,
                  char const* numpy_func,
                  TensorTensorFn tensor_tensor_fn,
                  TensorScalarFn tensor_scalar_fn,
                  ScalarTensorFn scalar_tensor_fn)
{
    x = py_number_as_builtin(x);
    y = py_number_as_builtin(y);
    bool const x_is_tensor = py::isinstance<DiagonalTensor>(x);
    bool const y_is_tensor = py::isinstance<DiagonalTensor>(y);
    if (x_is_tensor && y_is_tensor) {
        return py::cast(
          tensor_tensor_fn(py::cast<DiagonalTensorCPtr>(x), py::cast<DiagonalTensorCPtr>(y)));
    }
    if (x_is_tensor && is_number_or_scalar(y)) {
        auto tensor = py::cast<DiagonalTensorCPtr>(x);
        auto bb = tensor->backend->block_backend;
        BlockBackend::Scalar scalar =
          py::isinstance<BlockBackend::Scalar>(y)
            ? bb->as_scalar(py::cast<BlockBackend::Scalar>(y))
            : BlockBackend::Scalar(bb->as_block(y, std::nullopt, std::nullopt));
        return py::cast(tensor_scalar_fn(tensor, scalar));
    }
    if (y_is_tensor && is_number_or_scalar(x)) {
        auto tensor = py::cast<DiagonalTensorCPtr>(y);
        auto bb = tensor->backend->block_backend;
        BlockBackend::Scalar scalar =
          py::isinstance<BlockBackend::Scalar>(x)
            ? bb->as_scalar(py::cast<BlockBackend::Scalar>(x))
            : BlockBackend::Scalar(bb->as_block(x, std::nullopt, std::nullopt));
        return py::cast(scalar_tensor_fn(scalar, tensor));
    }
    if (x_is_tensor || y_is_tensor) {
        throw_elementwise_type_error(x_is_tensor ? y : x);
    }
    return numpy().attr(numpy_func)(x, y);
}

} // namespace

void
bind_tensors_ops_elementwise(nb::module_& m)
{
    m.def(
      "angle",
      [](py::object x) {
          return dispatch_elementwise(
            x,
            [](DiagonalTensorCPtr t) { return angle(t); },
            [](py::object n) { return numpy().attr("angle")(n); });
      },
      nb::arg("x"),
      doc_plus(DOC(lasyten, angle),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~lasyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "cutoff_inverse",
      [](py::object x, float64 cutoff) {
          return dispatch_elementwise(
            x,
            [cutoff](DiagonalTensorCPtr t) { return cutoff_inverse(t, cutoff); },
            [cutoff](py::object n) {
                py::object abs_x = py::module_::import("builtins").attr("abs")(n);
                if (py::cast<float64>(abs_x) < cutoff) {
                    return py::object(py::int_(0));
                }
                return py::object(py::float_(1.0) / n);
            });
      },
      nb::arg("x"),
      nb::arg("cutoff") = 1e-15,
      doc_plus(DOC(lasyten, cutoff_inverse),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~lasyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "complex_conj",
      [](py::object x) {
          return dispatch_elementwise(
            x,
            [](DiagonalTensorCPtr t) { return complex_conj(t); },
            [](py::object n) {
                if (py::isinstance<BlockBackend::Scalar>(n))
                    return py::cast(py::cast<BlockBackend::Scalar>(n).conj());
                return numpy().attr("conj")(n);
            });
      },
      nb::arg("x"),
      doc_plus(DOC(lasyten, complex_conj),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~lasyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "imag",
      [](py::object x) {
          return dispatch_elementwise(
            x,
            [](DiagonalTensorCPtr t) { return imag(t); },
            [](py::object n) { return numpy().attr("imag")(n); });
      },
      nb::arg("x"),
      doc_plus(DOC(lasyten, imag),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~lasyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "real",
      [](py::object x) {
          return dispatch_elementwise(
            x,
            [](DiagonalTensorCPtr t) { return real(t); },
            [](py::object n) { return numpy().attr("real")(n); });
      },
      nb::arg("x"),
      doc_plus(DOC(lasyten, real),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~lasyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "real_if_close",
      [](py::object x, float64 tol) {
          return dispatch_elementwise(
            x,
            [tol](DiagonalTensorCPtr t) { return real_if_close(t, tol); },
            [tol](py::object n) {
                if (py::isinstance<BlockBackend::Scalar>(n))
                    return py::cast(py::cast<BlockBackend::Scalar>(n).real_if_close(tol));
                return numpy().attr("real_if_close")(n, nb::arg("tol") = tol);
            });
      },
      nb::arg("x"),
      nb::arg("tol") = 100.,
      doc_plus(DOC(lasyten, real_if_close),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~lasyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "sqrt",
      [](py::object x) {
          return dispatch_elementwise(
            x,
            [](DiagonalTensorCPtr t) { return sqrt(t); },
            [](py::object n) { return numpy().attr("sqrt")(n); });
      },
      nb::arg("x"),
      doc_plus(DOC(lasyten, sqrt),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~lasyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "stable_log",
      [](py::object x, float64 cutoff) {
          return dispatch_elementwise(
            x,
            [cutoff](DiagonalTensorCPtr t) { return stable_log(t, cutoff); },
            [cutoff](py::object n) {
                auto np = numpy();
                return np.attr("where")(np.attr("greater")(n, cutoff), np.attr("log")(n), 0.0);
            });
      },
      nb::arg("x"),
      nb::arg("cutoff") = 1e-30,
      doc_plus(DOC(lasyten, stable_log),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~lasyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "minimum",
      [](py::object x, py::object y) {
          return dispatch_pairwise(
            x,
            y,
            "minimum",
            [](DiagonalTensorCPtr a, DiagonalTensorCPtr b) { return minimum(a, b); },
            [](DiagonalTensorCPtr a, BlockBackend::Scalar const& b) { return minimum(a, b); },
            [](BlockBackend::Scalar const& a, DiagonalTensorCPtr b) { return minimum(a, b); });
      },
      nb::arg("x"),
      nb::arg("y"),
      doc_plus(DOC(lasyten, minimum),
               R"pydoc(
In Python, scalar and array inputs are forwarded to ``numpy.minimum``.
)pydoc"));

    m.def(
      "maximum",
      [](py::object x, py::object y) {
          return dispatch_pairwise(
            x,
            y,
            "maximum",
            [](DiagonalTensorCPtr a, DiagonalTensorCPtr b) { return maximum(a, b); },
            [](DiagonalTensorCPtr a, BlockBackend::Scalar const& b) { return maximum(a, b); },
            [](BlockBackend::Scalar const& a, DiagonalTensorCPtr b) { return maximum(a, b); });
      },
      nb::arg("x"),
      nb::arg("y"),
      doc_plus(DOC(lasyten, maximum),
               R"pydoc(
In Python, scalar and array inputs are forwarded to ``numpy.maximum``.
)pydoc"));

    m.def(
      "exp",
      [](py::object obj) {
          if (py::isinstance<Tensor>(obj)) {
              return py::cast(exp(py::cast<TensorCPtr>(obj)));
          }
          return numpy().attr("exp")(obj);
      },
      nb::arg("obj"),
      doc_plus(DOC(lasyten, exp),
               R"pydoc(
In Python, non-tensor inputs are forwarded to ``numpy.exp``.
)pydoc"));
}

} // namespace lasyten
