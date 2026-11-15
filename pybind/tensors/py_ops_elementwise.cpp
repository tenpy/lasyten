#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/ops_elementwise.h>
#include <lasyten/tensors/tensor.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"

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
    return py::isinstance(x, py::module_::import("numbers").attr("Number")) ||
           py::isinstance(x, py::module_::import("lasyten.block_backends").attr("Scalar")) ||
           (lasyten::py_isinstance<Tensor>(x) &&
            py::module_::import("lasyten.tensors._tensors").attr("is_scalar")(x).cast<bool>());
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
    if (lasyten::py_isinstance<DiagonalTensor>(x)) {
        return py::cast(tensor_fn(x.cast<DiagonalTensorCPtr>()));
    }
    if (is_number_or_scalar(x)) {
        return scalar_py_fn(x);
    }
    throw_elementwise_type_error(x);
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
                if (abs_x.cast<float64>() < cutoff) {
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
                if (lasyten::py_isinstance<BlockBackend::Scalar>(n))
                    return py::cast(n.cast<BlockBackend::Scalar>().conj());
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
                if (lasyten::py_isinstance<BlockBackend::Scalar>(n))
                    return py::cast(n.cast<BlockBackend::Scalar>().real_if_close(tol));
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
      "exp",
      [](py::object obj) {
          if (lasyten::py_isinstance<Tensor>(obj)) {
              return py::cast(exp(obj.cast<TensorCPtr>()));
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
