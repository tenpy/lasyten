#include <lasyten/tensors/decompositions.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/tensor.h>
#include <lasyten/tools.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"

#include "docstrings/tensors/decompositions.h"

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <tuple>

namespace lasyten {

namespace {

LegLabelList
py_leg_labels(py::object seq)
{
    LegLabelList out;
    for (auto item : py::reinterpret_borrow<py::iterable>(seq)) {
        if (item.is_none()) {
            out.push_back("");
        } else {
            out.push_back(py::cast<std::string>(item));
        }
    }
    return out;
}

std::optional<LegLabelList>
py_opt_labels(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return py_leg_labels(to_iterable(obj));
}

std::optional<std::string>
py_opt_string(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return py::cast<std::string>(obj);
}

py::object
entropy_numpy(py::object p, py::object n)
{
    auto np = py::module_::import("numpy");
    p = np.attr("asarray")(p);
    p = np.attr("real_if_close")(p);
    p = p.attr("__getitem__")(p.attr("__gt__")(1e-30));
    if (n.equal(py::int_(1))) {
        return -np.attr("inner")(np.attr("log")(p), p);
    }
    bool inf = false;
    try {
        inf = std::isinf(py::cast<float64>(n));
    } catch (...) {
        inf = false;
    }
    if (inf) {
        return -np.attr("log")(np.attr("max")(p));
    }
    float64 n_f = py::cast<float64>(n);
    return np.attr("log")(np.attr("sum")(p.attr("__pow__")(n))).attr("__truediv__")(1.0 - n_f);
}

} // namespace

void
bind_tensors_decompositions(nb::module_& m)
{
    m.def("apply_mask_DiagonalTensor",
          &apply_mask_DiagonalTensor,
          nb::arg("tensor"),
          nb::arg("mask"),
          DOC(lasyten, apply_mask_DiagonalTensor));

    m.def(
      "eigh",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, py::object sort) {
          return eigh(std::move(tensor),
                      py_leg_labels(to_iterable(new_labels)),
                      new_leg_dual,
                      py_opt_string(sort));
      },
      nb::arg("tensor"),
      nb::arg("new_labels"),
      nb::arg("new_leg_dual") = false,
      nb::arg("sort").none() = nb::none(),
      DOC(lasyten, eigh));

    m.def(
      "eig",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, py::object sort) {
          return eig(std::move(tensor),
                     py_leg_labels(to_iterable(new_labels)),
                     new_leg_dual,
                     py_opt_string(sort));
      },
      nb::arg("tensor"),
      nb::arg("new_labels"),
      nb::arg("new_leg_dual") = false,
      nb::arg("sort").none() = nb::none(),
      DOC(lasyten, eig));

    m.def(
      "eigvalsh",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, py::object sort) {
          return eigvalsh(std::move(tensor),
                          py_leg_labels(to_iterable(new_labels)),
                          new_leg_dual,
                          py_opt_string(sort));
      },
      nb::arg("tensor"),
      nb::arg("new_labels"),
      nb::arg("new_leg_dual") = false,
      nb::arg("sort").none() = nb::none(),
      DOC(lasyten, eigvalsh));

    m.def(
      "eigvals",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, py::object sort) {
          return eigvals(std::move(tensor),
                         py_leg_labels(to_iterable(new_labels)),
                         new_leg_dual,
                         py_opt_string(sort));
      },
      nb::arg("tensor"),
      nb::arg("new_labels"),
      nb::arg("new_leg_dual") = false,
      nb::arg("sort").none() = nb::none(),
      DOC(lasyten, eigvals));

    m.def(
      "entropy",
      [](py::object p, py::object n) {
          if (py::isinstance<DiagonalTensor>(p)) {
              float64 n_f = 1.;
              if (n.is(py::none())) {
                  n_f = 1.;
              } else if (py::isinstance<py::float_>(n) || py::isinstance<py::int_>(n)) {
                  n_f = py::cast<float64>(n);
                  if (std::isinf(n_f)) {
                      n_f = std::numeric_limits<float64>::infinity();
                  }
              } else {
                  n_f = py::cast<float64>(n);
              }
              return py::cast(entropy(py::cast<DiagonalTensorCPtr>(p), n_f)).attr("to_numpy")();
          }
          return entropy_numpy(std::move(p), std::move(n));
      },
      nb::arg("p"),
      nb::arg("n") = 1,
      doc_plus(DOC(lasyten, entropy),
               R"pydoc(
In Python, ``p`` may also be a NumPy array / sequence of floats. The DiagonalTensor path
returns a NumPy scalar via ``.to_numpy()``.
)pydoc"));

    m.def(
      "lq",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, bool charge_leg_top) {
          return lq(std::move(tensor), py_opt_labels(new_labels), new_leg_dual, charge_leg_top);
      },
      nb::arg("tensor"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("new_leg_dual") = false,
      nb::arg("charge_leg_top") = true,
      DOC(lasyten, lq));

    m.def(
      "qr",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, bool charge_leg_top) {
          return qr(std::move(tensor), py_opt_labels(new_labels), new_leg_dual, charge_leg_top);
      },
      nb::arg("tensor"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("new_leg_dual") = false,
      nb::arg("charge_leg_top") = true,
      DOC(lasyten, qr));

    m.def(
      "svd",
      [](TensorCPtr tensor,
         py::object new_labels,
         bool new_leg_dual,
         bool charge_leg_top,
         py::object algorithm) {
          return svd(std::move(tensor),
                     py_opt_labels(new_labels),
                     new_leg_dual,
                     charge_leg_top,
                     py_opt_string(algorithm));
      },
      nb::arg("tensor"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("new_leg_dual") = false,
      nb::arg("charge_leg_top") = true,
      nb::arg("algorithm").none() = nb::none(),
      DOC(lasyten, svd));

    m.def("svd_apply_mask",
          &svd_apply_mask,
          nb::arg("U"),
          nb::arg("S"),
          nb::arg("Vh"),
          nb::arg("mask"),
          DOC(lasyten, svd_apply_mask));

    m.def(
      "truncate_singular_values",
      [](DiagonalTensorCPtr S,
         py::object chi_max,
         py::object chi_min,
         py::object degeneracy_tol,
         py::object trunc_cut,
         py::object svd_min,
         bool minimize_error,
         py::object mask_labels) {
          std::optional<int64> chi_max_opt;
          if (!chi_max.is_none()) {
              chi_max_opt = py::cast<int64>(chi_max);
          }
          int64 chi_min_v = chi_min.is_none() ? 1 : py::cast<int64>(chi_min);
          float64 degeneracy_tol_v =
            degeneracy_tol.is_none() ? 0. : py::cast<float64>(degeneracy_tol);
          float64 trunc_cut_v = trunc_cut.is_none() ? 0. : py::cast<float64>(trunc_cut);
          float64 svd_min_v = svd_min.is_none() ? 0. : py::cast<float64>(svd_min);
          return truncate_singular_values(std::move(S),
                                          chi_max_opt,
                                          chi_min_v,
                                          degeneracy_tol_v,
                                          trunc_cut_v,
                                          svd_min_v,
                                          minimize_error,
                                          py_opt_labels(mask_labels));
      },
      nb::arg("S"),
      nb::arg("chi_max").none() = nb::none(),
      nb::arg("chi_min") = 1,
      nb::arg("degeneracy_tol") = 0,
      nb::arg("trunc_cut") = 0,
      nb::arg("svd_min") = 0,
      nb::arg("minimize_error") = true,
      nb::arg("mask_labels").none() = nb::none(),
      DOC(lasyten, truncate_singular_values));

    m.def(
      "truncated_svd",
      [](TensorCPtr tensor,
         py::object new_labels,
         bool new_leg_dual,
         bool charge_leg_top,
         py::object algorithm,
         py::object normalize_to,
         py::object chi_max,
         py::object chi_min,
         py::object degeneracy_tol,
         py::object trunc_cut,
         py::object svd_min) {
          std::optional<float64> normalize_opt;
          if (!normalize_to.is_none()) {
              normalize_opt = py::cast<float64>(normalize_to);
          }
          std::optional<int64> chi_max_opt;
          if (!chi_max.is_none()) {
              chi_max_opt = py::cast<int64>(chi_max);
          }
          int64 chi_min_v = chi_min.is_none() ? 1 : py::cast<int64>(chi_min);
          float64 degeneracy_tol_v =
            degeneracy_tol.is_none() ? 0. : py::cast<float64>(degeneracy_tol);
          float64 trunc_cut_v = trunc_cut.is_none() ? 0. : py::cast<float64>(trunc_cut);
          float64 svd_min_v = svd_min.is_none() ? 0. : py::cast<float64>(svd_min);
          return truncated_svd(std::move(tensor),
                               py_opt_labels(new_labels),
                               new_leg_dual,
                               charge_leg_top,
                               py_opt_string(algorithm),
                               normalize_opt,
                               chi_max_opt,
                               chi_min_v,
                               degeneracy_tol_v,
                               trunc_cut_v,
                               svd_min_v);
      },
      nb::arg("tensor"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("new_leg_dual") = false,
      nb::arg("charge_leg_top") = true,
      nb::arg("algorithm").none() = nb::none(),
      nb::arg("normalize_to").none() = nb::none(),
      nb::arg("chi_max").none() = nb::none(),
      nb::arg("chi_min") = 1,
      nb::arg("degeneracy_tol") = 0,
      nb::arg("trunc_cut") = 0,
      nb::arg("svd_min") = 0,
      DOC(lasyten, truncated_svd));
}

} // namespace lasyten
