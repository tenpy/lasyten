#include <lasyten/tensors/decompositions.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/tensor.h>
#include <lasyten/tools.h>

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"

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
            out.push_back(item.cast<std::string>());
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
    return obj.cast<std::string>();
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
        inf = std::isinf(n.cast<float64>());
    } catch (...) {
        inf = false;
    }
    if (inf) {
        return -np.attr("log")(np.attr("max")(p));
    }
    float64 n_f = n.cast<float64>();
    return np.attr("log")(np.attr("sum")(p.attr("__pow__")(n))).attr("__truediv__")(1.0 - n_f);
}

} // namespace

void
bind_tensors_decompositions(py::module_& m)
{
    m.def("apply_mask_DiagonalTensor",
          &apply_mask_DiagonalTensor,
          py::arg("tensor"),
          py::arg("mask"),
          DOC(lasyten, apply_mask_DiagonalTensor));

    m.def(
      "eigh",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, py::object sort) {
          return eigh(std::move(tensor),
                      py_leg_labels(to_iterable(new_labels)),
                      new_leg_dual,
                      py_opt_string(sort));
      },
      py::arg("tensor"),
      py::arg("new_labels"),
      py::arg("new_leg_dual") = false,
      py::arg("sort") = py::none(),
      DOC(lasyten, eigh));

    m.def(
      "eig",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, py::object sort) {
          return eig(std::move(tensor),
                     py_leg_labels(to_iterable(new_labels)),
                     new_leg_dual,
                     py_opt_string(sort));
      },
      py::arg("tensor"),
      py::arg("new_labels"),
      py::arg("new_leg_dual") = false,
      py::arg("sort") = py::none(),
      DOC(lasyten, eig));

    m.def(
      "eigvalsh",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, py::object sort) {
          return eigvalsh(std::move(tensor),
                          py_leg_labels(to_iterable(new_labels)),
                          new_leg_dual,
                          py_opt_string(sort));
      },
      py::arg("tensor"),
      py::arg("new_labels"),
      py::arg("new_leg_dual") = false,
      py::arg("sort") = py::none(),
      DOC(lasyten, eigvalsh));

    m.def(
      "eigvals",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, py::object sort) {
          return eigvals(std::move(tensor),
                         py_leg_labels(to_iterable(new_labels)),
                         new_leg_dual,
                         py_opt_string(sort));
      },
      py::arg("tensor"),
      py::arg("new_labels"),
      py::arg("new_leg_dual") = false,
      py::arg("sort") = py::none(),
      DOC(lasyten, eigvals));

    m.def(
      "entropy",
      [](py::object p, py::object n) {
          if (py::isinstance<DiagonalTensor>(p)) {
              float64 n_f = 1.;
              if (n.is(py::none())) {
                  n_f = 1.;
              } else if (py::isinstance<py::float_>(n) || py::isinstance<py::int_>(n)) {
                  n_f = n.cast<float64>();
                  if (std::isinf(n_f)) {
                      n_f = std::numeric_limits<float64>::infinity();
                  }
              } else {
                  n_f = n.cast<float64>();
              }
              return py::cast(entropy(p.cast<DiagonalTensorCPtr>(), n_f)).attr("to_numpy")();
          }
          return entropy_numpy(std::move(p), std::move(n));
      },
      py::arg("p"),
      py::arg("n") = 1,
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
      py::arg("tensor"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("charge_leg_top") = true,
      DOC(lasyten, lq));

    m.def(
      "qr",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, bool charge_leg_top) {
          return qr(std::move(tensor), py_opt_labels(new_labels), new_leg_dual, charge_leg_top);
      },
      py::arg("tensor"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("charge_leg_top") = true,
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
      py::arg("tensor"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("charge_leg_top") = true,
      py::arg("algorithm") = py::none(),
      DOC(lasyten, svd));

    m.def("svd_apply_mask",
          &svd_apply_mask,
          py::arg("U"),
          py::arg("S"),
          py::arg("Vh"),
          py::arg("mask"),
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
              chi_max_opt = chi_max.cast<int64>();
          }
          int64 chi_min_v = chi_min.is_none() ? 1 : chi_min.cast<int64>();
          float64 degeneracy_tol_v =
            degeneracy_tol.is_none() ? 0. : degeneracy_tol.cast<float64>();
          float64 trunc_cut_v = trunc_cut.is_none() ? 0. : trunc_cut.cast<float64>();
          float64 svd_min_v = svd_min.is_none() ? 0. : svd_min.cast<float64>();
          return truncate_singular_values(std::move(S),
                                          chi_max_opt,
                                          chi_min_v,
                                          degeneracy_tol_v,
                                          trunc_cut_v,
                                          svd_min_v,
                                          minimize_error,
                                          py_opt_labels(mask_labels));
      },
      py::arg("S"),
      py::arg("chi_max") = py::none(),
      py::arg("chi_min") = 1,
      py::arg("degeneracy_tol") = 0,
      py::arg("trunc_cut") = 0,
      py::arg("svd_min") = 0,
      py::arg("minimize_error") = true,
      py::arg("mask_labels") = py::none(),
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
              normalize_opt = normalize_to.cast<float64>();
          }
          std::optional<int64> chi_max_opt;
          if (!chi_max.is_none()) {
              chi_max_opt = chi_max.cast<int64>();
          }
          int64 chi_min_v = chi_min.is_none() ? 1 : chi_min.cast<int64>();
          float64 degeneracy_tol_v =
            degeneracy_tol.is_none() ? 0. : degeneracy_tol.cast<float64>();
          float64 trunc_cut_v = trunc_cut.is_none() ? 0. : trunc_cut.cast<float64>();
          float64 svd_min_v = svd_min.is_none() ? 0. : svd_min.cast<float64>();
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
      py::arg("tensor"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("charge_leg_top") = true,
      py::arg("algorithm") = py::none(),
      py::arg("normalize_to") = py::none(),
      py::arg("chi_max") = py::none(),
      py::arg("chi_min") = 1,
      py::arg("degeneracy_tol") = 0,
      py::arg("trunc_cut") = 0,
      py::arg("svd_min") = 0,
      DOC(lasyten, truncated_svd));
}

} // namespace lasyten
