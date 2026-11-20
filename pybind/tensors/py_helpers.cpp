#include <lasyten/tensors/helpers.h>
#include <lasyten/tools.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"

#include "docstrings/tensors/helpers.h"

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

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

void
py_check_compatible_legs(py::sequence legs1, py::sequence legs2, bool expect_equal)
{
    bool all_space = true;
    for (auto item : legs1) {
        if (!py::isinstance<Space>(py::reinterpret_borrow<py::object>(item))) {
            all_space = false;
            break;
        }
    }
    if (all_space) {
        for (auto item : legs2) {
            if (!py::isinstance<Space>(py::reinterpret_borrow<py::object>(item))) {
                all_space = false;
                break;
            }
        }
    }
    if (all_space) {
        std::vector<Space::Ptr> a;
        std::vector<Space::Ptr> b;
        for (auto item : legs1) {
            a.push_back(py::cast<Space::Ptr>(item));
        }
        for (auto item : legs2) {
            b.push_back(py::cast<Space::Ptr>(item));
        }
        _check_compatible_legs(a, b, expect_equal);
        return;
    }
    std::vector<Leg::Ptr> a;
    std::vector<Leg::Ptr> b;
    for (auto item : legs1) {
        a.push_back(py::cast_non_null<Leg::Ptr>(item, "elements of legs1"));
    }
    for (auto item : legs2) {
        b.push_back(py::cast_non_null<Leg::Ptr>(item, "elements of legs2"));
    }
    _check_compatible_legs(a, b, expect_equal);
}

py::object
py_from_compose_sym(std::variant<SymmetricTensorPtr, BlockBackend::Scalar> const& v)
{
    return std::visit([](auto const& x) -> py::object { return py::cast(x); }, v);
}

} // namespace

void
bind_tensors_helpers(nb::module_& m)
{
    m.def("_check_compatible_legs",
          &py_check_compatible_legs,
          nb::arg("legs1"),
          nb::arg("legs2"),
          nb::arg("expect_equal") = true,
          doc_plus(DOC(lasyten, _check_compatible_legs),
                   R"pydoc(
Accepts sequences of either ``Leg`` or ``Space`` objects (dispatches to the matching
C++ overload).
)pydoc"));

    m.def("_compose_with_Mask",
          &_compose_with_Mask,
          nb::arg("tensor"),
          nb::arg("mask"),
          nb::arg("leg_idx"),
          DOC(lasyten, _compose_with_Mask));

    m.def(
      "_compose_SymmetricTensors",
      [](py::object tensor1, py::object tensor2, py::object relabel1, py::object relabel2) {
          std::optional<std::map<std::string, std::string>> r1;
          std::optional<std::map<std::string, std::string>> r2;
          if (!relabel1.is_none()) {
              r1 = py::cast<std::map<std::string, std::string>>(relabel1);
          }
          if (!relabel2.is_none()) {
              r2 = py::cast<std::map<std::string, std::string>>(relabel2);
          }
          return py_from_compose_sym(
            _compose_SymmetricTensors(py::cast_non_null<SymmetricTensorCPtr>(tensor1, "tensor1"),
                                      py::cast_non_null<SymmetricTensorCPtr>(tensor2, "tensor2"),
                                      r1,
                                      r2));
      },
      nb::arg("tensor1"),
      nb::arg("tensor2"),
      nb::arg("relabel1").none() = nb::none(),
      nb::arg("relabel2").none() = nb::none(),
      doc_plus(DOC(lasyten, _compose_SymmetricTensors),
               R"pydoc(
In Python, ``relabel1`` / ``relabel2`` are ``dict | None`` (``None`` = no relabel).
)pydoc"));

    m.def("_convert_abelian_to_FT",
          &_convert_abelian_to_FT,
          nb::arg("tensor"),
          nb::arg("backend"),
          nb::arg("dtype"),
          nb::arg("device"),
          DOC(lasyten, _convert_abelian_to_FT));

    m.def("_convert_FT_to_abelian",
          &_convert_FT_to_abelian,
          nb::arg("tensor"),
          nb::arg("backend"),
          nb::arg("dtype"),
          nb::arg("device"),
          DOC(lasyten, _convert_FT_to_abelian));

    m.def("_decomposition_prepare",
          &_decomposition_prepare,
          nb::arg("tensor"),
          nb::arg("new_leg_dual"),
          DOC(lasyten, _decomposition_prepare));

    m.def(
      "_decomposition_labels",
      [](py::object new_labels) {
          return _decomposition_labels(py_leg_labels(to_iterable(new_labels)));
      },
      nb::arg("new_labels"),
      DOC(lasyten, _decomposition_labels));

    m.def(
      "_svd_new_labels",
      [](py::object new_labels) {
          if (new_labels.is_none()) {
              return _svd_new_labels(std::nullopt);
          }
          return _svd_new_labels(py_leg_labels(to_iterable(new_labels)));
      },
      nb::arg("new_labels"),
      doc_plus(DOC(lasyten, _svd_new_labels),
               R"pydoc(
In Python, ``new_labels`` may be ``None`` (all unlabeled).
)pydoc"));
}

} // namespace lasyten
