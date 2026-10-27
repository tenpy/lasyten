#include <lasyten/tensors/constructors.h>
#include <lasyten/tensors/hidden_leg_tensor.h>
#include <lasyten/tensors/tensor.h>

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"
#include "py_factory_parse.hpp"

#include "docstrings/tensors/constructors.h"

#include <optional>
#include <string>
#include <vector>

namespace lasyten {

void
bind_tensors_constructors(py::module_& m)
{
    m.def(
      "eye",
      [](py::object leg,
         TensorBackend::Ptr backend,
         py::object labels,
         Dtype dtype,
         py::object device,
         bool diagonal) {
          std::optional<std::string> device_opt;
          if (!device.is_none()) {
              device_opt = device.cast<std::string>();
          }
          auto init = py_parse_diag(leg, std::move(backend), labels);
          return eye(py_as_space_leg(leg),
                     init.backend,
                     init.labels,
                     dtype,
                     std::move(device_opt),
                     diagonal);
      },
      py::arg("leg"),
      py::arg("backend") = py::none(),
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Float64,
      py::arg("device") = py::none(),
      py::arg("diagonal") = true,
      DOC(lasyten, eye));

    m.def(
      "tensor",
      [](py::object obj,
         py::object codomain,
         py::object domain,
         TensorBackend::Ptr backend,
         py::object labels,
         py::object dtype,
         py::object device,
         bool understood_braiding) {
          std::optional<Dtype> dtype_opt;
          if (!dtype.is_none()) {
              dtype_opt = dtype.cast<Dtype>();
          }
          std::optional<std::string> device_opt;
          if (!device.is_none()) {
              device_opt = device.cast<std::string>();
          }
          if (py::isinstance<Tensor>(obj)) {
              auto t = obj.cast<TensorCPtr>();
              auto cod = tensor_product_from_python(codomain, t->symmetry);
              TensorProduct::Ptr dom;
              if (!domain.is_none()) {
                  dom = tensor_product_from_python(domain, t->symmetry);
              }
              std::optional<OptionalLabels> labs;
              if (!labels.is_none()) {
                  labs = parse_tensor_init_labels(labels, t->codomain, t->domain);
              }
              return py::cast(tensor(t,
                                     std::move(cod),
                                     std::move(dom),
                                     std::move(backend),
                                     labs,
                                     dtype_opt,
                                     device_opt));
          }
          auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
          auto block = init.backend->block_backend->as_block(obj, dtype_opt, device_opt);
          return py::cast(tensor(block,
                                 init.codomain,
                                 init.domain,
                                 init.backend,
                                 init.labels,
                                 dtype_opt,
                                 device_opt,
                                 understood_braiding));
      },
      py::arg("obj"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = py::none(),
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      py::arg("understood_braiding") = false,
      doc_plus(DOC(lasyten, tensor),
               R"pydoc(
Also accepts a dense block (second C++ overload). ``None`` for optional arguments
matches C++ null / ``nullopt``. ``understood_braiding`` applies only to the block path.
)pydoc"));

    m.def(
      "add_trivial_leg",
      [](py::object tens,
         py::object legs_pos,
         py::object codomain_pos,
         py::object domain_pos,
         py::object label,
         bool is_dual) {
          std::optional<int64> legs_pos_opt;
          std::optional<int64> codomain_pos_opt;
          std::optional<int64> domain_pos_opt;
          OptionalLabel label_opt = std::nullopt;
          if (!legs_pos.is_none()) {
              legs_pos_opt = legs_pos.cast<int64>();
          }
          if (!codomain_pos.is_none()) {
              codomain_pos_opt = codomain_pos.cast<int64>();
          }
          if (!domain_pos.is_none()) {
              domain_pos_opt = domain_pos.cast<int64>();
          }
          if (!label.is_none()) {
              label_opt = label.cast<std::string>();
          }
          return add_trivial_leg(tens.cast<TensorCPtr>(),
                                 legs_pos_opt,
                                 codomain_pos_opt,
                                 domain_pos_opt,
                                 std::move(label_opt),
                                 is_dual);
      },
      py::arg("tens"),
      py::arg("legs_pos") = py::none(),
      py::kw_only(),
      py::arg("codomain_pos") = py::none(),
      py::arg("domain_pos") = py::none(),
      py::arg("label") = py::none(),
      py::arg("is_dual") = false,
      DOC(lasyten, add_trivial_leg));

    m.def(
      "zero_like",
      [](py::object tensor) { return zero_like(tensor.cast<TensorCPtr>()); },
      py::arg("tensor"),
      DOC(lasyten, zero_like));

    m.def(
      "tensor_from_grid",
      [](py::object grid,
         py::object labels,
         py::object dtype,
         std::optional<OptionalLabels> row_labels,
         std::optional<OptionalLabels> col_labels,
         std::optional<std::vector<ElementarySpace::Ptr>> row_summands,
         std::optional<std::vector<ElementarySpace::Ptr>> col_summands) {
          std::optional<Dtype> dtype_opt;
          if (!dtype.is_none()) {
              dtype_opt = dtype.cast<Dtype>();
          }
          std::vector<std::vector<TensorPtr>> g;
          for (auto row_h : py::reinterpret_borrow<py::iterable>(grid)) {
              std::vector<TensorPtr> row;
              for (auto item : py::reinterpret_borrow<py::iterable>(row_h)) {
                  py::object obj = py::reinterpret_borrow<py::object>(item);
                  row.push_back(obj.is_none() ? nullptr : obj.cast<TensorPtr>());
              }
              g.push_back(std::move(row));
          }
          auto res = tensor_from_grid(
            std::move(g),
            std::nullopt,
            dtype_opt,
            std::move(row_labels),
            std::move(col_labels),
            std::move(row_summands),
            std::move(col_summands));
          if (!labels.is_none()) {
              res->set_labels(parse_tensor_init_labels(labels, res->codomain, res->domain));
          }
          return HiddenLegTensor::maybe_wrap(std::dynamic_pointer_cast<SymmetricTensor>(res));
      },
      py::arg("grid"),
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::kw_only(),
      py::arg("row_labels") = py::none(),
      py::arg("col_labels") = py::none(),
      py::arg("row_summands") = py::none(),
      py::arg("col_summands") = py::none(),
      doc_plus(DOC(lasyten, tensor_from_grid),
               R"pydoc(
In Python, ``grid`` is ``list[list[SymmetricTensor | None]]`` (``None`` = zero cell);
``labels`` / ``dtype`` / ``row_labels`` / ``col_labels`` / ``row_summands`` /
``col_summands`` use ``None`` for C++ ``nullopt``.
)pydoc"));

    m.def(
      "tensor_grid_cell",
      [](py::object tensor,
         py::object row,
         py::object col,
         py::object row_leg,
         py::object col_leg) {
          auto as_summand_ref = [](py::object obj) -> DirectSumSpace::SummandRef {
              if (py::isinstance<py::str>(obj)) {
                  return obj.cast<std::string>();
              }
              return obj.cast<int64>();
          };
          LegRef row_ref = int64{ 0 };
          LegRef col_ref = int64{ -1 };
          if (!row_leg.is_none()) {
              if (py::isinstance<py::str>(row_leg)) {
                  row_ref = row_leg.cast<std::string>();
              } else {
                  row_ref = row_leg.cast<int64>();
              }
          }
          if (!col_leg.is_none()) {
              if (py::isinstance<py::str>(col_leg)) {
                  col_ref = col_leg.cast<std::string>();
              } else {
                  col_ref = col_leg.cast<int64>();
              }
          }
          return tensor_grid_cell(
            tensor.cast<TensorCPtr>(), as_summand_ref(row), as_summand_ref(col), row_ref, col_ref);
      },
      py::arg("tensor"),
      py::arg("row"),
      py::arg("col"),
      py::arg("row_leg") = py::none(),
      py::arg("col_leg") = py::none(),
      doc_plus(DOC(lasyten, tensor_grid_cell),
               R"pydoc(
In Python, ``row`` / ``col`` accept ``int | str`` (a summand label, resolved via
``DirectSumSpace.get_summand_idx``).
)pydoc"));

    m.def(
      "tensor_subgrid",
      [](py::object tensor,
         py::object rows,
         py::object cols,
         py::object row_leg,
         py::object col_leg) {
          auto as_summand_refs = [](py::object seq) {
              std::vector<DirectSumSpace::SummandRef> out;
              for (auto item : py::reinterpret_borrow<py::iterable>(seq)) {
                  if (py::isinstance<py::str>(item)) {
                      out.emplace_back(item.cast<std::string>());
                  } else {
                      out.emplace_back(item.cast<int64>());
                  }
              }
              return out;
          };
          auto as_leg_ref = [](py::object obj, int64 default_idx) -> LegRef {
              if (obj.is_none()) {
                  return default_idx;
              }
              if (py::isinstance<py::str>(obj)) {
                  return obj.cast<std::string>();
              }
              return obj.cast<int64>();
          };
          return tensor_subgrid(tensor.cast<TensorCPtr>(),
                                as_summand_refs(rows),
                                as_summand_refs(cols),
                                as_leg_ref(row_leg, 0),
                                as_leg_ref(col_leg, -1));
      },
      py::arg("tensor"),
      py::arg("rows"),
      py::arg("cols"),
      py::arg("row_leg") = py::none(),
      py::arg("col_leg") = py::none(),
      doc_plus(DOC(lasyten, tensor_subgrid),
               R"pydoc(
In Python, ``rows`` / ``cols`` are sequences of ``int | str`` (summand labels).
)pydoc"));

    m.def(
      "grid_project",
      [](py::object tensor, py::object legs, py::object cells, bool squeeze) {
          auto as_leg_ref = [](py::handle item) -> LegRef {
              py::object obj = py::reinterpret_borrow<py::object>(item);
              if (py::isinstance<py::str>(obj)) {
                  return obj.cast<std::string>();
              }
              return obj.cast<int64>();
          };
          auto as_summand_ref = [](py::handle item) -> DirectSumSpace::SummandRef {
              py::object obj = py::reinterpret_borrow<py::object>(item);
              if (py::isinstance<py::str>(obj)) {
                  return obj.cast<std::string>();
              }
              return obj.cast<int64>();
          };
          std::vector<LegRef> leg_refs;
          for (auto item : py::reinterpret_borrow<py::iterable>(legs)) {
              leg_refs.push_back(as_leg_ref(item));
          }
          std::vector<DirectSumSpace::SummandRef> cell_refs;
          for (auto item : py::reinterpret_borrow<py::iterable>(cells)) {
              cell_refs.push_back(as_summand_ref(item));
          }
          return grid_project(
            tensor.cast<TensorCPtr>(), std::move(leg_refs), std::move(cell_refs), squeeze);
      },
      py::arg("tensor"),
      py::arg("legs"),
      py::arg("cells"),
      py::arg("squeeze") = false,
      doc_plus(DOC(lasyten, grid_project),
               R"pydoc(
In Python, ``legs`` and ``cells`` are parallel sequences of ``str | int`` values.
)pydoc"));
}

} // namespace lasyten
