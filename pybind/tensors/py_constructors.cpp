#include <lasyten/tensors/constructors.h>
#include <lasyten/tensors/hidden_leg_tensor.h>
#include <lasyten/tensors/tensor.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "py_factory_parse.hpp"

#include "docstrings/tensors/constructors.h"

#include <optional>
#include <string>
#include <vector>

namespace lasyten {

void
bind_tensors_constructors(nb::module_& m)
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
              device_opt = py::cast<std::string>(device);
          }
          auto init = py_parse_diag(leg, std::move(backend), labels);
          return eye(py_as_space_leg(leg),
                     init.backend,
                     init.labels,
                     dtype,
                     std::move(device_opt),
                     diagonal);
      },
      nb::arg("leg"),
      nb::arg("backend").none() = nb::none(),
      nb::arg("labels").none() = nb::none(),
      nb::arg("dtype") = Dtype::Float64,
      nb::arg("device").none() = nb::none(),
      nb::arg("diagonal") = true,
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
              dtype_opt = py::cast<Dtype>(dtype);
          }
          std::optional<std::string> device_opt;
          if (!device.is_none()) {
              device_opt = py::cast<std::string>(device);
          }
          if (py::isinstance<Tensor>(obj)) {
              auto t = py::cast<TensorCPtr>(obj);
              auto cod = tensor_product_from_python(codomain, t->symmetry);
              TensorProduct::Ptr dom;
              if (!domain.is_none()) {
                  dom = tensor_product_from_python(domain, t->symmetry);
              }
              std::optional<LegLabelList> labs;
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
      nb::arg("obj"),
      nb::arg("codomain"),
      nb::arg("domain").none() = nb::none(),
      nb::arg("backend").none() = nb::none(),
      nb::arg("labels").none() = nb::none(),
      nb::arg("dtype").none() = nb::none(),
      nb::arg("device").none() = nb::none(),
      nb::arg("understood_braiding") = false,
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
          std::string label_opt = "";
          if (!legs_pos.is_none()) {
              legs_pos_opt = py::cast<int64>(legs_pos);
          }
          if (!codomain_pos.is_none()) {
              codomain_pos_opt = py::cast<int64>(codomain_pos);
          }
          if (!domain_pos.is_none()) {
              domain_pos_opt = py::cast<int64>(domain_pos);
          }
          if (!label.is_none()) {
              label_opt = py::cast<std::string>(label);
          }
          return add_trivial_leg(py::cast_non_null<TensorCPtr>(tens, "tens"),
                                 legs_pos_opt,
                                 codomain_pos_opt,
                                 domain_pos_opt,
                                 std::move(label_opt),
                                 is_dual);
      },
      nb::arg("tens"),
      nb::arg("legs_pos").none() = nb::none(),
      nb::kw_only(),
      nb::arg("codomain_pos").none() = nb::none(),
      nb::arg("domain_pos").none() = nb::none(),
      nb::arg("label").none() = nb::none(),
      nb::arg("is_dual") = false,
      DOC(lasyten, add_trivial_leg));

    m.def(
      "zero_like",
      [](py::object tensor) { return zero_like(py::cast_non_null<TensorCPtr>(tensor, "tensor")); },
      nb::arg("tensor"),
      DOC(lasyten, zero_like));

    m.def(
      "tensor_from_grid",
      [](py::object grid,
         py::object stack_legs,
         py::object dtype,
         py::object stack_labels,
         py::object stack_summands) {
          std::optional<Dtype> dtype_opt;
          if (!dtype.is_none()) {
              dtype_opt = py::cast<Dtype>(dtype);
          }
          auto as_leg_ref = [](py::handle item) -> LegRef {
              py::object obj = py::reinterpret_borrow<py::object>(item);
              if (py::isinstance<py::str>(obj)) {
                  return py::cast<std::string>(obj);
              }
              return py::cast<int64>(obj);
          };
          std::vector<LegRef> leg_refs;
          if (py::isinstance<py::str>(stack_legs)) {
              leg_refs.push_back(as_leg_ref(stack_legs));
          } else {
              for (auto item : py::reinterpret_borrow<py::iterable>(stack_legs)) {
                  leg_refs.push_back(as_leg_ref(item));
              }
          }

          std::vector<std::vector<TensorPtr>> g;
          std::vector<py::object> grid_items;
          bool is_one_dimensional = true;
          for (auto item : py::reinterpret_borrow<py::iterable>(grid)) {
              py::object obj = py::reinterpret_borrow<py::object>(item);
              if (!obj.is_none() && !py::hasattr(obj, "num_legs")) {
                  is_one_dimensional = false;
              }
              grid_items.push_back(std::move(obj));
          }
          if (is_one_dimensional) {
              std::vector<TensorPtr> row;
              row.reserve(grid_items.size());
              for (auto const& obj : grid_items) {
                  row.push_back(obj.is_none() ? nullptr : py::cast<TensorPtr>(obj));
              }
              g.push_back(std::move(row));
          } else {
              for (auto const& row_obj : grid_items) {
                  std::vector<TensorPtr> row;
                  for (auto item : py::reinterpret_borrow<py::iterable>(row_obj)) {
                      py::object obj = py::reinterpret_borrow<py::object>(item);
                      row.push_back(obj.is_none() ? nullptr : py::cast<TensorPtr>(obj));
                  }
                  g.push_back(std::move(row));
              }
          }

          std::vector<std::optional<LegLabelList>> labels;
          if (!stack_labels.is_none()) {
              for (auto dim_labels : py::reinterpret_borrow<py::iterable>(stack_labels)) {
                  if (dim_labels.is_none()) {
                      labels.emplace_back(std::nullopt);
                      continue;
                  }
                  LegLabelList axis_labels;
                  for (auto label : py::reinterpret_borrow<py::iterable>(dim_labels)) {
                      axis_labels.push_back(py::cast<std::string>(label));
                  }
                  labels.emplace_back(std::move(axis_labels));
              }
          }

          std::vector<std::optional<std::vector<ElementarySpace::Ptr>>> summands;
          if (!stack_summands.is_none()) {
              for (auto dim_summands : py::reinterpret_borrow<py::iterable>(stack_summands)) {
                  if (dim_summands.is_none()) {
                      summands.emplace_back(std::nullopt);
                      continue;
                  }
                  std::vector<ElementarySpace::Ptr> axis_summands;
                  for (auto space : py::reinterpret_borrow<py::iterable>(dim_summands)) {
                      axis_summands.push_back(
                        py::cast_non_null<ElementarySpace::Ptr>(space, "stack_summands"));
                  }
                  summands.emplace_back(std::move(axis_summands));
              }
          }
          auto res = tensor_from_grid(
            std::move(g), std::move(leg_refs), dtype_opt, std::move(labels), std::move(summands));
          return HiddenLegTensor::maybe_wrap(std::dynamic_pointer_cast<SymmetricTensor>(res));
      },
      nb::arg("grid"),
      nb::arg("stack_legs"),
      nb::arg("dtype").none() = nb::none(),
      nb::kw_only(),
      nb::arg("stack_labels").none() = nb::none(),
      nb::arg("stack_summands").none() = nb::none(),
      doc_plus(DOC(lasyten, tensor_from_grid),
               R"pydoc(
In Python, ``grid`` is a flat list for one stacking leg, or a list of rows for two;
``None`` denotes a zero cell. ``stack_legs`` contains leg labels or indices.
``stack_labels`` and ``stack_summands`` contain one entry per stacking dimension;
use ``None`` for an unspecified dimension.
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
                  return py::cast<std::string>(obj);
              }
              return py::cast<int64>(obj);
          };
          LegRef row_ref = int64{ 0 };
          LegRef col_ref = int64{ -1 };
          if (!row_leg.is_none()) {
              if (py::isinstance<py::str>(row_leg)) {
                  row_ref = py::cast<std::string>(row_leg);
              } else {
                  row_ref = py::cast<int64>(row_leg);
              }
          }
          if (!col_leg.is_none()) {
              if (py::isinstance<py::str>(col_leg)) {
                  col_ref = py::cast<std::string>(col_leg);
              } else {
                  col_ref = py::cast<int64>(col_leg);
              }
          }
          return tensor_grid_cell(py::cast_non_null<TensorCPtr>(tensor, "tensor"),
                                  as_summand_ref(row),
                                  as_summand_ref(col),
                                  row_ref,
                                  col_ref);
      },
      nb::arg("tensor"),
      nb::arg("row"),
      nb::arg("col"),
      nb::arg("row_leg").none() = nb::none(),
      nb::arg("col_leg").none() = nb::none(),
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
                      out.emplace_back(py::cast<std::string>(item));
                  } else {
                      out.emplace_back(py::cast<int64>(item));
                  }
              }
              return out;
          };
          auto as_leg_ref = [](py::object obj, int64 default_idx) -> LegRef {
              if (obj.is_none()) {
                  return default_idx;
              }
              if (py::isinstance<py::str>(obj)) {
                  return py::cast<std::string>(obj);
              }
              return py::cast<int64>(obj);
          };
          return tensor_subgrid(py::cast_non_null<TensorCPtr>(tensor, "tensor"),
                                as_summand_refs(rows),
                                as_summand_refs(cols),
                                as_leg_ref(row_leg, 0),
                                as_leg_ref(col_leg, -1));
      },
      nb::arg("tensor"),
      nb::arg("rows"),
      nb::arg("cols"),
      nb::arg("row_leg").none() = nb::none(),
      nb::arg("col_leg").none() = nb::none(),
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
                  return py::cast<std::string>(obj);
              }
              return py::cast<int64>(obj);
          };
          auto as_summand_ref = [](py::handle item) -> DirectSumSpace::SummandRef {
              py::object obj = py::reinterpret_borrow<py::object>(item);
              if (py::isinstance<py::str>(obj)) {
                  return py::cast<std::string>(obj);
              }
              return py::cast<int64>(obj);
          };
          std::vector<LegRef> leg_refs;
          for (auto item : py::reinterpret_borrow<py::iterable>(legs)) {
              leg_refs.push_back(as_leg_ref(item));
          }
          std::vector<DirectSumSpace::SummandRef> cell_refs;
          for (auto item : py::reinterpret_borrow<py::iterable>(cells)) {
              cell_refs.push_back(as_summand_ref(item));
          }
          return grid_project(py::cast_non_null<TensorCPtr>(tensor, "tensor"),
                              std::move(leg_refs),
                              std::move(cell_refs),
                              squeeze);
      },
      nb::arg("tensor"),
      nb::arg("legs"),
      nb::arg("cells"),
      nb::arg("squeeze") = false,
      doc_plus(DOC(lasyten, grid_project),
               R"pydoc(
In Python, ``legs`` and ``cells`` are parallel sequences of ``str | int`` values.
)pydoc"));
}

} // namespace lasyten
