#include <lasyten/symmetries/sector.h>
#include <lasyten/tensors/charged_tensor.h>
#include <lasyten/tensors/hidden_leg_tensor.h>
#include <lasyten/tensors/ops_legs.h>
#include <lasyten/tensors/tensor.h>

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"

#include "docstrings/tensors/ops_legs.h"

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lasyten {

namespace {

LegRef
py_as_leg_ref(py::object obj)
{
    if (py::isinstance<py::str>(obj)) {
        return obj.cast<std::string>();
    }
    return obj.cast<int64>();
}

std::vector<LegRef>
py_as_leg_refs(py::object obj)
{
    std::vector<LegRef> out;
    if (py::isinstance<py::str>(obj) || !py::isinstance<py::iterable>(obj) ||
        py::isinstance<py::dict>(obj)) {
        out.push_back(py_as_leg_ref(obj));
        return out;
    }
    for (auto item : py::reinterpret_borrow<py::iterable>(obj)) {
        out.push_back(py_as_leg_ref(py::reinterpret_borrow<py::object>(item)));
    }
    return out;
}

std::optional<std::vector<LegRef>>
py_opt_leg_refs(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return py_as_leg_refs(obj);
}

bool
py_is_public_idx(py::handle obj)
{
    if (py::isinstance<py::int_>(obj)) {
        return true;
    }
    if (py::isinstance<Sector>(obj)) {
        return false;
    }
    if (py::isinstance<py::array>(obj)) {
        return obj.cast<py::array>().ndim() == 0;
    }
    return false;
}

HiddenLegTensorPtr
py_slice_leg(TensorCPtr tensor, py::object leg, py::object idx_or_sector, py::object multiplicity)
{
    LegRef l = py_as_leg_ref(leg);
    if (!multiplicity.is_none() || !py_is_public_idx(idx_or_sector)) {
        int64 m = multiplicity.is_none() ? 0 : multiplicity.cast<int64>();
        return slice_leg(std::move(tensor), std::move(l), idx_or_sector.cast<Sector>(), m);
    }
    return slice_leg(std::move(tensor), std::move(l), idx_or_sector.cast<int64>());
}

int64
py_leg_idx(TensorCPtr const& tensor, py::object key)
{
    if (py::isinstance<py::str>(key)) {
        return tensor->labels.index(key.cast<std::string>());
    }
    return tensor->labels.index(key.cast<int64>());
}

std::optional<LevelsSpec>
py_as_levels(TensorCPtr const& tensor, py::object levels)
{
    if (levels.is_none()) {
        return std::nullopt;
    }
    LevelsSpec out;
    if (py::isinstance<py::dict>(levels)) {
        out.assign(static_cast<std::size_t>(tensor->num_legs), std::nullopt);
        py::dict d = py::reinterpret_borrow<py::dict>(levels);
        for (auto item : d) {
            py::object key = py::reinterpret_borrow<py::object>(item.first);
            py::object val = py::reinterpret_borrow<py::object>(item.second);
            int64 idx = py_leg_idx(tensor, key);
            if (val.is_none()) {
                out[static_cast<std::size_t>(idx)] = std::nullopt;
            } else {
                out[static_cast<std::size_t>(idx)] = val.cast<int64>();
            }
        }
        return out;
    }
    for (auto item : levels) {
        py::object o = py::reinterpret_borrow<py::object>(item);
        if (o.is_none()) {
            out.push_back(std::nullopt);
        } else {
            out.push_back(o.cast<int64>());
        }
    }
    return out;
}

std::optional<BendRight>
py_as_bend_right(TensorCPtr const& tensor, py::object bend_right)
{
    if (bend_right.is_none()) {
        return std::nullopt;
    }
    if (py::isinstance<py::bool_>(bend_right) ||
        py::isinstance(bend_right, py::module_::import("numpy").attr("bool_"))) {
        return bend_right.cast<bool>();
    }
    if (py::isinstance<py::dict>(bend_right)) {
        std::vector<std::optional<bool>> out(static_cast<std::size_t>(tensor->num_legs),
                                             std::nullopt);
        py::dict d = py::reinterpret_borrow<py::dict>(bend_right);
        for (auto item : d) {
            py::object key = py::reinterpret_borrow<py::object>(item.first);
            py::object val = py::reinterpret_borrow<py::object>(item.second);
            int64 idx = py_leg_idx(tensor, key);
            if (val.is_none()) {
                out[static_cast<std::size_t>(idx)] = std::nullopt;
            } else {
                out[static_cast<std::size_t>(idx)] = val.cast<bool>();
            }
        }
        return out;
    }
    if (py::isinstance<py::iterable>(bend_right) && !py::isinstance<py::str>(bend_right)) {
        std::vector<std::optional<bool>> out;
        for (auto item : bend_right) {
            py::object o = py::reinterpret_borrow<py::object>(item);
            if (o.is_none()) {
                out.push_back(std::nullopt);
            } else {
                out.push_back(o.cast<bool>());
            }
        }
        return out;
    }
    return bend_right.cast<bool>();
}

std::optional<PipeDualities>
py_as_pipe_dualities(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    if (py::isinstance<py::bool_>(obj) ||
        py::isinstance(obj, py::module_::import("numpy").attr("bool_")) ||
        !py::isinstance<py::iterable>(obj) || py::isinstance<py::str>(obj)) {
        return obj.cast<bool>();
    }
    std::vector<bool> out;
    for (auto item : obj) {
        out.push_back(py::reinterpret_borrow<py::object>(item).cast<bool>());
    }
    return out;
}

std::optional<std::vector<Leg::Ptr>>
py_as_pipes(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    std::vector<Leg::Ptr> out;
    for (auto item : obj) {
        py::object o = py::reinterpret_borrow<py::object>(item);
        if (o.is_none()) {
            out.push_back(nullptr);
        } else {
            out.push_back(o.cast<Leg::Ptr>());
        }
    }
    return out;
}

} // namespace

void
bind_tensors_ops_legs(py::module_& m)
{
    m.def(
      "bend_legs",
      [](TensorCPtr tensor, py::object num_codomain_legs, py::object num_domain_legs) {
          std::optional<int64> n_cod;
          std::optional<int64> n_dom;
          if (!num_codomain_legs.is_none()) {
              n_cod = num_codomain_legs.cast<int64>();
          }
          if (!num_domain_legs.is_none()) {
              n_dom = num_domain_legs.cast<int64>();
          }
          return bend_legs(std::move(tensor), n_cod, n_dom);
      },
      py::arg("tensor"),
      py::arg("num_codomain_legs") = py::none(),
      py::arg("num_domain_legs") = py::none(),
      doc_plus(DOC(lasyten, bend_legs),
               R"pydoc(
In Python, ``num_codomain_legs`` / ``num_domain_legs`` are ``int | None`` (``None`` = unspecified).
)pydoc"));

    m.def("check_same_legs",
          &check_same_legs,
          py::arg("t1"),
          py::arg("t2"),
          DOC(lasyten, check_same_legs));

    m.def(
      "combine_legs",
      [](TensorCPtr tensor,
         py::args which_legs,
         py::object pipe_dualities,
         py::object pipes,
         py::object levels) {
          std::vector<std::vector<LegRef>> groups;
          groups.reserve(static_cast<std::size_t>(which_legs.size()));
          for (auto item : which_legs) {
              groups.push_back(py_as_leg_refs(py::reinterpret_borrow<py::object>(item)));
          }
          auto dualities = py_as_pipe_dualities(pipe_dualities);
          auto pipes_opt = py_as_pipes(pipes);
          auto levels_opt = py_as_levels(tensor, levels);
          return combine_legs(std::move(tensor),
                              std::move(groups),
                              std::move(dualities),
                              std::move(pipes_opt),
                              std::move(levels_opt));
      },
      py::arg("tensor"),
      py::kw_only(),
      py::arg("pipe_dualities") = false,
      py::arg("pipes") = py::none(),
      py::arg("levels") = py::none(),
      doc_plus(DOC(lasyten, combine_legs),
               R"pydoc(
In Python, groups are passed as ``*which_legs`` (each a sequence of ``int | str``);
``pipes`` / ``levels`` use ``None`` for unspecified; ``levels`` may also be a ``dict``.
)pydoc"));

    m.def(
      "combine_to_matrix",
      [](TensorCPtr tensor, py::object codomain, py::object domain, py::object levels) {
          return combine_to_matrix(tensor,
                                   py_opt_leg_refs(codomain),
                                   py_opt_leg_refs(domain),
                                   py_as_levels(tensor, levels));
      },
      py::arg("tensor"),
      py::arg("codomain") = py::none(),
      py::arg("domain") = py::none(),
      py::arg("levels") = py::none(),
      doc_plus(DOC(lasyten, combine_to_matrix),
               R"pydoc(
In Python, ``codomain`` / ``domain`` are ``int``, ``str``, a sequence thereof, or ``None``;
``levels`` may be a ``list``, ``dict``, or ``None``.
)pydoc"));

    m.def(
      "move_leg",
      [](TensorCPtr tensor,
         py::object which_leg,
         py::object codomain_pos,
         py::object domain_pos,
         py::object levels,
         py::object bend_right) {
          std::optional<int64> cpos;
          std::optional<int64> dpos;
          if (!codomain_pos.is_none()) {
              cpos = codomain_pos.cast<int64>();
          }
          if (!domain_pos.is_none()) {
              dpos = domain_pos.cast<int64>();
          }
          auto levels_opt = py_as_levels(tensor, levels);
          auto bend_opt = py_as_bend_right(tensor, bend_right);
          return move_leg(std::move(tensor),
                          py_as_leg_ref(which_leg),
                          cpos,
                          dpos,
                          std::move(levels_opt),
                          std::move(bend_opt));
      },
      py::arg("tensor"),
      py::arg("which_leg"),
      py::arg("codomain_pos") = py::none(),
      py::kw_only(),
      py::arg("domain_pos") = py::none(),
      py::arg("levels") = py::none(),
      py::arg("bend_right") = py::none(),
      doc_plus(DOC(lasyten, move_leg),
               R"pydoc(
In Python, ``which_leg`` is ``int | str``; optional args use ``None``; ``levels`` /
``bend_right`` may also be a ``dict``.
)pydoc"));

    m.def(
      "flatten_pipe_leg",
      [](TensorCPtr tensor, py::object which_leg) {
          return flatten_pipe_leg(std::move(tensor), py_as_leg_ref(which_leg));
      },
      py::arg("tensor"),
      py::arg("which_leg"),
      doc_plus(DOC(lasyten, flatten_pipe_leg),
               R"pydoc(
In Python, ``which_leg`` is ``int | str``.
)pydoc"));

    m.def(
      "move_hidden_leg",
      [](HiddenLegTensorCPtr A,
         TensorCPtr B,
         py::object axis_A,
         py::object axis_B,
         std::string hidden_leg_label,
         py::object target_codomain_pos,
         py::object target_domain_pos) {
          std::optional<int64> cpos;
          std::optional<int64> dpos;
          if (!target_codomain_pos.is_none()) {
              cpos = target_codomain_pos.cast<int64>();
          }
          if (!target_domain_pos.is_none()) {
              dpos = target_domain_pos.cast<int64>();
          }
          return move_hidden_leg(std::move(A),
                                 std::move(B),
                                 py_as_leg_ref(axis_A),
                                 py_as_leg_ref(axis_B),
                                 std::move(hidden_leg_label),
                                 cpos,
                                 dpos);
      },
      py::arg("A"),
      py::arg("B"),
      py::arg("axis_A"),
      py::arg("axis_B"),
      py::arg("hidden_leg_label"),
      py::arg("target_codomain_pos") = py::none(),
      py::kw_only(),
      py::arg("target_domain_pos") = py::none(),
      doc_plus(DOC(lasyten, move_hidden_leg),
               R"pydoc(
In Python, ``axis_A`` / ``axis_B`` are ``int | str``; optional position args use ``None``.
)pydoc"));

    m.def(
      "permute_legs",
      [](TensorCPtr tensor,
         py::object codomain,
         py::object domain,
         py::object levels,
         py::object bend_right) {
          auto levels_opt = py_as_levels(tensor, levels);
          auto bend_opt = py_as_bend_right(tensor, bend_right);
          return permute_legs(std::move(tensor),
                              py_opt_leg_refs(codomain),
                              py_opt_leg_refs(domain),
                              std::move(levels_opt),
                              std::move(bend_opt));
      },
      py::arg("tensor"),
      py::arg("codomain") = py::none(),
      py::arg("domain") = py::none(),
      py::arg("levels") = py::none(),
      py::arg("bend_right") = py::none(),
      doc_plus(DOC(lasyten, permute_legs),
               R"pydoc(
In Python, ``codomain`` / ``domain`` are ``int``, ``str``, a sequence thereof, or ``None``;
``levels`` / ``bend_right`` may be a ``list``, ``dict``, or ``None``.
)pydoc"));

    m.def(
      "split_legs",
      [](TensorCPtr tensor, py::object legs) {
          std::optional<std::vector<LegRef>> legs_opt;
          if (!legs.is_none()) {
              legs_opt = py_as_leg_refs(legs);
          }
          return split_legs(std::move(tensor), legs_opt);
      },
      py::arg("tensor"),
      py::arg("legs") = py::none(),
      doc_plus(DOC(lasyten, split_legs),
               R"pydoc(
In Python, ``legs`` is ``int``, ``str``, a sequence thereof, or ``None`` (split all pipes).
)pydoc"));

    char const* slice_leg_py_doc = doc_plus(DOC(lasyten, slice_leg),
                                            R"pydoc(
In Python, both overloads are exposed as one function:
``slice_leg(tensor, leg, idx_or_sector, multiplicity=None)``.
``leg`` is ``int | str``; pass a public-basis index or a :class:`~lasyten.symmetries.Sector`.
)pydoc");

    m.def("slice_leg",
          &py_slice_leg,
          py::arg("tensor"),
          py::arg("leg"),
          py::arg("idx_or_sector"),
          py::arg("multiplicity") = py::none(),
          slice_leg_py_doc);

    py::object tensor_cls = m.attr("Tensor");
    tensor_cls.attr("slice_leg") = py::cpp_function(
      [](TensorCPtr self, py::object leg, py::object idx_or_sector, py::object multiplicity) {
          return py_slice_leg(
            std::move(self), std::move(leg), std::move(idx_or_sector), std::move(multiplicity));
      },
      py::name("slice_leg"),
      py::is_method(tensor_cls),
      py::arg("leg"),
      py::arg("idx_or_sector"),
      py::arg("multiplicity") = py::none(),
      slice_leg_py_doc);

    m.def(
      "squeeze_legs",
      [](TensorCPtr tensor, py::object legs) {
          std::optional<std::vector<LegRef>> legs_opt;
          if (!legs.is_none()) {
              legs_opt = py_as_leg_refs(legs);
          }
          return squeeze_legs(std::move(tensor), legs_opt);
      },
      py::arg("tensor"),
      py::arg("legs") = py::none(),
      doc_plus(DOC(lasyten, squeeze_legs),
               R"pydoc(
In Python, ``legs`` is ``int``, ``str``, a sequence thereof, or ``None`` (squeeze all trivial).
)pydoc"));
}

} // namespace lasyten
