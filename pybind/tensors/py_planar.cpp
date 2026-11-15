#include <lasyten/tensors/planar.h>

#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/ops_legs.h>
#include <lasyten/tensors/sparse.h>
#include <lasyten/tensors/tensor.h>
#include <lasyten/tools.h>
#include <lasyten/tools/cost_polynomials.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"

#include "docstrings/tensors/planar.h"

#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

namespace {

/// nanobind trampoline so Python subclasses can override ``matvec`` / ``to_tensor``.
class PyPlanarLinearOperator : public PlanarLinearOperator
{
  public:
    NB_TRAMPOLINE(PlanarLinearOperator, 2);

    VectorLike::Ptr matvec(VectorLike::CPtr vec) override { NB_OVERRIDE(matvec, vec); }

    TensorPtr to_tensor(TensorBackend::Ptr backend) override { NB_OVERRIDE(to_tensor, backend); }
};

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

py::object
py_from_tensor_or_scalar(std::variant<TensorPtr, BlockBackend::Scalar> const& v)
{
    return std::visit([](auto const& x) -> py::object { return py::cast(x); }, v);
}

py::object
eval_result_to_py(PlanarResult const& v)
{
    return py_from_tensor_or_scalar(v);
}

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

std::optional<float64>
py_opt_float(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<float64>();
}

std::optional<int64>
py_opt_int(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<int64>();
}

std::map<std::string, std::string>
py_relabel_map(py::object obj)
{
    if (obj.is_none()) {
        return {};
    }
    return obj.cast<std::map<std::string, std::string>>();
}

BigOPolynomial
as_polynomial(py::object obj)
{
    if (obj.is_none()) {
        return BigOPolynomial::from_str("None");
    }
    if (lasyten::py_isinstance<BigOPolynomial>(obj)) {
        return obj.cast<BigOPolynomial>();
    }
    return BigOPolynomial::from_str(std::string(py::str(obj)));
}

std::vector<BigOPolynomial>
parse_dims_list(py::object dims, std::size_t /*n_labels*/)
{
    if (dims.is_none()) {
        return {};
    }
    std::vector<BigOPolynomial> out;
    for (auto item : dims) {
        out.push_back(as_polynomial(py::reinterpret_borrow<py::object>(item)));
    }
    return out;
}

std::optional<std::map<std::string, std::vector<std::string>>>
parse_dims_dict(py::object dims)
{
    if (dims.is_none()) {
        return std::nullopt;
    }
    std::map<std::string, std::vector<std::string>> out;
    for (auto item : dims.cast<py::dict>()) {
        std::vector<std::string> labels;
        for (auto lab : py::reinterpret_borrow<py::iterable>(
               py::reinterpret_borrow<py::object>(item.second))) {
            labels.push_back(py::reinterpret_borrow<py::object>(lab).cast<std::string>());
        }
        out.emplace(item.first.cast<std::string>(), std::move(labels));
    }
    return out;
}

std::vector<DiagramInstruction>
parse_definition_py(py::object obj)
{
    if (obj.is_none()) {
        return {};
    }
    if (py::isinstance<py::str>(obj)) {
        auto s = obj.cast<std::string>();
        if (s.empty()) {
            return {};
        }
        return PlanarDiagram::parse_definition(std::move(s));
    }
    std::vector<DiagramInstruction> out;
    for (auto item : obj) {
        auto seq = py::reinterpret_borrow<py::sequence>(item);
        if (seq.size() != 4) {
            throw py::value_error("Each definition entry must be a 4-tuple");
        }
        auto t1 = seq[0].cast<std::string>();
        auto l1 = seq[1].cast<std::string>();
        std::optional<std::string> t2;
        py::object t2obj = seq[2];
        if (!t2obj.is_none()) {
            t2 = t2obj.cast<std::string>();
        }
        auto l2 = seq[3].cast<std::string>();
        out.emplace_back(std::move(t1), std::move(l1), std::move(t2), std::move(l2));
    }
    return PlanarDiagram::parse_definition(std::move(out));
}

TensorPlaceholderMap
parse_tensors_py(py::object tensors, py::object dims, std::vector<std::string>* names)
{
    auto dims_opt = parse_dims_dict(dims);
    if (py::isinstance<py::dict>(tensors)) {
        TensorPlaceholderMap map;
        if (names) {
            names->clear();
        }
        for (auto item : tensors.cast<py::dict>()) {
            auto key = item.first.cast<std::string>();
            if (names) {
                names->push_back(key);
            }
            map.emplace(key, item.second.cast<TensorPlaceholder>());
        }
        return PlanarDiagram::parse_tensors(std::move(map), dims_opt, nullptr);
    }
    if (py::isinstance<py::str>(tensors)) {
        return PlanarDiagram::parse_tensors(tensors.cast<std::string>(), dims_opt, names);
    }
    throw py::type_error("Expected dict or str for tensors");
}

ContractionTree
from_nested_containers_py(py::object tree)
{
    if (!py::isinstance<py::tuple>(tree) && !py::isinstance<py::list>(tree)) {
        return ContractionTree::from_single_node(tree.cast<std::string>());
    }
    auto seq = py::reinterpret_borrow<py::sequence>(tree);
    if (seq.size() != 2) {
        throw py::value_error("Nested contraction-order containers must have length 2");
    }
    auto left = from_nested_containers_py(seq[0]);
    auto right = from_nested_containers_py(seq[1]);
    return left.fuse(right, std::nullopt);
}

ContractionTree
parse_order_py(PlanarDiagram const& self, py::object order)
{
    if (py::isinstance<py::str>(order)) {
        return self.parse_order(order.cast<std::string>());
    }
    if (lasyten::py_isinstance<ContractionTree>(order)) {
        return self.parse_order(order.cast<ContractionTree>());
    }
    if (self.tensors.size() == 1) {
        auto name =
          self.tensor_names_.empty() ? self.tensors.begin()->first : self.tensor_names_.front();
        return ContractionTree::from_single_node(name);
    }
    return from_nested_containers_py(order);
}

PlanarDiagram
make_planar_diagram(py::object tensors,
                    py::object definition,
                    py::object dims,
                    py::object order,
                    bool allow)
{
    std::vector<std::string> names;
    auto tens = parse_tensors_py(tensors, dims, &names);
    auto defn = parse_definition_py(definition);
    if (tens.size() <= 1 || py::isinstance<py::str>(order)) {
        std::string order_str =
          py::isinstance<py::str>(order) ? order.cast<std::string>() : std::string("definition");
        return PlanarDiagram(std::move(tens), std::move(names), std::move(defn), order_str, allow);
    }
    if (lasyten::py_isinstance<ContractionTree>(order)) {
        return PlanarDiagram(std::move(tens),
                             std::move(names),
                             std::move(defn),
                             order.cast<ContractionTree>(),
                             allow);
    }
    auto tree = from_nested_containers_py(order);
    return PlanarDiagram(
      std::move(tens), std::move(names), std::move(defn), std::move(tree), allow);
}

PlanarDiagram
with_order(PlanarDiagram diag, py::object order)
{
    if (py::isinstance<py::str>(order)) {
        return PlanarDiagram(std::move(diag.tensors),
                             std::move(diag.tensor_names_),
                             std::move(diag.definition),
                             order.cast<std::string>(),
                             diag.allow_multiple_charged_tensors);
    }
    auto tree = parse_order_py(diag, order);
    return PlanarDiagram(std::move(diag.tensors),
                         std::move(diag.tensor_names_),
                         std::move(diag.definition),
                         std::move(tree),
                         diag.allow_multiple_charged_tensors);
}

py::dict
placeholder_map_to_dict(PlanarDiagram const& self)
{
    py::dict out;
    auto names = self.tensor_names();
    if (names.empty()) {
        for (auto const& [k, v] : self.tensors) {
            out[py::cast(k)] = py::cast(v);
        }
        return out;
    }
    for (auto const& n : names) {
        out[py::cast(n)] = py::cast(self.tensors.at(n));
    }
    return out;
}

py::list
definition_to_py(std::vector<DiagramInstruction> const& defn)
{
    py::list out;
    for (auto const& [t1, l1, t2, l2] : defn) {
        py::object t2py = t2 ? py::cast(*t2) : py::none();
        out.append(py::make_tuple(t1, l1, t2py, l2));
    }
    return out;
}

py::object
evaluate_diagram(PlanarDiagram const& self, py::object tensors_obj)
{
    py::dict tensors = py::dict(std::move(tensors_obj));
    if (tensors.size() == 0) {
        return eval_result_to_py(self.evaluate(std::map<std::string, TensorPtr>{}));
    }
    py::object first;
    for (auto item : tensors) {
        first = py::reinterpret_borrow<py::object>(item.second);
        break;
    }
    if (lasyten::py_isinstance<TensorPlaceholder>(first)) {
        std::map<std::string, TensorPlaceholder> map;
        for (auto item : tensors) {
            map.emplace(item.first.cast<std::string>(), item.second.cast<TensorPlaceholder>());
        }
        return py::cast(self.evaluate(std::move(map)));
    }
    std::map<std::string, TensorPtr> map;
    for (auto item : tensors) {
        map.emplace(item.first.cast<std::string>(), item.second.cast<TensorPtr>());
    }
    return eval_result_to_py(self.evaluate(std::move(map)));
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

std::vector<std::string>
labels_as_strings(py::object labels)
{
    std::vector<std::string> out;
    for (auto item : labels) {
        out.push_back(py::reinterpret_borrow<py::object>(item).cast<std::string>());
    }
    return out;
}

} // namespace

void
bind_tensors_planar(nb::module_& m)
{
    nb::class_<TensorPlaceholder> tensor_placeholder(m, "TensorPlaceholder");
    tensor_placeholder.doc() = DOC(lasyten, TensorPlaceholder);

    tensor_placeholder
      .def(nb::new_([](py::object labels, py::object dims, py::object cost_to_make) {
               auto labs = labels_as_strings(labels);
               auto d = parse_dims_list(dims, labs.size());
               BigOPolynomial cost;
               if (!cost_to_make.is_none()) {
                   cost = as_polynomial(cost_to_make);
               }
               return TensorPlaceholder(std::move(labs), std::move(d), std::move(cost));
           }),
           nb::arg("labels"),
           nb::arg("dims").none() = nb::none(),
           nb::arg("cost_to_make").none() = nb::none())
      .def_prop_rw(
        "labels",
        [](TensorPlaceholder& self) -> LegLabels& { return self.labels; },
        [](TensorPlaceholder& self, py::object value) {
            if (lasyten::py_isinstance<LegLabels>(value)) {
                self.labels.set(value.cast<LegLabels const&>().as_list());
            } else {
                self.labels.set(value.cast<LegLabelList>());
            }
            self.num_legs = self.labels.num_legs();
        },
        nb::rv_policy::reference_internal)
      .def_rw("num_legs", &TensorPlaceholder::num_legs)
      .def_rw("dims", &TensorPlaceholder::dims)
      .def_rw("cost_to_make", &TensorPlaceholder::cost_to_make)
      .def("copy", &TensorPlaceholder::copy, nb::arg("deep") = true)
      .def("string_labels", &TensorPlaceholder::string_labels)
      .def("__repr__", &TensorPlaceholder::__repr__);

    nb::class_<ContractionTreeNode> contraction_tree_node(m, "ContractionTreeNode");
    contraction_tree_node.doc() = DOC(lasyten, ContractionTreeNode);

    contraction_tree_node
      .def(
        nb::new_(
          [](py::object parent, py::object left_child, py::object right_child, py::object value) {
              ContractionTreeNode::Ptr p;
              ContractionTreeNode::Ptr left;
              ContractionTreeNode::Ptr right;
              if (!parent.is_none()) {
                  p = parent.cast<ContractionTreeNode::Ptr>();
              }
              if (!left_child.is_none()) {
                  left = left_child.cast<ContractionTreeNode::Ptr>();
              }
              if (!right_child.is_none()) {
                  right = right_child.cast<ContractionTreeNode::Ptr>();
              }
              std::optional<std::string> val;
              if (!value.is_none()) {
                  val = value.cast<std::string>();
              }
              return std::make_shared<ContractionTreeNode>(
                std::move(p), std::move(left), std::move(right), std::move(val));
          }),
        nb::arg("parent"),
        nb::arg("left_child"),
        nb::arg("right_child"),
        nb::arg("value"))
      .def_prop_rw(
        "parent",
        [](ContractionTreeNode const& self) -> py::object {
            auto p = self.parent.lock();
            if (!p) {
                return py::none();
            }
            return py::cast(p);
        },
        [](ContractionTreeNode& self, py::object parent) {
            if (parent.is_none()) {
                self.parent.reset();
            } else {
                self.parent = parent.cast<ContractionTreeNode::Ptr>();
            }
        })
      .def_rw("left_child", &ContractionTreeNode::left_child)
      .def_rw("right_child", &ContractionTreeNode::right_child)
      .def_rw("value", &ContractionTreeNode::value)
      .def_prop_ro("is_leaf", &ContractionTreeNode::is_leaf)
      .def("test_sanity", &ContractionTreeNode::test_sanity, "Perform sanity checks.")
      .def("copy",
           &ContractionTreeNode::copy,
           nb::arg("parent").none() = nullptr,
           "Implement :meth:`ContractionTree.copy` recursively.")
      .def("get_leaves", &ContractionTreeNode::get_leaves, "Returns ``leaves, num_nodes_below``.")
      .def("remove_children",
           &ContractionTreeNode::remove_children,
           "Remove both children and return their values.")
      .def("pop_contraction",
           &ContractionTreeNode::pop_contraction,
           "Implement :meth:`ContractionTree.pop_contraction` recursively.")
      .def("show_whole_tree",
           &ContractionTreeNode::show_whole_tree,
           "Return a graphic representation of the full contraction tree.");

    nb::class_<ContractionTree> contraction_tree(m, "ContractionTree");
    contraction_tree.doc() = DOC(lasyten, ContractionTree);

    contraction_tree
      .def(lasyten::ctor<decltype(contraction_tree), ContractionTreeNode::Ptr>(), nb::arg("root"))
      .def_rw("root", &ContractionTree::root)
      .def_prop_ro("leaves", &ContractionTree::leaves)
      .def_prop_ro("num_leaves", &ContractionTree::num_leaves)
      .def_prop_ro("num_nodes", &ContractionTree::num_nodes)
      .def_prop_ro("num_inner_nodes", &ContractionTree::num_inner_nodes)
      .def("test_sanity", &ContractionTree::test_sanity, "Perform sanity checks.")
      .def("copy", &ContractionTree::copy)
      .def(
        "fuse",
        [](ContractionTree& self, ContractionTree& other, py::object value) {
            std::optional<std::string> v;
            if (!value.is_none()) {
                v = value.cast<std::string>();
            }
            return self.fuse(other, std::move(v));
        },
        nb::arg("other"),
        nb::arg("value").none() = nb::none(),
        DOC(lasyten, ContractionTree, fuse))
      .def("pop_contraction",
           &ContractionTree::pop_contraction,
           DOC(lasyten, ContractionTree, pop_contraction))
      .def("__str__", &ContractionTree::str)
      .def_static(
        "from_contraction_order", &ContractionTree::from_contraction_order, nb::arg("order"))
      .def_static("from_single_node",
                  &ContractionTree::from_single_node,
                  nb::arg("node"),
                  "Contraction tree from a single node, i.e., without any child and parent nodes.")
      .def_static(
        "from_nested_containers",
        [](py::object tree) { return from_nested_containers_py(tree); },
        nb::arg("tree"));

    nb::class_<PlanarDiagram> planar_diagram(m, "PlanarDiagram");
    planar_diagram.doc() = DOC(lasyten, PlanarDiagram);

    planar_diagram
      .def(nb::new_([](py::object tensors,
                       py::object definition,
                       py::object dims,
                       py::object order,
                       bool allow_multiple_charged_tensors) {
               return make_planar_diagram(
                 tensors, definition, dims, order, allow_multiple_charged_tensors);
           }),
           nb::arg("tensors"),
           nb::arg("definition"),
           nb::arg("dims").none() = nb::none(),
           nb::arg("order") = "definition",
           nb::arg("allow_multiple_charged_tensors") = false)
      .def_prop_rw(
        "tensors",
        [](PlanarDiagram const& self) { return placeholder_map_to_dict(self); },
        [](PlanarDiagram& self, py::object obj) {
            std::vector<std::string> names;
            self.tensors = parse_tensors_py(obj, py::none(), &names);
            if (!names.empty()) {
                self.tensor_names_ = std::move(names);
            }
        })
      .def_prop_rw(
        "definition",
        [](PlanarDiagram const& self) { return definition_to_py(self.definition); },
        [](PlanarDiagram& self, py::object obj) { self.definition = parse_definition_py(obj); })
      .def_rw("order", &PlanarDiagram::order)
      .def_rw("open_legs", &PlanarDiagram::open_legs)
      .def_rw("contraction_cost", &PlanarDiagram::contraction_cost)
      .def_prop_ro("tensor_names", &PlanarDiagram::tensor_names)
      .def_rw("allow_multiple_charged_tensors", &PlanarDiagram::allow_multiple_charged_tensors)
      .def(
        "add_tensor",
        [](PlanarDiagram const& self,
           py::object tensor,
           py::object extra_definition,
           py::object extra_dims,
           py::object order) {
            std::vector<std::string> names;
            auto extra = parse_tensors_py(tensor, extra_dims, &names);
            auto extra_def = parse_definition_py(extra_definition);
            if (py::isinstance<py::str>(order)) {
                return self.add_tensor(
                  std::move(extra), std::move(extra_def), order.cast<std::string>());
            }
            auto tmp = self.add_tensor(std::move(extra), std::move(extra_def), "definition");
            return with_order(std::move(tmp), order);
        },
        nb::arg("tensor"),
        nb::arg("extra_definition"),
        nb::arg("extra_dims").none() = nb::none(),
        nb::arg("order") = "definition",
        DOC(lasyten, PlanarDiagram, add_tensor))
      .def(
        "remove_tensor",
        [](PlanarDiagram const& self,
           std::string name,
           py::object extra_definition,
           py::object order) {
            auto extra_def = parse_definition_py(extra_definition);
            if (py::isinstance<py::str>(order)) {
                return self.remove_tensor(name, std::move(extra_def), order.cast<std::string>());
            }
            auto tmp = self.remove_tensor(name, extra_def, "definition");
            return with_order(std::move(tmp), order);
        },
        nb::arg("name"),
        nb::arg("extra_definition") = py::list(),
        nb::arg("order") = "greedy",
        DOC(lasyten, PlanarDiagram, remove_tensor))
      .def(
        "evaluate",
        [](PlanarDiagram const& self, py::dict tensors) {
            return evaluate_diagram(self, tensors);
        },
        nb::arg("tensors"),
        DOC(lasyten, PlanarDiagram, evaluate))
      .def("__call__",
           [](PlanarDiagram const& self, nb::kwargs kwargs_nb) {
               auto kwargs = pykwargs(kwargs_nb);
               return evaluate_diagram(self, kwargs);
           })
      .def("optimize_order",
           &PlanarDiagram::optimize_order,
           nb::arg("strategy"),
           DOC(lasyten, PlanarDiagram, optimize_order))
      .def_static(
        "parse_definition",
        [](py::object definition) { return definition_to_py(parse_definition_py(definition)); },
        nb::arg("definition"),
        "Parse the input format for the ``definition`` arg to :class:`PlanarDiagram`.")
      .def(
        "parse_order",
        [](PlanarDiagram const& self, py::object order) { return parse_order_py(self, order); },
        nb::arg("order"),
        "Parse the input format for the ``order`` arg to :class:`PlanarDiagram`.")
      .def_static(
        "parse_tensors",
        [](py::object tensors, py::object dims) {
            std::vector<std::string> names;
            auto map = parse_tensors_py(tensors, dims, &names);
            py::dict out;
            if (names.empty()) {
                for (auto const& [k, v] : map) {
                    out[py::cast(k)] = py::cast(v);
                }
            } else {
                for (auto const& n : names) {
                    out[py::cast(n)] = py::cast(map.at(n));
                }
            }
            return out;
        },
        nb::arg("tensors"),
        nb::arg("dims").none() = nb::none(),
        "Parse the input format for the ``tensors`` arg to :class:`PlanarDiagram`.")
      .def("verify_diagram",
           &PlanarDiagram::verify_diagram,
           DOC(lasyten, PlanarDiagram, verify_diagram));

    // `nb::dynamic_attr` plus no data descriptors for `op_diagram` / `matvec_diagram`: subclasses
    // are documented to store those as *class* variables, and `self.op_diagram` in an
    // uninitialized C++ instance would otherwise hit `def_readwrite` and crash.
    nb::class_<PlanarLinearOperator, LinearOperator, PyPlanarLinearOperator>
      planar_linear_operator(m, "PlanarLinearOperator", nb::dynamic_attr());
    planar_linear_operator.doc() = DOC(lasyten, PlanarLinearOperator);

    planar_linear_operator
      .def(lasyten::ctor<decltype(planar_linear_operator),
                         PlanarDiagram const&,
                         PlanarDiagram const&,
                         std::map<std::string, TensorPtr>,
                         std::string>(),
           nb::arg("op_diagram"),
           nb::arg("matvec_diagram"),
           nb::arg("op_tensors"),
           nb::arg("vec_name"))
      .def_rw("op_tensors", &PlanarLinearOperator::op_tensors)
      .def_rw("vec_name", &PlanarLinearOperator::vec_name)
      .def("matvec", &PlanarLinearOperator::matvec, nb::arg("vec"))
      .def("to_tensor", &PlanarLinearOperator::to_tensor, nb::arg("backend").none() = nullptr);

    m.def("parse_leg_bipartition",
          &parse_leg_bipartition,
          nb::arg("legs"),
          nb::arg("num_legs"),
          DOC(lasyten, parse_leg_bipartition));

    m.def(
      "planar_contraction",
      [](py::object tensor1,
         py::object tensor2,
         py::object legs1,
         py::object legs2,
         py::object relabel1,
         py::object relabel2) -> py::object {
          auto l1 = py_as_leg_refs(legs1);
          auto l2 = py_as_leg_refs(legs2);
          if (lasyten::py_isinstance<TensorPlaceholder>(tensor1) ||
              lasyten::py_isinstance<TensorPlaceholder>(tensor2)) {
              return py::cast(planar_contraction(tensor1.cast<TensorPlaceholder>(),
                                                 tensor2.cast<TensorPlaceholder>(),
                                                 std::move(l1),
                                                 std::move(l2)));
          }
          return eval_result_to_py(planar_contraction(tensor1.cast<TensorCPtr>(),
                                                      tensor2.cast<TensorCPtr>(),
                                                      std::move(l1),
                                                      std::move(l2),
                                                      py_relabel_map(relabel1),
                                                      py_relabel_map(relabel2)));
      },
      nb::arg("tensor1"),
      nb::arg("tensor2"),
      nb::arg("legs1"),
      nb::arg("legs2"),
      nb::arg("relabel1") = std::map<std::string, std::string>{},
      nb::arg("relabel2") = std::map<std::string, std::string>{},
      DOC(lasyten, planar_contraction));

    m.def(
      "planar_contract_mask",
      [](TensorCPtr tensor1,
         MaskCPtr mask2,
         py::object legs1,
         py::object legs2,
         py::object relabel1,
         py::object relabel2) {
          return planar_contract_mask(std::move(tensor1),
                                      std::move(mask2),
                                      py_as_leg_refs(legs1),
                                      py_as_leg_refs(legs2),
                                      py_relabel_map(relabel1),
                                      py_relabel_map(relabel2));
      },
      nb::arg("tensor1"),
      nb::arg("mask2"),
      nb::arg("legs1"),
      nb::arg("legs2"),
      nb::arg("relabel1") = std::map<std::string, std::string>{},
      nb::arg("relabel2") = std::map<std::string, std::string>{},
      DOC(lasyten, planar_contract_mask));

    m.def(
      "planar_partial_trace",
      [](py::object tensor, nb::args pairs_nb) -> py::object {
          auto pairs = pyargs(pairs_nb);
          std::vector<std::vector<LegRef>> ps;
          ps.reserve(static_cast<std::size_t>(pairs.size()));
          for (auto item : pairs) {
              ps.push_back(py_as_leg_refs(py::reinterpret_borrow<py::object>(item)));
          }
          if (lasyten::py_isinstance<TensorPlaceholder>(tensor)) {
              return py::cast(
                planar_partial_trace(tensor.cast<TensorPlaceholder>(), std::move(ps)));
          }
          return eval_result_to_py(planar_partial_trace(tensor.cast<TensorCPtr>(), std::move(ps)));
      },
      nb::arg("tensor"),
      nb::arg("pairs"),
      DOC(lasyten, planar_partial_trace));

    m.def(
      "planar_permute_legs",
      [](TensorCPtr T, py::object codomain, py::object domain) {
          return planar_permute_legs(
            std::move(T), py_opt_leg_refs(codomain), py_opt_leg_refs(domain));
      },
      nb::arg("T"),
      nb::kw_only(),
      nb::arg("codomain").none() = nb::none(),
      nb::arg("domain").none() = nb::none(),
      DOC(lasyten, planar_permute_legs));

    m.def(
      "planar_combine_legs",
      [](TensorCPtr T, nb::args which_legs_nb, py::object pipe_dualities, py::object pipes) {
          auto which_legs = pyargs(which_legs_nb);
          std::vector<std::vector<LegRef>> groups;
          groups.reserve(static_cast<std::size_t>(which_legs.size()));
          for (auto item : which_legs) {
              groups.push_back(py_as_leg_refs(py::reinterpret_borrow<py::object>(item)));
          }
          return planar_combine_legs(std::move(T),
                                     std::move(groups),
                                     py_as_pipe_dualities(pipe_dualities),
                                     py_as_pipes(pipes));
      },
      nb::arg("T"),
      nb::arg("which_legs"),
      nb::kw_only(),
      nb::arg("pipe_dualities") = false,
      nb::arg("pipes").none() = nb::none(),
      DOC(lasyten, planar_combine_legs));

    m.def(
      "planar_eigh",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         bool new_leg_dual,
         py::object sort) {
          return planar_eigh(std::move(tensor),
                             codomain_cut,
                             domain_cut,
                             py_opt_labels(new_labels),
                             new_leg_dual,
                             py_opt_string(sort));
      },
      nb::arg("tensor"),
      nb::arg("codomain_cut"),
      nb::arg("domain_cut"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("new_leg_dual") = false,
      nb::arg("sort").none() = nb::none(),
      DOC(lasyten, planar_eigh));

    m.def(
      "planar_eig",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         bool new_leg_dual,
         py::object sort) {
          return planar_eig(std::move(tensor),
                            codomain_cut,
                            domain_cut,
                            py_opt_labels(new_labels),
                            new_leg_dual,
                            py_opt_string(sort));
      },
      nb::arg("tensor"),
      nb::arg("codomain_cut"),
      nb::arg("domain_cut"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("new_leg_dual") = false,
      nb::arg("sort").none() = nb::none(),
      DOC(lasyten, planar_eig));

    m.def(
      "planar_eigvals",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         bool new_leg_dual,
         py::object sort) {
          return planar_eigvals(std::move(tensor),
                                codomain_cut,
                                domain_cut,
                                py_opt_labels(new_labels),
                                new_leg_dual,
                                py_opt_string(sort));
      },
      nb::arg("tensor"),
      nb::arg("codomain_cut"),
      nb::arg("domain_cut"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("new_leg_dual") = false,
      nb::arg("sort").none() = nb::none(),
      DOC(lasyten, planar_eigvals));

    m.def(
      "planar_lq",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         bool new_leg_dual) {
          return planar_lq(
            std::move(tensor), codomain_cut, domain_cut, py_opt_labels(new_labels), new_leg_dual);
      },
      nb::arg("tensor"),
      nb::arg("codomain_cut"),
      nb::arg("domain_cut"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("new_leg_dual") = false,
      DOC(lasyten, planar_lq));

    m.def(
      "planar_qr",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         bool new_leg_dual) {
          return planar_qr(
            std::move(tensor), codomain_cut, domain_cut, py_opt_labels(new_labels), new_leg_dual);
      },
      nb::arg("tensor"),
      nb::arg("codomain_cut"),
      nb::arg("domain_cut"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("new_leg_dual") = false,
      DOC(lasyten, planar_qr));

    m.def(
      "planar_svd",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         bool new_leg_dual,
         py::object algorithm) {
          return planar_svd(std::move(tensor),
                            codomain_cut,
                            domain_cut,
                            py_opt_labels(new_labels),
                            new_leg_dual,
                            py_opt_string(algorithm));
      },
      nb::arg("tensor"),
      nb::arg("codomain_cut"),
      nb::arg("domain_cut"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("new_leg_dual") = false,
      nb::arg("algorithm").none() = nb::none(),
      DOC(lasyten, planar_svd));

    m.def(
      "planar_truncated_svd",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         bool new_leg_dual,
         py::object algorithm,
         py::object normalize_to,
         py::object chi_max,
         int64 chi_min,
         float64 degeneracy_tol,
         float64 trunc_cut,
         float64 svd_min) {
          return planar_truncated_svd(std::move(tensor),
                                      codomain_cut,
                                      domain_cut,
                                      py_opt_labels(new_labels),
                                      new_leg_dual,
                                      py_opt_string(algorithm),
                                      py_opt_float(normalize_to),
                                      py_opt_int(chi_max),
                                      chi_min,
                                      degeneracy_tol,
                                      trunc_cut,
                                      svd_min);
      },
      nb::arg("tensor"),
      nb::arg("codomain_cut"),
      nb::arg("domain_cut"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("new_leg_dual") = false,
      nb::arg("algorithm").none() = nb::none(),
      nb::arg("normalize_to").none() = nb::none(),
      nb::arg("chi_max").none() = nb::none(),
      nb::arg("chi_min") = 1,
      nb::arg("degeneracy_tol") = 0.,
      nb::arg("trunc_cut") = 0.,
      nb::arg("svd_min") = 0.,
      DOC(lasyten, planar_truncated_svd));

    m.def("planar_almost_equal",
          &planar_almost_equal,
          nb::arg("tensor_1"),
          nb::arg("tensor_2"),
          nb::arg("rtol") = 1e-5,
          nb::arg("atol") = 1e-8,
          DOC(lasyten, planar_almost_equal));

    m.def(
      "horizontal_factorization",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         py::object cutoff_singular_values) {
          return horizontal_factorization(std::move(tensor),
                                          codomain_cut,
                                          domain_cut,
                                          py_opt_labels(new_labels),
                                          py_opt_float(cutoff_singular_values));
      },
      nb::arg("tensor"),
      nb::arg("codomain_cut"),
      nb::arg("domain_cut"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("cutoff_singular_values").none() = nb::none(),
      DOC(lasyten, horizontal_factorization));

    m.def(
      "planar_decomposition",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         std::string which,
         py::object new_labels,
         bool new_leg_dual,
         nb::kwargs kwargs_nb) -> py::object {
          auto kwargs = pykwargs(kwargs_nb);
          auto labs = py_opt_labels(new_labels);
          auto kw = [&](char const* key) -> py::object {
              return kwargs.contains(key) ? py::object(kwargs[key]) : py::none();
          };
          if (which == "eigh") {
              return py::cast(planar_eigh(
                tensor, codomain_cut, domain_cut, labs, new_leg_dual, py_opt_string(kw("sort"))));
          }
          if (which == "eig") {
              return py::cast(planar_eig(
                tensor, codomain_cut, domain_cut, labs, new_leg_dual, py_opt_string(kw("sort"))));
          }
          if (which == "eigvals") {
              return py::cast(planar_eigvals(
                tensor, codomain_cut, domain_cut, labs, new_leg_dual, py_opt_string(kw("sort"))));
          }
          if (which == "lq") {
              return py::cast(planar_lq(tensor, codomain_cut, domain_cut, labs, new_leg_dual));
          }
          if (which == "qr") {
              return py::cast(planar_qr(tensor, codomain_cut, domain_cut, labs, new_leg_dual));
          }
          if (which == "svd") {
              return py::cast(planar_svd(tensor,
                                         codomain_cut,
                                         domain_cut,
                                         labs,
                                         new_leg_dual,
                                         py_opt_string(kw("algorithm"))));
          }
          if (which == "truncated_svd") {
              int64 chi_min = 1;
              if (!kw("chi_min").is_none()) {
                  chi_min = kw("chi_min").cast<int64>();
              }
              float64 degeneracy_tol = 0.;
              if (!kw("degeneracy_tol").is_none()) {
                  degeneracy_tol = kw("degeneracy_tol").cast<float64>();
              }
              float64 trunc_cut = 0.;
              if (!kw("trunc_cut").is_none()) {
                  trunc_cut = kw("trunc_cut").cast<float64>();
              }
              float64 svd_min = 0.;
              if (!kw("svd_min").is_none()) {
                  svd_min = kw("svd_min").cast<float64>();
              }
              return py::cast(planar_truncated_svd(tensor,
                                                   codomain_cut,
                                                   domain_cut,
                                                   labs,
                                                   new_leg_dual,
                                                   py_opt_string(kw("algorithm")),
                                                   py_opt_float(kw("normalize_to")),
                                                   py_opt_int(kw("chi_max")),
                                                   chi_min,
                                                   degeneracy_tol,
                                                   trunc_cut,
                                                   svd_min));
          }
          throw py::value_error(std::string("Invalid decomposition \"") + which + '"');
      },
      nb::arg("tensor"),
      nb::arg("codomain_cut"),
      nb::arg("domain_cut"),
      nb::arg("which"),
      nb::arg("new_labels").none() = nb::none(),
      nb::arg("new_leg_dual") = false,
      nb::arg("kwargs"),
      R"pydoc(Planar generalization of eigen, QR, LQ, SV, and truncated SV decompositions.

See the respective docstrings of :func:`planar_eigh`, :func:`planar_eig`, :func:`planar_eigvals`,
:func:`planar_qr`, :func:`planar_lq`, :func:`planar_svd`, and :func:`planar_truncated_svd` for
more details.
)pydoc");
}

} // namespace lasyten
