#include <lasyten/tensors/planar.h>

#include <lasyten/tensors/ops_legs.h>
#include <lasyten/tensors/sparse.h>
#include <lasyten/tensors/tensor.h>
#include <lasyten/tools.h>
#include <lasyten/tools/cost_polynomials.h>

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"

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

class PyPlanarLinearOperator
  : public PlanarLinearOperator
  , public py::trampoline_self_life_support
{
  public:
    using PlanarLinearOperator::PlanarLinearOperator;

    VectorLike::Ptr matvec(VectorLike::CPtr vec) override
    {
        PYBIND11_OVERRIDE(VectorLike::Ptr, PlanarLinearOperator, matvec, vec);
    }

    TensorPtr to_tensor(TensorBackend::Ptr backend) override
    {
        PYBIND11_OVERRIDE(TensorPtr, PlanarLinearOperator, to_tensor, backend);
    }
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
    if (py::isinstance<BigOPolynomial>(obj)) {
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
    if (py::isinstance<ContractionTree>(order)) {
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
    if (py::isinstance<ContractionTree>(order)) {
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
    if (py::isinstance<TensorPlaceholder>(first)) {
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
bind_tensors_planar(py::module_& m)
{
    py::class_<TensorPlaceholder, py::smart_holder> tensor_placeholder(m, "TensorPlaceholder");
    tensor_placeholder.doc() = DOC(lasyten, TensorPlaceholder);

    tensor_placeholder
      .def(py::init([](py::object labels, py::object dims, py::object cost_to_make) {
               auto labs = labels_as_strings(labels);
               auto d = parse_dims_list(dims, labs.size());
               BigOPolynomial cost;
               if (!cost_to_make.is_none()) {
                   cost = as_polynomial(cost_to_make);
               }
               return TensorPlaceholder(std::move(labs), std::move(d), std::move(cost));
           }),
           py::arg("labels"),
           py::arg("dims") = py::none(),
           py::arg("cost_to_make") = py::none())
      .def_property(
        "labels",
        py::cpp_function([](TensorPlaceholder& self) -> LegLabels& { return self.labels; },
                         py::return_value_policy::reference_internal),
        py::cpp_function([](TensorPlaceholder& self, py::object value) {
            if (py::isinstance<LegLabels>(value)) {
                self.labels.set(value.cast<LegLabels const&>().as_list());
            } else {
                self.labels.set(value.cast<LegLabelList>());
            }
            self.num_legs = self.labels.num_legs();
        }))
      .def_readwrite("num_legs", &TensorPlaceholder::num_legs)
      .def_readwrite("dims", &TensorPlaceholder::dims)
      .def_readwrite("cost_to_make", &TensorPlaceholder::cost_to_make)
      .def("copy", &TensorPlaceholder::copy, py::arg("deep") = true)
      .def("string_labels", &TensorPlaceholder::string_labels)
      .def("__repr__", &TensorPlaceholder::__repr__);

    py::class_<ContractionTreeNode, py::smart_holder> contraction_tree_node(m,
                                                                            "ContractionTreeNode");
    contraction_tree_node.doc() = DOC(lasyten, ContractionTreeNode);

    contraction_tree_node
      .def(
        py::init(
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
        py::arg("parent"),
        py::arg("left_child"),
        py::arg("right_child"),
        py::arg("value"))
      .def_property(
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
      .def_readwrite("left_child", &ContractionTreeNode::left_child)
      .def_readwrite("right_child", &ContractionTreeNode::right_child)
      .def_readwrite("value", &ContractionTreeNode::value)
      .def_property_readonly("is_leaf", &ContractionTreeNode::is_leaf)
      .def("test_sanity", &ContractionTreeNode::test_sanity, "Perform sanity checks.")
      .def("copy",
           &ContractionTreeNode::copy,
           py::arg("parent") = nullptr,
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

    py::class_<ContractionTree, py::smart_holder> contraction_tree(m, "ContractionTree");
    contraction_tree.doc() = DOC(lasyten, ContractionTree);

    contraction_tree.def(py::init<ContractionTreeNode::Ptr>(), py::arg("root"))
      .def_readwrite("root", &ContractionTree::root)
      .def_property_readonly("leaves", &ContractionTree::leaves)
      .def_property_readonly("num_leaves", &ContractionTree::num_leaves)
      .def_property_readonly("num_nodes", &ContractionTree::num_nodes)
      .def_property_readonly("num_inner_nodes", &ContractionTree::num_inner_nodes)
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
        py::arg("other"),
        py::arg("value") = py::none(),
        DOC(lasyten, ContractionTree, fuse))
      .def("pop_contraction",
           &ContractionTree::pop_contraction,
           DOC(lasyten, ContractionTree, pop_contraction))
      .def("__str__", &ContractionTree::str)
      .def_static(
        "from_contraction_order", &ContractionTree::from_contraction_order, py::arg("order"))
      .def_static("from_single_node",
                  &ContractionTree::from_single_node,
                  py::arg("node"),
                  "Contraction tree from a single node, i.e., without any child and parent nodes.")
      .def_static(
        "from_nested_containers",
        [](py::object tree) { return from_nested_containers_py(tree); },
        py::arg("tree"));

    py::class_<PlanarDiagram, py::smart_holder> planar_diagram(m, "PlanarDiagram");
    planar_diagram.doc() = DOC(lasyten, PlanarDiagram);

    planar_diagram
      .def(py::init([](py::object tensors,
                       py::object definition,
                       py::object dims,
                       py::object order,
                       bool allow_multiple_charged_tensors) {
               return make_planar_diagram(
                 tensors, definition, dims, order, allow_multiple_charged_tensors);
           }),
           py::arg("tensors"),
           py::arg("definition"),
           py::arg("dims") = py::none(),
           py::arg("order") = "definition",
           py::arg("allow_multiple_charged_tensors") = false)
      .def_property(
        "tensors",
        [](PlanarDiagram const& self) { return placeholder_map_to_dict(self); },
        [](PlanarDiagram& self, py::object obj) {
            std::vector<std::string> names;
            self.tensors = parse_tensors_py(obj, py::none(), &names);
            if (!names.empty()) {
                self.tensor_names_ = std::move(names);
            }
        })
      .def_property(
        "definition",
        [](PlanarDiagram const& self) { return definition_to_py(self.definition); },
        [](PlanarDiagram& self, py::object obj) { self.definition = parse_definition_py(obj); })
      .def_readwrite("order", &PlanarDiagram::order)
      .def_readwrite("open_legs", &PlanarDiagram::open_legs)
      .def_readwrite("contraction_cost", &PlanarDiagram::contraction_cost)
      .def_property_readonly("tensor_names", &PlanarDiagram::tensor_names)
      .def_readwrite("allow_multiple_charged_tensors",
                     &PlanarDiagram::allow_multiple_charged_tensors)
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
        py::arg("tensor"),
        py::arg("extra_definition"),
        py::arg("extra_dims") = py::none(),
        py::arg("order") = "definition",
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
        py::arg("name"),
        py::arg("extra_definition") = py::list(),
        py::arg("order") = "greedy",
        DOC(lasyten, PlanarDiagram, remove_tensor))
      .def(
        "evaluate",
        [](PlanarDiagram const& self, py::dict tensors) {
            return evaluate_diagram(self, tensors);
        },
        py::arg("tensors"),
        DOC(lasyten, PlanarDiagram, evaluate))
      .def("__call__",
           [](PlanarDiagram const& self, py::kwargs kwargs) {
               return evaluate_diagram(self, kwargs);
           })
      .def("optimize_order",
           &PlanarDiagram::optimize_order,
           py::arg("strategy"),
           DOC(lasyten, PlanarDiagram, optimize_order))
      .def_static(
        "parse_definition",
        [](py::object definition) { return definition_to_py(parse_definition_py(definition)); },
        py::arg("definition"),
        "Parse the input format for the ``definition`` arg to :class:`PlanarDiagram`.")
      .def(
        "parse_order",
        [](PlanarDiagram const& self, py::object order) { return parse_order_py(self, order); },
        py::arg("order"),
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
        py::arg("tensors"),
        py::arg("dims") = py::none(),
        "Parse the input format for the ``tensors`` arg to :class:`PlanarDiagram`.")
      .def("verify_diagram",
           &PlanarDiagram::verify_diagram,
           DOC(lasyten, PlanarDiagram, verify_diagram));

    // `py::dynamic_attr` plus no data descriptors for `op_diagram` / `matvec_diagram`: subclasses
    // are documented to store those as *class* variables, and `self.op_diagram` in an
    // uninitialized C++ instance would otherwise hit `def_readwrite` and crash.
    py::class_<PlanarLinearOperator, LinearOperator, PyPlanarLinearOperator, py::smart_holder>
      planar_linear_operator(m, "PlanarLinearOperator", py::dynamic_attr());
    planar_linear_operator.doc() = DOC(lasyten, PlanarLinearOperator);

    planar_linear_operator
      .def(py::init<PlanarDiagram const&,
                    PlanarDiagram const&,
                    std::map<std::string, TensorPtr>,
                    std::string>(),
           py::arg("op_diagram"),
           py::arg("matvec_diagram"),
           py::arg("op_tensors"),
           py::arg("vec_name"))
      .def_readwrite("op_tensors", &PlanarLinearOperator::op_tensors)
      .def_readwrite("vec_name", &PlanarLinearOperator::vec_name)
      .def("matvec", &PlanarLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &PlanarLinearOperator::to_tensor, py::arg("backend") = nullptr);

    m.def("parse_leg_bipartition",
          &parse_leg_bipartition,
          py::arg("legs"),
          py::arg("num_legs"),
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
          if (py::isinstance<TensorPlaceholder>(tensor1) ||
              py::isinstance<TensorPlaceholder>(tensor2)) {
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
      py::arg("tensor1"),
      py::arg("tensor2"),
      py::arg("legs1"),
      py::arg("legs2"),
      py::arg("relabel1") = std::map<std::string, std::string>{},
      py::arg("relabel2") = std::map<std::string, std::string>{},
      DOC(lasyten, planar_contraction));

    m.def(
      "planar_partial_trace",
      [](py::object tensor, py::args pairs) -> py::object {
          std::vector<std::vector<LegRef>> ps;
          ps.reserve(static_cast<std::size_t>(pairs.size()));
          for (auto item : pairs) {
              ps.push_back(py_as_leg_refs(py::reinterpret_borrow<py::object>(item)));
          }
          if (py::isinstance<TensorPlaceholder>(tensor)) {
              return py::cast(
                planar_partial_trace(tensor.cast<TensorPlaceholder>(), std::move(ps)));
          }
          return eval_result_to_py(planar_partial_trace(tensor.cast<TensorCPtr>(), std::move(ps)));
      },
      py::arg("tensor"),
      DOC(lasyten, planar_partial_trace));

    m.def(
      "planar_permute_legs",
      [](TensorCPtr T, py::object codomain, py::object domain) {
          return planar_permute_legs(
            std::move(T), py_opt_leg_refs(codomain), py_opt_leg_refs(domain));
      },
      py::arg("T"),
      py::kw_only(),
      py::arg("codomain") = py::none(),
      py::arg("domain") = py::none(),
      DOC(lasyten, planar_permute_legs));

    m.def(
      "planar_combine_legs",
      [](TensorCPtr T, py::args which_legs, py::object pipe_dualities, py::object pipes) {
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
      py::arg("T"),
      py::kw_only(),
      py::arg("pipe_dualities") = false,
      py::arg("pipes") = py::none(),
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
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("sort") = py::none(),
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
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("sort") = py::none(),
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
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("sort") = py::none(),
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
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
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
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
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
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("algorithm") = py::none(),
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
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("algorithm") = py::none(),
      py::arg("normalize_to") = py::none(),
      py::arg("chi_max") = py::none(),
      py::arg("chi_min") = 1,
      py::arg("degeneracy_tol") = 0.,
      py::arg("trunc_cut") = 0.,
      py::arg("svd_min") = 0.,
      DOC(lasyten, planar_truncated_svd));

    m.def("planar_almost_equal",
          &planar_almost_equal,
          py::arg("tensor_1"),
          py::arg("tensor_2"),
          py::arg("rtol") = 1e-5,
          py::arg("atol") = 1e-8,
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
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("cutoff_singular_values") = py::none(),
      DOC(lasyten, horizontal_factorization));

    m.def(
      "planar_decomposition",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         std::string which,
         py::object new_labels,
         bool new_leg_dual,
         py::kwargs kwargs) -> py::object {
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
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("which"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      R"pydoc(Planar generalization of eigen, QR, LQ, SV, and truncated SV decompositions.

See the respective docstrings of :func:`planar_eigh`, :func:`planar_eig`, :func:`planar_eigvals`,
:func:`planar_qr`, :func:`planar_lq`, :func:`planar_svd`, and :func:`planar_truncated_svd` for
more details.
)pydoc");
}

} // namespace lasyten
