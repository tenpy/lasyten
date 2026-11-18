#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/direct_sum.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tensors/tensor.h>
#include <lasyten/tensors/vector_like.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"

#include "docstrings/tensors/ops_algebra.h"

#include <format>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lasyten {

namespace {

std::optional<std::map<std::string, std::string>>
optional_relabel(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return py::cast<std::map<std::string, std::string>>(obj);
}

LegRef
py_as_leg_ref(py::object obj)
{
    if (py::isinstance<py::str>(obj)) {
        return py::cast<std::string>(obj);
    }
    return py::cast<int64>(obj);
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

py::object
py_from_tensor_or_scalar(std::variant<TensorPtr, BlockBackend::Scalar> const& v)
{
    return std::visit([](auto const& x) -> py::object { return py::cast(x); }, v);
}

/// Downcast a VectorLike pointer so Tensor arithmetic still returns a Tensor.
py::object
py_cast_vector_like(VectorLikePtr p)
{
    if (!p) {
        return py::none();
    }
    if (auto t = std::dynamic_pointer_cast<Tensor>(p)) {
        return py::cast(std::move(t));
    }
    if (auto ds = std::dynamic_pointer_cast<DirectSum>(p)) {
        return py::cast(std::move(ds));
    }
    return py::cast(std::move(p));
}

BlockBackend::Scalar
py_as_scalar(py::object obj, TensorCPtr hint)
{
    try {
        return py::cast<BlockBackend::Scalar>(obj);
    } catch (py::cast_error const&) {
    }
    return hint->backend->block_backend->as_scalar(obj, hint->dtype);
}

} // namespace

void
bind_tensors_ops_algebra(nb::module_& m)
{
    m.def("almost_equal",
          &almost_equal,
          nb::arg("tensor_1"),
          nb::arg("tensor_2"),
          nb::arg("rtol") = 1e-5,
          nb::arg("atol") = 1e-8,
          nb::arg("allow_different_types") = false,
          DOC(lasyten, almost_equal));

    m.def(
      "apply_mask",
      [](TensorCPtr tensor, MaskCPtr mask, py::object leg) {
          return apply_mask(std::move(tensor), std::move(mask), py_as_leg_ref(leg));
      },
      nb::arg("tensor"),
      nb::arg("mask"),
      nb::arg("leg"),
      DOC(lasyten, apply_mask));

    m.def(
      "enlarge_leg",
      [](TensorCPtr tensor, MaskCPtr mask, py::object leg) {
          return enlarge_leg(std::move(tensor), std::move(mask), py_as_leg_ref(leg));
      },
      nb::arg("tensor"),
      nb::arg("mask"),
      nb::arg("leg"),
      DOC(lasyten, enlarge_leg));

    m.def("dagger", &dagger, nb::arg("tensor"), DOC(lasyten, dagger));

    m.def(
      "compose",
      [](TensorCPtr tensor1, TensorCPtr tensor2, py::object relabel1, py::object relabel2) {
          return py_from_tensor_or_scalar(compose(std::move(tensor1),
                                                  std::move(tensor2),
                                                  optional_relabel(relabel1),
                                                  optional_relabel(relabel2)));
      },
      nb::arg("tensor1"),
      nb::arg("tensor2"),
      nb::arg("relabel1").none() = nb::none(),
      nb::arg("relabel2").none() = nb::none(),
      doc_plus(DOC(lasyten, compose),
               R"pydoc(
In Python, ``relabel1`` / ``relabel2`` are ``dict | None`` (``None`` = no relabel).
)pydoc"));

    m.def(
      "get_same_device",
      [](nb::args tensors_nb, std::string error_msg) {
          auto tensors = pyargs(tensors_nb);
          std::vector<TensorCPtr> vec;
          vec.reserve(static_cast<std::size_t>(tensors.size()));
          for (auto item : tensors) {
              vec.push_back(py::cast<TensorCPtr>(item));
          }
          return get_same_device(vec, std::move(error_msg));
      },
      nb::arg("tensors"),
      nb::kw_only(),
      nb::arg("error_msg") = "Incompatible devices.",
      DOC(lasyten, get_same_device));

    // Bind the VectorLike overload (DOC(..., 2)); Tensor overload is DOC(lasyten, inner).
    m.def("inner",
          static_cast<BlockBackend::Scalar (*)(VectorLikeCPtr, VectorLikeCPtr, bool)>(&inner),
          nb::arg("A"),
          nb::arg("B"),
          nb::arg("do_dagger") = true,
          DOC(lasyten, inner, 2));

    m.def(
      "is_scalar",
      [](py::object obj) {
          if (py::hasattr(obj, "domain") && py::hasattr(obj, "codomain")) {
              if (py::cast<int64>(obj.attr("domain").attr("num_sectors")) != 1) {
                  return false;
              }
              if (py::cast<int64>(obj.attr("codomain").attr("num_sectors")) != 1) {
                  return false;
              }
              auto np = py::module_::import("numpy");
              if (!py::to_bool(
                    np.attr("array_equal")(obj.attr("domain").attr("sector_decomposition"),
                                           obj.attr("codomain").attr("sector_decomposition")))) {
                  return false;
              }
              if (!py::to_bool(
                    np.attr("all")(obj.attr("domain").attr("multiplicities").attr("__eq__")(1)))) {
                  return false;
              }
              if (!py::to_bool(np.attr("all")(
                    obj.attr("codomain").attr("multiplicities").attr("__eq__")(1)))) {
                  return false;
              }
              return true;
          }
          return py::isinstance(obj, py::module_::import("numbers").attr("Number"));
      },
      nb::arg("obj"),
      doc_plus(DOC(lasyten, is_scalar),
               R"pydoc(
Also accepts Python numbers and duck-typed tensors with ``domain`` / ``codomain``.
)pydoc"));

    m.def("item", &item, nb::arg("tensor"), DOC(lasyten, item));

    m.def(
      "linear_combination",
      [](py::object a, VectorLikeCPtr v, py::object b, VectorLikeCPtr w) {
          auto is_ok = [](py::object o) {
              if (py::isinstance(o, py::module_::import("numbers").attr("Number"))) {
                  return true;
              }
              try {
                  (void)py::cast<BlockBackend::Scalar>(o);
                  return true;
              } catch (py::cast_error const&) {
                  return false;
              }
          };
          if (!v || !w) {
              throw py::type_error("linear_combination() v and w must be VectorLike");
          }
          if (!is_ok(a) || !is_ok(b)) {
              throw py::type_error(
                std::format("unsupported scalar types: {}, {}",
                            std::string(py::str(py::type::of(a).attr("__name__"))),
                            std::string(py::str(py::type::of(b).attr("__name__")))));
          }
          auto sa = py::cast<BlockBackend::Scalar>(
            py::cast(v->vector_backend()->block_backend).attr("as_scalar")(a));
          auto sb = py::cast<BlockBackend::Scalar>(
            py::cast(w->vector_backend()->block_backend).attr("as_scalar")(b));
          return py_cast_vector_like(
            linear_combination(std::move(sa), std::move(v), std::move(sb), std::move(w)));
      },
      nb::arg("a").none(),
      nb::arg("v"),
      nb::arg("b").none(),
      nb::arg("w"),
      doc_cpp_ref(R"pydoc(The linear combination ``a * v + b * w``)pydoc",
                  "lasyten::linear_combination(BlockBackend::Scalar const &, VectorLikeCPtr, "
                  "BlockBackend::Scalar const &, VectorLikeCPtr)"));

    m.def("norm",
          static_cast<BlockBackend::Scalar (*)(VectorLikeCPtr)>(&norm),
          nb::arg("tensor"),
          DOC(lasyten, norm, 2));

    m.def("on_device",
          &on_device,
          nb::arg("tensor"),
          nb::arg("device"),
          nb::arg("copy") = true,
          DOC(lasyten, on_device));

    m.def(
      "outer",
      [](TensorCPtr tensor1, TensorCPtr tensor2, py::object relabel1, py::object relabel2) {
          return outer(std::move(tensor1),
                       std::move(tensor2),
                       optional_relabel(relabel1),
                       optional_relabel(relabel2));
      },
      nb::arg("tensor1"),
      nb::arg("tensor2"),
      nb::arg("relabel1").none() = nb::none(),
      nb::arg("relabel2").none() = nb::none(),
      doc_plus(DOC(lasyten, outer),
               R"pydoc(
In Python, ``relabel1`` / ``relabel2`` are ``dict | None`` (``None`` = no relabel).
)pydoc"));

    m.def(
      "partial_compose",
      [](TensorCPtr tensor1,
         TensorCPtr tensor2,
         py::object tensor1_first_leg,
         py::object relabel1,
         py::object relabel2) {
          return partial_compose(std::move(tensor1),
                                 std::move(tensor2),
                                 py_as_leg_ref(tensor1_first_leg),
                                 optional_relabel(relabel1),
                                 optional_relabel(relabel2));
      },
      nb::arg("tensor1"),
      nb::arg("tensor2"),
      nb::arg("tensor1_first_leg"),
      nb::arg("relabel1").none() = nb::none(),
      nb::arg("relabel2").none() = nb::none(),
      doc_cpp_ref(
        R"pydoc(Tensor contraction / composition involving only a part of the full (co)domain.)pydoc",
        "lasyten::partial_compose()"));

    m.def(
      "partial_trace",
      [](TensorCPtr tensor, nb::args pairs_nb, py::object levels) {
          auto pairs = pyargs(pairs_nb);
          std::vector<std::vector<LegRef>> pair_vec;
          pair_vec.reserve(static_cast<std::size_t>(pairs.size()));
          for (auto item : pairs) {
              pair_vec.push_back(py_as_leg_refs(py::reinterpret_borrow<py::object>(item)));
          }
          std::optional<LevelsSpec> levels_opt;
          if (!levels.is_none()) {
              LevelsSpec spec;
              for (auto item : levels) {
                  py::object o = py::reinterpret_borrow<py::object>(item);
                  if (o.is_none()) {
                      spec.push_back(std::nullopt);
                  } else {
                      spec.push_back(py::cast<int64>(o));
                  }
              }
              levels_opt = std::move(spec);
          }
          return py_from_tensor_or_scalar(
            partial_trace(std::move(tensor), std::move(pair_vec), levels_opt));
      },
      nb::arg("tensor"),
      nb::arg("pairs"),
      nb::kw_only(),
      nb::arg("levels").none() = nb::none(),
      DOC(lasyten, partial_trace));

    m.def("pinv", &pinv, nb::arg("tensor"), nb::arg("cutoff") = 1e-15, DOC(lasyten, pinv));

    m.def(
      "scalar_multiply",
      [](py::object a, py::object v) {
          if (!py::isinstance<VectorLike>(v)) {
              throw py::type_error("scalar_multiply() v must be VectorLike");
          }
          auto vec = py::cast<VectorLikeCPtr>(v);
          bool ok = py::isinstance(a, py::module_::import("numbers").attr("Number"));
          if (!ok) {
              try {
                  (void)py::cast<BlockBackend::Scalar>(a);
                  ok = true;
              } catch (py::cast_error const&) {
              }
          }
          if (!ok) {
              throw py::type_error(
                std::format("unsupported scalar type: {}",
                            std::string(py::str(py::type::of(a).attr("__name__")))));
          }
          auto s = py::cast<BlockBackend::Scalar>(
            py::cast(vec->vector_backend()->block_backend).attr("as_scalar")(a));
          return py_cast_vector_like(scalar_multiply(std::move(s), std::move(vec)));
      },
      nb::arg("a").none(),
      nb::arg("v"),
      doc_cpp_ref(R"pydoc(The scalar multiplication ``a * v``)pydoc",
                  "lasyten::scalar_multiply(BlockBackend::Scalar const &, VectorLikeCPtr)"));

    m.def(
      "scale_axis",
      [](TensorCPtr tensor, DiagonalTensorCPtr diag, py::object leg) {
          return scale_axis(std::move(tensor), std::move(diag), py_as_leg_ref(leg));
      },
      nb::arg("tensor"),
      nb::arg("diag"),
      nb::arg("leg"),
      DOC(lasyten, scale_axis));

    m.def(
      "tdot",
      [](TensorCPtr tensor1,
         TensorCPtr tensor2,
         py::object legs1,
         py::object legs2,
         py::object relabel1,
         py::object relabel2) {
          return py_from_tensor_or_scalar(tdot(std::move(tensor1),
                                               std::move(tensor2),
                                               py_as_leg_refs(legs1),
                                               py_as_leg_refs(legs2),
                                               optional_relabel(relabel1),
                                               optional_relabel(relabel2)));
      },
      nb::arg("tensor1"),
      nb::arg("tensor2"),
      nb::arg("legs1"),
      nb::arg("legs2"),
      nb::arg("relabel1").none() = nb::none(),
      nb::arg("relabel2").none() = nb::none(),
      doc_plus(DOC(lasyten, tdot),
               R"pydoc(
In Python, ``legs1`` / ``legs2`` are ``int``, ``str``, or a sequence thereof;
``relabel1`` / ``relabel2`` are ``dict | None``.
)pydoc"));

    m.def("trace", &trace, nb::arg("tensor"), DOC(lasyten, trace));

    m.def("transpose", &transpose, nb::arg("tensor"), DOC(lasyten, transpose));
}

} // namespace lasyten
