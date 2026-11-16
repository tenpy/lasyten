#include <lasyten/tensors/tensor.h>

#include "py_trampolines.hpp"

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"

#include "docstrings/tensors/tensor.h"

#include <cmath>
#include <format>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lasyten {

namespace {

std::optional<std::vector<std::variant<int64, std::string>>>
optional_leg_order(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    std::vector<std::variant<int64, std::string>> out;
    for (auto item : to_iterable(obj)) {
        if (py::isinstance<py::str>(item)) {
            out.emplace_back(py::cast<std::string>(item));
        } else {
            out.emplace_back(py::cast<int64>(item));
        }
    }
    return out;
}

std::variant<int64, std::string>
as_leg_ref(py::handle obj)
{
    if (py::isinstance<py::str>(obj)) {
        return py::cast<std::string>(obj);
    }
    return py::cast<int64>(obj);
}

std::optional<nb::class_<Tensor, VectorLike>> g_tensor;

} // namespace

void
declare_tensors_tensor(nb::module_& m)
{
    g_tensor = nb::class_<Tensor, VectorLike>(m, "Tensor");
    (*g_tensor).doc() = DOC(lasyten, Tensor);
}

void
bind_tensors_tensor(nb::module_& m)
{
    auto& cls = *g_tensor;

    // ``Tensor`` is abstract and (per the nanobind port policy) has no trampoline, so it can
    // neither be instantiated directly nor through a Python subclass; use the concrete classes.
    cls.def(nb::new_([](py::object, py::object, TensorBackend::Ptr, py::object, Dtype, std::string)
                       -> std::shared_ptr<Tensor> {
                throw nb::type_error(
                  "Tensor is abstract and cannot be instantiated or subclassed from Python "
                  "(no trampoline); use SymmetricTensor, DiagonalTensor, ... instead.");
            }),
            nb::arg("codomain"),
            nb::arg("domain"),
            nb::arg("backend"),
            nb::arg("labels"),
            nb::arg("dtype"),
            nb::arg("device"));

    cls.def_rw("codomain", &Tensor::codomain)
      .def_rw("domain", &Tensor::domain)
      .def_rw("backend", &Tensor::backend)
      .def_rw("symmetry", &Tensor::symmetry)
      .def_rw("dtype", &Tensor::dtype)
      .def_rw("device", &Tensor::device)
      .def_rw("num_legs", &Tensor::num_legs)
      .def_prop_rw(
        "labels",
        [](Tensor& self) -> LegLabels& { return self.labels; },
        [](Tensor& self, py::object value) {
            if (py::isinstance<LegLabels>(value)) {
                self.labels.set(py::cast<LegLabels const&>(value).as_list());
            } else {
                self.labels.set(py::cast<LegLabelList>(value));
            }
        },
        nb::rv_policy::reference_internal)
      .def_prop_rw(
        "shape",
        [](Tensor const& self) {
            // Match Space.dim / Leg.dim: whole-number dims as int (np.zeros etc.).
            py::list out;
            for (float64 d : self.shape) {
                if (std::isfinite(d) && std::floor(d) == d) {
                    out.append(py::int_(static_cast<long long>(d)));
                } else {
                    out.append(py::float_(d));
                }
            }
            return py::tuple(out);
        },
        [](Tensor& self, py::object shape_obj) {
            self.shape = py::cast<std::vector<float64>>(shape_obj);
        });

    cls.def_static(
      "_init_parse_args",
      [](py::object codomain, py::object domain, TensorBackend::Ptr backend) {
          return parse_tensor_init_args(codomain, domain, std::move(backend));
      },
      nb::arg("codomain"),
      nb::arg("domain"),
      nb::arg("backend"),
      DOC(lasyten, Tensor, _init_parse_args));

    cls.def_static(
      "_init_parse_labels",
      [](py::object labels,
         TensorProduct::Ptr const& codomain,
         TensorProduct::Ptr const& domain,
         bool is_endomorphism) {
          return parse_tensor_init_labels(labels, codomain, domain, is_endomorphism);
      },
      nb::arg("labels"),
      nb::arg("codomain"),
      nb::arg("domain"),
      nb::arg("is_endomorphism") = false,
      DOC(lasyten, Tensor, _init_parse_labels));

    cls.def("test_sanity", &Tensor::test_sanity, DOC(lasyten, Tensor, test_sanity));

    cls.def_prop_ro("ascii_diagram", &Tensor::ascii_diagram, DOC(lasyten, Tensor, ascii_diagram));

    cls.def("as_dtype", &Tensor::as_dtype, nb::arg("dtype"), DOC(lasyten, Tensor, as_dtype));

    cls.def(
      "as_SymmetricTensor",
      [](Tensor& self, bool guarantee_copy, py::object warning) {
          std::optional<std::string> w;
          if (!warning.is_none()) {
              w = py::cast<std::string>(warning);
          }
          return self.as_SymmetricTensor(guarantee_copy, w);
      },
      nb::arg("guarantee_copy") = false,
      nb::arg("warning").none() = nb::none(),
      DOC(lasyten, Tensor, as_SymmetricTensor));

    cls.def(
      "copy",
      [](Tensor& self, bool deep, py::object device, py::object dtype) {
          std::optional<std::string> dev;
          std::optional<Dtype> dt;
          if (!device.is_none()) {
              dev = py::cast<std::string>(device);
          }
          if (!dtype.is_none()) {
              dt = py::cast<Dtype>(dtype);
          }
          return self.copy(deep, dev, dt);
      },
      nb::arg("deep") = true,
      nb::arg("device").none() = nb::none(),
      nb::arg("dtype").none() = nb::none(),
      DOC(lasyten, Tensor, copy));

    cls.def(
      "to_backend",
      [](Tensor& self, TensorBackend::Ptr backend, py::object dtype, py::object device) {
          std::optional<Dtype> dt;
          std::optional<std::string> dev;
          if (!dtype.is_none()) {
              dt = py::cast<Dtype>(dtype);
          }
          if (!device.is_none()) {
              dev = py::cast<std::string>(device);
          }
          return self.to_backend(std::move(backend), dt, dev);
      },
      nb::arg("backend"),
      nb::arg("dtype").none() = nb::none(),
      nb::arg("device").none() = nb::none(),
      DOC(lasyten, Tensor, to_backend));

    cls.def(
      "to_dense_block",
      [](Tensor& self, py::object leg_order, py::object dtype, bool understood_braiding) {
          std::optional<Dtype> dt;
          if (!dtype.is_none()) {
              dt = py::cast<Dtype>(dtype);
          }
          return self.to_dense_block(optional_leg_order(leg_order), dt, understood_braiding);
      },
      nb::arg("leg_order").none() = nb::none(),
      nb::arg("dtype").none() = nb::none(),
      nb::arg("understood_braiding") = false,
      DOC(lasyten, Tensor, to_dense_block));

    cls.def_prop_ro(
      "codomain_labels", &Tensor::codomain_labels, DOC(lasyten, Tensor, codomain_labels));
    cls.def_prop_ro(
      "dagger",
      [](py::object self) {
          return py::module_::import("lasyten.tensors._tensors").attr("dagger")(self);
      },
      DOC(lasyten, Tensor, dagger));
    cls.def_prop_ro("domain_labels", &Tensor::domain_labels, DOC(lasyten, Tensor, domain_labels));
    cls.def_prop_ro("has_pipes", &Tensor::has_pipes, DOC(lasyten, Tensor, has_pipes));
    cls.def_prop_ro(
      "hc",
      [](py::object self) {
          return py::module_::import("lasyten.tensors._tensors").attr("dagger")(self);
      },
      DOC(lasyten, Tensor, dagger));
    cls.def_prop_ro("legs", &Tensor::legs, DOC(lasyten, Tensor, legs));

    cls.def("move_to_device",
            &Tensor::move_to_device,
            nb::arg("device"),
            DOC(lasyten, Tensor, move_to_device));

    cls.def_prop_ro(
      "num_codomain_legs", &Tensor::num_codomain_legs, DOC(lasyten, Tensor, num_codomain_legs));
    cls.def_prop_ro(
      "num_domain_legs", &Tensor::num_domain_legs, DOC(lasyten, Tensor, num_domain_legs));
    cls.def_prop_ro("num_codomain_flat_legs",
                    &Tensor::num_codomain_flat_legs,
                    DOC(lasyten, Tensor, num_codomain_flat_legs));
    cls.def_prop_ro("num_domain_flat_legs",
                    &Tensor::num_domain_flat_legs,
                    DOC(lasyten, Tensor, num_domain_flat_legs));
    cls.def_prop_ro("num_flat_legs", &Tensor::num_flat_legs, DOC(lasyten, Tensor, num_flat_legs));
    cls.def_prop_ro(
      "num_parameters", &Tensor::num_parameters, DOC(lasyten, Tensor, num_parameters));
    cls.def_prop_ro("size", &Tensor::size, DOC(lasyten, Tensor, size));
    cls.def_prop_ro(
      "T",
      [](py::object self) {
          return py::module_::import("lasyten.tensors._tensors").attr("transpose")(self);
      },
      DOC(lasyten, Tensor, T));

    cls.def(
      "__getitem__",
      [](Tensor& self, py::object idx) {
          if (!self.symmetry->can_be_dropped()) {
              throw SymmetryError(std::format(
                "Can not access elements for tensor with symmetry {}", self.symmetry->repr()));
          }
          auto it = to_iterable(idx);
          std::vector<int64> parsed;
          for (auto item : it) {
              parsed.push_back(py::cast<int64>(item));
          }
          if (static_cast<int64>(parsed.size()) != self.num_legs) {
              throw py::index_error(std::format(
                "Expected {} indices (one per leg). Got {}", self.num_legs, parsed.size()));
          }
          for (std::size_t i = 0; i < parsed.size(); ++i) {
              parsed[i] = to_valid_idx(parsed[i], static_cast<int64>(self.shape[i]));
          }
          return self._get_item(parsed);
      },
      nb::arg("idx"));

    cls
      .def("__setitem__",
           [](Tensor&, py::object, py::object) {
               throw py::type_error("Tensors do not support item assignment.");
           })
      .def("__eq__",
           [](Tensor& self, py::object) -> bool {
               throw py::type_error(std::format(
                 "{} does not support == comparison. Use lasyten.almost_equal instead.",
                 self.class_name()));
           })
      .def("__complex__",
           [](Tensor&) {
               throw py::type_error(
                 "complex() of a tensor is not defined. Use lasyten.item() instead.");
           })
      .def("__float__", [](Tensor&) {
          throw py::type_error("float() of a tensor is not defined. Use lasyten.item() instead.");
      });

    // Arithmetic / composition: defer to Python free functions until those are converted.
    cls
      .def("__add__",
           [](py::object self, py::object other) {
               return py::module_::import("lasyten.tensors._tensors")
                 .attr("linear_combination")(1.0, self, 1.0, other);
           })
      .def("__sub__",
           [](py::object self, py::object other) {
               return py::module_::import("lasyten.tensors._tensors")
                 .attr("linear_combination")(1.0, self, -1.0, other);
           })
      .def("__mul__",
           [](py::object self, py::object other) {
               return py::module_::import("lasyten.tensors._tensors")
                 .attr("scalar_multiply")(other, self);
           })
      .def("__rmul__",
           [](py::object self, py::object other) {
               return py::module_::import("lasyten.tensors._tensors")
                 .attr("scalar_multiply")(other, self);
           })
      .def("__truediv__",
           [](py::object self, py::object other) {
               py::object inv;
               try {
                   inv = py::float_(1.0) / other;
               } catch (py::error_already_set&) {
                   throw py::value_error("Tensor can only be divided by invertible scalars.");
               }
               return py::module_::import("lasyten.tensors._tensors")
                 .attr("scalar_multiply")(inv, self);
           })
      .def("__neg__",
           [](py::object self) {
               return py::module_::import("lasyten.tensors._tensors")
                 .attr("scalar_multiply")(-1, self);
           })
      .def("__pos__", [](Tensor::Ptr self) { return self; })
      .def("__matmul__", [](py::object self, py::object other) {
          return py::module_::import("lasyten.tensors._tensors").attr("compose")(self, other);
      });

    cls.def("__repr__", &Tensor::__repr__).def("__str__", &Tensor::__str__);

    cls
      .def(
        "_as_codomain_leg",
        [](Tensor const& self, py::object idx) { return self._as_codomain_leg(as_leg_ref(idx)); },
        nb::arg("idx"),
        DOC(lasyten, Tensor, _as_codomain_leg))
      .def(
        "_as_domain_leg",
        [](Tensor const& self, py::object idx) { return self._as_domain_leg(as_leg_ref(idx)); },
        nb::arg("idx"),
        DOC(lasyten, Tensor, _as_codomain_leg))
      .def("dbg", &Tensor::dbg)
      .def("_get_item", &Tensor::_get_item, nb::arg("idx"), DOC(lasyten, Tensor, _as_codomain_leg))
      .def(
        "_parse_leg_idx",
        [](Tensor const& self, py::object which_leg) {
            return self._parse_leg_idx(as_leg_ref(which_leg));
        },
        nb::arg("which_leg"),
        DOC(lasyten, Tensor, _as_codomain_leg))
      .def("_repr_header_lines",
           &Tensor::_repr_header_lines,
           nb::arg("indent"),
           nb::arg("use_symm_str") = false);

    cls
      .def(
        "get_leg",
        [](Tensor const& self, py::object which_leg) -> py::object {
            if (!(py::isinstance<py::int_>(which_leg) || py::isinstance<py::str>(which_leg))) {
                std::vector<std::variant<int64, std::string>> refs;
                for (auto item : which_leg) {
                    refs.push_back(as_leg_ref(item));
                }
                py::list out;
                for (auto const& leg : self.get_leg(refs)) {
                    out.append(leg);
                }
                return out;
            }
            return py::cast(self.get_leg(as_leg_ref(which_leg)));
        },
        nb::arg("which_leg"),
        DOC(lasyten, Tensor, get_leg))
      .def(
        "get_leg_co_domain",
        [](Tensor const& self, py::object which_leg) -> py::object {
            if (!(py::isinstance<py::int_>(which_leg) || py::isinstance<py::str>(which_leg))) {
                std::vector<std::variant<int64, std::string>> refs;
                for (auto item : which_leg) {
                    refs.push_back(as_leg_ref(item));
                }
                py::list out;
                for (auto const& leg : self.get_leg_co_domain(refs)) {
                    out.append(leg);
                }
                return out;
            }
            return py::cast(self.get_leg_co_domain(as_leg_ref(which_leg)));
        },
        nb::arg("which_leg"),
        DOC(lasyten, Tensor, get_leg_co_domain));

    // Always an owning NumPy copy (dual path: use BlockBackend.to_ndarray for DLPack /
    // framework-tagged views). Torch block_to_ndarray defaults to a torch.Tensor and breaks
    // callers that expect np.ndarray.
    cls.def(
      "to_numpy",
      [](Tensor& self, py::object leg_order, py::object numpy_dtype, bool understood_braiding) {
          return self.to_numpy(optional_leg_order(leg_order), numpy_dtype, understood_braiding);
      },
      nb::arg("leg_order").none() = nb::none(),
      nb::arg("numpy_dtype").none() = nb::none(),
      nb::arg("understood_braiding") = false,
      DOC(lasyten, Tensor, to_numpy));
}

} // namespace lasyten
