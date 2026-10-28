#include <lasyten/tensors/tensor.h>

#include "py_trampolines.hpp"

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"

#include "docstrings/tensors/tensor.h"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

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
            out.emplace_back(item.cast<std::string>());
        } else {
            out.emplace_back(item.cast<int64>());
        }
    }
    return out;
}

std::variant<int64, std::string>
as_leg_ref(py::handle obj)
{
    if (py::isinstance<py::str>(obj)) {
        return obj.cast<std::string>();
    }
    return obj.cast<int64>();
}

std::optional<py::class_<Tensor, VectorLike, PyTensor, py::smart_holder>> g_tensor;

} // namespace

void
declare_tensors_tensor(py::module_& m)
{
    g_tensor = py::class_<Tensor, VectorLike, PyTensor, py::smart_holder>(m, "Tensor");
    (*g_tensor).doc() = DOC(lasyten, Tensor);
}

void
bind_tensors_tensor(py::module_& m)
{
    auto& cls = *g_tensor;

    cls.def(py::init([](py::object codomain,
                        py::object domain,
                        TensorBackend::Ptr backend,
                        py::object labels,
                        Dtype dtype,
                        std::string device) {
                auto [c, d, b, s] = parse_tensor_init_args(codomain, domain, std::move(backend));
                auto labs = parse_tensor_init_labels(labels, c, d);
                return std::make_shared<PyTensor>(std::move(c),
                                                  std::move(d),
                                                  std::move(b),
                                                  std::move(s),
                                                  std::move(labs),
                                                  dtype,
                                                  device);
            }),
            py::arg("codomain"),
            py::arg("domain"),
            py::arg("backend"),
            py::arg("labels"),
            py::arg("dtype"),
            py::arg("device"));

    cls.def_readwrite("codomain", &Tensor::codomain)
      .def_readwrite("domain", &Tensor::domain)
      .def_readwrite("backend", &Tensor::backend)
      .def_readwrite("symmetry", &Tensor::symmetry)
      .def_readwrite("dtype", &Tensor::dtype)
      .def_readwrite("device", &Tensor::device)
      .def_readwrite("num_legs", &Tensor::num_legs)
      .def_property("labels",
                    py::cpp_function([](Tensor& self) -> LegLabels& { return self.labels; },
                                     py::return_value_policy::reference_internal),
                    py::cpp_function([](Tensor& self, py::object value) {
                        if (py::isinstance<LegLabels>(value)) {
                            self.labels.set(value.cast<LegLabels const&>().as_list());
                        } else {
                            self.labels.set(value.cast<LegLabelList>());
                        }
                    }))
      .def_property(
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
            self.shape = shape_obj.cast<std::vector<float64>>();
        });

    cls.def_static(
      "_init_parse_args",
      [](py::object codomain, py::object domain, TensorBackend::Ptr backend) {
          return parse_tensor_init_args(codomain, domain, std::move(backend));
      },
      py::arg("codomain"),
      py::arg("domain"),
      py::arg("backend"),
      DOC(lasyten, Tensor, _init_parse_args));

    cls.def_static(
      "_init_parse_labels",
      [](py::object labels,
         TensorProduct::Ptr const& codomain,
         TensorProduct::Ptr const& domain,
         bool is_endomorphism) {
          return parse_tensor_init_labels(labels, codomain, domain, is_endomorphism);
      },
      py::arg("labels"),
      py::arg("codomain"),
      py::arg("domain"),
      py::arg("is_endomorphism") = false,
      DOC(lasyten, Tensor, _init_parse_labels));

    cls.def("test_sanity", &Tensor::test_sanity, DOC(lasyten, Tensor, test_sanity));

    cls.def_property_readonly(
      "ascii_diagram", &Tensor::ascii_diagram, DOC(lasyten, Tensor, ascii_diagram));

    cls.def("as_dtype", &Tensor::as_dtype, py::arg("dtype"), DOC(lasyten, Tensor, as_dtype));

    cls.def(
      "as_SymmetricTensor",
      [](Tensor& self, bool guarantee_copy, py::object warning) {
          std::optional<std::string> w;
          if (!warning.is_none()) {
              w = warning.cast<std::string>();
          }
          return self.as_SymmetricTensor(guarantee_copy, w);
      },
      py::arg("guarantee_copy") = false,
      py::arg("warning") = py::none(),
      DOC(lasyten, Tensor, as_SymmetricTensor));

    cls.def(
      "copy",
      [](Tensor& self, bool deep, py::object device, py::object dtype) {
          std::optional<std::string> dev;
          std::optional<Dtype> dt;
          if (!device.is_none()) {
              dev = device.cast<std::string>();
          }
          if (!dtype.is_none()) {
              dt = dtype.cast<Dtype>();
          }
          return self.copy(deep, dev, dt);
      },
      py::arg("deep") = true,
      py::arg("device") = py::none(),
      py::arg("dtype") = py::none(),
      DOC(lasyten, Tensor, copy));

    cls.def(
      "to_backend",
      [](Tensor& self, TensorBackend::Ptr backend, py::object dtype, py::object device) {
          std::optional<Dtype> dt;
          std::optional<std::string> dev;
          if (!dtype.is_none()) {
              dt = dtype.cast<Dtype>();
          }
          if (!device.is_none()) {
              dev = device.cast<std::string>();
          }
          return self.to_backend(std::move(backend), dt, dev);
      },
      py::arg("backend"),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      DOC(lasyten, Tensor, to_backend));

    cls.def(
      "to_dense_block",
      [](Tensor& self, py::object leg_order, py::object dtype, bool understood_braiding) {
          std::optional<Dtype> dt;
          if (!dtype.is_none()) {
              dt = dtype.cast<Dtype>();
          }
          return self.to_dense_block(optional_leg_order(leg_order), dt, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false,
      DOC(lasyten, Tensor, to_dense_block));

    cls.def_property_readonly(
      "codomain_labels", &Tensor::codomain_labels, DOC(lasyten, Tensor, codomain_labels));
    cls.def_property_readonly(
      "dagger",
      [](py::object self) {
          return py::module_::import("lasyten.tensors._tensors").attr("dagger")(self);
      },
      DOC(lasyten, Tensor, dagger));
    cls.def_property_readonly(
      "domain_labels", &Tensor::domain_labels, DOC(lasyten, Tensor, domain_labels));
    cls.def_property_readonly("has_pipes", &Tensor::has_pipes, DOC(lasyten, Tensor, has_pipes));
    cls.def_property_readonly(
      "hc",
      [](py::object self) {
          return py::module_::import("lasyten.tensors._tensors").attr("dagger")(self);
      },
      DOC(lasyten, Tensor, dagger));
    cls.def_property_readonly("legs", &Tensor::legs, DOC(lasyten, Tensor, legs));

    cls.def("move_to_device",
            &Tensor::move_to_device,
            py::arg("device"),
            DOC(lasyten, Tensor, move_to_device));

    cls.def_property_readonly(
      "num_codomain_legs", &Tensor::num_codomain_legs, DOC(lasyten, Tensor, num_codomain_legs));
    cls.def_property_readonly(
      "num_domain_legs", &Tensor::num_domain_legs, DOC(lasyten, Tensor, num_domain_legs));
    cls.def_property_readonly("num_codomain_flat_legs",
                              &Tensor::num_codomain_flat_legs,
                              DOC(lasyten, Tensor, num_codomain_flat_legs));
    cls.def_property_readonly("num_domain_flat_legs",
                              &Tensor::num_domain_flat_legs,
                              DOC(lasyten, Tensor, num_domain_flat_legs));
    cls.def_property_readonly(
      "num_flat_legs", &Tensor::num_flat_legs, DOC(lasyten, Tensor, num_flat_legs));
    cls.def_property_readonly(
      "num_parameters", &Tensor::num_parameters, DOC(lasyten, Tensor, num_parameters));
    cls.def_property_readonly("size", &Tensor::size, DOC(lasyten, Tensor, size));
    cls.def_property_readonly(
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
              parsed.push_back(item.cast<int64>());
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
      py::arg("idx"));

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
        py::arg("idx"),
        DOC(lasyten, Tensor, _as_codomain_leg))
      .def(
        "_as_domain_leg",
        [](Tensor const& self, py::object idx) { return self._as_domain_leg(as_leg_ref(idx)); },
        py::arg("idx"),
        DOC(lasyten, Tensor, _as_codomain_leg))
      .def("dbg", &Tensor::dbg)
      .def("_get_item", &Tensor::_get_item, py::arg("idx"), DOC(lasyten, Tensor, _as_codomain_leg))
      .def(
        "_parse_leg_idx",
        [](Tensor const& self, py::object which_leg) {
            return self._parse_leg_idx(as_leg_ref(which_leg));
        },
        py::arg("which_leg"),
        DOC(lasyten, Tensor, _as_codomain_leg))
      .def("_repr_header_lines",
           &Tensor::_repr_header_lines,
           py::arg("indent"),
           py::arg("use_symm_str") = false);

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
        py::arg("which_leg"),
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
        py::arg("which_leg"),
        DOC(lasyten, Tensor, get_leg_co_domain));

    cls.def(
      "to_numpy",
      [](Tensor& self, py::object leg_order, py::object numpy_dtype, bool understood_braiding) {
          return self.to_numpy(optional_leg_order(leg_order), numpy_dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("numpy_dtype") = py::none(),
      py::arg("understood_braiding") = false,
      DOC(lasyten, Tensor, to_numpy));
}

} // namespace lasyten
