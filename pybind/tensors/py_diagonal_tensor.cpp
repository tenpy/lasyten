#include <lasyten/backends/no_symmetry.h>
#include <lasyten/tensors/diagonal_tensor.h>

#include "py_callbacks.hpp"
#include "py_factory_parse.hpp"
#include "py_trampolines.hpp"

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"

#include "docstrings/tensors/diagonal_tensor.h"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "tools/hdf5_bind.h"
#include <format>
#include <optional>
#include <stdexcept>
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

py::object
tensors_mod()
{
    return py::module_::import("lasyten.tensors._tensors");
}

bool
is_py_tensor(py::object obj)
{
    return py::isinstance(obj, tensors_mod().attr("Tensor")) || py::isinstance<Tensor>(obj);
}

bool
is_py_diagonal(py::object obj)
{
    return py::isinstance(obj, tensors_mod().attr("DiagonalTensor")) ||
           py::isinstance<DiagonalTensor>(obj);
}

bool
is_py_number_or_scalar(py::object obj, std::shared_ptr<BlockBackend> const& bb)
{
    return py::isinstance(obj, py::module_::import("numbers").attr("Number")) ||
           py::isinstance<BlockBackend::Scalar>(obj) ||
           py::isinstance(obj, py::type::of(py::cast(bb)).attr("Scalar"));
}

BlockBackend::Scalar
scalar_from_py(std::shared_ptr<BlockBackend> const& bb, py::object obj)
{
    // Match Python: as_scalar(other) without forcing self.dtype (complex * float tensor).
    return py::cast(bb).attr("as_scalar")(obj).cast<BlockBackend::Scalar>();
}

BlockBinaryFn
block_op_from_name(char const* op)
{
    std::string s = op;
    if (s == "add") {
        return [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
            return (*a) + (*b);
        };
    }
    if (s == "sub") {
        return [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
            return (*a) - (*b);
        };
    }
    if (s == "mul") {
        return [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
            return (*a) * (*b);
        };
    }
    if (s == "truediv") {
        return [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
            return (*a) / (*b);
        };
    }
    if (s == "pow") {
        return [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
            return a->pow(*b);
        };
    }
    if (s == "lt") {
        return [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
            return (*a) < (*b);
        };
    }
    if (s == "le") {
        return [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
            return (*a) <= (*b);
        };
    }
    if (s == "gt") {
        return [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
            return (*a) > (*b);
        };
    }
    if (s == "ge") {
        return [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
            return (*a) >= (*b);
        };
    }
    throw std::logic_error(std::format("unknown block operator '{}'", s));
}

py::object
py_diagonal_binary_operand(DiagonalTensor& self,
                           py::object other,
                           BlockBinaryFn func,
                           std::string const& operand,
                           bool return_NotImplemented,
                           bool right)
{
    auto bb = self.backend->block_backend;
    if (is_py_number_or_scalar(other, bb)) {
        return py::cast(
          self._binary_operand(scalar_from_py(bb, other), std::move(func), operand, right));
    }
    if (is_py_diagonal(other)) {
        return py::cast(
          self._binary_operand(other.cast<DiagonalTensorCPtr>(), std::move(func), operand, right));
    }
    if (return_NotImplemented && !is_py_tensor(other)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    if (right) {
        throw std::invalid_argument(std::format("Invalid types for operand \"{}\": {} and {}",
                                                operand,
                                                py::str(py::type::of(other)).cast<std::string>(),
                                                self.class_name()));
    }
    throw std::invalid_argument(std::format("Invalid types for operand \"{}\": {} and {}",
                                            operand,
                                            self.class_name(),
                                            py::str(py::type::of(other)).cast<std::string>()));
}

std::optional<py::class_<DiagonalTensor, SymmetricTensor, PyDiagonalTensor, py::smart_holder>>
  g_diagonal_tensor;

} // namespace

void
declare_tensors_diagonal_tensor(py::module_& m)
{
    g_diagonal_tensor =
      py::class_<DiagonalTensor, SymmetricTensor, PyDiagonalTensor, py::smart_holder>(
        m, "DiagonalTensor");
    (*g_diagonal_tensor).doc() = DOC(lasyten, DiagonalTensor);
}

void
bind_tensors_diagonal_tensor(py::module_& m)
{
    auto& cls = *g_diagonal_tensor;

    cls.def(py::init([](TensorBackend::DataPtr data,
                        py::object leg,
                        TensorBackend::Ptr backend,
                        py::object labels) {
                auto sp = py_as_space_leg(leg);
                auto init = py_parse_diag(leg, std::move(backend), labels);
                return std::make_shared<DiagonalTensor>(
                  std::move(data), sp, init.backend, init.symmetry, init.labels);
            }),
            py::arg("data"),
            py::arg("leg"),
            py::arg("backend") = nullptr,
            py::arg("labels") = py::none());

    cls.def_property(
      "data",
      [](DiagonalTensor& self) -> py::object {
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              return py::cast(NoSymmetryBackend::unwrap(self.data));
          }
          return py::cast(self.data);
      },
      [](DiagonalTensor& self, py::object obj) {
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              self.data = NoSymmetryBackend::wrap(obj.cast<BlockBackend::BlockPtr>());
          } else {
              self.data = obj.cast<TensorBackend::DataPtr>();
          }
      });

    cls.def_property_readonly("leg", &DiagonalTensor::leg, DOC(lasyten, DiagonalTensor, leg));

    cls.def(
      "test_sanity", &DiagonalTensor::test_sanity, DOC(lasyten, DiagonalTensor, test_sanity));
    cls.def("verify_dtype", &DiagonalTensor::verify_dtype);

    cls.def_static(
      "from_block_func",
      [](py::function func,
         py::object leg,
         TensorBackend::Ptr backend,
         py::object labels,
         py::object func_kwargs,
         std::optional<std::string> shape_kw,
         std::optional<Dtype> dtype,
         std::optional<std::string> device) {
          auto init = py_parse_diag(leg, std::move(backend), labels);
          auto wrapped = block_factory_from_python(
            func, func_kwargs, shape_kw, init.backend->block_backend, dtype, device);
          return DiagonalTensor::from_block_func(
            std::move(wrapped), py_as_space_leg(leg), init.backend, init.labels, dtype, device);
      },
      py::arg("func"),
      py::arg("leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("func_kwargs") = py::none(),
      py::arg("shape_kw") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      DOC(lasyten, DiagonalTensor, from_block_func));

    cls.def_static(
      "from_dense_block",
      [](py::object block,
         py::object leg,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<Dtype> dtype,
         float64 tol,
         std::optional<std::string> device,
         bool understood_braiding) {
          auto init = py_parse_diag(leg, std::move(backend), labels);
          auto block_ptr = init.backend->block_backend->as_block(block, dtype, device);
          return DiagonalTensor::from_dense_block(block_ptr,
                                                  py_as_space_leg(leg),
                                                  init.backend,
                                                  init.labels,
                                                  dtype,
                                                  tol,
                                                  device,
                                                  understood_braiding);
      },
      py::arg("block"),
      py::arg("leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("tol") = 1e-6,
      py::arg("device") = py::none(),
      py::arg("understood_braiding") = false,
      DOC(lasyten, DiagonalTensor, from_dense_block));

    cls.def_static(
      "from_diag_block",
      [](py::object diag,
         py::object leg,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<Dtype> dtype,
         std::optional<std::string> device,
         float64 tol) {
          auto init = py_parse_diag(leg, std::move(backend), labels);
          auto diag_ptr = init.backend->block_backend->as_block(diag, dtype, device);
          return DiagonalTensor::from_diag_block(
            diag_ptr, py_as_space_leg(leg), init.backend, init.labels, dtype, device, tol);
      },
      py::arg("diag"),
      py::arg("leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      py::arg("tol") = 1e-6,
      DOC(lasyten, DiagonalTensor, from_diag_block));

    cls.def_static(
      "from_eye",
      [](py::object leg,
         TensorBackend::Ptr backend,
         py::object labels,
         Dtype dtype,
         std::optional<std::string> device) {
          auto init = py_parse_diag(leg, std::move(backend), labels);
          return DiagonalTensor::from_eye(
            py_as_space_leg(leg), init.backend, init.labels, dtype, device);
      },
      py::arg("leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Float64,
      py::arg("device") = py::none(),
      DOC(lasyten, DiagonalTensor, from_eye));

    cls.def_static(
      "from_random_normal",
      [](py::object leg,
         py::object mean,
         float64 sigma,
         TensorBackend::Ptr backend,
         py::object labels,
         Dtype dtype,
         std::optional<std::string> device) {
          auto mean_t = py_optional_tensor(mean);
          Space::Ptr sp;
          std::optional<LegLabelList> labs;
          if (!leg.is_none()) {
              auto init = py_parse_diag(leg, std::move(backend), labels);
              sp = py_as_space_leg(leg);
              backend = init.backend;
              labs = init.labels;
          } else if (mean_t) {
              if (!labels.is_none()) {
                  labs = parse_tensor_init_labels(labels, mean_t->codomain, mean_t->domain);
              }
          }
          return DiagonalTensor::from_random_normal(
            sp, mean_t, sigma, std::move(backend), std::move(labs), dtype, device);
      },
      py::arg("leg"),
      py::arg("mean") = py::none(),
      py::arg("sigma") = 1.0,
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      DOC(lasyten, DiagonalTensor, from_random_normal));

    cls.def_static(
      "from_random_uniform",
      [](py::object leg,
         TensorBackend::Ptr backend,
         py::object labels,
         Dtype dtype,
         std::optional<std::string> device) {
          auto init = py_parse_diag(leg, std::move(backend), labels);
          return DiagonalTensor::from_random_uniform(
            py_as_space_leg(leg), init.backend, init.labels, dtype, device);
      },
      py::arg("leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      DOC(lasyten, DiagonalTensor, from_random_uniform));

    cls.def_static(
      "from_sector_block_func",
      [](py::function func,
         py::object leg,
         TensorBackend::Ptr backend,
         py::object labels,
         py::object func_kwargs,
         std::optional<Dtype> dtype,
         std::optional<std::string> device) {
          auto init = py_parse_diag(leg, std::move(backend), labels);
          auto wrapped = sector_block_factory_from_python(
            func, func_kwargs, init.backend->block_backend, dtype, device);
          return DiagonalTensor::from_sector_block_func(
            std::move(wrapped), py_as_space_leg(leg), init.backend, init.labels, dtype, device);
      },
      py::arg("func"),
      py::arg("leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("func_kwargs") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      DOC(lasyten, DiagonalTensor, from_sector_block_func));

    cls.def_static(
      "from_tensor",
      [](py::object tens, std::optional<float64> tol) {
          return DiagonalTensor::from_tensor(tens.cast<SymmetricTensorCPtr>(), tol);
      },
      py::arg("tens"),
      py::arg("tol") = 1e-12,
      DOC(lasyten, DiagonalTensor, from_tensor));

    cls.def_static(
      "from_zero",
      [](py::object leg,
         TensorBackend::Ptr backend,
         py::object labels,
         Dtype dtype,
         std::optional<std::string> device) {
          auto init = py_parse_diag(leg, std::move(backend), labels);
          return DiagonalTensor::from_zero(
            py_as_space_leg(leg), init.backend, init.labels, dtype, device);
      },
      py::arg("leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      DOC(lasyten, DiagonalTensor, from_zero));

    cls.def_static("from_hdf5",
                   lasyten::hdf5::wrap_from_hdf5<DiagonalTensor>(),
                   py::arg("hdf5_loader"),
                   py::arg("h5gr"),
                   py::arg("subpath"),
                   DOC(lasyten, DiagonalTensor, from_hdf5));

    cls.def("as_dtype",
            &DiagonalTensor::as_dtype,
            py::arg("dtype"),
            DOC(lasyten, DiagonalTensor, as_dtype));
    cls.def("as_SymmetricTensor",
            &DiagonalTensor::as_SymmetricTensor,
            py::arg("guarantee_copy") = false,
            py::arg("warning") = py::none(),
            DOC(lasyten, DiagonalTensor, as_SymmetricTensor));
    cls.def("as_DiagonalTensor",
            &DiagonalTensor::as_DiagonalTensor,
            py::arg("guarantee_copy") = false,
            py::arg("warning") = py::none());
    cls.def("copy",
            &DiagonalTensor::copy,
            py::arg("deep") = true,
            py::arg("device") = py::none(),
            py::arg("dtype") = py::none(),
            DOC(lasyten, DiagonalTensor, copy));
    cls.def("diagonal",
            &DiagonalTensor::diagonal,
            py::arg("check_offdiagonal") = false,
            DOC(lasyten, DiagonalTensor, diagonal));
    cls.def(
      "diagonal_as_block", &DiagonalTensor::diagonal_as_block, py::arg("dtype") = py::none());
    cls.def("diagonal_as_numpy",
            &DiagonalTensor::diagonal_as_numpy,
            py::arg("numpy_dtype") = py::none());
    cls.def(
      "elementwise_almost_equal",
      [](DiagonalTensor& self, py::object other, float64 rtol, float64 atol) {
          return self.elementwise_almost_equal(
            other.attr("as_DiagonalTensor")().cast<DiagonalTensorCPtr>(), rtol, atol);
      },
      py::arg("other"),
      py::arg("rtol") = 1e-5,
      py::arg("atol") = 1e-8);
    cls.def(
      "_elementwise_unary",
      [](DiagonalTensor& self, py::function func, py::object func_kwargs, bool maps_zero_to_zero) {
          return self._elementwise_unary(block_unary_from_python(func, func_kwargs),
                                         maps_zero_to_zero);
      },
      py::arg("func"),
      py::arg("func_kwargs") = py::none(),
      py::arg("maps_zero_to_zero") = false,
      DOC(lasyten, DiagonalTensor, _elementwise_unary));
    cls.def(
      "_elementwise_binary",
      [](DiagonalTensor& self,
         py::object other,
         py::function func,
         py::object func_kwargs,
         bool partial_zero_is_zero) {
          if (!is_py_diagonal(other)) {
              throw std::invalid_argument("Expected a DiagonalTensor");
          }
          return self._elementwise_binary(
            other.attr("as_DiagonalTensor")().cast<DiagonalTensorCPtr>(),
            block_binary_from_python(func, func_kwargs),
            partial_zero_is_zero);
      },
      py::arg("other"),
      py::arg("func"),
      py::arg("func_kwargs") = py::none(),
      py::arg("partial_zero_is_zero") = false,
      DOC(lasyten, DiagonalTensor, _elementwise_binary));
    cls.def(
      "_binary_operand",
      [](DiagonalTensor& self,
         py::object other,
         py::function func,
         std::string operand,
         bool return_NotImplemented,
         bool right) {
          return py_diagonal_binary_operand(
            self, other, block_binary_from_python(func), operand, return_NotImplemented, right);
      },
      py::arg("other"),
      py::arg("func"),
      py::arg("operand"),
      py::arg("return_NotImplemented") = false,
      py::arg("right") = false,
      DOC(lasyten, DiagonalTensor, _binary_operand));
    cls.def("_get_item",
            &DiagonalTensor::_get_item,
            py::arg("idx"),
            DOC(lasyten, DiagonalTensor, _get_item));
    cls.def("all", &DiagonalTensor::all, DOC(lasyten, DiagonalTensor, all));
    cls.def("any", &DiagonalTensor::any, DOC(lasyten, DiagonalTensor, any));
    cls.def("max", &DiagonalTensor::max);
    cls.def("min", &DiagonalTensor::min);
    cls.def("sector_argmin",
            &DiagonalTensor::sector_argmin,
            py::arg("s") = py::none(),
            DOC(lasyten, DiagonalTensor, sector_argmin));
    cls.def("argmin",
            &DiagonalTensor::argmin,
            py::arg("s") = py::none(),
            DOC(lasyten, DiagonalTensor, argmin));
    cls.def("move_to_device",
            &DiagonalTensor::move_to_device,
            py::arg("device"),
            DOC(lasyten, DiagonalTensor, move_to_device));
    cls.def("to_backend",
            &DiagonalTensor::to_backend,
            py::arg("backend"),
            py::arg("dtype") = py::none(),
            py::arg("device") = py::none(),
            DOC(lasyten, DiagonalTensor, to_backend));
    cls.def(
      "to_dense_block",
      [](DiagonalTensor& self,
         py::object leg_order,
         std::optional<Dtype> dtype,
         bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false,
      DOC(lasyten, DiagonalTensor, to_dense_block));
    cls.def("save_hdf5",
            lasyten::hdf5::wrap_save_hdf5_const<DiagonalTensor>(),
            py::arg("hdf5_saver"),
            py::arg("h5gr"),
            py::arg("subpath"),
            DOC(lasyten, DiagonalTensor, save_hdf5));

    // Elementwise dunders
    cls.def("__abs__", &DiagonalTensor::abs);
    cls.def("__bool__", [](DiagonalTensor& self) {
        auto tensors_mod = py::module_::import("lasyten.tensors._tensors");
        if (self.dtype == Dtype::Bool && tensors_mod.attr("is_scalar")(self).cast<bool>()) {
            return tensors_mod.attr("item")(self).cast<bool>();
        }
        throw std::invalid_argument(
          "The truth value of a non-scalar DiagonalTensor is ambiguous. Use a.any() or a.all()");
    });

    auto bind_binop = [&](char const* dunder, char const* op, bool right = false) {
        std::string dunder_s = dunder;
        cls.def(
          dunder,
          [dunder_s, op, right](DiagonalTensor& self, py::object other) -> py::object {
              if (is_py_tensor(other)) {
                  auto tm = tensors_mod();
                  if (dunder_s == "__add__" || dunder_s == "__radd__") {
                      return tm.attr("linear_combination")(1.0, py::cast(self), 1.0, other);
                  }
                  if (dunder_s == "__sub__") {
                      return tm.attr("linear_combination")(1.0, py::cast(self), -1.0, other);
                  }
                  if (dunder_s == "__rsub__") {
                      return tm.attr("linear_combination")(1.0, other, -1.0, py::cast(self));
                  }
              }
              return py_diagonal_binary_operand(
                self, other, block_op_from_name(op), op, /*return_NotImplemented=*/true, right);
          },
          py::arg("other"));
    };
    bind_binop("__add__", "add");
    bind_binop("__radd__", "add", true);
    bind_binop("__sub__", "sub");
    bind_binop("__rsub__", "sub", true);
    bind_binop("__mul__", "mul");
    bind_binop("__rmul__", "mul", true);
    bind_binop("__truediv__", "truediv");
    bind_binop("__rtruediv__", "truediv", true);
    bind_binop("__pow__", "pow");
    bind_binop("__rpow__", "pow", true);
    bind_binop("__lt__", "lt");
    bind_binop("__le__", "le");
    bind_binop("__gt__", "gt");
    bind_binop("__ge__", "ge");

    // --- Identity ---
    py::class_<Identity, DiagonalTensor, py::smart_holder> id_cls(m, "Identity");
    id_cls.doc() = DOC(lasyten, Identity);

    id_cls.def(py::init([](py::object leg,
                           TensorBackend::Ptr backend,
                           std::optional<Dtype> dtype,
                           std::optional<std::string> device,
                           py::object labels) {
                   auto sp = py_as_space_leg(leg);
                   auto init = py_parse_diag(leg, std::move(backend), labels);
                   auto dt = SymmetricTensor::_parse_default_dtype(dtype, init.symmetry);
                   if (!dt.has_value()) {
                       dt = Dtype::Float64;
                   }
                   std::string device_s =
                     device.has_value() ? *device : init.backend->block_backend->default_device;
                   return std::make_shared<Identity>(
                     sp, init.backend, init.symmetry, init.labels, *dt, std::move(device_s));
               }),
               py::arg("leg"),
               py::arg("backend") = nullptr,
               py::arg("dtype") = py::none(),
               py::arg("device") = py::none(),
               py::arg("labels") = py::none());

    id_cls.def("test_sanity", &Identity::test_sanity, DOC(lasyten, Identity, test_sanity));

    auto bind_unsupported = [&](char const* name) {
        id_cls.def_static(name, [name](py::args, py::kwargs) {
            Identity::unsupported_factory(name);
            return Identity::Ptr{};
        });
    };
    bind_unsupported("from_block_func");
    bind_unsupported("from_dense_block");
    bind_unsupported("from_diag_block");
    bind_unsupported("from_random_normal");
    bind_unsupported("from_random_uniform");
    bind_unsupported("from_sector_block_func");
    bind_unsupported("from_tensor");
    bind_unsupported("from_zero");

    id_cls.def_static(
      "from_eye",
      [](py::object leg,
         TensorBackend::Ptr backend,
         py::object labels,
         Dtype dtype,
         std::optional<std::string> device) {
          auto init = py_parse_diag(leg, std::move(backend), labels);
          return Identity::from_eye(
            py_as_space_leg(leg), init.backend, init.labels, dtype, device);
      },
      py::arg("leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Float64,
      py::arg("device") = py::none(),
      DOC(lasyten, Identity, from_eye));

    id_cls.def_static("from_hdf5",
                      lasyten::hdf5::wrap_from_hdf5<Identity>(),
                      py::arg("hdf5_loader"),
                      py::arg("h5gr"),
                      py::arg("subpath"),
                      DOC(lasyten, Identity, from_hdf5));

    id_cls.def(
      "as_dtype", &Identity::as_dtype, py::arg("dtype"), DOC(lasyten, Identity, as_dtype));
    id_cls.def("as_SymmetricTensor",
               &Identity::as_SymmetricTensor,
               py::arg("guarantee_copy") = false,
               py::arg("warning") = py::none(),
               DOC(lasyten, Identity, as_SymmetricTensor));
    id_cls.def("as_DiagonalTensor",
               &Identity::as_DiagonalTensor,
               py::arg("guarantee_copy") = false,
               py::arg("warning") = py::none());
    id_cls.def("copy",
               &Identity::copy,
               py::arg("deep") = true,
               py::arg("device") = py::none(),
               py::arg("dtype") = py::none(),
               DOC(lasyten, Identity, copy));
    id_cls.def("diagonal",
               &Identity::diagonal,
               py::arg("check_offdiagonal") = false,
               DOC(lasyten, Identity, diagonal));
    id_cls.def("diagonal_as_block", &Identity::diagonal_as_block, py::arg("dtype") = py::none());
    id_cls.def(
      "diagonal_as_numpy", &Identity::diagonal_as_numpy, py::arg("numpy_dtype") = py::none());
    id_cls.def(
      "_get_item", &Identity::_get_item, py::arg("idx"), DOC(lasyten, Identity, _get_item));
    id_cls.def("all", &Identity::all, DOC(lasyten, Identity, all));
    id_cls.def("any", &Identity::any, DOC(lasyten, Identity, any));
    id_cls.def("max", &Identity::max);
    id_cls.def("min", &Identity::min);
    id_cls.def(
      "argmin", &Identity::argmin, py::arg("s") = py::none(), DOC(lasyten, Identity, argmin));
    id_cls.def("move_to_device",
               &Identity::move_to_device,
               py::arg("device"),
               DOC(lasyten, Identity, move_to_device));
    id_cls.def("to_backend",
               &Identity::to_backend,
               py::arg("backend"),
               py::arg("dtype") = py::none(),
               py::arg("device") = py::none(),
               DOC(lasyten, Identity, to_backend));
    id_cls.def(
      "to_dense_block",
      [](Identity& self,
         py::object leg_order,
         std::optional<Dtype> dtype,
         bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false,
      DOC(lasyten, Identity, to_dense_block));

    id_cls.def("__abs__", &Identity::abs);
    id_cls.def("__bool__", [](Identity& self) {
        auto tensors_mod = py::module_::import("lasyten.tensors._tensors");
        if (self.dtype == Dtype::Bool && tensors_mod.attr("is_scalar")(self).cast<bool>()) {
            return true;
        }
        throw std::invalid_argument(
          "The truth value of a non-scalar DiagonalTensor is ambiguous. Use a.any() or a.all()");
    });
}

} // namespace lasyten
