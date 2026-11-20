#include <lasyten/backends/no_symmetry.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/tensors/mask.h>

#include "py_callbacks.hpp"
#include "py_factory_parse.hpp"

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "../py_numbers.h"

#include "docstrings/tensors/mask.h"

#include "tensor_pickle.h"
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
            out.emplace_back(py::cast<std::string>(item));
        } else {
            out.emplace_back(py::cast<int64>(item));
        }
    }
    return out;
}

} // namespace

namespace {

std::optional<nb::class_<Mask, Tensor>> g_mask;

} // namespace

void
declare_tensors_mask(nb::module_& m)
{
    g_mask = nb::class_<Mask, Tensor>(m, "Mask");
    (*g_mask).doc() = DOC(lasyten, Mask);
}

void
bind_tensors_mask(nb::module_& m)
{
    auto& cls = *g_mask;

    cls.def(nb::new_([](TensorBackend::DataPtr data,
                        py::object space_in_obj,
                        py::object space_out_obj,
                        std::optional<bool> is_projection,
                        TensorBackend::Ptr backend,
                        py::object labels) {
                auto space_in = py_as_space_leg(space_in_obj);
                auto space_out = py_as_space_leg(space_out_obj);
                bool proj = false;
                if (!is_projection.has_value()) {
                    if (space_in->Space::dim == space_out->Space::dim) {
                        throw std::invalid_argument(
                          "Need to specify is_projection for equal spaces.");
                    }
                    proj = space_in->Space::dim > space_out->Space::dim;
                } else {
                    proj = *is_projection;
                }
                auto init = parse_tensor_init(py::make_tuple(space_out_obj),
                                              py::make_tuple(space_in_obj),
                                              std::move(backend),
                                              labels);
                auto device_s = init.backend->get_device_from_data(data);
                return std::make_shared<Mask>(std::move(data),
                                              space_in,
                                              space_out,
                                              proj,
                                              init.backend,
                                              init.symmetry,
                                              init.labels,
                                              std::move(device_s));
            }),
            nb::arg("data"),
            nb::arg("space_in"),
            nb::arg("space_out"),
            nb::arg("is_projection").none() = nb::none(),
            nb::arg("backend").none() = nullptr,
            nb::arg("labels").none() = nb::none());

    cls.def_ro_static("_forbidden_dtypes", &Mask::_forbidden_dtypes);

    cls.def_rw("is_projection", &Mask::is_projection);

    cls.def_prop_rw(
      "data",
      [](Mask& self) -> py::object {
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              return py::cast(NoSymmetryBackend::unwrap(self.data));
          }
          return py::cast(self.data);
      },
      [](Mask& self, py::object obj) {
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              self.data =
                NoSymmetryBackend::wrap(py::cast_non_null<BlockBackend::BlockPtr>(obj, "data"));
          } else {
              self.data = py::cast_non_null<TensorBackend::DataPtr>(obj, "data");
          }
      });

    cls.def_prop_ro("large_leg", &Mask::large_leg);
    cls.def_prop_ro("small_leg", &Mask::small_leg);

    cls.def("test_sanity", &Mask::test_sanity, DOC(lasyten, Mask, test_sanity));

    cls.def_static(
      "from_eye",
      [](py::object leg,
         bool is_projection,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<std::string> device) {
          auto init = py_parse_diag(leg, std::move(backend), labels);
          return Mask::from_eye(
            py_as_space_leg(leg), is_projection, init.backend, init.labels, device);
      },
      nb::arg("leg"),
      nb::arg("is_projection") = true,
      nb::arg("backend").none() = nullptr,
      nb::arg("labels").none() = nb::none(),
      nb::arg("device").none() = nb::none(),
      DOC(lasyten, Mask, from_eye));

    cls.def_static(
      "from_block_mask",
      [](py::object block_mask,
         py::object large_leg,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<std::string> device) {
          auto init = py_parse_diag(large_leg, std::move(backend), labels);
          auto block = init.backend->block_backend->as_block(block_mask, Dtype::Bool, device);
          return Mask::from_block_mask(
            block, py_as_space_leg(large_leg), init.backend, init.labels, device);
      },
      nb::arg("block_mask"),
      nb::arg("large_leg"),
      nb::arg("backend").none() = nullptr,
      nb::arg("labels").none() = nb::none(),
      nb::arg("device").none() = nb::none(),
      DOC(lasyten, Mask, from_block_mask));

    cls.def_static(
      "from_DiagonalTensor",
      [](py::object diag) {
          return Mask::from_DiagonalTensor(py::cast_non_null<DiagonalTensorCPtr>(diag, "diag"));
      },
      nb::arg("diag"),
      DOC(lasyten, Mask, from_DiagonalTensor));

    cls.def_static(
      "from_sectors_kept",
      [](py::object large_leg, SectorArray keep) {
          auto leg = std::dynamic_pointer_cast<ElementarySpace>(py_as_space_leg(large_leg));
          if (!leg) {
              throw std::invalid_argument("large_leg must be an ElementarySpace");
          }
          return Mask::from_sectors_kept(std::move(leg), std::move(keep));
      },
      nb::arg("large_leg"),
      nb::arg("keep"),
      DOC(lasyten, Mask, from_sectors_kept));

    cls.def_static(
      "from_sectors_discarded",
      [](py::object large_leg, SectorArray discard) {
          auto leg = std::dynamic_pointer_cast<ElementarySpace>(py_as_space_leg(large_leg));
          if (!leg) {
              throw std::invalid_argument("large_leg must be an ElementarySpace");
          }
          return Mask::from_sectors_discarded(std::move(leg), std::move(discard));
      },
      nb::arg("large_leg"),
      nb::arg("discard"),
      DOC(lasyten, Mask, from_sectors_discarded));

    cls.def_static(
      "from_indices",
      [](py::object indices,
         py::object large_leg,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<std::string> device) {
          auto init = py_parse_diag(large_leg, std::move(backend), labels);
          return Mask::from_indices(
            indices, py_as_space_leg(large_leg), init.backend, init.labels, device);
      },
      nb::arg("indices"),
      nb::arg("large_leg"),
      nb::arg("backend").none() = nullptr,
      nb::arg("labels").none() = nb::none(),
      nb::arg("device").none() = nb::none(),
      DOC(lasyten, Mask, from_indices));

    cls.def_static(
      "from_random",
      [](py::object large_leg,
         py::object small_leg,
         TensorBackend::Ptr backend,
         float64 p_keep,
         int64 min_keep,
         py::object labels,
         std::optional<std::string> device,
         py::object np_random) {
          auto init = py_parse_diag(large_leg, std::move(backend), labels);
          Space::Ptr small;
          if (!small_leg.is_none()) {
              small = py_as_space_leg(small_leg);
          }
          return Mask::from_random(py_as_space_leg(large_leg),
                                   small,
                                   init.backend,
                                   p_keep,
                                   min_keep,
                                   init.labels,
                                   device,
                                   np_random);
      },
      nb::arg("large_leg"),
      nb::arg("small_leg").none() = nb::none(),
      nb::arg("backend").none() = nullptr,
      nb::arg("p_keep") = 0.5,
      nb::arg("min_keep") = 0,
      nb::arg("labels").none() = nb::none(),
      nb::arg("device").none() = nb::none(),
      nb::arg("np_random").none() = nb::none(),
      DOC(lasyten, Mask, from_random));

    cls.def_static(
      "from_zero",
      [](py::object large_leg,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<std::string> device) {
          auto init = py_parse_diag(large_leg, std::move(backend), labels);
          return Mask::from_zero(py_as_space_leg(large_leg), init.backend, init.labels, device);
      },
      nb::arg("large_leg"),
      nb::arg("backend").none() = nullptr,
      nb::arg("labels").none() = nb::none(),
      nb::arg("device").none() = nb::none(),
      DOC(lasyten, Mask, from_zero));

    cls.def_static("from_hdf5",
                   lasyten::hdf5::wrap_from_hdf5<Mask>(),
                   nb::arg("hdf5_loader"),
                   nb::arg("h5gr"),
                   nb::arg("subpath"),
                   DOC(lasyten, Mask, from_hdf5));

    cls.def("save_hdf5",
            lasyten::hdf5::wrap_save_hdf5_const<Mask>(),
            nb::arg("hdf5_saver"),
            nb::arg("h5gr"),
            nb::arg("subpath"),
            DOC(lasyten, Mask, save_hdf5));

    // Pickle state: the HDF5 sub-groups ``domain``, ``codomain``, ``backend``, ``data``,
    // ``symmetry``, ``dtype``, ``shape``, ``labels`` plus the ``is_projection`` attribute.
    lasyten::pickle::def_state_pickle(
      cls,
      [](Mask const& self) {
          nb::dict d;
          lasyten::pickle::tensor_common_to_dict(self, d);
          d["data"] = self.data;
          d["is_projection"] = self.is_projection;
          return d;
      },
      [](nb::dict state) {
          auto c = lasyten::pickle::tensor_common_from_dict(state);
          auto data = lasyten::pickle::get_item<TensorBackend::DataPtr>(state, "data");
          auto obj =
            std::make_shared<Mask>(std::move(data),
                                   as_space(c.domain->factors[0]),
                                   as_space(c.codomain->factors[0]),
                                   lasyten::pickle::get_item<bool>(state, "is_projection"),
                                   c.backend,
                                   c.symmetry,
                                   std::move(c.labels),
                                   c.device);
          obj->shape = std::move(c.shape);
          return obj;
      });

    cls.def("as_dtype", &Mask::as_dtype, nb::arg("dtype"), DOC(lasyten, Mask, as_dtype));

    cls.def(
      "as_SymmetricTensor",
      [](Mask& self,
         bool guarantee_copy,
         std::optional<std::string> warning,
         std::optional<Dtype> dtype) {
          if (dtype.has_value()) {
              return self.as_SymmetricTensor(guarantee_copy, warning, *dtype);
          }
          return self.as_SymmetricTensor(guarantee_copy, warning);
      },
      nb::arg("guarantee_copy") = false,
      nb::arg("warning").none() = nb::none(),
      nb::arg("dtype") = Dtype::Complex128,
      DOC(lasyten, Mask, as_SymmetricTensor));

    cls.def("as_DiagonalTensor", &Mask::as_DiagonalTensor, nb::arg("dtype") = Dtype::Complex128);

    cls.def("as_block_mask", &Mask::as_block_mask);
    cls.def("as_numpy_mask", &Mask::as_numpy_mask);

    cls.def("all", &Mask::all, DOC(lasyten, Mask, all));
    cls.def("any", &Mask::any, DOC(lasyten, Mask, any));

    cls.def("copy",
            &Mask::copy,
            nb::arg("deep") = true,
            nb::arg("device").none() = nb::none(),
            nb::arg("dtype").none() = nb::none(),
            DOC(lasyten, Mask, copy));

    // Override Tensor.dagger / hc properties (which delegate to the Python free function).
    cls.def_prop_ro("dagger", &Mask::dagger, DOC(lasyten, Mask, dagger));
    cls.def_prop_ro("hc", &Mask::dagger, DOC(lasyten, Mask, dagger));

    cls.def("_get_item", &Mask::_get_item, nb::arg("idx"), DOC(lasyten, Mask, _get_item));

    cls.def("logical_not", &Mask::logical_not, DOC(lasyten, Mask, logical_not));
    cls.def("orthogonal_complement",
            &Mask::orthogonal_complement,
            DOC(lasyten, Mask, orthogonal_complement));

    cls.def("move_to_device",
            &Mask::move_to_device,
            nb::arg("device"),
            DOC(lasyten, Mask, move_to_device));

    cls.def("to_backend",
            &Mask::to_backend,
            nb::arg("backend"),
            nb::arg("dtype").none() = nb::none(),
            nb::arg("device").none() = nb::none(),
            DOC(lasyten, Mask, to_backend));

    cls.def(
      "to_dense_block",
      [](Mask& self, py::object leg_order, std::optional<Dtype> dtype, bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      nb::arg("leg_order").none() = nb::none(),
      nb::arg("dtype").none() = nb::none(),
      nb::arg("understood_braiding") = false,
      DOC(lasyten, Mask, to_dense_block));

    cls.def(
      "to_numpy",
      [](Mask& self, py::object leg_order, py::object numpy_dtype, bool understood_braiding) {
          return self.to_numpy(optional_leg_order(leg_order), numpy_dtype, understood_braiding);
      },
      nb::arg("leg_order").none() = nb::none(),
      nb::arg("numpy_dtype").none() = nb::none(),
      nb::arg("understood_braiding") = false,
      DOC(lasyten, Mask, to_numpy));

    cls.def(
      "_binary_operand",
      [](Mask& self,
         py::object other,
         py::function func,
         std::string operand,
         bool return_NotImplemented) -> py::object {
          other =
            py_number_as_builtin(other); // np.bool_, 0-d array / tensor: as the Python number
          auto adapted = adapt_block_bool_binary(func, self.backend->block_backend);
          if (py::isinstance<py::bool_>(other)) {
              return py::cast(self._binary_operand(py::to_bool(other), adapted, operand));
          }
          if (py::isinstance<Mask>(other) ||
              py::isinstance(other,
                             py::module_::import("lasyten.tensors._tensors").attr("Mask"))) {
              return py::cast(self._binary_operand(py::cast<MaskCPtr>(other), adapted, operand));
          }
          if (return_NotImplemented &&
              !(py::isinstance<Tensor>(other) ||
                py::isinstance(other,
                               py::module_::import("lasyten.tensors._tensors").attr("Tensor")) ||
                is_py_number(other))) {
              return py::reinterpret_borrow<py::object>(Py_NotImplemented);
          }
          throw std::invalid_argument(std::format("Invalid types for operand \"{}\": Mask and {}",
                                                  operand,
                                                  std::string(py::str(py::type::of(other)))));
      },
      nb::arg("other"),
      nb::arg("func"),
      nb::arg("operand"),
      nb::arg("return_NotImplemented") = true,
      DOC(lasyten, Mask, _binary_operand));

    cls.def(
      "_unary_operand",
      [](Mask& self, py::function func) {
          return self._unary_operand(adapt_block_bool_unary(func, self.backend->block_backend));
      },
      nb::arg("func"));

    cls.def("__bool__", [](Mask&) {
        throw py::type_error("The truth value of a Mask is ambiguous. Use a.any() or a.all()");
    });

    cls.def("__invert__", [](Mask& self) {
        return self._unary_operand(adapt_block_bool_unary(
          py::module_::import("operator").attr("invert"), self.backend->block_backend));
    });

    auto bind_bool_binop = [&](char const* name, char const* op_name, char const* operand) {
        cls.def(
          name,
          [op_name, operand](Mask& self, py::object other) -> py::object {
              other =
                py_number_as_builtin(other); // np.bool_, 0-d array / tensor: as the Python number
              auto func = adapt_block_bool_binary(py::module_::import("operator").attr(op_name),
                                                  self.backend->block_backend);
              if (py::isinstance<py::bool_>(other)) {
                  return py::cast(self._binary_operand(py::to_bool(other), func, operand));
              }
              if (py::isinstance<Mask>(other) ||
                  py::isinstance(other,
                                 py::module_::import("lasyten.tensors._tensors").attr("Mask"))) {
                  return py::cast(self._binary_operand(py::cast<MaskCPtr>(other), func, operand));
              }
              if (!(py::isinstance<Tensor>(other) ||
                    py::isinstance(
                      other, py::module_::import("lasyten.tensors._tensors").attr("Tensor")) ||
                    is_py_number(other))) {
                  return py::reinterpret_borrow<py::object>(Py_NotImplemented);
              }
              throw std::invalid_argument(
                std::format("Invalid types for operand \"{}\": Mask and {}",
                            operand,
                            std::string(py::str(py::type::of(other)))));
          },
          nb::arg("other"));
    };
    bind_bool_binop("__and__", "and_", "&");
    bind_bool_binop("__rand__", "and_", "&");
    bind_bool_binop("__or__", "or_", "|");
    bind_bool_binop("__ror__", "or_", "|");
    bind_bool_binop("__xor__", "xor", "^");
    bind_bool_binop("__rxor__", "xor", "^");
    bind_bool_binop("__eq__", "eq", "==");
    bind_bool_binop("__ne__", "ne", "!=");
}

} // namespace lasyten
