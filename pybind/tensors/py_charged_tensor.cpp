#include <lasyten/tensors/charged_tensor.h>

#include "py_callbacks.hpp"
#include "py_factory_parse.hpp"

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"

#include "docstrings/tensors/charged_tensor.h"

#include <pybind11/stl.h>

#include "tools/hdf5_bind.h"
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

} // namespace

void
bind_tensors_charged_tensor(py::module_& m)
{
    py::class_<ChargedTensor, Tensor, py::smart_holder> cls(m, "ChargedTensor");
    cls.doc() = DOC(lasyten, ChargedTensor);

    cls.def(py::init([](py::object invariant_part, py::object charged_state) {
                auto inv = invariant_part.cast<SymmetricTensor::Ptr>();
                inv->allow_charge_leg_label = true;
                if (charged_state.is_none()) {
                    throw std::invalid_argument(
                      "ChargedTensor requires a charged_state. Use HiddenLegTensor to hide legs.");
                }
                auto cs = py_optional_block(charged_state, inv->backend, inv->dtype, inv->device);
                return std::make_shared<ChargedTensor>(inv, cs);
            }),
            py::arg("invariant_part"),
            py::arg("charged_state"));

    cls.def_readonly_static("_CHARGE_LEG_LABEL", &ChargedTensor::_CHARGE_LEG_LABEL);

    cls.def_readwrite("invariant_part", &ChargedTensor::invariant_part);
    cls.def_property(
      "charged_state",
      [](ChargedTensor& self) -> py::object { return py::cast(self.charged_state); },
      [](ChargedTensor& self, py::object obj) {
          if (obj.is_none()) {
              throw std::invalid_argument("ChargedTensor.charged_state cannot be set to None");
          }
          self.charged_state = obj.cast<BlockBackend::BlockPtr>();
      });
    cls.def_readonly("charge_leg", &ChargedTensor::charge_leg);

    cls.def("test_sanity", &ChargedTensor::test_sanity, DOC(lasyten, ChargedTensor, test_sanity));

    cls.def_static(
      "_parse_inv_domain",
      [](TensorProduct::Ptr domain, py::object charge) {
          return ChargedTensor::_parse_inv_domain(std::move(domain), py_as_charge(charge));
      },
      py::arg("domain"),
      py::arg("charge"),
      DOC(lasyten, ChargedTensor, _parse_inv_domain));
    cls.def_static(
      "_parse_inv_labels",
      [](py::object labels, TensorProduct::Ptr codomain, TensorProduct::Ptr domain) {
          auto labs = parse_tensor_init_labels(labels, codomain, domain);
          return ChargedTensor::_parse_inv_labels(labs, codomain, domain);
      },
      py::arg("labels"),
      py::arg("codomain"),
      py::arg("domain"),
      DOC(lasyten, ChargedTensor, _parse_inv_labels));
    cls.def_static("supports_symmetry",
                   &ChargedTensor::supports_symmetry,
                   py::arg("symmetry"),
                   DOC(lasyten, ChargedTensor, supports_symmetry));

    cls.def_static(
      "from_block_func",
      [](py::function func,
         py::object charge,
         py::object codomain,
         py::object domain,
         py::object charged_state,
         TensorBackend::Ptr backend,
         py::object labels,
         py::object func_kwargs,
         std::optional<std::string> shape_kw,
         std::optional<Dtype> dtype,
         std::optional<std::string> device) {
          auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
          auto cs = py_optional_block(charged_state, init.backend, dtype, device);
          auto dt = SymmetricTensor::_parse_default_dtype(dtype, init.symmetry);
          auto wrapped = block_factory_from_python(
            func, func_kwargs, shape_kw, init.backend->block_backend, dt, device);
          return ChargedTensor::from_block_func(std::move(wrapped),
                                                py_as_charge(charge),
                                                init.codomain,
                                                init.domain,
                                                cs,
                                                init.backend,
                                                init.labels,
                                                dt,
                                                device);
      },
      py::arg("func"),
      py::arg("charge"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("charged_state"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("func_kwargs") = py::none(),
      py::arg("shape_kw") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      DOC(lasyten, ChargedTensor, from_block_func));

    cls.def_static(
      "from_dense_block",
      [](py::object block,
         py::object codomain,
         py::object domain,
         py::object charge,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<Dtype> dtype,
         std::optional<std::string> device,
         float64 tol,
         bool understood_braiding) {
          auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
          auto block_ptr = init.backend->block_backend->as_block(block, dtype, device);
          return ChargedTensor::from_dense_block(block_ptr,
                                                 init.codomain,
                                                 init.domain,
                                                 py_optional_charge(charge),
                                                 init.backend,
                                                 init.labels,
                                                 dtype,
                                                 device,
                                                 tol,
                                                 understood_braiding);
      },
      py::arg("block"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("charge") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      py::arg("tol") = 1e-6,
      py::arg("understood_braiding") = false,
      DOC(lasyten, ChargedTensor, from_dense_block));

    cls.def_static(
      "from_dense_block_single_sector",
      [](py::object vector,
         py::object space,
         Sector sector,
         TensorBackend::Ptr backend,
         std::optional<std::string> label,
         std::optional<std::string> device) {
          auto sp = space.cast<Leg::Ptr>();
          if (!backend) {
              backend = get_backend(sp->symmetry);
          }
          auto vec = backend->block_backend->as_block(vector, std::nullopt, device);
          return ChargedTensor::from_dense_block_single_sector(
            vec, sp, sector, backend, label, device);
      },
      py::arg("vector"),
      py::arg("space"),
      py::arg("sector"),
      py::arg("backend") = nullptr,
      py::arg("label") = py::none(),
      py::arg("device") = py::none(),
      DOC(lasyten, ChargedTensor, from_dense_block_single_sector));

    cls.def_static(
      "from_invariant_part",
      [](py::object invariant_part, py::object charged_state) {
          auto inv = invariant_part.cast<SymmetricTensor::Ptr>();
          auto cs = py_optional_block(charged_state, inv->backend, inv->dtype, inv->device);
          return py_from_charged_or_scalar(ChargedTensor::from_invariant_part(inv, cs));
      },
      py::arg("invariant_part"),
      py::arg("charged_state"),
      DOC(lasyten, ChargedTensor, from_invariant_part));

    cls.def_static(
      "from_two_charge_legs",
      [](py::object invariant_part, py::object state1, py::object state2) {
          auto inv = invariant_part.cast<SymmetricTensor::Ptr>();
          if (state1.is_none() || state2.is_none()) {
              throw std::invalid_argument("from_two_charge_legs requires both state1 and state2");
          }
          auto s1 = py_optional_block(state1, inv->backend, inv->dtype, inv->device);
          auto s2 = py_optional_block(state2, inv->backend, inv->dtype, inv->device);
          return py_from_charged_or_scalar(ChargedTensor::from_two_charge_legs(inv, s1, s2));
      },
      py::arg("invariant_part"),
      py::arg("state1"),
      py::arg("state2"),
      DOC(lasyten, ChargedTensor, from_two_charge_legs));

    cls.def_static(
      "from_zero",
      [](py::object codomain,
         py::object domain,
         py::object charge,
         py::object charged_state,
         TensorBackend::Ptr backend,
         py::object labels,
         Dtype dtype,
         std::optional<std::string> device) {
          auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
          auto cs = py_optional_block(charged_state, init.backend, dtype, device);
          return ChargedTensor::from_zero(init.codomain,
                                          init.domain,
                                          py_as_charge(charge),
                                          cs,
                                          init.backend,
                                          init.labels,
                                          dtype,
                                          device);
      },
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("charge") = py::none(),
      py::arg("charged_state") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      DOC(lasyten, ChargedTensor, from_zero));

    cls.def_static("from_hdf5",
                   lasyten::hdf5::wrap_from_hdf5<ChargedTensor>(),
                   py::arg("hdf5_loader"),
                   py::arg("h5gr"),
                   py::arg("subpath"),
                   DOC(lasyten, ChargedTensor, from_hdf5));
    cls.def("save_hdf5",
            lasyten::hdf5::wrap_save_hdf5_const<ChargedTensor>(),
            py::arg("hdf5_saver"),
            py::arg("h5gr"),
            py::arg("subpath"),
            DOC(lasyten, ChargedTensor, save_hdf5));

    cls.def("as_dtype",
            &ChargedTensor::as_dtype,
            py::arg("dtype"),
            DOC(lasyten, ChargedTensor, as_dtype));
    cls.def("as_SymmetricTensor",
            &ChargedTensor::as_SymmetricTensor,
            py::arg("guarantee_copy") = false,
            py::arg("warning") = py::none(),
            DOC(lasyten, ChargedTensor, as_SymmetricTensor));

    cls.def("copy",
            &ChargedTensor::copy,
            py::arg("deep") = true,
            py::arg("device") = py::none(),
            py::arg("dtype") = py::none(),
            DOC(lasyten, ChargedTensor, copy));

    cls.def_property_readonly(
      "dagger", &ChargedTensor::dagger, DOC(lasyten, ChargedTensor, dagger));
    cls.def_property_readonly("hc", &ChargedTensor::dagger, DOC(lasyten, ChargedTensor, dagger));

    cls.def("_get_item",
            &ChargedTensor::_get_item,
            py::arg("idx"),
            DOC(lasyten, ChargedTensor, _get_item));
    cls.def("move_to_device",
            &ChargedTensor::move_to_device,
            py::arg("device"),
            DOC(lasyten, ChargedTensor, move_to_device));

    cls.def("to_backend",
            &ChargedTensor::to_backend,
            py::arg("backend"),
            py::arg("dtype") = py::none(),
            py::arg("device") = py::none(),
            DOC(lasyten, ChargedTensor, to_backend));

    cls.def(
      "to_dense_block",
      [](ChargedTensor& self,
         py::object leg_order,
         std::optional<Dtype> dtype,
         bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false,
      DOC(lasyten, ChargedTensor, to_dense_block));

    cls.def("to_dense_block_single_sector",
            &ChargedTensor::to_dense_block_single_sector,
            DOC(lasyten, ChargedTensor, to_dense_block_single_sector));
}

} // namespace lasyten
