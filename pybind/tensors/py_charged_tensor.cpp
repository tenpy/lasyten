#include <lasyten/tensors/charged_tensor.h>

#include "py_callbacks.hpp"
#include "py_factory_parse.hpp"

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"

#include "docstrings/tensors/charged_tensor.h"

#include "tensor_pickle.h"
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
            out.emplace_back(py::cast<std::string>(item));
        } else {
            out.emplace_back(py::cast<int64>(item));
        }
    }
    return out;
}

} // namespace

void
bind_tensors_charged_tensor(nb::module_& m)
{
    nb::class_<ChargedTensor, Tensor> cls(m, "ChargedTensor");
    cls.doc() = DOC(lasyten, ChargedTensor);

    cls.def(nb::new_([](py::object invariant_part, py::object charged_state) {
                auto inv =
                  py::cast_non_null<SymmetricTensor::Ptr>(invariant_part, "invariant_part");
                inv->allow_charge_leg_label = true;
                if (charged_state.is_none()) {
                    throw std::invalid_argument(
                      "ChargedTensor requires a charged_state. Use HiddenLegTensor to hide legs.");
                }
                auto cs = py_optional_block(charged_state, inv->backend, inv->dtype, inv->device);
                return std::make_shared<ChargedTensor>(inv, cs);
            }),
            nb::arg("invariant_part"),
            nb::arg("charged_state"));

    cls.def_ro_static("_CHARGE_LEG_LABEL", &ChargedTensor::_CHARGE_LEG_LABEL);

    cls.def_rw("invariant_part", &ChargedTensor::invariant_part);
    cls.def_prop_rw(
      "charged_state",
      [](ChargedTensor& self) -> py::object { return py::cast(self.charged_state); },
      [](ChargedTensor& self, py::object obj) {
          if (obj.is_none()) {
              throw std::invalid_argument("ChargedTensor.charged_state cannot be set to None");
          }
          self.charged_state = py::cast<BlockBackend::BlockPtr>(obj);
      });
    cls.def_ro("charge_leg", &ChargedTensor::charge_leg);

    cls.def("test_sanity", &ChargedTensor::test_sanity, DOC(lasyten, ChargedTensor, test_sanity));

    cls.def_static(
      "_parse_inv_domain",
      [](TensorProduct::Ptr domain, py::object charge) {
          return ChargedTensor::_parse_inv_domain(std::move(domain), py_as_charge(charge));
      },
      nb::arg("domain"),
      nb::arg("charge"),
      DOC(lasyten, ChargedTensor, _parse_inv_domain));
    cls.def_static(
      "_parse_inv_labels",
      [](py::object labels, TensorProduct::Ptr codomain, TensorProduct::Ptr domain) {
          auto labs = parse_tensor_init_labels(labels, codomain, domain);
          return ChargedTensor::_parse_inv_labels(labs, codomain, domain);
      },
      nb::arg("labels"),
      nb::arg("codomain"),
      nb::arg("domain"),
      DOC(lasyten, ChargedTensor, _parse_inv_labels));
    cls.def_static("supports_symmetry",
                   &ChargedTensor::supports_symmetry,
                   nb::arg("symmetry"),
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
      nb::arg("func"),
      nb::arg("charge"),
      nb::arg("codomain"),
      nb::arg("domain").none() = nb::none(),
      nb::arg("charged_state"),
      nb::arg("backend").none() = nullptr,
      nb::arg("labels").none() = nb::none(),
      nb::arg("func_kwargs").none() = nb::none(),
      nb::arg("shape_kw").none() = nb::none(),
      nb::arg("dtype").none() = nb::none(),
      nb::arg("device").none() = nb::none(),
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
      nb::arg("block"),
      nb::arg("codomain"),
      nb::arg("domain").none() = nb::none(),
      nb::arg("charge").none() = nb::none(),
      nb::arg("backend").none() = nullptr,
      nb::arg("labels").none() = nb::none(),
      nb::arg("dtype").none() = nb::none(),
      nb::arg("device").none() = nb::none(),
      nb::arg("tol") = 1e-6,
      nb::arg("understood_braiding") = false,
      DOC(lasyten, ChargedTensor, from_dense_block));

    cls.def_static(
      "from_dense_block_single_sector",
      [](py::object vector,
         py::object space,
         Sector sector,
         TensorBackend::Ptr backend,
         std::optional<std::string> label,
         std::optional<std::string> device) {
          auto sp = py::cast_non_null<Leg::Ptr>(space, "space");
          if (!backend) {
              backend = get_backend(sp.symmetry());
          }
          auto vec = backend->block_backend->as_block(vector, std::nullopt, device);
          return ChargedTensor::from_dense_block_single_sector(
            vec, sp, sector, backend, label, device);
      },
      nb::arg("vector"),
      nb::arg("space"),
      nb::arg("sector"),
      nb::arg("backend").none() = nullptr,
      nb::arg("label").none() = nb::none(),
      nb::arg("device").none() = nb::none(),
      DOC(lasyten, ChargedTensor, from_dense_block_single_sector));

    cls.def_static(
      "from_invariant_part",
      [](py::object invariant_part, py::object charged_state) {
          auto inv = py::cast_non_null<SymmetricTensor::Ptr>(invariant_part, "invariant_part");
          auto cs = py_optional_block(charged_state, inv->backend, inv->dtype, inv->device);
          return py_from_charged_or_scalar(ChargedTensor::from_invariant_part(inv, cs));
      },
      nb::arg("invariant_part"),
      nb::arg("charged_state"),
      DOC(lasyten, ChargedTensor, from_invariant_part));

    cls.def_static(
      "from_two_charge_legs",
      [](py::object invariant_part, py::object state1, py::object state2) {
          auto inv = py::cast_non_null<SymmetricTensor::Ptr>(invariant_part, "invariant_part");
          if (state1.is_none() || state2.is_none()) {
              throw std::invalid_argument("from_two_charge_legs requires both state1 and state2");
          }
          auto s1 = py_optional_block(state1, inv->backend, inv->dtype, inv->device);
          auto s2 = py_optional_block(state2, inv->backend, inv->dtype, inv->device);
          return py_from_charged_or_scalar(ChargedTensor::from_two_charge_legs(inv, s1, s2));
      },
      nb::arg("invariant_part"),
      nb::arg("state1"),
      nb::arg("state2"),
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
      nb::arg("codomain"),
      nb::arg("domain").none() = nb::none(),
      nb::arg("charge").none() = nb::none(),
      nb::arg("charged_state").none() = nb::none(),
      nb::arg("backend").none() = nullptr,
      nb::arg("labels").none() = nb::none(),
      nb::arg("dtype") = Dtype::Complex128,
      nb::arg("device").none() = nb::none(),
      DOC(lasyten, ChargedTensor, from_zero));

    cls.def_static("from_hdf5",
                   lasyten::hdf5::wrap_from_hdf5<ChargedTensor>(),
                   nb::arg("hdf5_loader"),
                   nb::arg("h5gr"),
                   nb::arg("subpath"),
                   DOC(lasyten, ChargedTensor, from_hdf5));
    cls.def("save_hdf5",
            lasyten::hdf5::wrap_save_hdf5_const<ChargedTensor>(),
            nb::arg("hdf5_saver"),
            nb::arg("h5gr"),
            nb::arg("subpath"),
            DOC(lasyten, ChargedTensor, save_hdf5));

    // Pickle state: the HDF5 sub-groups ``invariant_part`` and ``charged_state`` (``dtype``,
    // ``shape`` and ``labels`` are derived from the invariant part by the constructor).
    lasyten::pickle::def_state_pickle(
      cls,
      [](ChargedTensor const& self) {
          nb::dict d;
          d["invariant_part"] = self.invariant_part;
          d["charged_state"] = self.charged_state;
          return d;
      },
      [](nb::dict state) {
          auto inv = lasyten::pickle::get_item<SymmetricTensor::Ptr>(state, "invariant_part");
          inv->allow_charge_leg_label = true;
          return std::make_shared<ChargedTensor>(
            std::move(inv),
            lasyten::pickle::get_item<BlockBackend::BlockPtr>(state, "charged_state"));
      });

    cls.def("as_dtype",
            &ChargedTensor::as_dtype,
            nb::arg("dtype"),
            DOC(lasyten, ChargedTensor, as_dtype));
    cls.def("as_SymmetricTensor",
            &ChargedTensor::as_SymmetricTensor,
            nb::arg("guarantee_copy") = false,
            nb::arg("warning").none() = nb::none(),
            DOC(lasyten, ChargedTensor, as_SymmetricTensor));

    cls.def("copy",
            &ChargedTensor::copy,
            nb::arg("deep") = true,
            nb::arg("device").none() = nb::none(),
            nb::arg("dtype").none() = nb::none(),
            DOC(lasyten, ChargedTensor, copy));

    cls.def_prop_ro("dagger", &ChargedTensor::dagger, DOC(lasyten, ChargedTensor, dagger));
    cls.def_prop_ro("hc", &ChargedTensor::dagger, DOC(lasyten, ChargedTensor, dagger));

    cls.def("_get_item",
            &ChargedTensor::_get_item,
            nb::arg("idx"),
            DOC(lasyten, ChargedTensor, _get_item));
    cls.def("move_to_device",
            &ChargedTensor::move_to_device,
            nb::arg("device"),
            DOC(lasyten, ChargedTensor, move_to_device));

    cls.def("to_backend",
            &ChargedTensor::to_backend,
            nb::arg("backend"),
            nb::arg("dtype").none() = nb::none(),
            nb::arg("device").none() = nb::none(),
            DOC(lasyten, ChargedTensor, to_backend));

    cls.def(
      "to_dense_block",
      [](ChargedTensor& self,
         py::object leg_order,
         std::optional<Dtype> dtype,
         bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      nb::arg("leg_order").none() = nb::none(),
      nb::arg("dtype").none() = nb::none(),
      nb::arg("understood_braiding") = false,
      DOC(lasyten, ChargedTensor, to_dense_block));

    cls.def("to_dense_block_single_sector",
            &ChargedTensor::to_dense_block_single_sector,
            DOC(lasyten, ChargedTensor, to_dense_block_single_sector));
}

} // namespace lasyten
