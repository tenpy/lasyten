#include <lasyten/tensors/hidden_leg_tensor.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"

#include "docstrings/tensors/hidden_leg_tensor.h"

#include "tensor_pickle.h"
#include "tools/hdf5_bind.h"
#include <string>
#include <variant>
#include <vector>

namespace lasyten {

void
bind_tensors_hidden_leg_tensor(nb::module_& m)
{
    nb::class_<HiddenLegTensor, SymmetricTensor> cls(m, "HiddenLegTensor");
    cls.doc() = DOC(lasyten, HiddenLegTensor);

    cls.def(
      nb::new_([](py::object tensor, py::object which_legs) {
          auto tens = py::cast_non_null<Tensor::Ptr>(tensor, "tensor");
          std::vector<std::variant<int64, std::string>> legs;
          for (auto item : which_legs) {
              if (py::isinstance<py::str>(item)) {
                  legs.emplace_back(py::cast<std::string>(item));
              } else {
                  legs.emplace_back(py::cast<int64>(item));
              }
          }
          return HiddenLegTensor::from_tensor(std::move(tens), std::move(legs));
      }),
      nb::arg("tensor"),
      nb::arg("which_legs"),
      "Construct a HiddenLegTensor by hiding the given legs (prefixes '!' to their labels).");

    cls.def_static(
      "from_tensor",
      [](py::object tensor, py::object which_legs) {
          auto tens = py::cast_non_null<Tensor::Ptr>(tensor, "tensor");
          std::vector<std::variant<int64, std::string>> legs;
          for (auto item : which_legs) {
              if (py::isinstance<py::str>(item)) {
                  legs.emplace_back(py::cast<std::string>(item));
              } else {
                  legs.emplace_back(py::cast<int64>(item));
              }
          }
          return HiddenLegTensor::from_tensor(std::move(tens), std::move(legs));
      },
      nb::arg("tensor"),
      nb::arg("which_legs"),
      DOC(lasyten, HiddenLegTensor, from_tensor));

    cls.def_static(
      "is_hidden_leg_label",
      [](py::object label) {
          if (label.is_none()) {
              return false;
          }
          return HiddenLegTensor::is_hidden_leg_label(py::cast<std::string>(label));
      },
      nb::arg("label"),
      DOC(lasyten, HiddenLegTensor, is_hidden_leg_label));

    cls.def_static(
      "is_charge_temp_label",
      [](py::object label) {
          if (label.is_none()) {
              return false;
          }
          return HiddenLegTensor::is_charge_temp_label(py::cast<std::string>(label));
      },
      nb::arg("label"),
      DOC(lasyten, HiddenLegTensor, is_charge_temp_label));

    cls.def("hidden_leg_idcs",
            &HiddenLegTensor::hidden_leg_idcs,
            DOC(lasyten, HiddenLegTensor, hidden_leg_idcs));
    cls.def("public_leg_idcs",
            &HiddenLegTensor::public_leg_idcs,
            DOC(lasyten, HiddenLegTensor, public_leg_idcs));
    cls.def(
      "unhide_legs", &HiddenLegTensor::unhide_legs, DOC(lasyten, HiddenLegTensor, unhide_legs));

    cls.def("test_sanity", &HiddenLegTensor::test_sanity);
    cls.def_prop_ro("dagger", &HiddenLegTensor::dagger);
    cls.def_prop_ro("hc", &HiddenLegTensor::dagger);

    cls.def(
      "as_SymmetricTensor",
      [](HiddenLegTensor& self, bool guarantee_copy, py::object warning) {
          std::optional<std::string> warn;
          if (!warning.is_none()) {
              warn = py::cast<std::string>(warning);
          }
          return self.as_SymmetricTensor(guarantee_copy, warn);
      },
      nb::arg("guarantee_copy") = false,
      nb::arg("warning").none() = nb::none(),
      DOC(lasyten, HiddenLegTensor, as_SymmetricTensor));

    cls.def_static("from_hdf5",
                   lasyten::hdf5::wrap_from_hdf5<HiddenLegTensor>(),
                   nb::arg("hdf5_loader"),
                   nb::arg("h5gr"),
                   nb::arg("subpath"),
                   DOC(lasyten, HiddenLegTensor, from_hdf5));
    cls.def("save_hdf5",
            lasyten::hdf5::wrap_save_hdf5_const<HiddenLegTensor>(),
            nb::arg("hdf5_saver"),
            nb::arg("h5gr"),
            nb::arg("subpath"),
            DOC(lasyten, HiddenLegTensor, save_hdf5));

    // Pickle state: as for ``SymmetricTensor``; rebuilt like ``HiddenLegTensor::from_hdf5``.
    lasyten::pickle::def_state_pickle(
      cls,
      [](HiddenLegTensor const& self) { return lasyten::pickle::symmetric_tensor_to_dict(self); },
      [](nb::dict state) {
          return std::make_shared<HiddenLegTensor>(
            lasyten::pickle::symmetric_tensor_from_dict(state));
      });
}

} // namespace lasyten
