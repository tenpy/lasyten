#include "../doc_plus.h"
#include "docstrings/symmetries/symmetry_factor.h"
#include "py_lasyten_nanobind.h"
#include "tools/hdf5_bind.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <lasyten/symmetries/symmetry.h>
#include <lasyten/symmetries/symmetry_factor.h>

#include <optional>
#include <string>

namespace lasyten {

void
bind_symmetry_factor(nb::module_& m)
{
    nb::class_<SymmetryFactor, BaseSymmetry> cls(
      m, "SymmetryFactor", DOC(lasyten, SymmetryFactor));

    cls.def(lasyten::ctor<decltype(cls),
                          FusionStyle,
                          BraidingStyle,
                          Sector,
                          std::string,
                          float64,
                          bool,
                          std::optional<std::string>,
                          bool>(),
            nb::arg("fusion_style"),
            nb::arg("braiding_style"),
            nb::arg("trivial_sector"),
            nb::arg("group_name"),
            nb::arg("num_sectors"),
            nb::arg("has_complex_topological_data"),
            nb::arg("descriptive_name").none() = nb::none(),
            nb::arg("trivial_shift") = true);

    cls.def_rw("group_name", &SymmetryFactor::group_name)
      .def_rw("descriptive_name", &SymmetryFactor::descriptive_name)
      .def_rw("fusion_tensor_dtype", &SymmetryFactor::fusion_tensor_dtype);

    cls
      .def(
        "is_valid_sector",
        [](SymmetryFactor const& self, py::object a) {
            if (py::isinstance<Sector>(a)) {
                return self.is_valid_sector(py::cast<Sector>(a));
            }
            if (!py::isinstance<py::array>(a)) {
                return false;
            }
            try {
                return self.is_valid_sector(sector_from_numpy(a));
            } catch (...) {
                return false;
            }
        },
        nb::arg("a"),
        doc_cpp_ref(R"pydoc(is_valid_sector)pydoc", "lasyten::BaseSymmetry::is_valid_sector()"))
      .def(
        "are_valid_sectors",
        [](SymmetryFactor const& self, py::object sectors) {
            if (py::isinstance<SectorArray>(sectors)) {
                return self.are_valid_sectors(py::cast<SectorArray>(sectors));
            }
            if (!py::isinstance<py::array>(sectors)) {
                return false;
            }
            try {
                return self.are_valid_sectors(sector_array_from_numpy(sectors));
            } catch (...) {
                return false;
            }
        },
        nb::arg("sectors"))
      .def("fusion_outcomes",
           &SymmetryFactor::fusion_outcomes,
           nb::arg("a"),
           nb::arg("b"),
           doc_cpp_ref(R"pydoc(fusion_outcomes)pydoc", "lasyten::BaseSymmetry::fusion_outcomes()"))
      .def("__repr__", &SymmetryFactor::repr)
      .def(
        "is_equivalent_to",
        [](SymmetryFactor& self, py::object other) {
            if (py::isinstance<Symmetry>(other)) {
                return py::to_bool(other.attr("is_equivalent_to")(py::cast(self)));
            }
            return self._is_equivalent_factor(py::cast<SymmetryFactor const&>(other));
        },
        nb::arg("other"))
      .def("_is_equivalent_factor",
           &SymmetryFactor::_is_equivalent_factor,
           nb::arg("other"),
           DOC(lasyten, SymmetryFactor, _is_equivalent_factor))
      .def(
        "as_Symmetry",
        [](py::object self) {
            // Use the Python-held shared_ptr; shared_from_this fails with smart_holder
            // trampolines.
            auto ptr = py::cast<SymmetryFactor::Ptr>(self);
            return std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ ptr });
        },
        DOC(lasyten, SymmetryFactor, as_Symmetry))
      .def("__str__", &SymmetryFactor::str)
      .def("__mul__",
           [](py::object self, py::object other) -> py::object {
               auto self_ptr = py::cast<SymmetryFactor::Ptr>(self);
               if (py::isinstance<SymmetryFactor>(other)) {
                   return py::cast(std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{
                     self_ptr, py::cast<SymmetryFactor::Ptr>(other) }));
               }
               if (py::isinstance<Symmetry>(other)) {
                   auto const& sym = py::cast<Symmetry const&>(other);
                   std::vector<SymmetryFactor::Ptr> factors;
                   factors.reserve(1 + sym.factors.size());
                   factors.push_back(self_ptr);
                   factors.insert(factors.end(), sym.factors.begin(), sym.factors.end());
                   return py::cast(std::make_shared<Symmetry>(std::move(factors)));
               }
               return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
           })
      .def("__eq__",
           [](SymmetryFactor const& self, py::object other) {
               if (!py::isinstance<SymmetryFactor>(other)) {
                   return false;
               }
               return self.equals(py::cast<SymmetryFactor const&>(other));
           })
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<SymmetryFactor>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"));
}

} // namespace lasyten
