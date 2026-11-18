#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/toric_code_category.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/toric_code_category.h>

#include "tools/hdf5_bind.h"
#include <optional>
#include <string>

namespace lasyten {

void
bind_toric_code_category(nb::module_& m)
{
    nb::class_<ToricCodeCategory, QuantumDoubleZNAnyonCategory> cls(
      m, "ToricCodeCategory", DOC(lasyten, ToricCodeCategory));
    cls
      .def(lasyten::ctor<decltype(cls), std::optional<std::string>>(),
           nb::arg("descriptive_name").none() = nb::none())
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<ToricCodeCategory>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
    cls.attr("vacuum") = ToricCodeCategory::vacuum;
    cls.attr("electric_charge") = ToricCodeCategory::electric_charge;
    cls.attr("magnetic_flux") = ToricCodeCategory::magnetic_flux;
    cls.attr("fermion") = ToricCodeCategory::fermion;
}

} // namespace lasyten
