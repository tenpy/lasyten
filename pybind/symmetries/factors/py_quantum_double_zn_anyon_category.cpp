#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/quantum_double_zn_anyon_category.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/quantum_double_zn_anyon_category.h>

#include "tools/hdf5_bind.h"
#include "tools/pickle_bind.h"
#include <optional>
#include <string>

namespace lasyten {

void
bind_quantum_double_zn_anyon_category(nb::module_& m)
{
    nb::class_<QuantumDoubleZNAnyonCategory, SymmetryFactor> cls(
      m, "QuantumDoubleZNAnyonCategory", DOC(lasyten, QuantumDoubleZNAnyonCategory));
    cls
      .def(lasyten::ctor_for<QuantumDoubleZNAnyonCategory, int, std::optional<std::string>>(),
           nb::arg("N"),
           nb::arg("descriptive_name").none() = nb::none())
      .def_ro("N", &QuantumDoubleZNAnyonCategory::N)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<QuantumDoubleZNAnyonCategory>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    lasyten::pickle::def_reduce_ctor(cls, [](QuantumDoubleZNAnyonCategory const& self) {
        return nb::make_tuple(self.N, self.descriptive_name);
    });
}

} // namespace lasyten
