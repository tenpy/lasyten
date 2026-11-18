#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/fermion_number.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/fermion_number.h>

#include "tools/hdf5_bind.h"
#include <optional>
#include <string>

namespace lasyten {

void
bind_fermion_number(nb::module_& m)
{
    nb::class_<FermionNumber, SymmetryFactor>(m, "FermionNumber", DOC(lasyten, FermionNumber))
      .def(lasyten::ctor_for<FermionNumber, std::optional<std::string>, bool>(),
           nb::arg("descriptive_name").none() = nb::none(),
           nb::arg("trivial_shift") = true)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<FermionNumber>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
}

} // namespace lasyten
