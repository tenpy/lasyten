#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/fermion_parity.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/fermion_parity.h>

#include "tools/hdf5_bind.h"
#include <optional>
#include <string>

namespace lasyten {

void
bind_fermion_parity(nb::module_& m)
{
    nb::class_<FermionParity, SymmetryFactor> cls(m, "FermionParity", DOC(lasyten, FermionParity));
    cls
      .def(lasyten::ctor<decltype(cls), std::optional<std::string>, bool>(),
           nb::arg("descriptive_name").none() = nb::none(),
           nb::arg("trivial_shift") = true)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<FermionParity>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
    cls.attr("even") = FermionParity::even;
    cls.attr("odd") = FermionParity::odd;
}

} // namespace lasyten
