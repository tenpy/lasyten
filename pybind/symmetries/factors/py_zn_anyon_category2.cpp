#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/zn_anyon_category2.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/zn_anyon_category2.h>

#include "tools/hdf5_bind.h"
#include <optional>
#include <string>

namespace lasyten {

void
bind_zn_anyon_category2(nb::module_& m)
{
    nb::class_<ZNAnyonCategory2, SymmetryFactor>(
      m, "ZNAnyonCategory2", DOC(lasyten, ZNAnyonCategory2))
      .def(lasyten::ctor_for<ZNAnyonCategory2, int, int, std::optional<std::string>>(),
           nb::arg("N"),
           nb::arg("n"),
           nb::arg("descriptive_name").none() = nb::none())
      .def_ro("N", &ZNAnyonCategory2::N)
      .def_ro("n", &ZNAnyonCategory2::n)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<ZNAnyonCategory2>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    m.def("semion_category", &semion_category, DOC(lasyten, semion_category));
    m.def("double_semion_category", &double_semion_category, DOC(lasyten, double_semion_category));
}

} // namespace lasyten
