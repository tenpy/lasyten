#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/zn_anyon_category.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/zn_anyon_category.h>

#include "tools/hdf5_bind.h"
#include <optional>
#include <string>

namespace lasyten {

void
bind_zn_anyon_category(nb::module_& m)
{
    nb::class_<ZNAnyonCategory, SymmetryFactor>(
      m, "ZNAnyonCategory", DOC(lasyten, ZNAnyonCategory))
      .def(lasyten::ctor_for<ZNAnyonCategory, int, int, std::optional<std::string>>(),
           nb::arg("N"),
           nb::arg("n"),
           nb::arg("descriptive_name").none() = nb::none())
      .def_ro("N", &ZNAnyonCategory::N)
      .def_ro("n", &ZNAnyonCategory::n)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<ZNAnyonCategory>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
}

} // namespace lasyten
