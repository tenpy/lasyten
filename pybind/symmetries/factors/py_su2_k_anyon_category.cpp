#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/su2_k_anyon_category.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/su2_k_anyon_category.h>

#include "tools/hdf5_bind.h"
#include <string>

namespace lasyten {

void
bind_su2_k_anyon_category(nb::module_& m)
{
    nb::class_<SU2_kAnyonCategory, SymmetryFactor> cls(
      m, "SU2_kAnyonCategory", DOC(lasyten, SU2_kAnyonCategory));
    cls
      .def(lasyten::ctor<decltype(cls), int, std::string>(),
           nb::arg("k"),
           nb::arg("handedness") = "left")
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<SU2_kAnyonCategory>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
    cls.attr("spin_zero") = SU2_kAnyonCategory::spin_zero;
    cls.attr("spin_half") = SU2_kAnyonCategory::spin_half;
    cls.def_ro("k", &SU2_kAnyonCategory::k)
      .def_ro("handedness", &SU2_kAnyonCategory::handedness)
      .def_prop_ro("spin_one", [](SU2_kAnyonCategory const& self) -> py::object {
          if (!self.spin_one.has_value()) {
              return py::none();
          }
          return py::cast(*self.spin_one);
      });
}

} // namespace lasyten
