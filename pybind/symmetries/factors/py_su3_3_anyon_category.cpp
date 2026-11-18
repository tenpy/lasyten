#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/su3_3_anyon_category.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include "tools/hdf5_bind.h"
#include "tools/pickle_bind.h"
#include <lasyten/symmetries/factors/su3_3_anyon_category.h>

namespace lasyten {

void
bind_su3_3_anyon_category(nb::module_& m)
{
    nb::class_<SU3_3AnyonCategory, SymmetryFactor> cls(
      m, "SU3_3AnyonCategory", DOC(lasyten, SU3_3AnyonCategory));
    cls.def(lasyten::ctor<decltype(cls)>())
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<SU3_3AnyonCategory>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
    cls.attr("one_irrep") = SU3_3AnyonCategory::one_irrep;
    cls.attr("eight_irrep") = SU3_3AnyonCategory::eight_irrep;
    cls.attr("ten_irrep") = SU3_3AnyonCategory::ten_irrep;
    cls.attr("ten_bar_irrep") = SU3_3AnyonCategory::ten_bar_irrep;

    // Pickle state mirrors ``save_hdf5``: the constructor fields plus ``descriptive_name`` (which
    // the C++ constructor does not take, but HDF5 restores as an attribute).
    lasyten::pickle::def_state_pickle(
      cls,
      [](SU3_3AnyonCategory const& self) {
          nb::dict d;
          d["descriptive_name"] = self.descriptive_name;
          return d;
      },
      [](nb::dict state) {
          auto obj = std::make_shared<SU3_3AnyonCategory>();
          obj->descriptive_name =
            lasyten::pickle::get_item<std::optional<std::string>>(state, "descriptive_name");
          return obj;
      });
}

} // namespace lasyten
