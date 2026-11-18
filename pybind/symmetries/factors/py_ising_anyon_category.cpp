#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/ising_anyon_category.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include "tools/hdf5_bind.h"
#include "tools/pickle_bind.h"
#include <lasyten/symmetries/factors/ising_anyon_category.h>

namespace lasyten {

void
bind_ising_anyon_category(nb::module_& m)
{
    nb::class_<IsingAnyonCategory, SymmetryFactor> cls(
      m, "IsingAnyonCategory", DOC(lasyten, IsingAnyonCategory));
    cls.def(lasyten::ctor<decltype(cls), int>(), nb::arg("nu") = 1)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<IsingAnyonCategory>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
    cls.attr("vacuum") = IsingAnyonCategory::vacuum;
    cls.attr("sigma") = IsingAnyonCategory::sigma;
    cls.attr("psi") = IsingAnyonCategory::psi;
    cls.def_ro("nu", &IsingAnyonCategory::nu);

    // Pickle state mirrors ``save_hdf5``: the constructor fields plus ``descriptive_name`` (which
    // the C++ constructor does not take, but HDF5 restores as an attribute).
    lasyten::pickle::def_state_pickle(
      cls,
      [](IsingAnyonCategory const& self) {
          nb::dict d;
          d["nu"] = self.nu;
          d["descriptive_name"] = self.descriptive_name;
          return d;
      },
      [](nb::dict state) {
          auto obj =
            std::make_shared<IsingAnyonCategory>(lasyten::pickle::get_item<int>(state, "nu"));
          obj->descriptive_name =
            lasyten::pickle::get_item<std::optional<std::string>>(state, "descriptive_name");
          return obj;
      });
}

} // namespace lasyten
