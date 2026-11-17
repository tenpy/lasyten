#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/no_symmetry.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include "tools/hdf5_bind.h"
#include <lasyten/symmetries/factors/no_symmetry.h>

namespace lasyten {

void
bind_no_symmetry(nb::module_& m)
{
    nb::class_<NoSymmetry, AbelianGroup>(m, "NoSymmetry", DOC(lasyten, NoSymmetry))
      .def(lasyten::ctor_for<NoSymmetry>())
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<NoSymmetry>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
}

} // namespace lasyten
