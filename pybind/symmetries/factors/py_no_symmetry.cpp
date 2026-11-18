#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/no_symmetry.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include "tools/hdf5_bind.h"
#include "tools/pickle_bind.h"
#include <lasyten/symmetries/factors/no_symmetry.h>

namespace lasyten {

void
bind_no_symmetry(nb::module_& m)
{
    nb::class_<NoSymmetry, AbelianGroup> cls(m, "NoSymmetry", DOC(lasyten, NoSymmetry));
    cls.def(lasyten::ctor_for<NoSymmetry>())
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<NoSymmetry>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    // Pickle: NoSymmetry has no state; unpickling calls ``NoSymmetry()``.
    lasyten::pickle::def_reduce_ctor(cls, [](NoSymmetry const&) { return nb::tuple(); });
}

} // namespace lasyten
