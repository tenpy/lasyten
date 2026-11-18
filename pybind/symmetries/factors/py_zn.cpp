#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/zn.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/zn.h>

#include "tools/hdf5_bind.h"
#include <optional>
#include <string>

namespace lasyten {

void
bind_zn(nb::module_& m)
{
    nb::class_<ZN, AbelianGroup>(m, "ZN", DOC(lasyten, ZN))
      .def(lasyten::ctor_for<ZN, int, std::optional<std::string>, bool>(),
           nb::arg("N"),
           nb::arg("descriptive_name").none() = nb::none(),
           nb::arg("trivial_shift") = true)
      .def_ro("N", &ZN::N)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<ZN>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
}

} // namespace lasyten
