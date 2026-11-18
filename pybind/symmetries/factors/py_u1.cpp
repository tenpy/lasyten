#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/u1.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/u1.h>

#include "tools/hdf5_bind.h"
#include "tools/pickle_bind.h"
#include <optional>
#include <string>

namespace lasyten {

void
bind_u1(nb::module_& m)
{
    nb::class_<U1, AbelianGroup> cls(m, "U1", DOC(lasyten, U1));
    cls
      .def(lasyten::ctor_for<U1, std::optional<std::string>, bool>(),
           nb::arg("descriptive_name").none() = nb::none(),
           nb::arg("trivial_shift") = true)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<U1>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    // Pickle: constructor args = the logical fields that HDF5 persists.
    lasyten::pickle::def_reduce_ctor(cls, [](U1 const& self) {
        return nb::make_tuple(self.descriptive_name, self.trivial_shift);
    });
}

} // namespace lasyten
