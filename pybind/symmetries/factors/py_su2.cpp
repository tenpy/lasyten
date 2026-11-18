#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/su2.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/su2.h>

#include "tools/hdf5_bind.h"
#include "tools/pickle_bind.h"
#include <optional>
#include <string>

namespace lasyten {

namespace {

constexpr char const* kSu2TestOnlyWarning =
  "_SU2 is a test-only API for symbol checks; not for production. "
  "Prefer SUN(N=2, ...) for real SU(2).";

} // namespace

void
bind_su2(nb::module_& m)
{
    nb::class_<_SU2, Group> cls(m, "_SU2", DOC(lasyten, _SU2));

    cls
      .def(nb::new_([](std::optional<std::string> descriptive_name) {
               py::module_::import("warnings")
                 .attr("warn")(
                   kSu2TestOnlyWarning, py::module_::import("builtins").attr("UserWarning"), 2);
               return std::make_shared<_SU2>(std::move(descriptive_name));
           }),
           nb::arg("descriptive_name").none() = nb::none())
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<_SU2>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    // Class-level convenience sectors (match Python ``_SU2.spin_half`` etc.).
    cls.attr("spin_zero") = _SU2::spin_zero;
    cls.attr("spin_half") = _SU2::spin_half;
    cls.attr("spin_one") = _SU2::spin_one;

    lasyten::pickle::def_reduce_ctor(
      cls, [](_SU2 const& self) { return nb::make_tuple(self.descriptive_name); });
}

} // namespace lasyten
