#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/fibonacci_anyon_category.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/fibonacci_anyon_category.h>

#include "tools/hdf5_bind.h"
#include <string>

namespace lasyten {

void
bind_fibonacci_anyon_category(nb::module_& m)
{
    nb::class_<FibonacciAnyonCategory, SymmetryFactor> cls(
      m, "FibonacciAnyonCategory", DOC(lasyten, FibonacciAnyonCategory));
    cls.def(lasyten::ctor<decltype(cls), std::string>(), nb::arg("handedness") = "left")
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<FibonacciAnyonCategory>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
    cls.attr("vacuum") = FibonacciAnyonCategory::vacuum;
    cls.attr("tau") = FibonacciAnyonCategory::tau;
    cls.def_ro("handedness", &FibonacciAnyonCategory::handedness);
}

} // namespace lasyten
