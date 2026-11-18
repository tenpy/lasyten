#include "../doc_plus.h"
#include "docstrings/symmetries/group.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <lasyten/symmetries/group.h>

#include <optional>
#include <string>

namespace lasyten {

void
bind_group(nb::module_& m)
{
    nb::class_<Group, SymmetryFactor> cls(m, "Group", DOC(lasyten, Group));

    cls.def(lasyten::ctor<decltype(cls),
                          FusionStyle,
                          Sector,
                          std::string,
                          float64,
                          bool,
                          std::optional<std::string>,
                          bool>(),
            nb::arg("fusion_style"),
            nb::arg("trivial_sector"),
            nb::arg("group_name"),
            nb::arg("num_sectors"),
            nb::arg("has_complex_topological_data"),
            nb::arg("descriptive_name").none() = nb::none(),
            nb::arg("trivial_shift") = true);

    // Group defaults; subclasses may still override via trampoline.
    cls
      .def("_fusion_tensor",
           &Group::_fusion_tensor,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           nb::arg("Z_a"),
           nb::arg("Z_b"),
           DOC(lasyten, Group, _fusion_tensor))
      .def("swap_gate",
           &Group::swap_gate,
           nb::arg("a"),
           nb::arg("b"),
           doc_cpp_ref(R"pydoc(swap_gate)pydoc", "lasyten::BaseSymmetry::swap_gate()"))
      .def("qdim",
           &Group::qdim,
           nb::arg("a"),
           doc_cpp_ref(R"pydoc(qdim)pydoc", "lasyten::BaseSymmetry::qdim()"))
      .def(
        "batch_qdim",
        [](Group const& self, SectorArray const& a) {
            return vector_f64_to_numpy(self.batch_qdim(a));
        },
        nb::arg("a"),
        DOC(lasyten, Group, batch_qdim))
      .def("topological_twist",
           &Group::topological_twist,
           nb::arg("a"),
           DOC(lasyten, Group, topological_twist));
}

} // namespace lasyten
