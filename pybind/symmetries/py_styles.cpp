#include "../doc_plus.h"
#include "docstrings/symmetries/styles.h"
#include "py_lasyten_nanobind.h"

#include <lasyten/symmetries/styles.h>

namespace lasyten {

void
bind_symmetries_styles(nb::module_& m)
{
    // nb::is_arithmetic(): Python enum.IntEnum (as the former py::native_enum(...,
    // "enum.IntEnum")).
    nb::enum_<FusionStyle> fusion_enum(
      m, "FusionStyle", nb::is_arithmetic(), DOC(lasyten, FusionStyle));
    fusion_enum.value("single", FusionStyle::single)
      .value("multiple_unique", FusionStyle::multiple_unique)
      .value("general", FusionStyle::general)
      .export_values();

    nb::enum_<BraidingStyle> braid_enum(
      m, "BraidingStyle", nb::is_arithmetic(), DOC(lasyten, BraidingStyle));
    braid_enum.value("bosonic", BraidingStyle::bosonic)
      .value("fermionic", BraidingStyle::fermionic)
      .value("anyonic", BraidingStyle::anyonic)
      .value("no_braiding", BraidingStyle::no_braiding)
      .export_values();
}

} // namespace lasyten
