#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "docstrings/backends/backend_factory.h"

#include <lasyten/backends/backend_factory.h>

namespace lasyten {

void
bind_backend_factory(nb::module_& m)
{
    m.def("get_backend",
          nb::overload_cast<py::object, py::object>(&get_backend),
          nb::arg("symmetry").none() = nb::none(),
          nb::arg("block_backend").none() = nb::none(),
          DOC(lasyten, get_backend));
}

} // namespace lasyten
