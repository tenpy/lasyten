#include "../doc_plus.h"
#include "docstrings/symmetries/exceptions.h"
#include "py_lasyten_nanobind.h"

#include <lasyten/symmetries/exceptions.h>

namespace lasyten {

void
bind_symmetries_exceptions(nb::module_& m)
{
    // nb::exception registers the C++ -> Python translator itself; docstrings are attached after
    // registration.
    static nb::exception<SymmetryError> symmetry_error(m, "SymmetryError", PyExc_Exception);
    symmetry_error.doc() = DOC(lasyten, SymmetryError);

    static nb::exception<BraidChiralityUnspecifiedError> braid_err(
      m, "BraidChiralityUnspecifiedError", symmetry_error);
    braid_err.doc() = DOC(lasyten, BraidChiralityUnspecifiedError);
}

} // namespace lasyten
