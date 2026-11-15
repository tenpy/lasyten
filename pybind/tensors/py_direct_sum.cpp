#include <lasyten/tensors/direct_sum.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tensors/tensor.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"

#include "docstrings/tensors/direct_sum.h"

#include <sstream>
#include <string>
#include <vector>

namespace lasyten {

void
bind_tensors_direct_sum(nb::module_& m)
{
    nb::class_<DirectSum, VectorLike> cls(m, "DirectSum");
    cls.doc() = DOC(lasyten, DirectSum);

    cls.def(lasyten::ctor<decltype(cls), std::vector<TensorPtr>>(),
            nb::arg("components"),
            DOC(lasyten, DirectSum, DirectSum));

    cls.def_prop_ro(
      "components",
      [](DirectSum const& self) { return self.components(); },
      DOC(lasyten, DirectSum, components));
    cls.def_prop_ro("dtype", &DirectSum::dtype);
    cls.def_prop_ro("device", &DirectSum::device);
    cls.def_prop_ro("backend", &DirectSum::backend);

    cls.def("copy", &DirectSum::copy, nb::arg("deep") = true, DOC(lasyten, DirectSum, copy));
    cls.def("__len__", &DirectSum::size);
    cls.def(
      "__getitem__", [](DirectSum const& self, int64 i) { return self.at(i); }, nb::arg("i"));
    cls.def("__repr__", [](DirectSum const& self) {
        std::ostringstream os;
        os << "<DirectSum of " << self.size() << " tensors, dtype=" << dtype::repr(self.dtype())
           << '>';
        return os.str();
    });
    // DirectSum is a sequence (__len__/__getitem__), so numpy would otherwise treat
    // ``np.float64 * ds`` as iterating the components into an object array.
    cls.attr("__array_ufunc__") = py::none();
}

} // namespace lasyten
