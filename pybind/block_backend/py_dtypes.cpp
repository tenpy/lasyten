#include "../doc_plus.h"
#include "docstrings/block_backend/dtypes.h"
// NOTE: this file is #included from block_backend.cpp

#include <lasyten/block_backend/dtypes.h>

#include <string>
#include <type_traits>
#include <vector>

namespace lasyten {

void
bind_block_backend_dtypes(nb::module_& m)
{
    // nb::enum_ creates a Python enum.Enum and (unlike py::native_enum) still allows .def(),
    // so the methods / properties are attached directly; no monkey-patching of the Python class.
    nb::enum_<Dtype> D(m, "Dtype", DOC(lasyten, Dtype));
    D.value("bool", Dtype::Bool)
      .value("float32", Dtype::Float32)
      .value("complex64", Dtype::Complex64)
      .value("float64", Dtype::Float64)
      .value("complex128", Dtype::Complex128)
      .value("int64", Dtype::Int64)
      .export_values();

    // Properties
    D.def_prop_ro("is_real", [](Dtype d) { return dtype::is_real(d); })
      .def_prop_ro("is_complex", [](Dtype d) { return dtype::is_complex(d); })
      .def_prop_ro("to_complex", [](Dtype d) { return dtype::to_complex(d); })
      .def_prop_ro("to_real", [](Dtype d) { return dtype::to_real(d); })
      .def_prop_ro("python_type", [](Dtype d) { return dtype::python_type(d); })
      .def_prop_ro("zero_scalar", [](Dtype d) { return dtype::zero_scalar(d); })
      .def_prop_ro("one_scalar", [](Dtype d) { return dtype::one_scalar(d); })
      .def_prop_ro("eps", [](Dtype d) { return dtype::eps(d); });

    // Methods
    D.def("to_numpy_dtype", [](Dtype d) { return dtype::to_numpy_dtype(d); });

    D.def(
      "convert_python_scalar",
      [](Dtype d, py::object value) { return dtype::convert_python_scalar(d, std::move(value)); },
      nb::arg("value"));

    // Supports both Dtype.common(a, b) and a.common(b) / a.common(b, c, ...).
    D.def("common", [](Dtype first, nb::args rest_nb) {
        auto rest = pyargs(rest_nb);
        std::vector<Dtype> dtypes;
        dtypes.reserve(1 + rest.size());
        dtypes.push_back(first);
        for (py::handle h : rest)
            dtypes.push_back(py::cast<Dtype>(h));
        return dtype::common(dtypes);
    });

    D.def_static(
      "from_numpy_dtype",
      [](py::object numpy_dtype) { return dtype::from_numpy_dtype(std::move(numpy_dtype)); },
      nb::arg("dtype"));

    D.def("__repr__", [](Dtype d) { return std::string("Dtype.") + dtype::repr(d); });

    D.def(
      "save_hdf5",
      [](Dtype self, py::object hdf5_saver, py::object /*h5gr*/, const std::string& subpath) {
          using Underlying = std::underlying_type_t<Dtype>;
          hdf5_saver.attr("save")(static_cast<Underlying>(self), subpath + "value");
      },
      nb::arg("hdf5_saver"),
      nb::arg("h5gr"),
      nb::arg("subpath"),
      "Export a Dtype enum member for lasyten.tools.hdf5_io");

    D.def_static(
      "from_hdf5",
      [](py::object hdf5_loader, py::object h5gr, const std::string& subpath) {
          py::object value = hdf5_loader.attr("load")(subpath + "value");
          nb::object obj = nb::type<Dtype>()(nb::handle(value.ptr()));
          hdf5_loader.attr("memorize_load")(h5gr, py::reinterpret_borrow<py::object>(obj.ptr()));
          return obj;
      },
      nb::arg("hdf5_loader"),
      nb::arg("h5gr"),
      nb::arg("subpath"),
      "Reconstruct a Dtype enum member from HDF5");
}

} // namespace lasyten
