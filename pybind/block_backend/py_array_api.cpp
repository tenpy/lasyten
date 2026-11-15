#include "../doc_plus.h"
#include "docstrings/block_backend/array_api.h"
// NOTE: this file is #included from py_block_backend.cpp

#include "py_trampolines.hpp"
#include "tools/hdf5_bind.h"
#include <lasyten/block_backend/array_api.h>

namespace lasyten {

void
bind_block_backend_array_api(nb::module_& m)
{
    nb::class_<ArrayApiBlockBackend, BlockBackend> array_api_block_backend(m,
                                                                           "ArrayApiBlockBackend");
    array_api_block_backend.doc() = DOC(lasyten, ArrayApiBlockBackend);
    array_api_block_backend.def(
      lasyten::ctor<decltype(array_api_block_backend), py::object, std::string>(),
      nb::arg("api_namespace"),
      nb::arg("default_device") = "cpu");
    array_api_block_backend.def_prop_ro(
      "api", &ArrayApiBlockBackend::api, "The Array API namespace this backend dispatches to.");
    array_api_block_backend.def_static(
      "from_hdf5",
      lasyten::hdf5::wrap_from_hdf5<ArrayApiBlockBackend>(),
      nb::arg("hdf5_loader"),
      nb::arg("h5gr"),
      nb::arg("subpath"),
      "Load an ArrayApiBlockBackend from an HDF5 file (not generally supported).");

    nb::class_<ArrayApiBlockBackend::Block, BlockBackend::Block>(
      array_api_block_backend,
      "BlockCls",
      "Block that holds an Array-API array as a Python object.")
      .def("to_numpy", nb::overload_cast<>(&ArrayApiBlockBackend::Block::to_numpy, nb::const_))
      .def("to_numpy",
           nb::overload_cast<Dtype>(&ArrayApiBlockBackend::Block::to_numpy, nb::const_),
           nb::arg("dtype"))
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5<ArrayApiBlockBackend::Block>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"),
           "Save block to HDF5 via numpy conversion.")
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<ArrayApiBlockBackend::Block>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"),
                  "Load block from HDF5 (requires Array API context).");
}

} // namespace lasyten
