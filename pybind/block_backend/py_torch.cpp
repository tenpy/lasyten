#include "../doc_plus.h"
#include "docstrings/block_backend/torch.h"
// NOTE: this file is #included from py_block_backend.cpp

#include "tools/hdf5_bind.h"
#include <lasyten/block_backend/torch.h>

namespace lasyten {

void
bind_block_backend_torch(nb::module_& m)
{
    nb::class_<TorchBlockBackend, BlockBackend> torch_block_backend(m, "TorchBlockBackend");
    torch_block_backend.doc() = DOC(lasyten, TorchBlockBackend);
    torch_block_backend.def_static(
      "from_factory",
      &TorchBlockBackend::from_factory,
      nb::arg("device") = "cpu:0",
      nb::rv_policy::reference,
      "Get the backend instance for the given device (nearly-singleton per device).");
    torch_block_backend.def_static("from_hdf5",
                                   lasyten::hdf5::wrap_from_hdf5<TorchBlockBackend>(),
                                   nb::arg("hdf5_loader"),
                                   nb::arg("h5gr"),
                                   nb::arg("subpath"),
                                   "Load a TorchBlockBackend from an HDF5 file.");

    nb::class_<TorchBlockBackend::Block, BlockBackend::Block>(
      torch_block_backend, "BlockCls", "Block that holds a torch::Tensor.")
      .def("to_numpy",
           nb::overload_cast<>(&TorchBlockBackend::Block::to_numpy, nb::const_),
           nb::rv_policy::reference_internal)
      .def("to_numpy",
           nb::overload_cast<Dtype>(&TorchBlockBackend::Block::to_numpy, nb::const_),
           nb::arg("dtype"),
           nb::rv_policy::reference_internal)
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5<TorchBlockBackend::Block>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"),
           "Save block (via numpy) to HDF5.")
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<TorchBlockBackend::Block>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"),
                  "Load block from HDF5.");
}

} // namespace lasyten
