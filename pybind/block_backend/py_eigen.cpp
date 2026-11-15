#include "../doc_plus.h"
#include "docstrings/block_backend/eigen.h"
// NOTE: this file is #included from py_block_backend.cpp

#include "eigen_ndarray.h"
#include "py_trampolines.hpp"
#include "tools/hdf5_bind.h"
#include <lasyten/block_backend/eigen.h>
#include <nanobind/stl/optional.h>

namespace lasyten {

void
bind_block_backend_eigen(nb::module_& m)
{
    nb::class_<EigenBlockBackend, BlockBackend> eigen_block_backend(m, "EigenBlockBackend");
    eigen_block_backend.doc() = DOC(lasyten, EigenBlockBackend);
    eigen_block_backend.def_static(
      "from_factory",
      &EigenBlockBackend::from_factory,
      nb::arg("device") = "cpu",
      nb::rv_policy::reference,
      "Get the backend instance for the given device (\"cpu\" or \"eigen\"; one per device).");
    eigen_block_backend.def_static("from_hdf5",
                                   lasyten::hdf5::wrap_from_hdf5<EigenBlockBackend>(),
                                   nb::arg("hdf5_loader"),
                                   nb::arg("h5gr"),
                                   nb::arg("subpath"),
                                   "Load an EigenBlockBackend from an HDF5 file.");
    eigen_block_backend.def_static("seed",
                                   &EigenBlockBackend::seed,
                                   nb::arg("seed"),
                                   "Seed the random number generator of the Eigen backend.");

    // nb::ndarray boundary (see eigen_ndarray.h): zero-copy export, copying import.
    eigen_block_backend.def(
      "block_from_ndarray",
      [](EigenBlockBackend& self, eigen_ndarray::AnyCpuArray arr, std::optional<Dtype> dtype) {
          return eigen_ndarray::block_from_ndarray(self, arr, dtype);
      },
      nb::arg("arr"),
      nb::arg("dtype") = nb::none(),
      "Copy a CPU array (NumPy, PyTorch, anything supporting DLPack or the buffer protocol) into "
      "a new block, optionally converting to `dtype`.");
    eigen_block_backend.def(
      "to_ndarray",
      [](EigenBlockBackend&, nb::object block) {
          auto& b = nb::cast<EigenBlockBackend::Block&>(block);
          return eigen_ndarray::block_to_ndarray(b, block);
      },
      nb::arg("block"),
      "Zero-copy writable NumPy view of the block's buffer (keeps the block alive).");

    nb::class_<EigenBlockBackend::Block, BlockBackend::Block>(
      eigen_block_backend, "BlockCls", "Block that owns a contiguous C-order buffer.")
      .def(lasyten::ctor_for<EigenBlockBackend::Block, py::array>(), nb::arg("arr"))
      .def(
        "to_ndarray",
        [](nb::object self) {
            auto& b = nb::cast<EigenBlockBackend::Block&>(self);
            return eigen_ndarray::block_to_ndarray(b, self);
        },
        "Zero-copy writable NumPy view of the block's buffer (keeps the block alive).")
      .def("to_numpy",
           nb::overload_cast<>(&EigenBlockBackend::Block::to_numpy, nb::const_),
           nb::rv_policy::reference_internal)
      .def("to_numpy",
           nb::overload_cast<Dtype>(&EigenBlockBackend::Block::to_numpy, nb::const_),
           nb::arg("dtype"),
           nb::rv_policy::reference_internal)
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5<EigenBlockBackend::Block>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"),
           "Save block (via numpy) to HDF5.")
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<EigenBlockBackend::Block>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"),
                  "Load block from HDF5.");
}

} // namespace lasyten
