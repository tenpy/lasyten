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
    // Prefer these over to_numpy / block_from_numpy at new call sites (leftover #10).
    eigen_block_backend.def(
      "block_from_ndarray",
      [](EigenBlockBackend& self, eigen_ndarray::AnyDeviceArray arr, std::optional<Dtype> dtype) {
          return eigen_ndarray::block_from_ndarray(self, arr, dtype);
      },
      nb::arg("arr"),
      nb::arg("dtype") = nb::none(),
      "Copy a CPU array (NumPy, PyTorch, anything supporting DLPack or the buffer protocol) into "
      "a new block, optionally converting to `dtype`.");
    eigen_block_backend.def(
      "to_ndarray",
      [](EigenBlockBackend&, nb::object block, bool writable, std::optional<std::string> framework)
        -> nb::object {
          auto& b = nb::cast<EigenBlockBackend::Block&>(block);
          auto const fw =
            eigen_ndarray::parse_nd_framework(framework, eigen_ndarray::NdFramework::Numpy);
          return eigen_ndarray::block_to_ndarray_framework(b, block, writable, fw);
      },
      nb::arg("block"),
      nb::arg("writable") = true,
      nb::arg("framework") = nb::none(),
      "Zero-copy array view of the block's buffer (keeps the block alive). "
      "framework=None|'numpy'|'torch'|'jax'|'tensorflow' selects the DLPack consumer tag. "
      "writable=True (default) aliases storage; writable=False returns a read-only view. "
      "Contrast with to_numpy(), which always returns an owning NumPy copy.");

    nb::class_<EigenBlockBackend::Block, BlockBackend::Block>(
      eigen_block_backend, "BlockCls", "Block that owns a contiguous C-order buffer.")
      .def(lasyten::ctor_for<EigenBlockBackend::Block, py::array>(),
           nb::arg("arr"),
           "Construct from a NumPy array (owning copy). Prefer "
           "EigenBlockBackend.block_from_ndarray for Torch/DLPack/buffer inputs.")
      .def(
        "to_ndarray",
        [](nb::object self, bool writable, std::optional<std::string> framework) -> nb::object {
            auto& b = nb::cast<EigenBlockBackend::Block&>(self);
            auto const fw =
              eigen_ndarray::parse_nd_framework(framework, eigen_ndarray::NdFramework::Numpy);
            return eigen_ndarray::block_to_ndarray_framework(b, self, writable, fw);
        },
        nb::arg("writable") = true,
        nb::arg("framework") = nb::none(),
        "Zero-copy array view (framework tag: numpy/torch/jax/tensorflow). "
        "writable=False returns a read-only view. to_numpy() always copies.")
      .def("to_numpy",
           nb::overload_cast<>(&EigenBlockBackend::Block::to_numpy, nb::const_),
           nb::rv_policy::reference_internal,
           "Owning NumPy copy of the block (never aliases Eigen storage).")
      .def("to_numpy",
           nb::overload_cast<Dtype>(&EigenBlockBackend::Block::to_numpy, nb::const_),
           nb::arg("dtype"),
           nb::rv_policy::reference_internal,
           "Owning NumPy copy converted to dtype.")
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
