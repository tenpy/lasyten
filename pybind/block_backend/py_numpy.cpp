#include "../doc_plus.h"
#include "docstrings/block_backend/numpy.h"
// NOTE: this file is #included from py_block_backend.cpp

#include "ndarray_dispatch.h"
#include "py_trampolines.hpp"
#include "tools/hdf5_bind.h"
#include <lasyten/block_backend/numpy.h>

namespace lasyten {

void
bind_block_backend_numpy(nb::module_& m)
{

    nb::class_<NumpyBlockBackend, BlockBackend> numpy_block_backend(m, "NumpyBlockBackend");
    numpy_block_backend.doc() = DOC(lasyten, NumpyBlockBackend);
    numpy_block_backend.def_static(
      "from_factory",
      &NumpyBlockBackend::from_factory,
      nb::arg("device") = "cpu",
      nb::rv_policy::reference,
      "Get the backend instance for the given device (nearly-singleton per device).");
    numpy_block_backend.def_static("from_hdf5",
                                   lasyten::hdf5::wrap_from_hdf5<NumpyBlockBackend>(),
                                   nb::arg("hdf5_loader"),
                                   nb::arg("h5gr"),
                                   nb::arg("subpath"),
                                   "Load a NumpyBlockBackend from an HDF5 file.");
    numpy_block_backend.def(
      "block_from_ndarray",
      [](NumpyBlockBackend& self,
         eigen_ndarray::AnyDeviceArray arr,
         std::optional<Dtype> dtype,
         std::optional<std::string> device) {
          return ndarray_dispatch::block_from_ndarray(self, arr, dtype, std::move(device));
      },
      nb::arg("arr"),
      nb::arg("dtype") = nb::none(),
      nb::arg("device") = nb::none(),
      "Import a CPU array into a Numpy block (zero-copy when already a NumPy array).");
    numpy_block_backend.def(
      "to_ndarray",
      [](NumpyBlockBackend& self, nb::object block, bool writable) {
          auto& b = nb::cast<BlockBackend::Block&>(block);
          return ndarray_dispatch::block_to_ndarray(self, b, block, writable);
      },
      nb::arg("block"),
      nb::arg("writable") = true,
      "Zero-copy NumPy view of the block (writable=False clears the writeable flag).");

    nb::class_<NumpyBlockBackend::Block, BlockBackend::Block>(
      numpy_block_backend, "BlockCls", "Block that holds a numpy array in a py::object.")
      .def(lasyten::ctor_for<NumpyBlockBackend::Block, py::array>(), nb::arg("arr"))
      .def(
        "to_ndarray",
        [](nb::object self, bool writable) {
            auto& np = nb::cast<NumpyBlockBackend::Block&>(self);
            return ndarray_dispatch::numpy_array_as_ndarray(np.to_numpy(), writable);
        },
        nb::arg("writable") = true,
        "Zero-copy NumPy view of the stored array.")
      .def("to_numpy",
           nb::overload_cast<>(&NumpyBlockBackend::Block::to_numpy, nb::const_),
           nb::rv_policy::reference_internal)
      .def("to_numpy",
           nb::overload_cast<Dtype>(&NumpyBlockBackend::Block::to_numpy, nb::const_),
           nb::arg("dtype"),
           nb::rv_policy::reference_internal)
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5<NumpyBlockBackend::Block>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"),
           "Save block (numpy array) to HDF5.")
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<NumpyBlockBackend::Block>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"),
                  "Load block from HDF5.");
    // NOTE: don't immplment __array__ since we don't want to allow Blocks to cast to numpy arrays
    // except via NOTE: no __mul__, __add__ etc since we have Block*Scalar defined in
    // BlockBackend.cpp
}

} // namespace lasyten
