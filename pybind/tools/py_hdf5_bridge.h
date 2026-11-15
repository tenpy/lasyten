#pragma once

#include <lasyten/lasyten.h>
#include <lasyten/tools/hdf5.h>

#include <highfive/highfive.hpp>
#include <string>
#include <utility>

namespace lasyten::hdf5 {

namespace py = lasyten::py;

/// h5py File/Group/Dataset → hid_t (``obj.id.id``).
hid_t hid_from_h5py(py::handle obj);

/// Borrow h5py File/Group as HighFive::Group (increments HDF5 refcount).
HighFive::Group wrap_group(py::handle obj);

/// Build a core Saver from a Python ``Hdf5Saver`` (uses ``saver.h5group``).
Saver saver_from_python(py::object hdf5_saver);

/// Build a core Loader from a Python ``Hdf5Loader`` (uses ``loader.h5group``).
Loader loader_from_python(py::object hdf5_loader);

/// Convert a contiguous numpy array into an owned ``Hdf5Buffer``.
::hdf5_io::Hdf5Buffer buffer_from_numpy(py::array arr);

/// Convert an ``Hdf5Buffer`` back to a numpy array (copies).
py::array numpy_from_buffer(::hdf5_io::Hdf5Buffer const& buf);

} // namespace lasyten::hdf5
