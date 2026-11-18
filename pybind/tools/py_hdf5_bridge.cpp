#include "py_hdf5_bridge.h"

#include <hdf5.h>
#include <tenpy_hdf5/exceptions.h>
#include <tenpy_hdf5/h5_ops.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace lasyten::hdf5 {

namespace {

void
check_hdf5(herr_t status, char const* what)
{
    if (status < 0)
        throw ::tenpy_hdf5::Hdf5ExportError(std::string("HDF5 error: ") + what);
}

} // namespace

hid_t
hid_from_h5py(py::handle obj)
{
    return py::cast<hid_t>(obj.attr("id").attr("id"));
}

HighFive::Group
wrap_group(py::handle obj)
{
    auto h5py = py::module_::import("h5py");
    if (py::isinstance(obj, h5py.attr("File"))) {
        hid_t fid = hid_from_h5py(obj);
        hid_t gid = H5Gopen2(fid, "/", H5P_DEFAULT);
        if (gid < 0)
            throw ::tenpy_hdf5::Hdf5ExportError("failed to open root group");
        return HighFive::detail::make_group(gid);
    }
    hid_t hid = hid_from_h5py(obj);
    check_hdf5(H5Iinc_ref(hid), "H5Iinc_ref");
    return HighFive::detail::make_group(hid);
}

Saver
saver_from_python(py::object hdf5_saver)
{
    return Saver(wrap_group(hdf5_saver.attr("h5group")));
}

Loader
loader_from_python(py::object hdf5_loader)
{
    bool ignore = true;
    if (py::hasattr(hdf5_loader, "ignore_unknown"))
        ignore = py::to_bool(hdf5_loader.attr("ignore_unknown"));
    return Loader(wrap_group(hdf5_loader.attr("h5group")), ignore);
}

::tenpy_hdf5::Hdf5Buffer
buffer_from_numpy(py::array arr)
{
    if (!py::to_bool(arr.attr("flags").attr("c_contiguous")))
        arr = py::reinterpret_steal<py::array>(arr.attr("copy")("C").release());
    ::tenpy_hdf5::Hdf5Buffer buf;
    py::dtype dt = arr.dtype();
    buf.kind = dt.kind();
    buf.itemsize = static_cast<int>(dt.itemsize());
    if (buf.kind == 'b' || (buf.kind == 'i' && py::cast<std::string>(dt.attr("name")) == "bool")) {
        buf.kind = 'b';
        buf.itemsize = 1;
    }
    buf.shape.clear();
    if (arr.ndim() > 0) {
        buf.shape.resize(static_cast<size_t>(arr.ndim()));
        for (py::ssize_t i = 0; i < arr.ndim(); ++i)
            buf.shape[static_cast<size_t>(i)] = static_cast<std::size_t>(arr.shape(i));
    }
    std::size_t nbytes = buf.nbytes();
    buf.storage = std::shared_ptr<std::uint8_t[]>(new std::uint8_t[nbytes ? nbytes : 1]);
    if (nbytes > 0)
        std::memcpy(buf.storage.get(), arr.data(), nbytes);
    return buf;
}

py::array
numpy_from_buffer(::tenpy_hdf5::Hdf5Buffer const& buf)
{
    std::string descr;
    if (buf.kind == 'f' && buf.itemsize == 4)
        descr = "float32";
    else if (buf.kind == 'f' && buf.itemsize == 8)
        descr = "float64";
    else if (buf.kind == 'c' && buf.itemsize == 8)
        descr = "complex64";
    else if (buf.kind == 'c' && buf.itemsize == 16)
        descr = "complex128";
    else if (buf.kind == 'i' && buf.itemsize == 1)
        descr = "int8";
    else if (buf.kind == 'i' && buf.itemsize == 2)
        descr = "int16";
    else if (buf.kind == 'i' && buf.itemsize == 4)
        descr = "int32";
    else if (buf.kind == 'i' && buf.itemsize == 8)
        descr = "int64";
    else if (buf.kind == 'u' && buf.itemsize == 1)
        descr = "uint8";
    else if (buf.kind == 'u' && buf.itemsize == 2)
        descr = "uint16";
    else if (buf.kind == 'u' && buf.itemsize == 4)
        descr = "uint32";
    else if (buf.kind == 'u' && buf.itemsize == 8)
        descr = "uint64";
    else if (buf.kind == 'b')
        descr = "bool";
    else
        throw std::runtime_error("unsupported Hdf5Buffer dtype for numpy");

    std::vector<py::ssize_t> shape(buf.shape.begin(), buf.shape.end());
    py::array arr(py::dtype(descr), shape);
    std::size_t nbytes = buf.nbytes();
    if (nbytes > 0 && buf.data())
        std::memcpy(arr.mutable_data(), buf.data(), nbytes);
    return arr;
}

} // namespace lasyten::hdf5
