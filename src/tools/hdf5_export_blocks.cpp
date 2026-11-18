// Python-free HDF5 export of block backends / blocks (split out of hdf5_export.cpp so that
// EigenBlockBackend can save/load without the tensor-level code, which still needs Python).

#include <lasyten/tools/hdf5_export.h>

#include <lasyten/block_backend/eigen.h>
#if LASYTEN_WITH_NUMPY_BACKEND
#include <lasyten/block_backend/array_api.h>
#include <lasyten/block_backend/numpy.h>
#endif
#if LASYTEN_WITH_TORCH_BACKEND
#include <lasyten/block_backend/torch.h>
#endif

#include <cstring>
#include <stdexcept>
#include <typeinfo>

namespace lasyten::hdf5_export {

std::string
class_attr(HighFive::Group& h5gr)
{
    auto cls = tenpy_hdf5::h5_get_attr_string(h5gr.getId(), tenpy_hdf5::ATTR_CLASS);
    if (!cls) {
        throw std::runtime_error("hdf5_export: missing class attribute");
    }
    return *cls;
}

#if LASYTEN_HAS_PYTHON
tenpy_hdf5::Hdf5Buffer
buffer_from_numpy(py::array arr)
{
    if (!py::to_bool(arr.attr("flags").attr("c_contiguous")))
        arr = py::reinterpret_steal<py::array>(arr.attr("copy")("C").release());
    tenpy_hdf5::Hdf5Buffer buf;
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
numpy_from_buffer(tenpy_hdf5::Hdf5Buffer const& buf)
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
#endif // LASYTEN_HAS_PYTHON

namespace {

char const*
block_class_name(BlockBackend::Block const& block)
{
    if (dynamic_cast<EigenBlockBackend::Block const*>(&block))
        return "EigenBlockBackend.BlockCls";
#if LASYTEN_WITH_NUMPY_BACKEND
    if (dynamic_cast<NumpyBlockBackend::Block const*>(&block))
        return "NumpyBlockBackend.BlockCls";
    if (dynamic_cast<ArrayApiBlockBackend::Block const*>(&block))
        return "ArrayApiBlockBackend.BlockCls";
#endif
#if LASYTEN_WITH_TORCH_BACKEND
    if (dynamic_cast<TorchBlockBackend::Block const*>(&block))
        return "TorchBlockBackend.BlockCls";
#endif
    throw std::runtime_error(std::string("hdf5_export: unknown Block type ") +
                             typeid(block).name());
}

char const*
block_backend_class_name(BlockBackend const& backend)
{
    if (dynamic_cast<EigenBlockBackend const*>(&backend))
        return "EigenBlockBackend";
#if LASYTEN_WITH_NUMPY_BACKEND
    if (dynamic_cast<NumpyBlockBackend const*>(&backend))
        return "NumpyBlockBackend";
    if (dynamic_cast<ArrayApiBlockBackend const*>(&backend))
        return "ArrayApiBlockBackend";
#endif
#if LASYTEN_WITH_TORCH_BACKEND
    if (dynamic_cast<TorchBlockBackend const*>(&backend))
        return "TorchBlockBackend";
#endif
    throw std::runtime_error(std::string("hdf5_export: unknown BlockBackend type ") +
                             typeid(backend).name());
}

} // namespace

void
save_block(lasyten::hdf5::Saver& saver, std::string const& path, BlockCPtr const& block)
{
    if (!block) {
        saver.save_none(path);
        return;
    }
    auto const* cls = block_class_name(*block);
    saver.save_instance(path,
                        kModule,
                        cls,
                        block.get(),
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            const_cast<BlockBackend::Block&>(*block).save_hdf5(s, g, sub);
                        });
}

BlockPtr
load_block(lasyten::hdf5::Loader& loader, std::string const& path)
{
    return load_instance<BlockBackend::Block>(
      loader, path, [&](HighFive::Group& g, std::string const& sub) -> BlockPtr {
          auto const cls = class_attr(g);
          if (cls == "EigenBlockBackend.BlockCls")
              return EigenBlockBackend::Block::from_hdf5(loader, g, sub);
#if LASYTEN_WITH_NUMPY_BACKEND
          if (cls == "NumpyBlockBackend.BlockCls")
              return NumpyBlockBackend::Block::from_hdf5(loader, g, sub);
          if (cls == "ArrayApiBlockBackend.BlockCls")
              return ArrayApiBlockBackend::Block::from_hdf5(loader, g, sub);
#else
          // The numpy-based classes store the same ``arr`` dataset: load as an Eigen block.
          if (cls == "NumpyBlockBackend.BlockCls" || cls == "ArrayApiBlockBackend.BlockCls")
              return EigenBlockBackend::Block::from_hdf5(loader, g, sub);
#endif
#if LASYTEN_WITH_TORCH_BACKEND
          if (cls == "TorchBlockBackend.BlockCls")
              return TorchBlockBackend::Block::from_hdf5(loader, g, sub);
#endif
          throw std::runtime_error("hdf5_export: unknown Block class " + cls);
      });
}

void
save_block_backend(lasyten::hdf5::Saver& saver,
                   std::string const& path,
                   std::shared_ptr<BlockBackend const> const& backend)
{
    if (!backend) {
        saver.save_none(path);
        return;
    }
    auto const* cls = block_backend_class_name(*backend);
    saver.save_instance(path,
                        kModule,
                        cls,
                        backend.get(),
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            // Non-const save_hdf5 on const object via const_cast: methods only
                            // write to HDF5.
                            const_cast<BlockBackend&>(*backend).save_hdf5(s, g, sub);
                        });
}

std::shared_ptr<BlockBackend>
load_block_backend(lasyten::hdf5::Loader& loader, std::string const& path)
{
    return load_instance<BlockBackend>(
      loader,
      path,
      [&](HighFive::Group& g, std::string const& sub) -> std::shared_ptr<BlockBackend> {
          auto const cls = class_attr(g);
          if (cls == "EigenBlockBackend")
              return EigenBlockBackend::from_hdf5(loader, g, sub);
#if LASYTEN_WITH_NUMPY_BACKEND
          if (cls == "NumpyBlockBackend")
              return NumpyBlockBackend::from_hdf5(loader, g, sub);
          if (cls == "ArrayApiBlockBackend")
              return ArrayApiBlockBackend::from_hdf5(loader, g, sub);
#else
          if (cls == "NumpyBlockBackend" || cls == "ArrayApiBlockBackend")
              return EigenBlockBackend::from_hdf5(loader, g, sub);
#endif
#if LASYTEN_WITH_TORCH_BACKEND
          if (cls == "TorchBlockBackend")
              return TorchBlockBackend::from_hdf5(loader, g, sub);
#endif
          throw std::runtime_error("hdf5_export: unknown BlockBackend class " + cls);
      });
}

} // namespace lasyten::hdf5_export
