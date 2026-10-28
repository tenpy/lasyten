#include <lasyten/tools/hdf5_export.h>

#include <lasyten/backends/abelian.h>
#include <lasyten/backends/fusion_tree_backend.h>
#include <lasyten/backends/no_symmetry.h>
#include <lasyten/block_backend/array_api.h>
#include <lasyten/block_backend/numpy.h>
#include <lasyten/block_backend/torch.h>
#include <lasyten/symmetries/factors/fermion_number.h>
#include <lasyten/symmetries/factors/fermion_parity.h>
#include <lasyten/symmetries/factors/fibonacci_anyon_category.h>
#include <lasyten/symmetries/factors/ising_anyon_category.h>
#include <lasyten/symmetries/factors/no_symmetry.h>
#include <lasyten/symmetries/factors/quantum_double_zn_anyon_category.h>
#include <lasyten/symmetries/factors/su2.h>
#include <lasyten/symmetries/factors/su2_k_anyon_category.h>
#include <lasyten/symmetries/factors/su3_3_anyon_category.h>
#include <lasyten/symmetries/factors/sun.h>
#include <lasyten/symmetries/factors/toric_code_category.h>
#include <lasyten/symmetries/factors/u1.h>
#include <lasyten/symmetries/factors/zn.h>
#include <lasyten/symmetries/factors/zn_anyon_category.h>
#include <lasyten/symmetries/factors/zn_anyon_category2.h>

#include <cstring>
#include <stdexcept>
#include <typeinfo>

namespace lasyten::hdf5_export {
namespace {

template<typename T>
hdf5_io::Hdf5Buffer
vector_to_buffer(std::vector<T> const& v, char kind, int itemsize)
{
    hdf5_io::Hdf5Buffer buf;
    buf.kind = kind;
    buf.itemsize = itemsize;
    buf.shape = { v.size() };
    auto const nbytes = buf.nbytes();
    buf.storage = std::shared_ptr<std::uint8_t[]>(new std::uint8_t[nbytes ? nbytes : 1]);
    if (nbytes > 0) {
        std::memcpy(buf.storage.get(), v.data(), nbytes);
    }
    return buf;
}

template<typename T>
std::vector<T>
vector_from_buffer(hdf5_io::Hdf5Buffer const& buf, char expect_kind, int expect_itemsize)
{
    if (buf.kind != expect_kind || buf.itemsize != expect_itemsize) {
        throw std::invalid_argument("hdf5_export: unexpected buffer dtype");
    }
    if (buf.shape.size() != 1) {
        throw std::invalid_argument("hdf5_export: expected 1D buffer");
    }
    std::vector<T> out(buf.shape[0]);
    if (!out.empty()) {
        std::memcpy(out.data(), buf.data(), buf.nbytes());
    }
    return out;
}

std::int64_t
sequence_len(lasyten::hdf5::Loader& loader, hid_t id)
{
    auto len = hdf5_io::h5_get_attr_int64(id, hdf5_io::ATTR_LEN);
    if (!len) {
        throw std::runtime_error("hdf5_export: sequence missing len attribute");
    }
    return *len;
}

} // namespace

hdf5_io::Hdf5Buffer
i64_vector_to_buffer(std::vector<std::int64_t> const& v)
{
    return vector_to_buffer(v, 'i', 8);
}

std::vector<std::int64_t>
i64_vector_from_buffer(hdf5_io::Hdf5Buffer const& buf)
{
    if (buf.kind == 'i' && buf.itemsize == 8) {
        return vector_from_buffer<std::int64_t>(buf, 'i', 8);
    }
    if (buf.kind == 'i' && buf.itemsize == 4) {
        auto tmp = vector_from_buffer<std::int32_t>(buf, 'i', 4);
        return { tmp.begin(), tmp.end() };
    }
    throw std::invalid_argument("hdf5_export: expected int buffer");
}

hdf5_io::Hdf5Buffer
i64_matrix_to_buffer(std::vector<std::int64_t> const& flat, std::size_t rows, std::size_t cols)
{
    if (flat.size() != rows * cols) {
        throw std::invalid_argument("hdf5_export: matrix size mismatch");
    }
    hdf5_io::Hdf5Buffer buf;
    buf.kind = 'i';
    buf.itemsize = 8;
    buf.shape = { rows, cols };
    auto const nbytes = buf.nbytes();
    buf.storage = std::shared_ptr<std::uint8_t[]>(new std::uint8_t[nbytes ? nbytes : 1]);
    if (nbytes > 0) {
        std::memcpy(buf.storage.get(), flat.data(), nbytes);
    }
    return buf;
}

std::pair<std::vector<std::int64_t>, std::pair<std::size_t, std::size_t>>
i64_matrix_from_buffer(hdf5_io::Hdf5Buffer const& buf)
{
    if (buf.shape.size() == 1) {
        auto v = i64_vector_from_buffer(buf);
        return { std::move(v), { v.size(), 1 } };
    }
    if (buf.shape.size() != 2) {
        throw std::invalid_argument("hdf5_export: expected 1D or 2D int matrix");
    }
    if (!(buf.kind == 'i' && (buf.itemsize == 8 || buf.itemsize == 4))) {
        throw std::invalid_argument("hdf5_export: expected int matrix");
    }
    std::size_t rows = buf.shape[0];
    std::size_t cols = buf.shape[1];
    std::vector<std::int64_t> flat(rows * cols);
    if (buf.itemsize == 8) {
        if (!flat.empty()) {
            std::memcpy(flat.data(), buf.data(), buf.nbytes());
        }
    } else {
        auto const* src = static_cast<std::int32_t const*>(buf.data());
        for (std::size_t i = 0; i < flat.size(); ++i) {
            flat[i] = src[i];
        }
    }
    return { std::move(flat), { rows, cols } };
}

hdf5_io::Hdf5Buffer
f64_vector_to_buffer(std::vector<double> const& v)
{
    return vector_to_buffer(v, 'f', 8);
}

std::vector<double>
f64_vector_from_buffer(hdf5_io::Hdf5Buffer const& buf)
{
    return vector_from_buffer<double>(buf, 'f', 8);
}

void
save_i64_vector(lasyten::hdf5::Saver& saver,
                std::string const& path,
                std::vector<std::int64_t> const& v)
{
    saver.save_array(path, i64_vector_to_buffer(v));
}

std::vector<std::int64_t>
load_i64_vector(lasyten::hdf5::Loader& loader, std::string const& path)
{
    hid_t id = loader.open(path);
    auto buf = loader.load_array(id);
    H5Idec_ref(id);
    return i64_vector_from_buffer(buf);
}

void
save_f64_vector(lasyten::hdf5::Saver& saver, std::string const& path, std::vector<double> const& v)
{
    saver.save_array(path, f64_vector_to_buffer(v));
}

std::vector<double>
load_f64_vector(lasyten::hdf5::Loader& loader, std::string const& path)
{
    hid_t id = loader.open(path);
    auto buf = loader.load_array(id);
    H5Idec_ref(id);
    return f64_vector_from_buffer(buf);
}

void
save_optional_i64_vector(lasyten::hdf5::Saver& saver,
                         std::string const& path,
                         std::optional<std::vector<std::int64_t>> const& v)
{
    if (!v) {
        saver.save_none(path);
        return;
    }
    save_i64_vector(saver, path, *v);
}

std::optional<std::vector<std::int64_t>>
load_optional_i64_vector(lasyten::hdf5::Loader& loader, std::string const& path)
{
    hid_t id = loader.open(path);
    auto type = loader.type_of(id);
    if (type == hdf5_io::REPR_NONE) {
        H5Idec_ref(id);
        return std::nullopt;
    }
    auto buf = loader.load_array(id);
    H5Idec_ref(id);
    return i64_vector_from_buffer(buf);
}

void
save_leg_labels(lasyten::hdf5::Saver& saver, std::string const& path, LegLabelList const& labels)
{
    HighFive::Group seq_g;
    std::string seq_sub;
    saver.save_sequence_begin(
      path, hdf5_io::REPR_LIST, static_cast<std::int64_t>(labels.size()), seq_g, seq_sub);
    lasyten::hdf5::Saver seq_saver(seq_g);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        // Empty string = unset; still stored as a string (not REPR_NONE).
        seq_saver.save_string(std::to_string(i), labels[i]);
    }
}

LegLabelList
load_leg_labels(lasyten::hdf5::Loader& loader, std::string const& path)
{
    hid_t id = loader.open(path);
    auto const n = sequence_len(loader, id);
    HighFive::Group seq_g = hdf5_io::group_from_hid(id);
    lasyten::hdf5::Loader seq_loader(seq_g);
    LegLabelList out;
    out.reserve(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        hid_t child = seq_loader.open(std::to_string(i));
        auto type = seq_loader.type_of(child);
        // Backward compat: older files stored unset labels as HDF5 None.
        if (type == hdf5_io::REPR_NONE) {
            out.emplace_back("");
        } else {
            out.emplace_back(seq_loader.load_string(child));
        }
        H5Idec_ref(child);
    }
    return out;
}

void
save_dtype_string(lasyten::hdf5::Saver& saver, std::string const& path, Dtype dt)
{
    saver.save_string(path, dtype::repr(dt));
}

Dtype
load_dtype_string(lasyten::hdf5::Loader& loader, std::string const& path)
{
    hid_t id = loader.open(path);
    auto s = loader.load_string(id);
    H5Idec_ref(id);
    return dtype::from_repr(s);
}

void
save_sector(lasyten::hdf5::Saver& saver, std::string const& path, Sector const& sector)
{
    saver.save_instance(path,
                        kModule,
                        "Sector",
                        &sector,
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            sector.save_hdf5(s, g, sub);
                        });
}

Sector
load_sector(lasyten::hdf5::Loader& loader, std::string const& path)
{
    hid_t id = loader.open(path);
    HighFive::Group g = hdf5_io::group_from_hid(id);
    return Sector::from_hdf5(loader, g, lasyten::hdf5::ensure_slash(path));
}

void
save_sector_array(lasyten::hdf5::Saver& saver, std::string const& path, SectorArray const& sectors)
{
    saver.save_instance(path,
                        kModule,
                        "SectorArray",
                        &sectors,
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            sectors.save_hdf5(s, g, sub);
                        });
}

SectorArray
load_sector_array(lasyten::hdf5::Loader& loader, std::string const& path)
{
    hid_t id = loader.open(path);
    HighFive::Group g = hdf5_io::group_from_hid(id);
    return SectorArray::from_hdf5(loader, g, lasyten::hdf5::ensure_slash(path));
}

void
save_block_inds(lasyten::hdf5::Saver& saver, std::string const& path, BlockInds const& bi)
{
    saver.save_instance(path,
                        kModule,
                        "BlockInds",
                        &bi,
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            bi.save_hdf5(s, g, sub);
                        });
}

BlockInds
load_block_inds(lasyten::hdf5::Loader& loader, std::string const& path)
{
    hid_t id = loader.open(path);
    HighFive::Group g = hdf5_io::group_from_hid(id);
    return BlockInds::from_hdf5(loader, g, lasyten::hdf5::ensure_slash(path));
}

std::string
class_attr(HighFive::Group& h5gr)
{
    auto cls = hdf5_io::h5_get_attr_string(h5gr.getId(), hdf5_io::ATTR_CLASS);
    if (!cls) {
        throw std::runtime_error("hdf5_export: missing class attribute");
    }
    return *cls;
}

hdf5_io::Hdf5Buffer
buffer_from_numpy(py::array arr)
{
    if (!arr.attr("flags").attr("c_contiguous").cast<bool>())
        arr = py::reinterpret_steal<py::array>(arr.attr("copy")("C").release());
    hdf5_io::Hdf5Buffer buf;
    py::dtype dt = arr.dtype();
    buf.kind = dt.kind();
    buf.itemsize = static_cast<int>(dt.itemsize());
    if (buf.kind == 'b' || (buf.kind == 'i' && dt.attr("name").cast<std::string>() == "bool")) {
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
numpy_from_buffer(hdf5_io::Hdf5Buffer const& buf)
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

namespace {

char const*
symmetry_factor_class_name(SymmetryFactor const& factor)
{
    if (dynamic_cast<ZN const*>(&factor))
        return "ZN";
    if (dynamic_cast<U1 const*>(&factor))
        return "U1";
    if (dynamic_cast<NoSymmetry const*>(&factor))
        return "NoSymmetry";
    if (dynamic_cast<_SU2 const*>(&factor))
        return "_SU2";
    if (dynamic_cast<SUN const*>(&factor))
        return "SUN";
    if (dynamic_cast<FermionParity const*>(&factor))
        return "FermionParity";
    if (dynamic_cast<FermionNumber const*>(&factor))
        return "FermionNumber";
    if (dynamic_cast<ToricCodeCategory const*>(&factor))
        return "ToricCodeCategory";
    if (dynamic_cast<QuantumDoubleZNAnyonCategory const*>(&factor))
        return "QuantumDoubleZNAnyonCategory";
    if (dynamic_cast<ZNAnyonCategory2 const*>(&factor))
        return "ZNAnyonCategory2";
    if (dynamic_cast<ZNAnyonCategory const*>(&factor))
        return "ZNAnyonCategory";
    if (dynamic_cast<FibonacciAnyonCategory const*>(&factor))
        return "FibonacciAnyonCategory";
    if (dynamic_cast<IsingAnyonCategory const*>(&factor))
        return "IsingAnyonCategory";
    if (dynamic_cast<SU2_kAnyonCategory const*>(&factor))
        return "SU2_kAnyonCategory";
    if (dynamic_cast<SU3_3AnyonCategory const*>(&factor))
        return "SU3_3AnyonCategory";
    throw std::runtime_error(std::string("hdf5_export: unknown SymmetryFactor type ") +
                             typeid(factor).name());
}

} // namespace

void
save_symmetry_factor(lasyten::hdf5::Saver& saver,
                     std::string const& path,
                     SymmetryFactor::CPtr const& factor)
{
    if (!factor) {
        saver.save_none(path);
        return;
    }
    auto const* cls = symmetry_factor_class_name(*factor);
    saver.save_instance(path,
                        kModule,
                        cls,
                        factor.get(),
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            factor->save_hdf5(s, g, sub);
                        });
}

SymmetryFactor::Ptr
load_symmetry_factor(lasyten::hdf5::Loader& loader, std::string const& path)
{
    return load_instance<SymmetryFactor>(
      loader, path, [&](HighFive::Group& g, std::string const& sub) {
          auto const cls = class_attr(g);
          if (cls == "ZN")
              return std::static_pointer_cast<SymmetryFactor>(ZN::from_hdf5(loader, g, sub));
          if (cls == "U1")
              return std::static_pointer_cast<SymmetryFactor>(U1::from_hdf5(loader, g, sub));
          if (cls == "NoSymmetry")
              return std::static_pointer_cast<SymmetryFactor>(
                NoSymmetry::from_hdf5(loader, g, sub));
          if (cls == "_SU2" || cls == "SU2")
              return std::static_pointer_cast<SymmetryFactor>(_SU2::from_hdf5(loader, g, sub));
          if (cls == "SUN")
              return std::static_pointer_cast<SymmetryFactor>(SUN::from_hdf5(loader, g, sub));
          if (cls == "FermionParity")
              return std::static_pointer_cast<SymmetryFactor>(
                FermionParity::from_hdf5(loader, g, sub));
          if (cls == "FermionNumber")
              return std::static_pointer_cast<SymmetryFactor>(
                FermionNumber::from_hdf5(loader, g, sub));
          if (cls == "ToricCodeCategory")
              return std::static_pointer_cast<SymmetryFactor>(
                ToricCodeCategory::from_hdf5(loader, g, sub));
          if (cls == "QuantumDoubleZNAnyonCategory")
              return std::static_pointer_cast<SymmetryFactor>(
                QuantumDoubleZNAnyonCategory::from_hdf5(loader, g, sub));
          if (cls == "ZNAnyonCategory2")
              return std::static_pointer_cast<SymmetryFactor>(
                ZNAnyonCategory2::from_hdf5(loader, g, sub));
          if (cls == "ZNAnyonCategory")
              return std::static_pointer_cast<SymmetryFactor>(
                ZNAnyonCategory::from_hdf5(loader, g, sub));
          if (cls == "FibonacciAnyonCategory")
              return std::static_pointer_cast<SymmetryFactor>(
                FibonacciAnyonCategory::from_hdf5(loader, g, sub));
          if (cls == "IsingAnyonCategory")
              return std::static_pointer_cast<SymmetryFactor>(
                IsingAnyonCategory::from_hdf5(loader, g, sub));
          if (cls == "SU2_kAnyonCategory")
              return std::static_pointer_cast<SymmetryFactor>(
                SU2_kAnyonCategory::from_hdf5(loader, g, sub));
          if (cls == "SU3_3AnyonCategory")
              return std::static_pointer_cast<SymmetryFactor>(
                SU3_3AnyonCategory::from_hdf5(loader, g, sub));
          throw std::runtime_error("hdf5_export: unknown SymmetryFactor class " + cls);
      });
}

void
save_symmetry(lasyten::hdf5::Saver& saver, std::string const& path, Symmetry::CPtr const& symmetry)
{
    if (!symmetry) {
        saver.save_none(path);
        return;
    }
    saver.save_instance(path,
                        kModule,
                        "Symmetry",
                        symmetry.get(),
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            symmetry->save_hdf5(s, g, sub);
                        });
}

Symmetry::Ptr
load_symmetry(lasyten::hdf5::Loader& loader, std::string const& path)
{
    return load_instance<Symmetry>(loader, path, [&](HighFive::Group& g, std::string const& sub) {
        return Symmetry::from_hdf5(loader, g, sub);
    });
}

namespace {

char const*
elementary_space_class_name(ElementarySpace const& space)
{
    if (dynamic_cast<AbelianLegPipe const*>(&space))
        return "AbelianLegPipe";
    if (dynamic_cast<DirectSumSpace const*>(&space))
        return "DirectSumSpace";
    return "ElementarySpace";
}

char const*
leg_class_name(Leg const& leg)
{
    if (auto const* es = dynamic_cast<ElementarySpace const*>(&leg)) {
        return elementary_space_class_name(*es);
    }
    throw std::runtime_error(std::string("hdf5_export: unknown Leg type ") + typeid(leg).name());
}

} // namespace

void
save_elementary_space(lasyten::hdf5::Saver& saver,
                      std::string const& path,
                      ElementarySpace::CPtr const& space)
{
    if (!space) {
        saver.save_none(path);
        return;
    }
    auto const* cls = elementary_space_class_name(*space);
    saver.save_instance(path,
                        kModule,
                        cls,
                        space.get(),
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            space->save_hdf5(s, g, sub);
                        });
}

ElementarySpace::Ptr
load_elementary_space(lasyten::hdf5::Loader& loader, std::string const& path)
{
    return load_instance<ElementarySpace>(
      loader, path, [&](HighFive::Group& g, std::string const& sub) -> ElementarySpace::Ptr {
          auto const cls = class_attr(g);
          if (cls == "AbelianLegPipe")
              return AbelianLegPipe::from_hdf5(loader, g, sub);
          if (cls == "DirectSumSpace")
              return DirectSumSpace::from_hdf5(loader, g, sub);
          if (cls == "ElementarySpace")
              return ElementarySpace::from_hdf5(loader, g, sub);
          throw std::runtime_error("hdf5_export: unknown ElementarySpace class " + cls);
      });
}

void
save_leg(lasyten::hdf5::Saver& saver, std::string const& path, Leg::CPtr const& leg)
{
    if (!leg) {
        saver.save_none(path);
        return;
    }
    auto const* cls = leg_class_name(*leg);
    saver.save_instance(path,
                        kModule,
                        cls,
                        leg.get(),
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            if (auto es = std::dynamic_pointer_cast<ElementarySpace const>(leg)) {
                                es->save_hdf5(s, g, sub);
                                return;
                            }
                            throw std::runtime_error("hdf5_export: Leg without save_hdf5");
                        });
}

Leg::Ptr
load_leg(lasyten::hdf5::Loader& loader, std::string const& path)
{
    return std::static_pointer_cast<Leg>(load_elementary_space(loader, path));
}

void
save_tensor_product(lasyten::hdf5::Saver& saver,
                    std::string const& path,
                    TensorProduct::CPtr const& tp)
{
    if (!tp) {
        saver.save_none(path);
        return;
    }
    saver.save_instance(path,
                        kModule,
                        "TensorProduct",
                        tp.get(),
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            tp->save_hdf5(s, g, sub);
                        });
}

TensorProduct::Ptr
load_tensor_product(lasyten::hdf5::Loader& loader, std::string const& path)
{
    return load_instance<TensorProduct>(
      loader, path, [&](HighFive::Group& g, std::string const& sub) {
          return TensorProduct::from_hdf5(loader, g, sub);
      });
}

namespace {

char const*
block_class_name(BlockBackend::Block const& block)
{
    if (dynamic_cast<NumpyBlockBackend::Block const*>(&block))
        return "NumpyBlockBackend.BlockCls";
    if (dynamic_cast<TorchBlockBackend::Block const*>(&block))
        return "TorchBlockBackend.BlockCls";
    if (dynamic_cast<ArrayApiBlockBackend::Block const*>(&block))
        return "ArrayApiBlockBackend.BlockCls";
    throw std::runtime_error(std::string("hdf5_export: unknown Block type ") +
                             typeid(block).name());
}

char const*
block_backend_class_name(BlockBackend const& backend)
{
    if (dynamic_cast<NumpyBlockBackend const*>(&backend))
        return "NumpyBlockBackend";
    if (dynamic_cast<TorchBlockBackend const*>(&backend))
        return "TorchBlockBackend";
    if (dynamic_cast<ArrayApiBlockBackend const*>(&backend))
        return "ArrayApiBlockBackend";
    throw std::runtime_error(std::string("hdf5_export: unknown BlockBackend type ") +
                             typeid(backend).name());
}

char const*
tensor_backend_class_name(TensorBackend const& backend)
{
    if (dynamic_cast<AbelianBackend const*>(&backend))
        return "AbelianBackend";
    if (dynamic_cast<FusionTreeBackend const*>(&backend))
        return "FusionTreeBackend";
    if (dynamic_cast<NoSymmetryBackend const*>(&backend))
        return "NoSymmetryBackend";
    throw std::runtime_error(std::string("hdf5_export: unknown TensorBackend type ") +
                             typeid(backend).name());
}

char const*
tensor_backend_data_class_name(TensorBackend::Data const& data)
{
    if (dynamic_cast<AbelianBackendData const*>(&data))
        return "AbelianBackendData";
    if (dynamic_cast<FusionTreeData const*>(&data))
        return "FusionTreeData";
    if (dynamic_cast<NoSymmetryBackend::BlockData const*>(&data))
        return "NoSymmetryBackend.BlockData";
    throw std::runtime_error(std::string("hdf5_export: unknown TensorBackend::Data type ") +
                             typeid(data).name());
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
          if (cls == "NumpyBlockBackend.BlockCls")
              return NumpyBlockBackend::Block::from_hdf5(loader, g, sub);
          if (cls == "TorchBlockBackend.BlockCls")
              return TorchBlockBackend::Block::from_hdf5(loader, g, sub);
          if (cls == "ArrayApiBlockBackend.BlockCls")
              return ArrayApiBlockBackend::Block::from_hdf5(loader, g, sub);
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
          if (cls == "NumpyBlockBackend")
              return NumpyBlockBackend::from_hdf5(loader, g, sub);
          if (cls == "TorchBlockBackend")
              return TorchBlockBackend::from_hdf5(loader, g, sub);
          if (cls == "ArrayApiBlockBackend")
              return ArrayApiBlockBackend::from_hdf5(loader, g, sub);
          throw std::runtime_error("hdf5_export: unknown BlockBackend class " + cls);
      });
}

void
save_tensor_backend(lasyten::hdf5::Saver& saver,
                    std::string const& path,
                    TensorBackend::CPtr const& backend)
{
    if (!backend) {
        saver.save_none(path);
        return;
    }
    auto const* cls = tensor_backend_class_name(*backend);
    saver.save_instance(path,
                        kModule,
                        cls,
                        backend.get(),
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            const_cast<TensorBackend&>(*backend).save_hdf5(s, g, sub);
                        });
}

TensorBackend::Ptr
load_tensor_backend(lasyten::hdf5::Loader& loader, std::string const& path)
{
    return load_instance<TensorBackend>(
      loader, path, [&](HighFive::Group& g, std::string const& sub) -> TensorBackend::Ptr {
          auto const cls = class_attr(g);
          auto block_backend = load_block_backend(loader, sub + "block_backend");
          if (cls == "AbelianBackend") {
              auto obj = std::make_shared<AbelianBackend>(std::move(block_backend));
              loader.memorize_load(g.getId(), std::static_pointer_cast<void>(obj));
              return obj;
          }
          if (cls == "FusionTreeBackend") {
              auto obj = std::make_shared<FusionTreeBackend>(std::move(block_backend));
              loader.memorize_load(g.getId(), std::static_pointer_cast<void>(obj));
              return obj;
          }
          if (cls == "NoSymmetryBackend") {
              auto obj = std::make_shared<NoSymmetryBackend>(std::move(block_backend));
              loader.memorize_load(g.getId(), std::static_pointer_cast<void>(obj));
              return obj;
          }
          (void)sub;
          throw std::runtime_error("hdf5_export: unknown TensorBackend class " + cls);
      });
}

void
save_tensor_backend_data(lasyten::hdf5::Saver& saver,
                         std::string const& path,
                         TensorBackend::DataCPtr const& data)
{
    if (!data) {
        saver.save_none(path);
        return;
    }
    auto const* cls = tensor_backend_data_class_name(*data);
    saver.save_instance(
      path,
      kModule,
      cls,
      data.get(),
      [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
          if (auto abd = std::dynamic_pointer_cast<AbelianBackendData const>(data)) {
              abd->save_hdf5(s, g, sub);
              return;
          }
          if (auto ftd = std::dynamic_pointer_cast<FusionTreeData const>(data)) {
              ftd->save_hdf5(s, g, sub);
              return;
          }
          throw std::runtime_error("hdf5_export: TensorBackend::Data without save_hdf5");
      });
}

TensorBackend::DataPtr
load_tensor_backend_data(lasyten::hdf5::Loader& loader, std::string const& path)
{
    return load_instance<TensorBackend::Data>(
      loader, path, [&](HighFive::Group& g, std::string const& sub) -> TensorBackend::DataPtr {
          auto const cls = class_attr(g);
          if (cls == "AbelianBackendData")
              return AbelianBackendData::from_hdf5(loader, g, sub);
          if (cls == "FusionTreeData")
              return FusionTreeData::from_hdf5(loader, g, sub);
          throw std::runtime_error("hdf5_export: unknown TensorBackend::Data class " + cls);
      });
}

} // namespace lasyten::hdf5_export
