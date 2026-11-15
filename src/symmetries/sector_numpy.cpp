#include <lasyten/symmetries/sector_numpy.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <lasyten/tools/hdf5.h>
#include <limits>
#include <memory>
#include <stdexcept>

namespace lasyten {

namespace {

bool
narrow_to_int16(std::int64_t v, std::int16_t& out)
{
    if (v < std::numeric_limits<std::int16_t>::min() ||
        v > std::numeric_limits<std::int16_t>::max()) {
        return false;
    }
    out = static_cast<std::int16_t>(v);
    return true;
}

template<typename T>
bool
load_sector_from_ptr(T const* ptr, std::size_t n, Sector& out)
{
    if (n > max_sector_ind_len) {
        return false;
    }
    std::array<std::int16_t, max_sector_ind_len> buf{};
    for (std::size_t i = 0; i < n; ++i) {
        if (!narrow_to_int16(static_cast<std::int64_t>(ptr[i]), buf[i])) {
            return false;
        }
    }
    out = Sector::from_span(std::span<const std::int16_t>(buf.data(), n));
    return true;
}

#if LASYTEN_HAS_PYTHON
template<typename T>
bool
load_sector_from_buffer(py::buffer_info const& info, Sector& out)
{
    if (info.ndim != 1) {
        return false;
    }
    auto const n = static_cast<std::size_t>(info.shape[0]);
    if (n > max_sector_ind_len) {
        return false;
    }
    auto const* ptr = static_cast<T const*>(info.ptr);
    auto const stride = info.strides[0] / static_cast<ssize_t>(sizeof(T));
    std::array<std::int16_t, max_sector_ind_len> buf{};
    for (std::size_t i = 0; i < n; ++i) {
        auto const v = static_cast<std::int64_t>(ptr[static_cast<ssize_t>(i) * stride]);
        if (!narrow_to_int16(v, buf[i])) {
            return false;
        }
    }
    out = Sector::from_span(std::span<const std::int16_t>(buf.data(), n));
    return true;
}

#endif // LASYTEN_HAS_PYTHON

template<typename T>
bool
load_sector_array_from_ptr(T const* ptr,
                           std::size_t num_sectors,
                           std::size_t sector_ind_len,
                           SectorArray& out)
{
    if (sector_ind_len > max_sector_ind_len) {
        return false;
    }
    out = SectorArray(num_sectors, static_cast<std::uint8_t>(sector_ind_len));
    for (std::size_t i = 0; i < num_sectors; ++i) {
        std::array<std::int16_t, max_sector_ind_len> buf{};
        for (std::size_t j = 0; j < sector_ind_len; ++j) {
            if (!narrow_to_int16(static_cast<std::int64_t>(ptr[i * sector_ind_len + j]), buf[j])) {
                return false;
            }
        }
        out[i] = Sector::from_span(std::span<const std::int16_t>(buf.data(), sector_ind_len));
    }
    return true;
}

#if LASYTEN_HAS_PYTHON
template<typename T>
bool
load_sector_array_from_buffer(py::buffer_info const& info, SectorArray& out)
{
    if (info.ndim != 2) {
        return false;
    }
    auto const num_sectors = static_cast<std::size_t>(info.shape[0]);
    auto const sector_ind_len = static_cast<std::size_t>(info.shape[1]);
    if (sector_ind_len > max_sector_ind_len) {
        return false;
    }
    auto const* ptr = static_cast<T const*>(info.ptr);
    auto const stride0 = info.strides[0] / static_cast<ssize_t>(sizeof(T));
    auto const stride1 = info.strides[1] / static_cast<ssize_t>(sizeof(T));
    out = SectorArray(num_sectors, static_cast<std::uint8_t>(sector_ind_len));
    for (std::size_t i = 0; i < num_sectors; ++i) {
        std::array<std::int16_t, max_sector_ind_len> buf{};
        for (std::size_t j = 0; j < sector_ind_len; ++j) {
            auto const v = static_cast<std::int64_t>(
              ptr[static_cast<ssize_t>(i) * stride0 + static_cast<ssize_t>(j) * stride1]);
            if (!narrow_to_int16(v, buf[j])) {
                return false;
            }
        }
        out[i] = Sector::from_span(std::span<const std::int16_t>(buf.data(), sector_ind_len));
    }
    return true;
}
#endif // LASYTEN_HAS_PYTHON (numpy buffer loaders)

hdf5_io::Hdf5Buffer
sector_to_hdf5_buffer(Sector const& src)
{
    hdf5_io::Hdf5Buffer buf;
    buf.kind = 'i';
    buf.itemsize = 8;
    buf.shape = { static_cast<std::size_t>(src.len()) };
    auto const nbytes = buf.nbytes();
    buf.storage = std::shared_ptr<std::uint8_t[]>(new std::uint8_t[nbytes]);
    auto* out = reinterpret_cast<std::int64_t*>(buf.storage.get());
    for (std::uint8_t i = 0; i < src.len(); ++i) {
        out[i] = src.q[i];
    }
    return buf;
}

hdf5_io::Hdf5Buffer
sector_array_to_hdf5_buffer(SectorArray const& src)
{
    hdf5_io::Hdf5Buffer buf;
    buf.kind = 'i';
    buf.itemsize = 8;
    buf.shape = { src.size(), static_cast<std::size_t>(src.sector_ind_len()) };
    auto const nbytes = buf.nbytes();
    buf.storage = std::shared_ptr<std::uint8_t[]>(new std::uint8_t[nbytes ? nbytes : 1]);
    auto* out = reinterpret_cast<std::int64_t*>(buf.storage.get());
    for (std::size_t i = 0; i < src.size(); ++i) {
        for (std::uint8_t j = 0; j < src.sector_ind_len(); ++j) {
            out[i * src.sector_ind_len() + j] = src[i][j];
        }
    }
    return buf;
}

Sector
sector_from_hdf5_buffer(hdf5_io::Hdf5Buffer const& buf)
{
    if (buf.shape.size() != 1) {
        throw std::invalid_argument("sector_from_hdf5_buffer: expected 1D array");
    }
    Sector out;
    bool ok = false;
    if (buf.kind == 'i' && buf.itemsize == 8) {
        ok = load_sector_from_ptr(static_cast<std::int64_t const*>(buf.data()), buf.shape[0], out);
    } else if (buf.kind == 'i' && buf.itemsize == 4) {
        ok = load_sector_from_ptr(static_cast<std::int32_t const*>(buf.data()), buf.shape[0], out);
    } else if (buf.kind == 'i' && buf.itemsize == 2) {
        ok = load_sector_from_ptr(static_cast<std::int16_t const*>(buf.data()), buf.shape[0], out);
    }
    if (!ok) {
        throw std::invalid_argument("sector_from_hdf5_buffer: invalid sector array");
    }
    return out;
}

SectorArray
sector_array_from_hdf5_buffer(hdf5_io::Hdf5Buffer const& buf)
{
    if (buf.shape.size() != 2) {
        throw std::invalid_argument("sector_array_from_hdf5_buffer: expected 2D array");
    }
    SectorArray out;
    bool ok = false;
    if (buf.kind == 'i' && buf.itemsize == 8) {
        ok = load_sector_array_from_ptr(
          static_cast<std::int64_t const*>(buf.data()), buf.shape[0], buf.shape[1], out);
    } else if (buf.kind == 'i' && buf.itemsize == 4) {
        ok = load_sector_array_from_ptr(
          static_cast<std::int32_t const*>(buf.data()), buf.shape[0], buf.shape[1], out);
    } else if (buf.kind == 'i' && buf.itemsize == 2) {
        ok = load_sector_array_from_ptr(
          static_cast<std::int16_t const*>(buf.data()), buf.shape[0], buf.shape[1], out);
    }
    if (!ok) {
        throw std::invalid_argument("sector_array_from_hdf5_buffer: invalid sector array");
    }
    return out;
}

} // namespace

#if LASYTEN_HAS_PYTHON
py::array
sector_to_numpy(Sector const& src)
{
    py::array_t<std::int64_t> arr(static_cast<ssize_t>(src.len()));
    auto r = arr.mutable_unchecked<1>();
    for (std::uint8_t i = 0; i < src.len(); ++i) {
        r(i) = src.q[i];
    }
    return arr;
}

py::array
sector_array_to_numpy(SectorArray const& src)
{
    py::array_t<std::int64_t> arr(
      { static_cast<ssize_t>(src.size()), static_cast<ssize_t>(src.sector_ind_len()) });
    auto r = arr.mutable_unchecked<2>();
    for (std::size_t i = 0; i < src.size(); ++i) {
        for (std::uint8_t j = 0; j < src.sector_ind_len(); ++j) {
            r(static_cast<ssize_t>(i), static_cast<ssize_t>(j)) = src[i][j];
        }
    }
    return arr;
}

Sector
sector_from_numpy(py::handle src)
{
    Sector out;
    py::array arr = py::array::ensure(src);
    if (!arr) {
        throw std::invalid_argument("sector_from_numpy: expected array-like");
    }
    auto const info = arr.request();
    bool ok = false;
    if (info.item_type_is_equivalent_to<std::int16_t>()) {
        ok = load_sector_from_buffer<std::int16_t>(info, out);
    } else if (info.item_type_is_equivalent_to<std::int32_t>()) {
        ok = load_sector_from_buffer<std::int32_t>(info, out);
    } else if (info.item_type_is_equivalent_to<std::int64_t>()) {
        ok = load_sector_from_buffer<std::int64_t>(info, out);
    } else {
        auto casted =
          py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(src);
        if (casted) {
            ok = load_sector_from_buffer<std::int64_t>(casted.request(), out);
        }
    }
    if (!ok) {
        throw std::invalid_argument("sector_from_numpy: invalid sector array");
    }
    return out;
}

SectorArray
sector_array_from_numpy(py::handle src)
{
    SectorArray out;
    py::array arr = py::array::ensure(src);
    if (!arr) {
        throw std::invalid_argument("sector_array_from_numpy: expected array-like");
    }
    auto const info = arr.request();
    bool ok = false;
    if (info.item_type_is_equivalent_to<std::int16_t>()) {
        ok = load_sector_array_from_buffer<std::int16_t>(info, out);
    } else if (info.item_type_is_equivalent_to<std::int32_t>()) {
        ok = load_sector_array_from_buffer<std::int32_t>(info, out);
    } else if (info.item_type_is_equivalent_to<std::int64_t>()) {
        ok = load_sector_array_from_buffer<std::int64_t>(info, out);
    } else {
        auto casted =
          py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(src);
        if (casted) {
            ok = load_sector_array_from_buffer<std::int64_t>(casted.request(), out);
        }
    }
    if (!ok) {
        throw std::invalid_argument("sector_array_from_numpy: invalid sector array");
    }
    return out;
}
#endif // LASYTEN_HAS_PYTHON (numpy conversions)

void
Sector::save_hdf5(lasyten::hdf5::Saver& saver,
                  HighFive::Group& /*h5gr*/,
                  std::string const& subpath) const
{
    // ``subpath`` is the group already created by ``saver``; store charges under ``values``.
    saver.save_array(subpath + "values", sector_to_hdf5_buffer(*this));
}

Sector
Sector::from_hdf5(lasyten::hdf5::Loader& loader,
                  HighFive::Group& /*h5gr*/,
                  std::string const& subpath)
{
    hid_t id = loader.open(subpath + "values");
    auto buf = loader.load_array(id);
    H5Idec_ref(id);
    return sector_from_hdf5_buffer(buf);
}

void
SectorArray::save_hdf5(lasyten::hdf5::Saver& saver,
                       HighFive::Group& /*h5gr*/,
                       std::string const& subpath) const
{
    saver.save_array(subpath + "values", sector_array_to_hdf5_buffer(*this));
}

SectorArray
SectorArray::from_hdf5(lasyten::hdf5::Loader& loader,
                       HighFive::Group& /*h5gr*/,
                       std::string const& subpath)
{
    hid_t id = loader.open(subpath + "values");
    auto buf = loader.load_array(id);
    H5Idec_ref(id);
    return sector_array_from_hdf5_buffer(buf);
}

} // namespace lasyten
