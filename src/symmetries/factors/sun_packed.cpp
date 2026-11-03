#include "sun_packed.h"

#include <lasyten/config.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <format>
#include <limits>
#include <stdexcept>

namespace lasyten {

namespace {

std::atomic<std::uint64_t> g_open_files{ 0 };
std::atomic<std::uint64_t> g_index_bytes{ 0 };
std::atomic<std::uint64_t> g_data_bytes_in_ram{ 0 };

} // namespace

PackedFileStats
packed_file_stats()
{
    return PackedFileStats{ g_open_files.load(),
                            g_index_bytes.load(),
                            g_data_bytes_in_ram.load() };
}

namespace sun_packed {

namespace {

/// Closes an HDF5 object id (``H5Idec_ref``) when leaving the scope.
struct ScopedId
{
    hid_t id = -1;
    explicit ScopedId(hid_t id_)
      : id(id_)
    {
    }
    ~ScopedId()
    {
        if (id >= 0) {
            H5Idec_ref(id);
        }
    }
    ScopedId(ScopedId const&) = delete;
    ScopedId& operator=(ScopedId const&) = delete;
};

void
expect_buffer(hdf5_io::Hdf5Buffer const& buf,
              char kind,
              int itemsize,
              std::vector<std::size_t> const& shape,
              char const* what,
              std::string const& path)
{
    if (buf.kind != kind || buf.itemsize != itemsize || buf.shape != shape) {
        throw std::runtime_error(
          std::format("SUN: the packed symbol file '{}' has an unexpected '{}' dataset (expected "
                      "kind '{}' with itemsize {} and {} dimensions).",
                      path,
                      what,
                      kind,
                      itemsize,
                      shape.size()));
    }
}

} // namespace

bool
is_packed_layout(HighFive::File& file, std::string const& path)
{
    auto const layout = hdf5_io::h5_get_attr_string(file.getId(), "Layout");
    if (!layout) {
        return false;
    }
    if (*layout == "packed_v1") {
        return true;
    }
    throw std::invalid_argument(std::format(
      "SUN: unsupported Layout '{}' of the symbol file '{}' (this version reads files without a "
      "'Layout' attribute and 'packed_v1').",
      *layout,
      path));
}

std::shared_ptr<PackedIndex const>
PackedIndex::load(std::shared_ptr<HighFive::File> file,
                  std::string const& path,
                  char kind,
                  int N,
                  bool read_data)
{
    if (kind != 'F' && kind != 'R') {
        throw std::invalid_argument("PackedIndex::load: kind must be 'F' or 'R'");
    }
    std::shared_ptr<PackedIndex> idx(new PackedIndex());
    idx->file_ = file;
    idx->path_ = path;
    idx->kind_ = kind;
    idx->k_ = kind == 'F' ? 6 : 3;
    idx->N_ = N;
    idx->rank_ = kind == 'F' ? 4 : 1;

    auto root = file->getGroup("/");
    std::string const group_name = std::string(1, kind) + "_packed";
    if (!hdf5_io::h5_contains(root, group_name)) {
        std::string present;
        for (auto const& name : hdf5_io::h5_link_names(root.getId())) {
            present += (present.empty() ? "" : ", ") + name;
        }
        throw std::invalid_argument(
          std::format("SUN: the symbol file '{}' is in the packed layout but has no group '/{}/' "
                      "(groups present: {}).",
                      path,
                      group_name,
                      present));
    }
    ScopedId group(hdf5_io::h5_open(root, group_name));
    auto read_array = [&](char const* name) {
        ScopedId dataset(hdf5_io::h5_open(group.id, name));
        return hdf5_io::h5_read_buffer(dataset.id);
    };

    auto const k = static_cast<std::size_t>(idx->k_);
    auto const len = static_cast<std::size_t>(N);
    auto const rank = static_cast<std::size_t>(idx->rank_);

    // labels (n, k, N) int32 -> compact int16 rows
    {
        auto const buf = read_array("labels");
        if (buf.shape.size() != 3) {
            throw std::runtime_error(std::format(
              "SUN: the packed symbol file '{}' has a 'labels' dataset that is not 3-dimensional.",
              path));
        }
        idx->n_ = buf.shape[0];
        expect_buffer(buf, 'i', 4, { idx->n_, k, len }, "labels", path);
        auto const* p = static_cast<std::int32_t const*>(buf.data());
        idx->labels_.resize(idx->n_ * k * len);
        for (std::size_t i = 0; i < idx->labels_.size(); ++i) {
            if (p[i] < 0 || p[i] > std::numeric_limits<std::int16_t>::max()) {
                throw std::runtime_error(
                  std::format("SUN: the packed symbol file '{}' has a label outside [0, {}].",
                              path,
                              std::numeric_limits<std::int16_t>::max()));
            }
            idx->labels_[i] = static_cast<std::int16_t>(p[i]);
        }
    }
    auto const n = idx->n_;
    // shapes (n, rank) int64 -> uint32
    {
        auto const buf = read_array("shapes");
        expect_buffer(buf, 'i', 8, { n, rank }, "shapes", path);
        auto const* p = static_cast<std::int64_t const*>(buf.data());
        idx->shapes_.resize(n * rank);
        for (std::size_t i = 0; i < idx->shapes_.size(); ++i) {
            if (p[i] < 0 || p[i] > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(std::format(
                  "SUN: the packed symbol file '{}' has a block extent that is out of range.",
                  path));
            }
            idx->shapes_[i] = static_cast<std::uint32_t>(p[i]);
        }
    }
    // offsets (n + 1,) int64
    {
        auto const buf = read_array("offsets");
        expect_buffer(buf, 'i', 8, { n + 1 }, "offsets", path);
        auto const* p = static_cast<std::int64_t const*>(buf.data());
        idx->offsets_.assign(p, p + n + 1);
    }

    // consistency of the index
    for (std::size_t i = 1; i < n; ++i) {
        auto const* prev = idx->labels_.data() + (i - 1) * k * len;
        auto const* cur = idx->labels_.data() + i * k * len;
        if (!std::lexicographical_compare(prev, prev + k * len, cur, cur + k * len)) {
            throw std::runtime_error(
              std::format("SUN: the labels of the packed symbol file '{}' are not strictly "
                          "increasing (they must be sorted and unique).",
                          path));
        }
    }
    if (idx->offsets_[0] != 0) {
        throw std::runtime_error(
          std::format("SUN: the offsets of the packed symbol file '{}' do not start at 0.", path));
    }
    for (std::size_t i = 0; i < n; ++i) {
        std::int64_t const count = idx->offsets_[i + 1] - idx->offsets_[i];
        // the shapes come from the file: the product must not overflow (signed overflow is UB)
        std::int64_t product = 1;
        bool overflow = false;
        for (std::size_t j = 0; j < rank; ++j) {
            auto const dim = static_cast<std::int64_t>(idx->shapes_[i * rank + j]);
            if (dim != 0 && product > std::numeric_limits<std::int64_t>::max() / dim) {
                overflow = true;
                break;
            }
            product *= dim;
        }
        if (overflow || count < 0 || count != product) {
            throw std::runtime_error(std::format("SUN: block {} of the packed symbol file '{}' "
                                                 "has offsets that do not match its shape.",
                                                 i,
                                                 path));
        }
    }
    auto const n_data = static_cast<std::size_t>(idx->offsets_[n]);

    // the block data
    {
        ScopedId dataset(hdf5_io::h5_open(group.id, "data"));
        ScopedId space(H5Dget_space(dataset.id));
        hsize_t dims[1] = { 0 };
        if (H5Sget_simple_extent_ndims(space.id) != 1 ||
            H5Sget_simple_extent_dims(space.id, dims, nullptr) < 0 || dims[0] != n_data) {
            throw std::runtime_error(std::format(
              "SUN: the 'data' of the packed symbol file '{}' does not have offsets[n] entries.",
              path));
        }
        // the block data is held in RAM if it is small enough (config option su_n_packed_ram_mb)
        auto const ram_max_bytes = megabytes_to_bytes(get_config().su_n_packed_ram_mb);
        if (read_data && n_data * sizeof(complex128) <= ram_max_bytes) {
            idx->data_buffer_ = hdf5_io::h5_read_buffer(dataset.id);
            expect_buffer(idx->data_buffer_, 'c', 16, { n_data }, "data", path);
            idx->data_ptr_ = static_cast<complex128 const*>(idx->data_buffer_.data());
            idx->data_bytes_in_ram_ = n_data * sizeof(complex128);
        } else if (read_data) {
            // keep the dataset open and read one block at a time
            ScopedId type(H5Dget_type(dataset.id));
            // complex128: a compound of exactly two float64 members (real, imaginary part)
            auto const is_float64_member = [&](unsigned i) {
                ScopedId member(H5Tget_member_type(type.id, i));
                return H5Tget_class(member.id) == H5T_FLOAT && H5Tget_size(member.id) == 8;
            };
            if (H5Tget_class(type.id) != H5T_COMPOUND || H5Tget_size(type.id) != 16 ||
                H5Tget_nmembers(type.id) != 2 || !is_float64_member(0) || !is_float64_member(1)) {
                throw std::runtime_error(std::format(
                  "SUN: the 'data' of the packed symbol file '{}' is not complex128.", path));
            }
            idx->data_type_ = H5Tget_native_type(type.id, H5T_DIR_ASCEND);
            idx->data_dataset_ = dataset.id;
            H5Iinc_ref(idx->data_dataset_); // survives the scope guard
        }
    }
    idx->index_bytes_ = idx->labels_.capacity() * sizeof(std::int16_t) +
                        idx->shapes_.capacity() * sizeof(std::uint32_t) +
                        idx->offsets_.capacity() * sizeof(std::int64_t);
    g_open_files.fetch_add(1);
    g_index_bytes.fetch_add(idx->index_bytes_);
    g_data_bytes_in_ram.fetch_add(idx->data_bytes_in_ram_);
    idx->counted_ = true;
    return idx;
}

PackedIndex::~PackedIndex()
{
    if (data_dataset_ >= 0) {
        H5Idec_ref(data_dataset_);
    }
    if (data_type_ >= 0) {
        H5Tclose(data_type_);
    }
    if (counted_) {
        g_open_files.fetch_sub(1);
        g_index_bytes.fetch_sub(index_bytes_);
        g_data_bytes_in_ram.fetch_sub(data_bytes_in_ram_);
    }
}

std::optional<std::size_t>
PackedIndex::find(std::span<Sector const> sectors) const
{
    auto const k = static_cast<std::size_t>(k_);
    auto const len = static_cast<std::size_t>(N_);
    if (sectors.size() != k) {
        throw std::invalid_argument("PackedIndex::find: wrong number of sectors");
    }
    std::array<std::int16_t, 6 * max_sector_ind_len> target{};
    for (std::size_t j = 0; j < k; ++j) {
        for (std::size_t t = 0; t < len; ++t) {
            target[j * len + t] = sectors[j].q[t];
        }
    }
    auto const row = k * len;
    auto compare = [&](std::size_t i) {
        auto const* r = labels_.data() + i * row;
        for (std::size_t t = 0; t < row; ++t) {
            if (r[t] != target[t]) {
                return r[t] < target[t] ? -1 : 1;
            }
        }
        return 0;
    };
    std::size_t lo = 0;
    std::size_t hi = n_;
    while (lo < hi) {
        auto const mid = lo + (hi - lo) / 2;
        if (compare(mid) < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < n_ && compare(lo) == 0) {
        return lo;
    }
    return std::nullopt;
}

FusionSymbol
PackedIndex::read(std::size_t i) const
{
    auto const rank = static_cast<std::size_t>(rank_);
    FusionSymbol::Shape shape{ { 1, 1, 1, 1 } };
    for (std::size_t j = 0; j < rank; ++j) {
        shape[j] = shapes_[i * rank + j];
    }
    auto const first = static_cast<std::size_t>(offsets_[i]);
    auto const count = static_cast<std::size_t>(offsets_[i + 1] - offsets_[i]);
    std::vector<complex128> values(count);
    if (data_ptr_ != nullptr) {
        std::copy(data_ptr_ + first, data_ptr_ + first + count, values.begin());
    } else if (count > 0) {
        ScopedId space(H5Dget_space(data_dataset_));
        hsize_t start = first;
        hsize_t extent = count;
        hdf5_io::check_hdf5(
          H5Sselect_hyperslab(space.id, H5S_SELECT_SET, &start, nullptr, &extent, nullptr),
          "H5Sselect_hyperslab");
        hid_t memory_space = H5Screate_simple(1, &extent, nullptr);
        herr_t const status =
          H5Dread(data_dataset_, data_type_, memory_space, space.id, H5P_DEFAULT, values.data());
        H5Sclose(memory_space);
        hdf5_io::check_hdf5(status, "H5Dread packed block");
    }
    return FusionSymbol::from_complex128(
      static_cast<std::uint8_t>(rank), shape, std::move(values));
}

void
PackedIndex::check(int64 highest_weight) const
{
    auto const k = static_cast<std::size_t>(k_);
    auto const len = static_cast<std::size_t>(N_);
    auto const row = k * len;
    std::string const group_name = std::string(1, kind_) + "_packed";
    if (n_ == 0) {
        throw std::invalid_argument(std::string("No valid ") + kind_ + "-symbol keys found in '/" +
                                    group_name + "/'.");
    }
    // the all-trivial key sorts first
    auto const* first = labels_.data();
    if (!std::all_of(first, first + row, [](std::int16_t v) { return v == 0; })) {
        std::string zero_key(1, kind_);
        for (std::size_t j = 0; j < k; ++j) {
            zero_key += "[0";
            for (std::size_t t = 1; t < len; ++t) {
                zero_key += ", 0";
            }
            zero_key += "]";
        }
        throw std::invalid_argument(std::string("Missing key for all-trivial-sector ") + kind_ +
                                    "-symbol: " + zero_key);
    }
    // a key with the highest weight in two neighbouring labels
    bool found = false;
    for (std::size_t i = 0; i < n_ && !found; ++i) {
        auto const* r = labels_.data() + i * row;
        for (std::size_t j = 0; j + 1 < k && !found; ++j) {
            bool both = true;
            for (std::size_t t = 0; t < len; ++t) {
                auto const expected = t == 0 ? static_cast<std::int64_t>(highest_weight) : 0;
                both = both && r[j * len + t] == expected && r[(j + 1) * len + t] == expected;
            }
            found = both;
        }
    }
    if (!found) {
        std::string h_bracket = "[" + std::to_string(highest_weight);
        for (std::size_t t = 1; t < len; ++t) {
            h_bracket += ", 0";
        }
        h_bracket += "]";
        throw std::invalid_argument("No key found containing " + h_bracket + h_bracket + ".");
    }
}

} // namespace sun_packed

} // namespace lasyten
