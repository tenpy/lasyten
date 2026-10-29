#include <lasyten/symmetries/factors/sun.h>

#include <lasyten/block_backend/numpy.h>
#include <lasyten/config.h>
#include <lasyten/symmetries/exceptions.h>
#include <lasyten/symmetries/fusion_symbol.h>
#include <lasyten/tools/lru_cache.h>

#include "sun_packed.h"

#include <hdf5_io/h5_ops.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <lasyten/tools/hdf5.h>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

namespace {

std::string
sector_slash_path(Sector const& a)
{
    std::string s;
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        if (i != 0) {
            s += '/';
        }
        s += std::to_string(a.q[i]);
    }
    s += '/';
    return s;
}

std::string
sector_concat(Sector const& a)
{
    std::string s;
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        s += std::to_string(a.q[i]);
    }
    return s;
}

std::string
sector_bracket(Sector const& a)
{
    std::string s = "[";
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        if (i != 0) {
            s += ", ";
        }
        s += std::to_string(static_cast<int>(a.q[i]));
    }
    s += "]";
    return s;
}

std::string
cg_key(int N, Sector const& a, Sector const& b)
{
    // Absolute-style path without leading slash (hdf5_io::normalized_path strips it).
    return "N_" + std::to_string(N) + "/" + sector_slash_path(a) + sector_slash_path(b);
}

HighFive::Group
file_root(HighFive::File& file)
{
    return file.getGroup("/");
}

bool
cg_key_usable(HighFive::File& CGfile, std::string const& key)
{
    auto root = file_root(CGfile);
    if (!hdf5_io::h5_contains(root, key)) {
        return false;
    }
    hid_t loc = hdf5_io::h5_open(root, key);
    auto names = hdf5_io::h5_link_names(loc);
    H5Idec_ref(loc);
    return !names.empty();
}

int64
file_attr_int64(HighFive::File& file, char const* name)
{
    auto v = hdf5_io::h5_get_attr_int64(file.getId(), name);
    if (!v) {
        throw std::runtime_error(std::string("SUN HDF5 file missing attribute '") + name + "'");
    }
    return *v;
}

Sector
zeros_sector(int N)
{
    std::array<int16_t, max_sector_ind_len> z{};
    return Sector::from_span(std::span<const int16_t>(z.data(), static_cast<std::size_t>(N)));
}

std::string
normalize_su_n_data_kind(std::string const& kind)
{
    std::string up = kind;
    for (char& c : up)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (up != "CG" && up != "F" && up != "R") {
        throw std::invalid_argument("SU(N) data kind must be one of 'CG', 'F', 'R'; got '" + kind +
                                    "'");
    }
    return up;
}

int64
binomial(int n, int k)
{
    if (k < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    if (k > n - k) {
        k = n - k;
    }
    int64 r = 1;
    for (int i = 1; i <= k; ++i) {
        r = r * (n - k + i) / i;
    }
    return r;
}

Sector
sector_from_int_buffer(void const* data, std::size_t n, char kind, int itemsize)
{
    std::vector<int16_t> vals(n);
    if (kind == 'i' || kind == 'u') {
        if (itemsize == 1) {
            auto const* p = static_cast<std::int8_t const*>(data);
            for (std::size_t i = 0; i < n; ++i)
                vals[i] = static_cast<int16_t>(p[i]);
        } else if (itemsize == 2) {
            auto const* p = static_cast<std::int16_t const*>(data);
            for (std::size_t i = 0; i < n; ++i)
                vals[i] = p[i];
        } else if (itemsize == 4) {
            auto const* p = static_cast<std::int32_t const*>(data);
            for (std::size_t i = 0; i < n; ++i)
                vals[i] = static_cast<int16_t>(p[i]);
        } else if (itemsize == 8) {
            auto const* p = static_cast<std::int64_t const*>(data);
            for (std::size_t i = 0; i < n; ++i)
                vals[i] = static_cast<int16_t>(p[i]);
        } else {
            throw std::runtime_error("SUN: unsupported integer itemsize for Irreplabel");
        }
    } else if (kind == 'f') {
        if (itemsize == 8) {
            auto const* p = static_cast<double const*>(data);
            for (std::size_t i = 0; i < n; ++i)
                vals[i] = static_cast<int16_t>(p[i]);
        } else if (itemsize == 4) {
            auto const* p = static_cast<float const*>(data);
            for (std::size_t i = 0; i < n; ++i)
                vals[i] = static_cast<int16_t>(p[i]);
        } else {
            throw std::runtime_error("SUN: unsupported float itemsize for Irreplabel");
        }
    } else {
        throw std::runtime_error("SUN: unsupported Irreplabel dtype kind");
    }
    return Sector::from_span(vals);
}

Sector
read_irrep_label_attr(hid_t loc)
{
    if (H5Aexists(loc, "Irreplabel") <= 0) {
        throw std::runtime_error("SUN: dataset missing Irreplabel attribute");
    }
    hid_t attr = H5Aopen(loc, "Irreplabel", H5P_DEFAULT);
    hdf5_io::check_hdf5(attr < 0 ? -1 : 0, "H5Aopen Irreplabel");
    hid_t type = H5Aget_type(attr);
    hid_t space = H5Aget_space(attr);
    int ndims = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims(static_cast<std::size_t>(std::max(ndims, 0)));
    if (ndims > 0) {
        H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    }
    std::size_t n = 1;
    if (ndims <= 0) {
        n = 1;
    } else {
        for (auto d : dims)
            n *= static_cast<std::size_t>(d);
    }

    H5T_class_t cls = H5Tget_class(type);
    size_t sz = H5Tget_size(type);
    char kind = 'i';
    if (cls == H5T_FLOAT)
        kind = 'f';
    else if (cls == H5T_INTEGER)
        kind = H5Tget_sign(type) == H5T_SGN_NONE ? 'u' : 'i';
    else
        throw std::runtime_error("SUN: Irreplabel has unsupported HDF5 type");

    std::vector<std::uint8_t> storage(n * sz);
    hid_t ntype = H5Tget_native_type(type, H5T_DIR_ASCEND);
    hdf5_io::check_hdf5(H5Aread(attr, ntype, storage.data()), "H5Aread Irreplabel");
    H5Tclose(ntype);
    H5Sclose(space);
    H5Tclose(type);
    H5Aclose(attr);
    return sector_from_int_buffer(storage.data(), n, kind, static_cast<int>(sz));
}

FusionSymbol
fusion_symbol_from_hdf5_buffer(hdf5_io::Hdf5Buffer const& buf)
{
    if (buf.shape.size() > 4) {
        throw std::invalid_argument("fusion_symbol_from_hdf5_buffer: rank must be <= 4");
    }
    FusionSymbol::Shape shape{ { 1, 1, 1, 1 } };
    auto rank = static_cast<std::uint8_t>(buf.shape.empty() ? 1 : buf.shape.size());
    for (std::size_t i = 0; i < buf.shape.size(); ++i) {
        shape[i] = buf.shape[i];
    }
    std::size_t n = 1;
    for (std::size_t i = 0; i < rank; ++i)
        n *= shape[i];

    if (buf.kind == 'f' && buf.itemsize == 8) {
        auto const* p = static_cast<double const*>(buf.data());
        std::vector<float64> data(p, p + n);
        return FusionSymbol::from_float64(rank, shape, std::move(data));
    }
    if (buf.kind == 'c' && buf.itemsize == 16) {
        auto const* p = static_cast<complex128 const*>(buf.data());
        std::vector<complex128> data(p, p + n);
        return FusionSymbol::from_complex128(rank, shape, std::move(data));
    }
    if (buf.kind == 'f' && buf.itemsize == 4) {
        auto const* p = static_cast<float const*>(buf.data());
        std::vector<float64> data(n);
        for (std::size_t i = 0; i < n; ++i)
            data[i] = static_cast<float64>(p[i]);
        return FusionSymbol::from_float64(rank, shape, std::move(data));
    }
    throw std::runtime_error("SUN: unsupported dataset dtype for FusionSymbol");
}

/// Read CG coefficient table as rows of [q_a, q_b, q_c, coeff] (float64).
std::vector<std::array<double, 4>>
read_cg_rows(HighFive::File& file, std::string const& group_key, std::string const& dset_name)
{
    auto root = file_root(file);
    hid_t grp = hdf5_io::h5_open(root, group_key);
    hid_t dset = hdf5_io::h5_open(grp, dset_name);
    auto buf = hdf5_io::h5_read_buffer(dset);
    H5Idec_ref(dset);
    H5Idec_ref(grp);

    if (buf.kind != 'f' || buf.itemsize != 8) {
        throw std::runtime_error("SUN: CG dataset must be float64");
    }
    auto const* p = static_cast<double const*>(buf.data());
    // Accept (nrows, 4) or (1, nrows, 4) as produced by some exporters.
    std::size_t nrows = 0;
    if (buf.shape.size() == 2 && buf.shape[1] == 4) {
        nrows = buf.shape[0];
    } else if (buf.shape.size() == 3 && buf.shape[0] == 1 && buf.shape[2] == 4) {
        nrows = buf.shape[1];
    } else if (buf.shape.size() == 1 && buf.shape[0] % 4 == 0) {
        nrows = buf.shape[0] / 4;
    } else {
        throw std::runtime_error("SUN: unexpected CG dataset shape");
    }
    std::vector<std::array<double, 4>> rows(nrows);
    for (std::size_t i = 0; i < nrows; ++i) {
        rows[i] = { p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3] };
    }
    return rows;
}

std::shared_ptr<HighFive::File>
open_su_n_data_file(std::string const& full_path, char const* kind, int N, int64 hweight)
{
    if (!std::filesystem::exists(std::filesystem::path(full_path))) {
        std::string msg = std::format(
          "SU(N) {} data file for N={}, hweight={} not found:\n"
          "    {}\n"
          "Generate it with the clebsch_gordan_coefficients package, or tell lasyten where your "
          "files are:\n"
          "    lasyten.set_options(su_n_data_path='/path/to/dir')\n"
          "    lasyten.set_options(su_n_data_filename_base='my_base')\n"
          "or set the environment variables LASYTEN_SU_N_DATA_PATH / "
          "LASYTEN_SU_N_DATA_FILENAME_BASE, or add the keys to ~/.lasytenconfig.yaml.\n"
          "The default location is the literal POSIX path "
          "'/home/<login-name>/.tenpy/su_n_symmetry_data' on all platforms.",
          kind,
          N,
          hweight,
          full_path);
        // Match the historical Python API: missing SU(N) data is FileNotFoundError.
        PyErr_SetString(PyExc_FileNotFoundError, msg.c_str());
        throw py::error_already_set();
    }
    auto file = std::make_shared<HighFive::File>(full_path, HighFive::File::ReadOnly);
    auto stored = file_attr_int64(*file, "Highest_Weight");
    if (stored != hweight) {
        throw std::invalid_argument(std::format(
          "SU(N) {} data file '{}' is named for hweight {} but has attrs['Highest_Weight'] = {}.",
          kind,
          full_path,
          hweight,
          stored));
    }
    return file;
}

std::string
load_string_child(lasyten::hdf5::Loader& loader, std::string const& path)
{
    hid_t id = loader.open(path);
    auto s = loader.load_string(id);
    H5Idec_ref(id);
    return s;
}

int64
load_int64_child(lasyten::hdf5::Loader& loader, std::string const& path)
{
    hid_t id = loader.open(path);
    auto v = loader.load_int64(id);
    H5Idec_ref(id);
    return v;
}

} // namespace

std::string
su_n_data_filename(int N,
                   std::string const& kind,
                   int64 hweight,
                   std::optional<std::string> filename_base)
{
    const std::string base = filename_base.value_or(get_config().su_n_data_filename_base);
    return std::format(
      "{}_N{}_{}_hweight{}.hdf5", base, N, normalize_su_n_data_kind(kind), hweight);
}

std::string
su_n_data_file_path(int N,
                    std::string const& kind,
                    int64 hweight,
                    std::optional<std::string> path,
                    std::optional<std::string> filename_base)
{
    std::string dir = expand_user(path.value_or(get_config().su_n_data_path));
    const std::string name = su_n_data_filename(N, kind, hweight, std::move(filename_base));
    if (dir.empty())
        return name;
    if (dir.back() != '/' && dir.back() != '\\')
        dir += '/';
    return dir + name;
}

HighFive::Group
SUN::cg_root() const
{
    return file_root(*CGfile_);
}

HighFive::Group
SUN::f_root() const
{
    return file_root(*Ffile_);
}

HighFive::Group
SUN::r_root() const
{
    return file_root(*Rfile_);
}

// ---------------------------------------------------------------------------------------------
// RAM cache for the data read from the HDF5 files: F symbols, R symbols and fusion outcomes.
// One process-wide least-recently-used cache whose size is capped by the config option
// ``su_n_symbol_cache_mb``. The files are opened read-only and never change while a `SUN` object
// lives, so everything derived from them can be cached; entries are keyed by an id that is unique
// per `SUN` object, and entries of destroyed objects simply age out.
namespace {

/// Fusion outcomes of ``a x b``: labels of all children of the (a, b) group of the CG file (in
/// file order) and the "Outer Multiplicity" of every irrep c (dataset ``Irrep<c>a1``).
struct SunOutcomes
{
    std::vector<Sector> labels;
    std::map<Sector, int64> multiplicity;
};

using SymbolValue = std::variant<FusionSymbol, SunOutcomes>;
using SymbolCache = ByteLRU<std::string, SymbolValue>;

std::size_t
symbol_value_bytes(SymbolValue const& value)
{
    if (auto const* f = std::get_if<FusionSymbol>(&value)) {
        return sizeof(FusionSymbol) +
               f->size() * (f->is_real() ? sizeof(float64) : sizeof(complex128));
    }
    constexpr std::size_t map_node_bytes = sizeof(Sector) + sizeof(int64) + 48;
    auto const& o = std::get<SunOutcomes>(value);
    return sizeof(SunOutcomes) + o.labels.capacity() * sizeof(Sector) +
           o.multiplicity.size() * map_node_bytes;
}

SymbolCache&
symbol_cache()
{
    static SymbolCache cache(
      [] {
          return static_cast<std::size_t>(get_config().su_n_symbol_cache_mb) *
                 (std::size_t{ 1 } << 20);
      },
      [](std::string const& key, SymbolValue const& value) {
          constexpr std::size_t node_overhead = 112; // list node + hash node + control block
          return 2 * (sizeof(std::string) + key.capacity()) + node_overhead +
                 symbol_value_bytes(value);
      });
    return cache;
}

/// Per-object state: unique id for the cache keys, weights of the three files (read once), and
/// whether the F file is a legacy file (no ``Gauge_Version`` attribute). Legacy files omit F
/// blocks that vanish; files from the current generator contain every allowed key.
struct SunState
{
    std::uint64_t id = 0;
    std::atomic<int64> h_cg{ -1 };
    std::atomic<int64> h_f{ -1 };
    std::atomic<int64> h_r{ -1 };
    bool legacy_f_file = true;
    // F / R file in the packed layout (null: folder layout)
    std::shared_ptr<sun_packed::PackedIndex const> f_packed;
    std::shared_ptr<sun_packed::PackedIndex const> r_packed;
};

/// Check the group ``F_sym`` / ``R_sym`` of a symbol file in a single pass over its keys: there is
/// a key for the all-trivial sectors and a key containing the highest weight twice. The number of
/// F keys grows like weight^5 (110k at weight 12), so this must stay linear: indexing the keys of
/// an old-style HDF5 group one by one (``H5Lget_name_by_idx``) is quadratic.
void
check_symbol_group(HighFive::Group& root, char kind, int64 highest_weight)
{
    struct State
    {
        std::string prefix;
        std::string zero_key;
        std::string h_key;
        char kind = 'F';
        int64 highest_weight = 0;
        bool have_first = false;
        bool found_h = false;
    };
    std::string const group_name = std::string(1, kind) + "_sym";
    if (!hdf5_io::h5_contains(root, group_name)) {
        throw std::invalid_argument("HDF5 file does not contain '/" + group_name + "/' group.");
    }
    State state;
    state.kind = kind;
    state.prefix = std::string(1, kind) + "[";
    state.highest_weight = highest_weight;
    auto callback =
      [](hid_t /*group*/, char const* name, H5L_info2_t const* /*info*/, void* data) {
          auto& st = *static_cast<State*>(data);
          std::string_view const key(name);
          if (!st.have_first && key.starts_with(st.prefix)) {
              st.have_first = true;
              auto const num_lists = static_cast<int>(std::ranges::count(key, '['));
              auto const commas = static_cast<int>(std::ranges::count(key, ','));
              st.zero_key = std::string(1, st.kind);
              for (int i = 0; i < num_lists; ++i) {
                  st.zero_key += "[0";
                  for (int j = 0; j < commas / num_lists; ++j) {
                      st.zero_key += ", 0";
                  }
                  st.zero_key += "]";
              }
              std::string h_bracket = "[" + std::to_string(st.highest_weight);
              for (int j = 0; j < commas / num_lists; ++j) {
                  h_bracket += ", 0";
              }
              h_bracket += "]";
              st.h_key = h_bracket + h_bracket;
          }
          if (st.have_first && key.find(st.h_key) != std::string_view::npos) {
              st.found_h = true;
              return herr_t{ 1 }; // found: stop the iteration
          }
          return herr_t{ 0 };
      };
    hid_t group = hdf5_io::h5_open(root, group_name);
    hsize_t position = 0;
    herr_t const status =
      H5Literate2(group, H5_INDEX_NAME, H5_ITER_INC, &position, callback, &state);
    if (status < 0) {
        H5Idec_ref(group);
        throw std::runtime_error("SUN: iterating over the keys of '/" + group_name + "/' failed");
    }
    bool const has_zero_key = state.have_first && hdf5_io::h5_contains(group, state.zero_key);
    H5Idec_ref(group);
    if (!state.have_first) {
        throw std::invalid_argument(std::string("No valid ") + kind + "-symbol keys found in '/" +
                                    group_name + "/'.");
    }
    if (!has_zero_key) {
        throw std::invalid_argument(std::string("Missing key for all-trivial-sector ") + kind +
                                    "-symbol: " + state.zero_key);
    }
    if (!state.found_h) {
        throw std::invalid_argument("No key found containing " + state.h_key + ".");
    }
}

void
append_sector_key(std::string& k, Sector const& s)
{
    k.push_back(static_cast<char>(s.len()));
    for (std::size_t i = 0; i < s.len(); ++i) {
        k.append(reinterpret_cast<char const*>(&s.q[i]), sizeof(int16_t));
    }
}

std::string
symbol_key(std::uint64_t id, char kind, std::initializer_list<Sector> sectors)
{
    std::string k;
    k.reserve(sizeof(id) + 1 + sectors.size() * (1 + 2 * max_sector_ind_len));
    k.append(reinterpret_cast<char const*>(&id), sizeof(id));
    k.push_back(kind);
    for (Sector const& s : sectors) {
        append_sector_key(k, s);
    }
    return k;
}

SunOutcomes
build_outcomes(HighFive::File& CGfile, int N, Sector a, Sector b)
{
    auto root = file_root(CGfile);
    auto key = cg_key(N, a, b);
    if (!cg_key_usable(CGfile, key)) {
        key = cg_key(N, b, a);
    }
    hid_t grp = hdf5_io::h5_open(root, key);
    SunOutcomes res;
    for (auto const& name : hdf5_io::h5_link_names(grp)) {
        hid_t child = hdf5_io::h5_open(grp, name);
        Sector lab = read_irrep_label_attr(child);
        res.labels.push_back(lab);
        if (name == std::string("Irrep") + sector_concat(lab) + "a1") {
            auto mult = hdf5_io::h5_get_attr_int64(child, "Outer Multiplicity");
            if (!mult) {
                H5Idec_ref(child);
                H5Idec_ref(grp);
                throw std::runtime_error("SUN: missing Outer Multiplicity attribute");
            }
            res.multiplicity[lab] = *mult;
        }
        H5Idec_ref(child);
    }
    H5Idec_ref(grp);
    return res;
}

std::shared_ptr<SunOutcomes const>
sun_outcomes(SunState const& state, HighFive::File& CGfile, int N, Sector a, Sector b)
{
    auto const entry = symbol_cache().get_or_make(symbol_key(state.id, 'O', { a, b }), [&] {
        return SymbolValue{ build_outcomes(CGfile, N, a, b) };
    });
    // aliasing constructor: keeps the cache entry alive while the outcomes are used
    return std::shared_ptr<SunOutcomes const>(entry, &std::get<SunOutcomes>(*entry));
}

} // namespace

CacheStats
symbol_cache_stats()
{
    return symbol_cache().stats();
}

void
clear_symbol_cache()
{
    symbol_cache().clear();
}

SUN::SUN(int N_,
         std::string cg_path,
         std::string f_path,
         std::string r_path,
         std::optional<std::string> descriptive_name)
  : Group(FusionStyle::general,
          zeros_sector(N_),
          "SU(" + std::to_string(N_) + ")",
          std::numeric_limits<float64>::infinity(),
          /*has_complex_topological_data=*/false,
          std::move(descriptive_name),
          /*trivial_shift=*/true)
  , N(N_)
  , CGpath(std::move(cg_path))
  , Fpath(std::move(f_path))
  , Rpath(std::move(r_path))
{
    if (N <= 1) {
        throw std::invalid_argument("Invalid N!");
    }
    if (static_cast<std::size_t>(N) > max_sector_ind_len) {
        throw std::invalid_argument("SUN: N exceeds max_sector_ind_len");
    }
    CGfile_ = std::make_shared<HighFive::File>(CGpath, HighFive::File::ReadOnly);
    Ffile_ = std::make_shared<HighFive::File>(Fpath, HighFive::File::ReadOnly);
    Rfile_ = std::make_shared<HighFive::File>(Rpath, HighFive::File::ReadOnly);

    auto n_cg = file_attr_int64(*CGfile_, "N");
    auto n_f = file_attr_int64(*Ffile_, "N");
    auto n_r = file_attr_int64(*Rfile_, "N");
    if (N != n_cg || N != n_f || N != n_r) {
        throw std::invalid_argument("Files must contain data for same N!");
    }
    sanity_check_hdf5(*CGfile_);
    // F and R files are in the folder layout (one dataset per key) or in the packed layout
    std::shared_ptr<sun_packed::PackedIndex const> f_packed;
    std::shared_ptr<sun_packed::PackedIndex const> r_packed;
    if (sun_packed::is_packed_layout(*Ffile_, Fpath)) {
        f_packed = sun_packed::PackedIndex::load(Ffile_, Fpath, 'F', N);
        f_packed->check(file_attr_int64(*Ffile_, "Highest_Weight"));
    } else {
        sanity_check_hdf5(*Ffile_);
    }
    if (sun_packed::is_packed_layout(*Rfile_, Rpath)) {
        r_packed = sun_packed::PackedIndex::load(Rfile_, Rpath, 'R', N);
        r_packed->check(file_attr_int64(*Rfile_, "Highest_Weight"));
    } else {
        sanity_check_hdf5(*Rfile_);
    }
    // Files written by the same generator run carry the same 'CG_Run_Id'; F and R symbols only
    // fit to the CG coefficients they were computed from.
    {
        auto const id_cg = hdf5_io::h5_get_attr_string(CGfile_->getId(), "CG_Run_Id");
        for (auto const& [file, what] :
             { std::pair{ Ffile_.get(), "F" }, std::pair{ Rfile_.get(), "R" } }) {
            auto const id = hdf5_io::h5_get_attr_string(file->getId(), "CG_Run_Id");
            if (id_cg && id && *id_cg != *id) {
                throw std::invalid_argument(std::format(
                  "SUN: the {} file and the CG file come from different generator runs "
                  "(CG_Run_Id {} vs {}); regenerate them together.",
                  what,
                  *id,
                  *id_cg));
            }
        }
    }
    fusion_tensor_dtype = Dtype::Float64;

    static std::atomic<std::uint64_t> next_id{ 1 };
    auto state = std::make_shared<SunState>();
    state->id = next_id.fetch_add(1);
    state->legacy_f_file = H5Aexists(Ffile_->getId(), "Gauge_Version") <= 0;
    state->f_packed = std::move(f_packed);
    state->r_packed = std::move(r_packed);
    state_ = state;
}

bool
SUN::is_valid_sector(Sector a) const
{
    if (a.len() != static_cast<std::uint8_t>(N)) {
        return false;
    }
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        if (a.q[i] < 0) {
            return false;
        }
    }
    for (std::uint8_t i = 0; i + 1 < a.len(); ++i) {
        if (a.q[i] < a.q[i + 1]) {
            return false;
        }
    }
    return a.q[a.len() - 1] == 0;
}

bool
SUN::_is_equivalent_factor(SymmetryFactor const& other) const
{
    if (auto const* sun = dynamic_cast<SUN const*>(&other)) {
        return sun->N == N;
    }
    return false;
}

int64
SUN::sector_dim(Sector a) const
{
    assert(is_valid_sector(a));
    // Weyl dimension formula  prod_{k<k'} (q_k - q_k' + k' - k) / (k' - k)  in exact integer
    // arithmetic: reduce every factor against the running fraction, so intermediate values stay
    // small and the result is exact (a float product is off by one for e.g. SU(4) weight >= 4).
    auto gcd128 = [](__int128 x, __int128 y) {
        while (y != 0) {
            auto t = x % y;
            x = y;
            y = t;
        }
        return x;
    };
    __int128 num = 1;
    __int128 den = 1;
    for (int kp = 2; kp <= N; ++kp) {
        for (int k = 1; k < kp; ++k) {
            __int128 n = static_cast<__int128>(a.q[k - 1]) - a.q[kp - 1] + (kp - k);
            __int128 d = kp - k;
            auto g1 = gcd128(n, den);
            n /= g1;
            den /= g1;
            auto g2 = gcd128(num, d);
            num /= g2;
            d /= g2;
            num *= n;
            den *= d;
        }
    }
    auto const dim = num / den;
    if (dim > std::numeric_limits<int64>::max()) {
        throw std::overflow_error("SUN::sector_dim: dimension does not fit into int64");
    }
    return static_cast<int64>(dim);
}

std::string
SUN::repr() const
{
    return "SUNSymmetry(N=" + std::to_string(N) + ")";
}

Sector
SUN::dual_sector(Sector a) const
{
    int16_t mx = a.q[0];
    for (std::uint8_t i = 1; i < a.len(); ++i) {
        mx = std::max(mx, a.q[i]);
    }
    std::array<int16_t, max_sector_ind_len> buf{};
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        auto v = static_cast<int16_t>(a.q[i] - mx);
        buf[a.len() - 1 - i] = static_cast<int16_t>(std::abs(v));
    }
    return Sector::from_span(std::span<const int16_t>(buf.data(), a.len()));
}

int64
SUN::hweight_from_CG_hdf5() const
{
    auto& state = *static_cast<SunState*>(state_.get());
    auto v = state.h_cg.load(std::memory_order_relaxed);
    if (v < 0) {
        v = file_attr_int64(*CGfile_, "Highest_Weight");
        state.h_cg.store(v, std::memory_order_relaxed);
    }
    return v;
}

int64
SUN::hweight_from_F_hdf5() const
{
    auto& state = *static_cast<SunState*>(state_.get());
    auto v = state.h_f.load(std::memory_order_relaxed);
    if (v < 0) {
        v = file_attr_int64(*Ffile_, "Highest_Weight");
        state.h_f.store(v, std::memory_order_relaxed);
    }
    return v;
}

int64
SUN::hweight_from_R_hdf5() const
{
    auto& state = *static_cast<SunState*>(state_.get());
    auto v = state.h_r.load(std::memory_order_relaxed);
    if (v < 0) {
        v = file_attr_int64(*Rfile_, "Highest_Weight");
        state.h_r.store(v, std::memory_order_relaxed);
    }
    return v;
}

bool
SUN::can_fuse_to(Sector a, Sector b, Sector c) const
{
    auto const hmax = hweight_from_CG_hdf5();
    if (a.q[0] > hmax || b.q[0] > hmax) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    if (c.q[0] > a.q[0] + b.q[0]) {
        return false;
    }
    auto const outcomes = sun_outcomes(*static_cast<SunState*>(state_.get()), *CGfile_, N, a, b);
    return std::ranges::find(outcomes->labels, c) != outcomes->labels.end();
}

int64
SUN::_n_symbol(Sector a, Sector b, Sector c) const
{
    auto const outcomes = sun_outcomes(*static_cast<SunState*>(state_.get()), *CGfile_, N, a, b);
    auto const it = outcomes->multiplicity.find(c);
    return it == outcomes->multiplicity.end() ? 0 : it->second;
}

int64
SUN::S_index_irrep_weight(Sector a) const
{
    int64 S = 0;
    for (int k = 1; k < N; ++k) {
        S += binomial(N - k + a.q[k - 1] - 1, N - k);
    }
    return S;
}

Sector
SUN::highest_irrep_in_decomp(Sector a, Sector b) const
{
    assert(a.len() == b.len());
    std::array<int16_t, max_sector_ind_len> buf{};
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        buf[i] = static_cast<int16_t>(a.q[i] + b.q[i]);
    }
    return Sector::from_span(std::span<const int16_t>(buf.data(), a.len()));
}

SectorArray
SUN::fusion_outcomes(Sector a, Sector b) const
{
    auto const hmax = hweight_from_CG_hdf5();
    if (a.q[0] > hmax || b.q[0] > hmax) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    auto const outcomes = sun_outcomes(*static_cast<SunState*>(state_.get()), *CGfile_, N, a, b);
    return SectorArray(std::vector<Sector>(outcomes->labels));
}

std::map<Sector, int64>
SUN::dims_of_irreps(Sector a, Sector b) const
{
    auto outcomes = fusion_outcomes(a, b);
    auto key = cg_key(N, a, b);
    auto root = cg_root();
    hid_t grp = hdf5_io::h5_open(root, key);
    std::map<Sector, int64> C;
    for (std::size_t i = 0; i < outcomes.size(); ++i) {
        Sector ir = outcomes[i];
        auto obj = std::string("Irrep") + sector_concat(ir) + "a1";
        hid_t dset = hdf5_io::h5_open(grp, obj);
        auto dim = hdf5_io::h5_get_attr_int64(dset, "Dimension");
        H5Idec_ref(dset);
        if (!dim) {
            H5Idec_ref(grp);
            throw std::runtime_error("SUN: missing Dimension attribute");
        }
        C[ir] = *dim;
    }
    H5Idec_ref(grp);
    return C;
}

std::map<Sector, int64>
SUN::outer_multiplicity_from_CG(Sector a, Sector b) const
{
    auto outcomes = fusion_outcomes(a, b);
    auto key = cg_key(N, a, b);
    auto root = cg_root();
    hid_t grp = hdf5_io::h5_open(root, key);
    std::map<Sector, int64> C;
    for (std::size_t i = 0; i < outcomes.size(); ++i) {
        Sector ir = outcomes[i];
        auto obj = std::string("Irrep") + sector_concat(ir) + "a1";
        hid_t dset = hdf5_io::h5_open(grp, obj);
        auto mult = hdf5_io::h5_get_attr_int64(dset, "Outer Multiplicity");
        H5Idec_ref(dset);
        if (!mult) {
            H5Idec_ref(grp);
            throw std::runtime_error("SUN: missing Outer Multiplicity attribute");
        }
        C[ir] = *mult;
    }
    H5Idec_ref(grp);
    return C;
}

float64
SUN::clebschgordan(Sector a, int64 q_a, Sector b, int64 q_b, Sector c, int64 q_c, int64 mu) const
{
    auto const hw = hweight_from_CG_hdf5();
    if (a.q[0] > hw || b.q[0] > hw || c.q[0] > hw) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    auto key1 = cg_key(N, a, b);
    auto key2 = std::string("Irrep") + sector_concat(c) + "a" + std::to_string(mu);
    double qa = static_cast<double>(q_a);
    double qb = static_cast<double>(q_b);
    double qc = static_cast<double>(q_c);
    if (!cg_key_usable(*CGfile_, key1)) {
        key1 = cg_key(N, b, a);
        std::swap(qa, qb);
    }
    auto rows = read_cg_rows(*CGfile_, key1, key2);
    for (auto const& row : rows) {
        if (row[0] == qa && row[1] == qb && row[2] == qc) {
            return row[3];
        }
    }
    return 0.0;
}

FusionSymbol
SUN::_fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const
{
    if (Z_a || Z_b) {
        throw std::runtime_error("SUN::_fusion_tensor: Z_a/Z_b not implemented");
    }
    auto const hw = hweight_from_CG_hdf5();
    if (a.q[0] > hw || b.q[0] > hw || c.q[0] > hw) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    auto dim_Sa = static_cast<std::size_t>(sector_dim(a));
    auto dim_Sb = static_cast<std::size_t>(sector_dim(b));
    auto dim_Sc = static_cast<std::size_t>(sector_dim(c));
    auto dim_mu = _n_symbol(a, b, c);
    if (dim_mu == 0) {
        return FusionSymbol::zeros(
          4, FusionSymbol::Shape{ { dim_Sa, dim_Sb, dim_Sc, 1 } }, Dtype::Float64);
    }
    FusionSymbol X(
      4,
      FusionSymbol::Shape{ { dim_Sa, dim_Sb, dim_Sc, static_cast<std::size_t>(dim_mu) } },
      Dtype::Float64);
    for (int64 m_a = 1; m_a <= static_cast<int64>(dim_Sa); ++m_a) {
        for (int64 m_b = 1; m_b <= static_cast<int64>(dim_Sb); ++m_b) {
            for (int64 m_c = 1; m_c <= static_cast<int64>(dim_Sc); ++m_c) {
                for (int64 mu = 1; mu <= dim_mu; ++mu) {
                    auto rr = clebschgordan(a, m_a, b, m_b, c, m_c, mu);
                    X.set(static_cast<std::size_t>(m_a - 1),
                          static_cast<std::size_t>(m_b - 1),
                          static_cast<std::size_t>(m_c - 1),
                          static_cast<std::size_t>(mu - 1),
                          complex128{ rr, 0.0 });
                }
            }
        }
    }
    return X.transpose(std::array<std::uint8_t, 4>{ { 3, 0, 1, 2 } });
}

FusionSymbol
SUN::_f_symbol_from_CG(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    auto const hw = hweight_from_CG_hdf5();
    if (a.q[0] > hw || b.q[0] > hw || c.q[0] > hw || d.q[0] > hw || e.q[0] > hw || f.q[0] > hw) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    auto& be = *static_cast<BlockBackend*>(NumpyBlockBackend::from_factory("cpu"));
    auto X1 =
      block_from_fusion_symbol(be,
                               _fusion_tensor(a, b, f, false, false)
                                 .transpose(std::array<std::uint8_t, 4>{ { 1, 2, 3, 0 } }));
    auto X2 =
      block_from_fusion_symbol(be,
                               _fusion_tensor(f, c, d, false, false)
                                 .transpose(std::array<std::uint8_t, 4>{ { 1, 2, 3, 0 } }));
    auto X3 =
      block_from_fusion_symbol(be,
                               _fusion_tensor(b, c, e, false, false)
                                 .transpose(std::array<std::uint8_t, 4>{ { 1, 2, 3, 0 } }));
    auto X4 =
      block_from_fusion_symbol(be,
                               _fusion_tensor(a, e, d, false, false)
                                 .transpose(std::array<std::uint8_t, 4>{ { 1, 2, 3, 0 } }));
    if (!be.any(X1) || !be.any(X2) || !be.any(X3) || !be.any(X4)) {
        return FusionSymbol::zeros(4, FusionSymbol::Shape{ { 1, 1, 1, 1 } }, Dtype::Complex128);
    }
    auto X12 = be.tdot(X1, X2, { 2 }, { 0 });
    X12 = be.permute_axes(X12, { 0, 1, 3, 4, 2, 5 });
    auto X34 = be.tdot(X3, X4, { 2 }, { 1 });
    X34 = be.permute_axes(X34, { 3, 0, 1, 4, 2, 5 });
    auto F = be.tdot(X12, be.conj(X34), { 0, 1, 2, 3 }, { 0, 1, 2, 3 });
    F = be.permute_axes(F, { 2, 3, 0, 1 });
    auto out = fusion_symbol_from_block(F).as_complex();
    auto span = out.as_complex128();
    for (auto& v : span) {
        if (std::abs(v) < 1e-12) {
            v = complex128{ 0.0, 0.0 };
        }
    }
    auto denom = static_cast<float64>(sector_dim(d));
    return out * (1.0 / denom);
}

FusionSymbol
SUN::_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    auto const hmax = hweight_from_F_hdf5();
    if (a.q[0] > hmax || b.q[0] > hmax || c.q[0] > hmax || d.q[0] > hmax || e.q[0] > hmax ||
        f.q[0] > hmax) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    auto const& state = *static_cast<SunState*>(state_.get());
    auto const cached =
      symbol_cache().get_or_make(symbol_key(state.id, 'F', { a, b, c, d, e, f }), [&] {
          std::string key = "F";
          for (Sector const& s : { a, b, c, d, e, f }) {
              key += sector_bracket(s);
          }
          auto root = f_root();
          if (state.f_packed) {
              if (auto const i = state.f_packed->find(std::array<Sector, 6>{ a, b, c, d, e, f })) {
                  return SymbolValue{ state.f_packed->read(*i) };
              }
          } else if (hdf5_io::h5_contains(root, std::string("F_sym/") + key)) {
              hid_t fsym = hdf5_io::h5_open(root, "F_sym");
              hid_t dset = hdf5_io::h5_open(fsym, key);
              auto buf = hdf5_io::h5_read_buffer(dset);
              H5Idec_ref(dset);
              H5Idec_ref(fsym);
              return SymbolValue{ fusion_symbol_from_hdf5_buffer(buf) };
          }
          // Not stored. The shape of the F symbol is given by the four fusion multiplicities.
          auto const n1 = _n_symbol(b, c, e);
          auto const n2 = _n_symbol(a, e, d);
          auto const n3 = _n_symbol(a, b, f);
          auto const n4 = _n_symbol(f, c, d);
          if (n1 == 0 || n2 == 0 || n3 == 0 || n4 == 0) {
              throw SymmetryError("Sectors are not consistent with fusion rules.");
          }
          if (state.legacy_f_file) {
              // Files without a 'Gauge_Version' attribute omit the F symbols that vanish
              // identically, so an allowed channel without a dataset is an all-zero block.
              return SymbolValue{ FusionSymbol::zeros(
                4,
                FusionSymbol::Shape{ { static_cast<std::size_t>(n1),
                                       static_cast<std::size_t>(n2),
                                       static_cast<std::size_t>(n3),
                                       static_cast<std::size_t>(n4) } },
                Dtype::Complex128) };
          }
          throw std::runtime_error(std::format(
            "SUN: the F-symbol file '{}' is incomplete: it has no dataset '{}' although all four "
            "fusion channels are allowed. Files written by the current generator contain every "
            "allowed key; please regenerate the file.",
            Fpath,
            key));
      });
    return std::get<FusionSymbol>(*cached);
}

FusionSymbol
SUN::_r_symbol_from_CG(Sector a, Sector b, Sector c) const
{
    auto const hw = hweight_from_CG_hdf5();
    if (a.q[0] > hw || b.q[0] > hw || c.q[0] > hw) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    BlockBackend& be = *NumpyBlockBackend::from_factory("cpu");
    auto X1 = block_from_fusion_symbol(be, fusion_tensor(a, b, c));
    auto Y1 = be.conj(block_from_fusion_symbol(be, fusion_tensor(b, a, c)));
    if (!be.any(X1) || !be.any(Y1)) {
        auto mult = static_cast<std::size_t>(n_symbol(a, b, c));
        return FusionSymbol::zeros(1, FusionSymbol::Shape{ { mult, 1, 1, 1 } }, Dtype::Complex128);
    }
    auto R = be.tdot(X1, Y1, { 0, 1, 2 }, { 1, 0, 2 });
    auto denom = static_cast<float64>(sector_dim(c));
    R = be.mul(1.0 / denom, be.permute_axes(R, { 1, 0 }));
    return fusion_symbol_from_block(be.get_diagonal(R, std::nullopt));
}

FusionSymbol
SUN::_r_symbol(Sector a, Sector b, Sector c) const
{
    auto const hmax = hweight_from_R_hdf5();
    if (a.q[0] > hmax || b.q[0] > hmax || c.q[0] > hmax) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    auto const& state = *static_cast<SunState*>(state_.get());
    auto const cached = symbol_cache().get_or_make(symbol_key(state.id, 'R', { a, b, c }), [&] {
        std::string key = "R";
        for (Sector const& s : { a, b, c }) {
            key += sector_bracket(s);
        }
        auto root = r_root();
        if (state.r_packed) {
            if (auto const i = state.r_packed->find(std::array<Sector, 3>{ a, b, c })) {
                return SymbolValue{ state.r_packed->read(*i) };
            }
        } else if (hdf5_io::h5_contains(root, std::string("R_sym/") + key)) {
            hid_t rsym = hdf5_io::h5_open(root, "R_sym");
            hid_t dset = hdf5_io::h5_open(rsym, key);
            auto buf = hdf5_io::h5_read_buffer(dset);
            H5Idec_ref(dset);
            H5Idec_ref(rsym);
            return SymbolValue{ fusion_symbol_from_hdf5_buffer(buf) };
        }
        if (_n_symbol(a, b, c) == 0) {
            throw SymmetryError("Sectors are not consistent with fusion rules.");
        }
        // R symbols are phases, they never vanish, so a missing allowed key means a broken file.
        throw std::runtime_error(std::format(
          "SUN: the R-symbol file '{}' is incomplete: it has no dataset '{}' although the fusion "
          "channel is allowed. Please regenerate the file.",
          Rpath,
          key));
    });
    return std::get<FusionSymbol>(*cached);
}

int64
SUN::frobenius_schur(Sector a) const
{
    if (N == 2) {
        return 1 - 2 * (static_cast<int64>(a.q[0]) % 2);
    }
    auto F = _f_symbol(a, dual_sector(a), a, a, trivial_sector, trivial_sector);
    auto const val = F.get_complex(0, 0, 0, 0);
    float64 const r = val.real();
    return static_cast<int64>((r > 0.0) - (r < 0.0));
}

bool
SUN::has_data_in_group(hid_t loc) const
{
    H5O_info2_t info{};
    if (H5Oget_info3(loc, &info, H5O_INFO_BASIC) < 0) {
        return false;
    }
    if (info.type == H5O_TYPE_DATASET) {
        hid_t space = H5Dget_space(loc);
        hssize_t n = H5Sget_simple_extent_npoints(space);
        H5Sclose(space);
        return n > 0;
    }
    if (info.type == H5O_TYPE_GROUP) {
        for (auto const& name : hdf5_io::h5_link_names(loc)) {
            hid_t child = hdf5_io::h5_open(loc, name);
            bool ok = has_data_in_group(child);
            H5Idec_ref(child);
            if (ok) {
                return true;
            }
        }
    }
    return false;
}

void
SUN::sanity_check_hdf5(HighFive::File const& file) const
{
    // HighFive::File::getId is non-const; cast away for read-only attr/group access.
    auto& f = const_cast<HighFive::File&>(file);
    auto H = file_attr_int64(f, "Highest_Weight");
    auto Nattr = file_attr_int64(f, "N");
    auto root = file_root(f);
    if (sun_packed::is_packed_layout(f, f.getName())) {
        // symbol file in the packed layout: check its index (the data is not needed)
        char const kind = hdf5_io::h5_contains(root, "F_packed") ? 'F' : 'R';
        std::shared_ptr<HighFive::File> borrowed(&f, [](HighFive::File*) {});
        sun_packed::PackedIndex::load(borrowed, f.getName(), kind, static_cast<int>(Nattr), false)
          ->check(H);
        return;
    }
    auto keys0 = hdf5_io::h5_link_names(root.getId());
    if (keys0.empty()) {
        throw std::invalid_argument("SUN sanity_check_hdf5: empty HDF5 file");
    }
    char ft = keys0[0].empty() ? '?' : keys0[0][0];

    if (ft == 'F' || ft == 'R') {
        check_symbol_group(root, ft, H);
    } else if (ft == 'N') {
        auto path = std::string("N_") + std::to_string(Nattr);
        if (!hdf5_io::h5_contains(root, path)) {
            throw std::invalid_argument("HDF5 file does not contain /" + path + "/ group.");
        }
        hid_t parent = hdf5_io::h5_open(root, path);
        auto keys = hdf5_io::h5_link_names(parent);
        if (static_cast<int64>(keys.size()) != H + 1) {
            H5Idec_ref(parent);
            throw std::runtime_error("SUN sanity_check_hdf5: unexpected CG key count");
        }
        for (auto const& idx : { keys.back(), keys.front() }) {
            hid_t group = hdf5_io::h5_open(parent, idx);
            if (hdf5_io::h5_link_names(group).empty()) {
                H5Idec_ref(group);
                H5Idec_ref(parent);
                throw std::runtime_error("SUN sanity_check_hdf5: empty weight group");
            }
            if (!has_data_in_group(group)) {
                H5Idec_ref(group);
                H5Idec_ref(parent);
                throw std::invalid_argument("Key exists but contains no data.");
            }
            H5Idec_ref(group);
        }
        H5Idec_ref(parent);
    }
}

void
SUN::save_hdf5(lasyten::hdf5::Saver& saver,
               HighFive::Group& h5gr,
               std::string const& subpath) const
{
    SymmetryFactor::save_hdf5(saver, h5gr, subpath);
    // Persist paths so from_hdf5 can reopen.
    // Prefer saver rooted at the instance group when possible; fall back to full paths.
    (void)h5gr;
    saver.save_int64(subpath + "N", N);
    saver.save_string(subpath + "CGfile", CGpath);
    saver.save_string(subpath + "Ffile", Fpath);
    saver.save_string(subpath + "Rfile", Rpath);
}

SUN::Ptr
SUN::from_hdf5(lasyten::hdf5::Loader& loader, HighFive::Group& h5gr, std::string const& subpath)
{
    int N = static_cast<int>(load_int64_child(loader, subpath + "N"));
    auto name = descriptive_name_from_hdf5_attrs(h5gr);
    auto cg = load_string_child(loader, subpath + "CGfile");
    auto ff = load_string_child(loader, subpath + "Ffile");
    auto rr = load_string_child(loader, subpath + "Rfile");
    auto obj = std::make_shared<SUN>(N, std::move(cg), std::move(ff), std::move(rr), name);
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

namespace {

/// Largest ``H <= limit`` for which ``<base>_N<N>_<kind>_hweight<H>.hdf5`` exists in `dir`.
std::optional<int64>
largest_su_n_weight(std::string const& dir,
                    std::string const& base,
                    int N,
                    std::string const& kind,
                    int64 limit)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return std::nullopt;
    }
    std::string const prefix = std::format("{}_N{}_{}_hweight", base, N, kind);
    std::string const suffix = ".hdf5";
    std::optional<int64> best;
    for (auto const& entry : fs::directory_iterator(dir, ec)) {
        auto const name = entry.path().filename().string();
        if (name.size() <= prefix.size() + suffix.size() || !name.starts_with(prefix) ||
            !name.ends_with(suffix)) {
            continue;
        }
        auto const digits =
          name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
        if (digits.size() > 9 ||
            !std::ranges::all_of(digits, [](unsigned char c) { return std::isdigit(c) != 0; })) {
            continue;
        }
        auto const h = static_cast<int64>(std::stoll(digits));
        if (h <= limit && (!best || h > *best)) {
            best = h;
        }
    }
    return best;
}

} // namespace

std::array<std::optional<int64>, 3>
su_n_detect_hweights(int N,
                     std::optional<std::string> path,
                     std::optional<std::string> filename_base,
                     std::optional<int64> cg_hweight,
                     std::optional<int64> f_hweight,
                     std::optional<int64> r_hweight)
{
    auto const& config = get_config();
    auto const dir = expand_user(path.value_or(config.su_n_data_path));
    auto const& base = filename_base.value_or(config.su_n_data_filename_base);
    auto const pinned = su_n_pinned_hweights(N);
    auto pick = [&](std::optional<int64> given, int64 pin, std::string const& kind, int64 limit) {
        if (given.has_value()) {
            return given;
        }
        if (pin > 0) {
            return std::optional<int64>(pin);
        }
        return largest_su_n_weight(dir, base, N, kind, limit);
    };
    auto const cg = pick(cg_hweight, pinned[0], "CG", std::numeric_limits<int64>::max());
    int64 const limit = cg.value_or(std::numeric_limits<int64>::max());
    return { cg, pick(f_hweight, pinned[1], "F", limit), pick(r_hweight, pinned[2], "R", limit) };
}

SUN::Ptr
SUN::from_config(int N,
                 std::optional<int64> hweight,
                 std::optional<int64> cg_hweight,
                 std::optional<int64> f_hweight,
                 std::optional<int64> r_hweight,
                 std::optional<std::string> path,
                 std::optional<std::string> filename_base,
                 std::optional<std::string> descriptive_name)
{
    int64 h_cg = 0;
    int64 h_f = 0;
    int64 h_r = 0;
    if (hweight.has_value()) {
        h_cg = cg_hweight.value_or(*hweight);
        h_f = f_hweight.value_or(*hweight);
        h_r = r_hweight.value_or(*hweight);
    } else {
        auto const detected =
          su_n_detect_hweights(N, path, filename_base, cg_hweight, f_hweight, r_hweight);
        for (std::size_t i = 0; i < 3; ++i) {
            if (!detected[i].has_value()) {
                constexpr std::array<char const*, 3> kinds{ "CG", "F", "R" };
                std::string const msg = std::format(
                  "No SU({}) {} data file found in '{}' (name pattern '{}_N{}_{}_hweight<H>.hdf5'); pass the "
                  "weights explicitly, set the config option 'su_n_hweights', or generate the data.",
                  N,
                  kinds[i],
                  expand_user(path.value_or(get_config().su_n_data_path)),
                  filename_base.value_or(get_config().su_n_data_filename_base),
                  N,
                  kinds[i]);
                PyErr_SetString(PyExc_FileNotFoundError, msg.c_str());
                throw py::error_already_set();
            }
        }
        h_cg = *detected[0];
        h_f = *detected[1];
        h_r = *detected[2];
    }
    for (auto [h, what] : { std::pair{ h_cg, "cg_hweight" },
                            std::pair{ h_f, "f_hweight" },
                            std::pair{ h_r, "r_hweight" } }) {
        if (h < 0) {
            throw std::invalid_argument(std::string("SUN: ") + what + " must be >= 0");
        }
    }
    if (h_cg < h_f || h_cg < h_r) {
        throw std::invalid_argument(std::format(
          "SUN: the CG hweight ({}) must be >= the F ({}) and R ({}) hweights.", h_cg, h_f, h_r));
    }
    auto CGpath = su_n_data_file_path(N, "CG", h_cg, path, filename_base);
    auto Fpath = su_n_data_file_path(N, "F", h_f, path, filename_base);
    auto Rpath = su_n_data_file_path(N, "R", h_r, path, filename_base);
    // Validate files exist and attrs match before constructing.
    (void)open_su_n_data_file(CGpath, "Clebsch-Gordan", N, h_cg);
    (void)open_su_n_data_file(Fpath, "F-symbol", N, h_f);
    (void)open_su_n_data_file(Rpath, "R-symbol", N, h_r);
    return std::make_shared<SUN>(
      N, std::move(CGpath), std::move(Fpath), std::move(Rpath), std::move(descriptive_name));
}

} // namespace lasyten
