#include <lasyten/symmetries/trees.h>

#include <lasyten/block_backend/numpy.h>
#include <lasyten/config.h>
#include <lasyten/symmetries/exceptions.h>
#include <lasyten/symmetries/fusion_symbol.h>
#include <lasyten/tools.h>
#include <lasyten/tools/lru_cache.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <format>
#include <functional>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace lasyten {

namespace {

SectorArray
concat_sector_arrays_many(std::span<SectorArray const* const> arrays)
{
    if (arrays.empty()) {
        return SectorArray{};
    }
    SectorArray res = *arrays[0];
    for (std::size_t i = 1; i < arrays.size(); ++i) {
        res = res.concat(*arrays[i]);
    }
    return res;
}

SectorArray
concat_sector_arrays_many(std::initializer_list<SectorArray> arrays)
{
    if (arrays.size() == 0) {
        return SectorArray{};
    }
    auto it = arrays.begin();
    SectorArray res = *it;
    for (++it; it != arrays.end(); ++it) {
        res = res.concat(*it);
    }
    return res;
}

SectorArray
sector_array_from_sectors(std::span<Sector const> sectors, std::uint8_t sector_ind_len)
{
    SectorArray out(sectors.size(), sector_ind_len);
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        out[i] = sectors[i];
    }
    return out;
}

template<typename T>
std::vector<T>
concat_vectors(std::span<T const> a, std::span<T const> b)
{
    std::vector<T> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

template<typename T>
std::vector<T>
concat_vectors(std::span<T const> a, std::span<T const> b, std::span<T const> c)
{
    std::vector<T> out;
    out.reserve(a.size() + b.size() + c.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    out.insert(out.end(), c.begin(), c.end());
    return out;
}

template<typename T>
std::vector<T>
vector_slice(std::vector<T> const& v, std::size_t start, std::size_t stop)
{
    // Match NumPy ``arr[start:stop]``: clamp ``stop``, empty if ``start >= stop``.
    if (stop > v.size()) {
        stop = v.size();
    }
    if (start >= stop) {
        return {};
    }
    return std::vector<T>(v.begin() + static_cast<std::ptrdiff_t>(start),
                          v.begin() + static_cast<std::ptrdiff_t>(stop));
}

std::string
join_strings(std::span<std::string const> parts, std::string_view sep)
{
    std::ostringstream oss;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            oss << sep;
        }
        oss << parts[i];
    }
    return oss.str();
}

std::string
sector_array_str(SectorArray const& arr)
{
    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < arr.size(); ++i) {
        if (i > 0) {
            oss << '\n';
        }
        oss << " [";
        Sector const& row = arr[i];
        for (std::uint8_t j = 0; j < arr.sector_ind_len(); ++j) {
            if (j > 0) {
                oss << ' ';
            }
            oss << row[j];
        }
        oss << ']';
    }
    oss << ']';
    return oss.str();
}

std::string
replace_all(std::string s, std::string_view from, std::string_view to)
{
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

void
hash_combine(std::size_t& seed, std::size_t value) noexcept
{
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

std::size_t
hash_sector(Sector const& s) noexcept
{
    return std::hash<Sector>{}(s);
}

std::size_t
hash_sector_array(SectorArray const& arr) noexcept
{
    std::size_t seed = arr.size();
    hash_combine(seed, arr.sector_ind_len());
    for (Sector const& s : arr) {
        hash_combine(seed, hash_sector(s));
    }
    return seed;
}

std::size_t
hash_int64_vector(std::vector<int64> const& v) noexcept
{
    std::size_t seed = v.size();
    for (auto x : v) {
        hash_combine(seed, static_cast<std::size_t>(x));
    }
    return seed;
}

std::size_t
hash_uint8_vector(std::vector<std::uint8_t> const& v) noexcept
{
    std::size_t seed = v.size();
    for (auto x : v) {
        hash_combine(seed, static_cast<std::size_t>(x));
    }
    return seed;
}

/// ``tensor[mu, :, :, :]`` for fusion tensors shaped ``[μ, a0, a1, c]``.
FusionSymbol
fusion_tensor_slice_mu(FusionSymbol const& tensor, std::size_t mu)
{
    FusionSymbol::Shape const sh{ { tensor.extent(1), tensor.extent(2), tensor.extent(3), 1 } };
    FusionSymbol out(3, sh, tensor.dtype());
    for (std::size_t i1 = 0; i1 < sh[0]; ++i1) {
        for (std::size_t i2 = 0; i2 < sh[1]; ++i2) {
            for (std::size_t i3 = 0; i3 < sh[2]; ++i3) {
                out.set(i1, i2, i3, tensor.get_complex(mu, i1, i2, i3));
            }
        }
    }
    return out;
}

std::vector<std::vector<std::string>>
make_char_grid(std::size_t cols, std::size_t rows, std::string fill = " ")
{
    return std::vector<std::vector<std::string>>(cols, std::vector<std::string>(rows, fill));
}

/// Advance one UTF-8 codepoint; returns byte length (1–4), or 1 on invalid input.
std::size_t
utf8_codepoint_len(std::string_view s, std::size_t pos)
{
    if (pos >= s.size()) {
        return 0;
    }
    auto const c = static_cast<unsigned char>(s[pos]);
    if ((c & 0x80u) == 0) {
        return 1;
    }
    if ((c & 0xE0u) == 0xC0u) {
        return 2;
    }
    if ((c & 0xF0u) == 0xE0u) {
        return 3;
    }
    if ((c & 0xF8u) == 0xF0u) {
        return 4;
    }
    return 1;
}

void
write_string_to_col(std::vector<std::vector<std::string>>& grid,
                    std::size_t col_start,
                    std::size_t row,
                    std::string_view s)
{
    std::size_t col = col_start;
    for (std::size_t pos = 0; pos < s.size();) {
        auto const n = utf8_codepoint_len(s, pos);
        assert(col < grid.size());
        assert(row < grid[col].size());
        grid[col][row] = std::string(s.substr(pos, n));
        pos += n;
        ++col;
    }
}

void
set_cell(std::vector<std::vector<std::string>>& grid,
         std::size_t col,
         std::size_t row,
         std::string_view s)
{
    assert(col < grid.size());
    assert(row < grid[col].size());
    grid[col][row] = std::string(s);
}

void
reverse_cols(std::vector<std::vector<std::string>>& grid)
{
    std::reverse(grid.begin(), grid.end());
}

std::vector<std::vector<std::string>>
prepend_rows(std::vector<std::vector<std::string>> const& extra_left,
             std::vector<std::vector<std::string>> const& ascii)
{
    std::vector<std::vector<std::string>> out;
    out.reserve(extra_left.size() + ascii.size());
    out.insert(out.end(), extra_left.begin(), extra_left.end());
    out.insert(out.end(), ascii.begin(), ascii.end());
    return out;
}

std::string
ascii_grid_to_string(std::vector<std::vector<std::string>> const& grid)
{
    if (grid.empty()) {
        return {};
    }
    auto const num_cols = grid.size();
    auto const num_rows = grid[0].size();
    std::ostringstream oss;
    for (std::size_t row = 0; row < num_rows; ++row) {
        if (row > 0) {
            oss << '\n';
        }
        for (std::size_t col = 0; col < num_cols; ++col) {
            oss << grid[col][row];
        }
    }
    return oss.str();
}

void
map_add_coeff(FusionTreeLinearCombination& dest, FusionTreeLinearCombination const& src)
{
    for (auto const& [tree, coeff] : src) {
        dest[tree] += coeff;
    }
}

} // namespace

FusionTree::FusionTree(Symmetry::Ptr symmetry,
                       SectorArray uncoupled,
                       Sector coupled,
                       std::vector<std::uint8_t> are_dual,
                       SectorArray inner_sectors,
                       std::optional<std::vector<int64>> multiplicities)
  : symmetry(std::move(symmetry))
  , uncoupled(std::move(uncoupled))
  , coupled(coupled)
  , are_dual(std::move(are_dual))
  , inner_sectors(std::move(inner_sectors))
{
    // --- hints from Python FusionTree.__init__ ---
    // N uncoupled sectors
    // N flags: is there a Z isomorphism above the uncoupled sector
    // N - 2 internal sectors
    // N - 1 multiplicity labels; all 0 per default
    // empty lists are by default converted to arrays with dtype=float, which leads to issues in
    // __hash__
    // ---
    // OPTIMIZE demand SectorArray / ndarray (not list) and skip conversions?
    // C++ ctor already takes SectorArray; Python bindings still accept/convert lists.
    num_uncoupled = this->uncoupled.size();
    num_vertices = num_uncoupled > 0 ? num_uncoupled - 1 : 0;
    num_inner_edges = num_uncoupled > 1 ? num_uncoupled - 2 : 0;

    if (this->inner_sectors.size() == 0) {
        // empty lists were converted to float arrays in Python, which broke __hash__;
        // keep a proper empty SectorArray with the right sector_ind_len instead.
        this->inner_sectors = this->symmetry->empty_sector_array;
    }

    if (!multiplicities.has_value()) {
        this->multiplicities.assign(num_vertices, 0);
    } else {
        this->multiplicities = std::move(*multiplicities);
    }

    fusion_style = this->symmetry->fusion_style;
    is_abelian = this->symmetry->is_abelian();
    braiding_style = this->symmetry->braiding_style;
}

void
FusionTree::test_sanity() const
{
    // --- hints from Python FusionTree.test_sanity ---
    // otherwise, check fusion rules at every vertex
    // ---
    assert(symmetry->are_valid_sectors(uncoupled));
    assert(symmetry->is_valid_sector(coupled));
    assert(are_dual.size() == num_uncoupled);
    assert(inner_sectors.size() == num_inner_edges);
    assert(symmetry->are_valid_sectors(inner_sectors));
    assert(multiplicities.size() == num_vertices);

    // special cases: no vertices
    if (num_uncoupled == 0) {
        assert(coupled == symmetry->trivial_sector);
    }
    if (num_uncoupled == 1) {
        assert(uncoupled[0] == coupled);
    }

    for (std::size_t n = 0; n < num_vertices; ++n) {
        auto [a, b, mu, c] = vertex_labels(static_cast<int64>(n));
        int64 const N = symmetry->n_symbol(a, b, c);
        assert(N > 0);
        assert(mu >= 0 && mu < N);
    }
}

FusionTree
FusionTree::from_abelian_symmetry(Symmetry::Ptr symmetry,
                                  SectorArray const& uncoupled,
                                  std::vector<std::uint8_t> const& are_dual)
{
    assert(symmetry->is_abelian());

    if (uncoupled.size() == 0) {
        return from_empty(symmetry);
    }
    if (uncoupled.size() == 1) {
        return from_sector(symmetry, uncoupled[0], are_dual[0] != 0);
    }

    std::vector<Sector> fusion_outcomes;
    fusion_outcomes.reserve(uncoupled.size() - 1);
    Sector last_sector = uncoupled[0];
    for (std::size_t i = 1; i < uncoupled.size(); ++i) {
        Sector const a = uncoupled[i];
        SectorArray const outcomes = symmetry->fusion_outcomes(last_sector, a);
        Sector const f = outcomes[0];
        fusion_outcomes.push_back(f);
        last_sector = f;
    }

    SectorArray inner = sector_array_from_sectors(
      std::span<Sector const>(fusion_outcomes).first(fusion_outcomes.size() - 1),
      symmetry->sector_ind_len);

    return FusionTree(symmetry, uncoupled, fusion_outcomes.back(), are_dual, inner, std::nullopt);
}

FusionTree
FusionTree::from_empty(Symmetry::Ptr symmetry)
{
    return FusionTree(symmetry,
                      symmetry->empty_sector_array,
                      symmetry->trivial_sector,
                      {},
                      symmetry->empty_sector_array,
                      std::vector<int64>{});
}

FusionTree
FusionTree::from_sector(Symmetry::Ptr symmetry, Sector sector, bool is_dual)
{
    return FusionTree(symmetry,
                      SectorArray::from_sector(sector),
                      sector,
                      { static_cast<std::uint8_t>(is_dual ? 1 : 0) },
                      symmetry->empty_sector_array,
                      std::vector<int64>{});
}

SectorArray
FusionTree::pre_Z_uncoupled() const
{
    SectorArray res = uncoupled;
    std::size_t num_dual = 0;
    for (std::uint8_t d : are_dual) {
        num_dual += static_cast<std::size_t>(d);
    }
    if (num_dual == 0) {
        return res;
    }

    SectorArray dual_input(num_dual, uncoupled.sector_ind_len());
    std::size_t j = 0;
    for (std::size_t i = 0; i < are_dual.size(); ++i) {
        if (are_dual[i]) {
            dual_input[j++] = uncoupled[i];
        }
    }
    SectorArray const duals = symmetry->dual_sectors(dual_input);
    j = 0;
    for (std::size_t i = 0; i < are_dual.size(); ++i) {
        if (are_dual[i]) {
            res[i] = duals[j++];
        }
    }
    return res;
}

std::size_t
FusionTree::hash() const
{
    std::size_t seed = 0;
    hash_combine(seed, hash_uint8_vector(are_dual));
    hash_combine(seed, hash_sector(coupled));
    hash_combine(seed, hash_sector_array(uncoupled));

    // if abelian: inner sectors are completely determined by uncoupled, all multiplicities are 0
    if (!symmetry->is_abelian()) {
        hash_combine(seed, hash_sector_array(inner_sectors));
    }
    // if has_unique_fusion: all multiplicities are 0
    if (!symmetry->has_unique_fusion()) {
        hash_combine(seed, hash_int64_vector(multiplicities));
    }
    return seed;
}

bool
FusionTree::operator==(FusionTree const& other) const
{
    return are_dual == other.are_dual && coupled == other.coupled &&
           (uncoupled == other.uncoupled) && (inner_sectors == other.inner_sectors) &&
           multiplicities == other.multiplicities;
}

bool
FusionTree::operator<(FusionTree const& other) const
{
    if (are_dual != other.are_dual) {
        return are_dual < other.are_dual;
    }
    if (coupled != other.coupled) {
        return coupled < other.coupled;
    }
    if (uncoupled.size() != other.uncoupled.size() ||
        uncoupled.sector_ind_len() != other.uncoupled.sector_ind_len()) {
        if (uncoupled.size() != other.uncoupled.size()) {
            return uncoupled.size() < other.uncoupled.size();
        }
        return uncoupled.sector_ind_len() < other.uncoupled.sector_ind_len();
    }
    for (std::size_t i = 0; i < uncoupled.size(); ++i) {
        if (uncoupled[i] != other.uncoupled[i]) {
            return uncoupled[i] < other.uncoupled[i];
        }
    }
    if (inner_sectors.size() != other.inner_sectors.size()) {
        return inner_sectors.size() < other.inner_sectors.size();
    }
    for (std::size_t i = 0; i < inner_sectors.size(); ++i) {
        if (inner_sectors[i] != other.inner_sectors[i]) {
            return inner_sectors[i] < other.inner_sectors[i];
        }
    }
    return multiplicities < other.multiplicities;
}

std::vector<std::vector<std::string>>
FusionTree::ascii_diagram_chars(bool dagger, int uncoupled_padding, int inner_sector_padding) const
{
    (void)inner_sector_padding;
    assert(uncoupled_padding > 0);
    assert(inner_sector_padding >= 0);

    std::vector<std::string> uncoupled_strs;
    std::vector<std::string> pre_Z_uncoupled_strs;
    uncoupled_strs.reserve(num_uncoupled);
    pre_Z_uncoupled_strs.reserve(num_uncoupled);

    SectorArray const pre_Z = pre_Z_uncoupled();
    for (std::size_t i = 0; i < num_uncoupled; ++i) {
        uncoupled_strs.push_back(symmetry->sector_str(uncoupled[i]));
        pre_Z_uncoupled_strs.push_back(symmetry->sector_str(pre_Z[i]));
    }

    // single-letter sectors dont work with the design choice of attaching wires to the
    // second character of a sector -> make them at least 2 characters
    auto utf8_len = [](std::string_view s) {
        std::size_t n = 0;
        for (std::size_t pos = 0; pos < s.size();) {
            auto const cp = utf8_codepoint_len(s, pos);
            pos += cp;
            ++n;
        }
        return n;
    };
    auto pad_left = [&](std::string s, std::size_t width) {
        while (utf8_len(s) < width) {
            s.insert(s.begin(), ' ');
        }
        return s;
    };
    auto pad_right = [&](std::string s, std::size_t width) {
        while (utf8_len(s) < width) {
            s.push_back(' ');
        }
        return s;
    };

    for (auto& s : uncoupled_strs) {
        s = pad_left(std::move(s), 2);
    }
    for (auto& s : pre_Z_uncoupled_strs) {
        s = pad_left(std::move(s), 2);
    }

    // pad the uncoupled sectors in a single column to a consistent width
    std::vector<std::size_t> uncoupled_widths(num_uncoupled);
    for (std::size_t i = 0; i < num_uncoupled; ++i) {
        uncoupled_widths[i] =
          std::max(utf8_len(uncoupled_strs[i]), utf8_len(pre_Z_uncoupled_strs[i]));
        uncoupled_strs[i] = pad_right(std::move(uncoupled_strs[i]), uncoupled_widths[i]);
        pre_Z_uncoupled_strs[i] =
          pad_right(std::move(pre_Z_uncoupled_strs[i]), uncoupled_widths[i]);
    }

    // special cases with no fusion vertices
    if (num_uncoupled == 0) {
        std::string const msg = "empty FusionTree";
        auto grid = make_char_grid(utf8_len(msg), 1);
        write_string_to_col(grid, 0, 0, msg);
        return grid;
    }

    if (num_uncoupled == 1) {
        auto ascii = make_char_grid(uncoupled_widths[0], 5);
        write_string_to_col(ascii, 0, 0, uncoupled_strs[0]);
        if (are_dual[0]) {
            set_cell(ascii, 1, 1, "v");
            set_cell(ascii, 1, 2, "Z");
            set_cell(ascii, 1, 3, "^");
        } else {
            set_cell(ascii, 1, 1, "v");
            set_cell(ascii, 1, 2, "│");
            set_cell(ascii, 1, 3, "v");
        }
        write_string_to_col(ascii, 0, 4, pre_Z_uncoupled_strs[0]);
        if (!dagger) {
            reverse_cols(ascii);
        }
        return ascii;
    }

    std::string const coupled_str = symmetry->sector_str(coupled);
    std::vector<std::string> inner_sector_strs;
    inner_sector_strs.reserve(num_inner_edges);
    for (std::size_t i = 0; i < num_inner_edges; ++i) {
        inner_sector_strs.push_back(symmetry->sector_str(inner_sectors[i]));
    }

    int const num_rows_uncoupled = 5;
    int const num_rows_coupled = 1;
    int const num_rows =
      num_rows_uncoupled + 2 * static_cast<int>(num_vertices) + num_rows_coupled;

    std::vector<int> uncoupled_pos(num_uncoupled);
    int pos_acc = 0;
    for (std::size_t i = 0; i < num_uncoupled; ++i) {
        uncoupled_pos[static_cast<std::size_t>(i)] = pos_acc;
        pos_acc += static_cast<int>(uncoupled_widths[i]) + uncoupled_padding;
    }

    int const num_cols = pos_acc - uncoupled_padding;
    auto ascii =
      make_char_grid(static_cast<std::size_t>(num_cols), static_cast<std::size_t>(num_rows));

    for (std::size_t i = 0; i < num_uncoupled; ++i) {
        write_string_to_col(ascii,
                            static_cast<std::size_t>(uncoupled_pos[i]),
                            static_cast<std::size_t>(num_rows - 1),
                            uncoupled_strs[i]);
    }

    for (std::size_t i = 0; i < num_uncoupled; ++i) {
        auto const p = static_cast<std::size_t>(uncoupled_pos[i]);
        if (are_dual[i]) {
            set_cell(ascii, p + 1, static_cast<std::size_t>(num_rows - 4), "v");
            set_cell(ascii, p + 1, static_cast<std::size_t>(num_rows - 3), "Z");
            set_cell(ascii, p + 1, static_cast<std::size_t>(num_rows - 2), "^");
        } else {
            set_cell(ascii, p + 1, static_cast<std::size_t>(num_rows - 4), "v");
            set_cell(ascii, p + 1, static_cast<std::size_t>(num_rows - 3), "│");
            set_cell(ascii, p + 1, static_cast<std::size_t>(num_rows - 2), "v");
        }
    }

    for (std::size_t i = 0; i < num_uncoupled; ++i) {
        write_string_to_col(ascii,
                            static_cast<std::size_t>(uncoupled_pos[i]),
                            static_cast<std::size_t>(num_rows - 5),
                            pre_Z_uncoupled_strs[i]);
    }

    std::vector<std::pair<int, int>> vertex_positions;
    vertex_positions.reserve(num_vertices);

    int row = num_rows - 1 - num_rows_uncoupled;
    int left_wire = uncoupled_pos[0] + 1;
    auto write_utf8_at = [&](int col, int r, std::string_view s) {
        set_cell(ascii, static_cast<std::size_t>(col), static_cast<std::size_t>(r), s);
    };

    for (std::size_t n = 0; n < num_vertices; ++n) {
        int const right_wire = uncoupled_pos[n + 1] + 1;
        for (int r = row + 1; r < num_rows - num_rows_uncoupled; ++r) {
            write_utf8_at(right_wire, r, "│");
        }
        int const vertex = (left_wire + right_wire) / 2;
        write_utf8_at(left_wire, row, dagger ? "╭" : "╰");
        for (int c = left_wire + 1; c < vertex; ++c) {
            write_utf8_at(c, row, "─");
        }
        write_utf8_at(vertex, row, dagger ? "┴" : "┬");
        vertex_positions.emplace_back(vertex, row);
        for (int c = vertex + 1; c < right_wire; ++c) {
            write_utf8_at(c, row, "─");
        }
        write_utf8_at(right_wire, row, dagger ? "╮" : "╯");
        write_utf8_at(vertex, row - 1, "│");
        // for next iteration:
        left_wire = vertex;
        row -= 2;
    }
    assert(row == 0);

    int const coupled_pos = left_wire - 1;
    write_string_to_col(ascii, static_cast<std::size_t>(coupled_pos), 0, coupled_str);

    std::map<int, std::string> left_overhangs;
    for (std::size_t i = 0; i + 1 < vertex_positions.size(); ++i) {
        auto [x, y] = vertex_positions[i];
        std::string const& s = inner_sector_strs[i];
        int const inner_row = y - 1;
        int const start = x - static_cast<int>(utf8_len(s));
        if (start < 0) {
            // Split by codepoints for overhang.
            std::vector<std::string> cps;
            for (std::size_t pos = 0; pos < s.size();) {
                auto const n = utf8_codepoint_len(s, pos);
                cps.emplace_back(s.substr(pos, n));
                pos += n;
            }
            auto const abs_start = static_cast<std::size_t>(-start);
            std::string overhang;
            for (std::size_t k = 0; k < abs_start && k < cps.size(); ++k) {
                overhang += cps[k];
            }
            left_overhangs[inner_row] = overhang;
            std::string rest;
            for (std::size_t k = abs_start; k < cps.size(); ++k) {
                rest += cps[k];
            }
            write_string_to_col(ascii, 0, static_cast<std::size_t>(inner_row), rest);
        } else {
            write_string_to_col(
              ascii, static_cast<std::size_t>(start), static_cast<std::size_t>(inner_row), s);
        }
    }

    std::vector<std::vector<std::string>> extra_left;
    if (!left_overhangs.empty()) {
        std::size_t max_len = 0;
        for (auto const& [r, s] : left_overhangs) {
            (void)r;
            max_len = std::max(max_len, utf8_len(s));
        }
        extra_left = make_char_grid(max_len, static_cast<std::size_t>(num_rows));
        for (auto const& [r, extra_s] : left_overhangs) {
            write_string_to_col(
              extra_left, max_len - utf8_len(extra_s), static_cast<std::size_t>(r), extra_s);
        }
    }

    if (!symmetry->has_unique_fusion()) {
        // need to print multiplicities
        for (std::size_t i = 0; i < vertex_positions.size(); ++i) {
            auto [x, y] = vertex_positions[i];
            std::string mult = std::to_string(multiplicities[i]);
            if (mult.size() == 1) {
                set_cell(ascii, static_cast<std::size_t>(x), static_cast<std::size_t>(y), mult);
            } else if (mult.size() == 2) {
                write_string_to_col(
                  ascii, static_cast<std::size_t>(x), static_cast<std::size_t>(y), mult);
            } else if (mult.size() == 3) {
                write_string_to_col(
                  ascii, static_cast<std::size_t>(x - 1), static_cast<std::size_t>(y), mult);
            } else {
                throw NotImplemented("FusionTree::ascii_diagram_chars multiplicity >3 digits");
            }
        }
    }

    ascii = prepend_rows(extra_left, ascii);
    if (!dagger) {
        reverse_cols(ascii);
    }
    return ascii;
}

std::string
FusionTree::ascii_diagram(bool dagger) const
{
    return ascii_grid_to_string(ascii_diagram_chars(dagger));
}

std::string
FusionTree::str_uncoupled_coupled(Symmetry const& symmetry,
                                  SectorArray const& uncoupled,
                                  Sector coupled,
                                  std::vector<std::uint8_t> const& are_dual)
{
    std::vector<std::string> uncoupled_1;
    std::vector<std::string> uncoupled_2;
    uncoupled_1.reserve(uncoupled.size());
    uncoupled_2.reserve(uncoupled.size());

    for (std::size_t i = 0; i < uncoupled.size(); ++i) {
        Sector const a = uncoupled[i];
        std::string const a_str = symmetry.sector_str(a);
        uncoupled_2.push_back(a_str);
        if (are_dual[i]) {
            uncoupled_1.push_back(
              std::format("dual({})", symmetry.sector_str(symmetry.dual_sector(a))));
        } else {
            uncoupled_1.push_back(a_str);
        }
    }

    std::string const before_Z = std::format("({})", join_strings(uncoupled_1, ", "));
    std::string const after_Z = std::format("({})", join_strings(uncoupled_2, ", "));
    std::string const final = symmetry.sector_str(coupled);
    return std::format("{} -> {} -> {}", before_Z, after_Z, final);
}

FusionTreePairLinearCombination
FusionTree::bend_leg(FusionTree const& X, FusionTree const& Y, bool bend_downward, bool do_conj)
{
    // --- hints from Python FusionTree.bend_leg ---
    // OPTIMIZE: do it explicitly instead?
    // bend_up(dagger(Y) @ X)
    // == dagger(dagger(bend_up(dagger(Y) @ X))
    // == dagger(bend_down(dagger(dagger(Y) @ X))))
    // == dagger(bend_down(dagger(X) @ Y))
    // == dagger(sum_i b_i (dagger(X_i) @ Y_i))
    // == sum_i conj(b_i) dagger(Y_i) @ X_i
    // i.e. we need to swap the order of inputs and invert bend_downward,
    // then for the result, swap the trees back and conj the coefficients (invert do_conj)
    // OPTIMIZE remove input checks?
    // ---
    if (!bend_downward) {
        // OPTIMIZE: do it explicitly instead?
        // bend_up(dagger(Y) @ X)
        // == dagger(dagger(bend_up(dagger(Y) @ X))
        // == dagger(bend_down(dagger(dagger(Y) @ X))))
        // == dagger(bend_down(dagger(X) @ Y))
        // == dagger(sum_i b_i (dagger(X_i) @ Y_i))
        // == sum_i conj(b_i) dagger(Y_i) @ X_i
        // i.e. we need to swap the order of inputs and invert bend_downward,
        // then for the result, swap the trees back and conj the coefficients (invert do_conj)
        FusionTreePairLinearCombination const other = bend_leg(Y, X, true, !do_conj);
        FusionTreePairLinearCombination res;
        for (auto const& [pair, b_i] : other) {
            res[{ pair.second, pair.first }] = b_i;
        }
        return res;
    }

    // OPTIMIZE remove input checks?
    // Compare by value: trees from different TensorProducts may hold distinct Symmetry
    // shared_ptrs that are mathematically equal (Python used ``==``, not ``is``).
    assert(Y.symmetry && X.symmetry && Y.symmetry->equals(*X.symmetry));
    Symmetry::Ptr const symmetry = Y.symmetry;
    assert(Y.coupled == X.coupled);
    Sector const c = Y.coupled;

    if (Y.num_uncoupled == 0) {
        throw std::invalid_argument("No leg to be bent.");
    }

    bool const is_dual = Y.are_dual.back() != 0;

    if (Y.num_uncoupled == 1) {
        FusionTree const X_i = from_empty(symmetry);
        FusionTree const Y_i =
          X.extended(symmetry->dual_sector(c), 0, symmetry->trivial_sector, !is_dual);
        complex128 b_i = symmetry->sqrt_qdim(c);
        if (is_dual) {
            b_i *= static_cast<complex128>(symmetry->frobenius_schur(c));
        }
        return { { { Y_i, X_i }, b_i } };
    }

    auto [X_i, c_split, mu, z] = Y.split_bottom_vertex();
    (void)c_split;

    if (X.num_uncoupled == 0) {
        Sector const e = X_i.coupled;
        FusionTree const Y_i = from_sector(symmetry, e, !is_dual);
        complex128 b_i = symmetry->inv_sqrt_qdim(e);
        if (!is_dual) {
            b_i *= static_cast<complex128>(symmetry->frobenius_schur(e));
        }
        return { { { Y_i, X_i }, b_i } };
    }

    FusionSymbol const B = symmetry->b_symbol(X_i.coupled, z, c);
    complex128 const chi_z = static_cast<complex128>(symmetry->frobenius_schur(z));
    Sector const zbar = symmetry->dual_sector(z);

    FusionTreePairLinearCombination res;
    std::size_t const n_nu = B.extent(1);
    for (std::size_t nu = 0; nu < n_nu; ++nu) {
        complex128 b_i = B.get_complex(static_cast<std::size_t>(mu), nu);
        FusionTree const Y_i = X.extended(zbar, static_cast<int64>(nu), X_i.coupled, !is_dual);
        if (is_dual) {
            b_i *= chi_z;
        }
        if (do_conj) {
            b_i = std::conj(b_i);
        }
        res[{ Y_i, X_i }] = b_i;
    }
    return res;
}

FusionTreeLinearCombination
FusionTree::braid(int64 j, bool overbraid, float64 cutoff, bool do_conj) const
{
    // --- hints from Python FusionTree.braid ---
    // R-move
    // C-move
    // underbraid compared to overbraid:
    // - conj
    // - b <-> c  [in args of c_symbol(...)]
    // - e <-> f  [in args of c_symbol(...)]
    // - (mu,nu) <-> (kappa,lambda)  [by indexing c_symbol(...) differently]
    // OPTIMIZE rm check
    // ---
    assert(j >= 0 && static_cast<std::size_t>(j) < num_uncoupled - 1);

    if (j == 0) { // R-move
        auto [a, b, mu, c] = vertex_labels(0);
        complex128 a_i;
        if (overbraid) {
            FusionSymbol const R = symmetry->r_symbol(a, b, c);
            a_i = R.get_complex(static_cast<std::size_t>(mu));
        } else {
            FusionSymbol const R = symmetry->r_symbol(b, a, c);
            a_i = std::conj(R.get_complex(static_cast<std::size_t>(mu)));
        }
        if (do_conj) {
            a_i = std::conj(a_i);
        }
        FusionTree X_i = copy(true);
        X_i.uncoupled[0] = b;
        X_i.uncoupled[1] = a;
        std::swap(X_i.are_dual[0], X_i.are_dual[1]);
        return { { X_i, a_i } };
    }

    // C-move
    FusionTreeLinearCombination res;
    auto [a, b, mu, e] = vertex_labels(j - 1);
    auto [_e, c, nu, d] = vertex_labels(j);
    (void)_e;

    FusionTree X_new = copy(true);
    X_new.uncoupled[static_cast<std::size_t>(j)] = c;
    X_new.uncoupled[static_cast<std::size_t>(j + 1)] = b;
    X_new.are_dual[static_cast<std::size_t>(j)] = are_dual[static_cast<std::size_t>(j + 1)];
    X_new.are_dual[static_cast<std::size_t>(j + 1)] = are_dual[static_cast<std::size_t>(j)];

    for (std::size_t fi = 0; fi < symmetry->fusion_outcomes(a, c).size(); ++fi) {
        Sector const f = symmetry->fusion_outcomes(a, c)[fi];
        if (!symmetry->can_fuse_to(f, b, d)) {
            continue;
        }

        FusionSymbol C_arr;
        if (overbraid) {
            C_arr = symmetry->c_symbol(a, b, c, d, e, f)
                      .slice2d(static_cast<std::size_t>(mu), static_cast<std::size_t>(nu));
        } else {
            // underbraid compared to overbraid:
            //  - conj
            //  - b <-> c  [in args of c_symbol(...)]
            //  - e <-> f  [in args of c_symbol(...)]
            //  - (mu,nu) <-> (kappa,lambda)  [by indexing c_symbol(...) differently]
            C_arr = symmetry->c_symbol(a, c, b, d, f, e)
                      .slice2d_trailing(static_cast<std::size_t>(mu), static_cast<std::size_t>(nu))
                      .conj();
        }
        if (do_conj) {
            C_arr = C_arr.conj();
        }

        C_arr.for_each2d([&](std::size_t kappa, std::size_t lambda_, complex128 a_i) {
            if (std::abs(a_i) < cutoff) {
                return;
            }
            FusionTree X_i = X_new.copy(true);
            X_i.inner_sectors[static_cast<std::size_t>(j - 1)] = f;
            X_i.multiplicities[static_cast<std::size_t>(j - 1)] = static_cast<int64>(kappa);
            X_i.multiplicities[static_cast<std::size_t>(j)] = static_cast<int64>(lambda_);
            assert(!res.contains(X_i)); // OPTIMIZE rm check
            res[X_i] = a_i;
        });
    }
    return res;
}

std::tuple<Sector, Sector, int64, Sector>
FusionTree::vertex_labels(int64 n) const
{
    Sector a;
    Sector b;
    if (n == 0) {
        a = uncoupled[0];
        b = uncoupled[1];
    } else {
        a = inner_sectors[static_cast<std::size_t>(n - 1)];
        b = uncoupled[static_cast<std::size_t>(n + 1)];
    }

    Sector c;
    if (static_cast<std::size_t>(n) == num_vertices - 1) {
        c = coupled;
    } else {
        c = inner_sectors[static_cast<std::size_t>(n)];
    }
    return { a, b, multiplicities[static_cast<std::size_t>(n)], c };
}

FusionTree
FusionTree::modify_vertex_labels(int64 n, Sector a, Sector b, int64 mu, Sector c, bool copy)
{
    if (copy) {
        return this->copy(true).modify_vertex_labels(n, a, b, mu, c, false);
    }
    if (n == 0) {
        uncoupled[0] = a;
    } else {
        inner_sectors[static_cast<std::size_t>(n - 1)] = a;
    }
    uncoupled[static_cast<std::size_t>(n + 1)] = b;
    if (static_cast<std::size_t>(n) == num_vertices - 1) {
        coupled = c;
    } else {
        inner_sectors[static_cast<std::size_t>(n)] = c;
    }
    multiplicities[static_cast<std::size_t>(n)] = mu;
    return *this;
}

std::string
FusionTree::str() const
{
    auto const ascii = ascii_diagram_chars(false);
    std::string res = std::format("<FusionTree   symmetry: {}>", symmetry->str());
    if (!ascii.empty()) {
        auto const num_cols = ascii.size();
        auto const num_rows = ascii[0].size();
        for (std::size_t row = 0; row < num_rows; ++row) {
            res += "\n    |   ";
            for (std::size_t col = 0; col < num_cols; ++col) {
                res += ascii[col][row];
            }
        }
    }
    return res;
}

std::string
FusionTree::repr() const
{
    std::string inner = replace_all(sector_array_str(inner_sectors), "\n", ",");
    std::string unc = replace_all(sector_array_str(uncoupled), "\n", ",");

    std::ostringstream dual_ss;
    dual_ss << "[";
    for (std::size_t i = 0; i < are_dual.size(); ++i) {
        if (i > 0) {
            dual_ss << " ";
        }
        dual_ss << static_cast<int>(are_dual[i]);
    }
    dual_ss << "]";

    std::ostringstream mult_ss;
    mult_ss << "[";
    for (std::size_t i = 0; i < multiplicities.size(); ++i) {
        if (i > 0) {
            mult_ss << " ";
        }
        mult_ss << multiplicities[i];
    }
    mult_ss << "]";

    return std::format("FusionTree({}, {}, {}, coupled={}, inner_sectors={}, multiplicities={})",
                       symmetry->repr(),
                       unc,
                       dual_ss.str(),
                       symmetry->sector_str(coupled),
                       inner,
                       mult_ss.str());
}

BlockBackend::BlockPtr
FusionTree::to_dense_block(BlockBackend* backend,
                           std::optional<Dtype> dtype,
                           bool understood_braiding) const
{
    // --- hints from Python FusionTree.to_dense_block ---
    // handle special cases of small trees
    // must be identity on the trivial sector. But since there is no uncoupled sector,
    // do not even give it an axis.
    // [m_c, m_a1] -> need to transpose!
    // OPTIMIZE should we offer a symmetry function to compute only the mu slice?
    // [a0, a1, c]
    // larger trees: iterate over vertices
    // [a0, a1, i0]
    // [a0, a1, ..., an, i{n-1}] & [i{n-1}, a{n+1}, in] -> [a0, a1, ..., a{n+1}, in]
    // ---
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("Can not convert to block for symmetry {}", symmetry->str()));
    }
    if (!symmetry->has_trivial_braid() && !understood_braiding) {
        throw SymmetryError(
          "If the symmetry has non-trivial braids, dense block representations do not "
          "consistently reproduce the braiding statistics. Make sure you understand what "
          "that means (read the docstring of from_dense_block). Then you can disable "
          "this error by setting ``understood_braiding=True``.");
    }

    BlockBackend* block_backend = backend;
    if (block_backend == nullptr) {
        block_backend = NumpyBlockBackend::from_factory("cpu");
    }

    if (!dtype.has_value()) {
        if (symmetry->fusion_tensor_dtype.has_value()) {
            dtype = symmetry->fusion_tensor_dtype;
        } else {
            dtype = Dtype::Complex128;
        }
    }

    // handle special cases of small trees
    if (num_uncoupled == 0) {
        // must be identity on the trivial sector. But since there is no uncoupled sector,
        // do not even give it an axis.
        return block_backend->ones_block({ 1 }, *dtype);
    }

    if (num_uncoupled == 1) {
        if (are_dual[0]) {
            FusionSymbol const Z = symmetry->Z_iso(symmetry->dual_sector(uncoupled[0]));
            // [m_c, m_a1] -> need to transpose!
            FusionSymbol const ZT = Z.transpose(std::array<std::uint8_t, 4>{ { 1, 0, 2, 3 } });
            return block_from_fusion_symbol(*block_backend, ZT, *dtype);
        }
        int64 const dim_c = symmetry->sector_dim(coupled);
        return block_backend->eye_block({ dim_c }, *dtype);
    }

    if (num_uncoupled == 2) {
        int64 const mu = multiplicities[0];
        // OPTIMIZE should we offer a symmetry function to compute only the mu slice?
        FusionSymbol const tensor = symmetry->fusion_tensor(
          uncoupled[0], uncoupled[1], coupled, are_dual[0] != 0, are_dual[1] != 0);
        FusionSymbol const X = fusion_tensor_slice_mu(tensor, static_cast<std::size_t>(mu));
        return block_from_fusion_symbol(*block_backend, X, *dtype); // [a0, a1, c]
    }

    // larger trees: iterate over vertices
    int64 const mu0 = multiplicities[0];
    FusionSymbol const tensor0 = symmetry->fusion_tensor(
      uncoupled[0], uncoupled[1], inner_sectors[0], are_dual[0] != 0, are_dual[1] != 0);
    FusionSymbol const X0 = fusion_tensor_slice_mu(tensor0, static_cast<std::size_t>(mu0));
    BlockBackend::BlockPtr res =
      block_from_fusion_symbol(*block_backend, X0, *dtype); // [a0, a1, i0]

    for (std::size_t vertex = 1; vertex < num_vertices; ++vertex) {
        int64 const mu = multiplicities[vertex];
        Sector const a = inner_sectors[vertex - 1];
        Sector const b = uncoupled[vertex + 1];
        Sector const c = vertex < num_inner_edges ? inner_sectors[vertex] : coupled;
        FusionSymbol const tensor =
          symmetry->fusion_tensor(a, b, c, false, are_dual[vertex + 1] != 0);
        FusionSymbol const X = fusion_tensor_slice_mu(tensor, static_cast<std::size_t>(mu));
        BlockBackend::BlockPtr const X_block = block_from_fusion_symbol(*block_backend, X, *dtype);
        // [a0, a1, ..., an, i{n-1}] & [i{n-1}, a{n+1}, in] -> [a0, a1, ..., a{n+1}, in]
        res = block_backend->tdot(res, X_block, { -1 }, { 0 });
    }
    return res;
}

FusionTree
FusionTree::copy(bool deep) const
{
    if (deep) {
        return FusionTree(symmetry, uncoupled, coupled, are_dual, inner_sectors, multiplicities);
    }
    return FusionTree(symmetry, uncoupled, coupled, are_dual, inner_sectors, multiplicities);
}

FusionTree
FusionTree::extended(Sector new_uncoupled, int64 mu, Sector new_coupled, bool is_dual) const
{
    // --- hints from Python FusionTree.extended ---
    // result has one vertex, and thus no inner sectors
    // ---
    std::vector<int64> new_multiplicities;
    if (num_uncoupled == 0) {
        assert(mu == 0);
    } else {
        new_multiplicities = multiplicities;
        new_multiplicities.push_back(mu);
    }

    SectorArray new_inner = inner_sectors;
    if (num_uncoupled >= 2) {
        // for num_uncoupled < 2: result has one vertex, and thus no inner sectors
        new_inner = new_inner.concat(SectorArray::from_sector(coupled));
    }

    std::vector<std::uint8_t> new_are_dual = are_dual;
    new_are_dual.push_back(static_cast<std::uint8_t>(is_dual ? 1 : 0));

    SectorArray const new_unc = uncoupled.concat(SectorArray::from_sector(new_uncoupled));

    return FusionTree(
      symmetry, new_unc, new_coupled, std::move(new_are_dual), new_inner, new_multiplicities);
}

FusionTree
FusionTree::insert(FusionTree const& t2) const
{
    SectorArray const new_unc = t2.uncoupled.concat(uncoupled.slice(1, num_uncoupled));
    std::vector<std::uint8_t> new_dual =
      concat_vectors(std::span<std::uint8_t const>(t2.are_dual),
                     std::span<std::uint8_t const>(are_dual).subspan(1));
    SectorArray const new_inners =
      concat_sector_arrays_many({ t2.inner_sectors, uncoupled.slice(0, 1), inner_sectors });
    std::vector<int64> new_mults = concat_vectors(std::span<int64 const>(t2.multiplicities),
                                                  std::span<int64 const>(multiplicities));

    return FusionTree(symmetry, new_unc, coupled, std::move(new_dual), new_inners, new_mults);
}

FusionTreeLinearCombination
FusionTree::insert_at(int64 n, FusionTree const& t2, float64 eps) const
{
    // --- hints from Python FusionTree.insert_at ---
    // t2 has no actual fusion, it is either identity or a Z iso
    // ---
    assert(symmetry && t2.symmetry && symmetry->equals(*t2.symmetry));
    assert(uncoupled[static_cast<std::size_t>(n)] == t2.coupled);
    assert(!are_dual[static_cast<std::size_t>(n)]);

    if (t2.num_uncoupled == 0) {
        // special case: empty tree with trivial coupled sector
        // -> effectively remove self.uncoupled[n] (replace with empty set of sectors)
        SectorArray const res_unc =
          uncoupled.slice(0, static_cast<std::size_t>(n))
            .concat(uncoupled.slice(static_cast<std::size_t>(n) + 1, num_uncoupled));
        std::vector<std::uint8_t> const res_dual = concat_vectors(
          std::span<std::uint8_t const>(vector_slice(are_dual, 0, static_cast<std::size_t>(n))),
          std::span<std::uint8_t const>(
            vector_slice(are_dual, static_cast<std::size_t>(n) + 1, are_dual.size())));
        // `idx` selects which inner_sectors entry becomes redundant (and is dropped) once
        // uncoupled[n] is removed. For n==0 or a middle position, that entry is
        // inner_sectors[n-1]. At the last position (n == num_uncoupled-1) there is no
        // inner_sectors[n-1] -- the last valid index is num_inner_edges-1 -- so that boundary must
        // be special-cased separately, mirroring the analogous n==num_uncoupled-1 special case in
        // the general branch below (see `d_initial`).
        std::size_t const idx =
          (n == 0) ? std::size_t{ 0 }
          : (static_cast<std::size_t>(n) == num_uncoupled - 1 && num_inner_edges > 0)
            ? num_inner_edges - 1
            : static_cast<std::size_t>(std::max<int64>(0, n - 1));
        SectorArray const res_inners =
          inner_sectors.slice(0, idx).concat(inner_sectors.slice(idx + 1, num_inner_edges));
        std::vector<int64> const res_mults = concat_vectors(
          std::span<int64 const>(vector_slice(multiplicities, 0, idx)),
          std::span<int64 const>(vector_slice(multiplicities, idx + 1, multiplicities.size())));
        FusionTree const res(symmetry, res_unc, coupled, res_dual, res_inners, res_mults);
        return { { res, complex128{ 1.0, 0.0 } } };
    }

    if (t2.num_vertices == 0) {
        if (t2.are_dual[0]) {
            FusionTree res = copy(true);
            res.are_dual[static_cast<std::size_t>(n)] = 1;
            return { { res, complex128{ 1.0, 0.0 } } };
        }
        return { { *this, complex128{ 1.0, 0.0 } } };
    }

    if (num_vertices == 0) {
        return { { t2, complex128{ 1.0, 0.0 } } };
    }

    if (n == 0) {
        // result is already a canonical tree -> no need to do F moves
        return { { insert(t2), complex128{ 1.0, 0.0 } } };
    }

    // should be more efficient than using recursion
    Symmetry::Ptr const sym = symmetry;
    FusionTreeLinearCombination coefficients;

    SectorArray const new_unc = concat_sector_arrays_many(
      { uncoupled.slice(0, static_cast<std::size_t>(n)),
        t2.uncoupled,
        uncoupled.slice(static_cast<std::size_t>(n) + 1, num_uncoupled) });

    std::vector<std::uint8_t> const new_dual = concat_vectors(
      std::span<std::uint8_t const>(vector_slice(are_dual, 0, static_cast<std::size_t>(n))),
      std::span<std::uint8_t const>(t2.are_dual),
      std::span<std::uint8_t const>(
        vector_slice(are_dual, static_cast<std::size_t>(n) + 1, are_dual.size())));

    SectorArray const new_inners_left = inner_sectors.slice(0, static_cast<std::size_t>(n - 1));
    SectorArray const new_inners_right =
      inner_sectors.slice(static_cast<std::size_t>(n - 1), num_inner_edges);
    std::vector<int64> const new_multis_left =
      vector_slice(multiplicities, 0, static_cast<std::size_t>(n - 1));
    std::vector<int64> const new_multis_right =
      vector_slice(multiplicities, static_cast<std::size_t>(n), multiplicities.size());

    Sector const a =
      new_inners_left.size() == 0 ? uncoupled[0] : new_inners_left[new_inners_left.size() - 1];
    Sector const d_initial =
      static_cast<std::size_t>(n) == num_uncoupled - 1 ? coupled : new_inners_right[0];

    // build the remaining parts (inner and multiplicities) from the right
    using TreePartsKey = std::pair<std::vector<Sector>, std::vector<int64>>;
    std::map<TreePartsKey, complex128> tree_parts;
    tree_parts[{ {}, { multiplicities[static_cast<std::size_t>(n - 1)] } }] =
      complex128{ 1.0, 0.0 };

    for (std::size_t i = t2.num_uncoupled - 1; i > 0; --i) {
        std::map<TreePartsKey, complex128>
          new_tree_parts; // contains new inner_sectors and multiplicities
        for (auto const& [parts, amplitude] : tree_parts) {
            auto const& [inners, multis] = parts;
            Sector const b = i > 1 ? t2.inner_sectors[i - 2] : t2.uncoupled[0];
            Sector const c = t2.uncoupled[i];
            Sector const d = inners.empty() ? d_initial : inners[0];
            Sector const e = inners.empty() ? t2.coupled : t2.inner_sectors[i - 1];
            int64 const multi = t2.multiplicities[i - 1];

            for (std::size_t fi = 0; fi < sym->fusion_outcomes(a, b).size(); ++fi) {
                Sector const f = sym->fusion_outcomes(a, b)[fi];
                if (!sym->can_fuse_to(f, c, d)) {
                    continue;
                }
                FusionSymbol const fs =
                  sym->_f_symbol(a, b, c, d, e, f)
                    .slice2d(static_cast<std::size_t>(multi), static_cast<std::size_t>(multis[0]));

                fs.for_each2d([&](std::size_t kap, std::size_t lam, complex128 factor) {
                    if (std::abs(factor) < eps) {
                        return;
                    }
                    std::vector<Sector> new_inners_vec;
                    new_inners_vec.push_back(f);
                    new_inners_vec.insert(new_inners_vec.end(), inners.begin(), inners.end());
                    std::vector<int64> new_multis_vec = { static_cast<int64>(kap),
                                                          static_cast<int64>(lam) };
                    new_multis_vec.insert(new_multis_vec.end(), multis.begin() + 1, multis.end());
                    TreePartsKey const key{ new_inners_vec, new_multis_vec };
                    new_tree_parts[key] += amplitude * factor;
                });
            }
        }
        tree_parts = std::move(new_tree_parts);
    }

    for (auto const& [parts, amplitude] : tree_parts) {
        auto const& [inners, multis] = parts;
        SectorArray const inners_arr = sector_array_from_sectors(inners, sym->sector_ind_len);
        SectorArray const new_inners =
          concat_sector_arrays_many({ new_inners_left, inners_arr, new_inners_right });
        std::vector<int64> const new_multis =
          concat_vectors(std::span<int64 const>(new_multis_left),
                         std::span<int64 const>(multis),
                         std::span<int64 const>(new_multis_right));
        FusionTree const new_tree(sym, new_unc, coupled, new_dual, new_inners, new_multis);
        coefficients[new_tree] = amplitude;
    }
    return coefficients;
}

FusionTreeLinearCombination
FusionTree::outer(FusionTree const& right_tree, float64 eps) const
{
    // --- hints from Python FusionTree.outer ---
    // trivial cases
    // use self.insert_at(right_tree) -> construct new tree with
    // right_tree.coupled as uncoupled sector at the end
    // ---
    // trivial cases
    if (num_uncoupled == 0) {
        return { { right_tree, complex128{ 1.0, 0.0 } } };
    }
    if (right_tree.num_uncoupled == 0) {
        return { { *this, complex128{ 1.0, 0.0 } } };
    }

    // use self.insert_at(right_tree) -> construct new tree with
    // right_tree.coupled as uncoupled sector at the end
    Symmetry::Ptr const sym = symmetry;
    FusionTreeLinearCombination res;

    SectorArray const unc = uncoupled.concat(SectorArray::from_sector(right_tree.coupled));
    std::vector<std::uint8_t> dual = are_dual;
    dual.push_back(0);

    SectorArray inner = num_uncoupled <= 1
                          ? sym->empty_sector_array
                          : inner_sectors.concat(SectorArray::from_sector(coupled));

    for (std::size_t ci = 0; ci < sym->fusion_outcomes(coupled, right_tree.coupled).size(); ++ci) {
        Sector const new_coupled = sym->fusion_outcomes(coupled, right_tree.coupled)[ci];
        int64 const n_sym = sym->_n_symbol(coupled, right_tree.coupled, new_coupled);
        for (int64 m = 0; m < n_sym; ++m) {
            std::vector<int64> multi = multiplicities;
            multi.push_back(m);
            FusionTree const tree(sym, unc, new_coupled, dual, inner, multi);
            map_add_coeff(res, tree.insert_at(static_cast<int64>(num_uncoupled), right_tree, eps));
        }
    }
    return res;
}

std::pair<FusionTree, FusionTree>
FusionTree::split(int64 n) const
{
    if (n < 2) {
        throw std::invalid_argument("Left tree has no vertices (n < 2)");
    }
    if (static_cast<std::size_t>(n) >= num_uncoupled) {
        throw std::invalid_argument("Right tree has no vertices (n >= num_uncoupled)");
    }

    Sector const cut_sector = inner_sectors[static_cast<std::size_t>(n - 2)];

    FusionTree t1(symmetry,
                  uncoupled.slice(0, static_cast<std::size_t>(n)),
                  cut_sector,
                  vector_slice(are_dual, 0, static_cast<std::size_t>(n)),
                  inner_sectors.slice(0, static_cast<std::size_t>(n - 2)),
                  vector_slice(multiplicities, 0, static_cast<std::size_t>(n - 1)));

    std::vector<std::uint8_t> t2_dual =
      vector_slice(are_dual, static_cast<std::size_t>(n), are_dual.size());
    t2_dual.insert(t2_dual.begin(), 0);

    FusionTree t2(
      symmetry,
      SectorArray::from_sector(cut_sector)
        .concat(uncoupled.slice(static_cast<std::size_t>(n), num_uncoupled)),
      coupled,
      std::move(t2_dual),
      inner_sectors.slice(static_cast<std::size_t>(n - 1), num_inner_edges),
      vector_slice(multiplicities, static_cast<std::size_t>(n - 1), multiplicities.size()));

    return { t1, t2 };
}

std::tuple<FusionTree, Sector, int64, Sector>
FusionTree::split_bottom_vertex() const
{
    if (num_uncoupled == 0) {
        throw std::invalid_argument("Cant split empty tree");
    }
    if (num_uncoupled == 1) {
        return { from_empty(symmetry), coupled, 0, coupled };
    }
    if (num_uncoupled == 2) {
        FusionTree const rest_tree = from_sector(symmetry, uncoupled[0], are_dual[0] != 0);
        return { rest_tree, coupled, multiplicities[0], uncoupled[1] };
    }

    FusionTree const rest_tree(symmetry,
                               uncoupled.slice(0, num_uncoupled - 1),
                               inner_sectors[num_inner_edges - 1],
                               vector_slice(are_dual, 0, are_dual.size() - 1),
                               inner_sectors.slice(0, num_inner_edges - 1),
                               vector_slice(multiplicities, 0, multiplicities.size() - 1));

    return { rest_tree, coupled, multiplicities.back(), uncoupled[num_uncoupled - 1] };
}

FusionTreeLinearCombination
FusionTree::twist(std::vector<int64> const& idcs, bool overtwist) const
{
    // --- hints from Python FusionTree.twist ---
    // twists are trivial
    // ---
    if (symmetry->has_trivial_braid()) {
        return { { *this, complex128{ 1.0, 0.0 } } };
    }
    if (idcs.empty()) {
        return { { *this, complex128{ 1.0, 0.0 } } };
    }
    if (idcs.size() == 1) {
        // single wire twist
        int64 const i = to_valid_idx(idcs[0], static_cast<int64>(num_uncoupled));
        complex128 theta = symmetry->topological_twist(uncoupled[static_cast<std::size_t>(i)]);
        if (!overtwist) {
            theta = std::conj(theta);
        }
        return { { *this, theta } };
    }

    std::vector<int64> sorted_idcs;
    sorted_idcs.reserve(idcs.size());
    for (int64 idx : idcs) {
        sorted_idcs.push_back(to_valid_idx(idx, static_cast<int64>(num_uncoupled)));
    }
    std::sort(sorted_idcs.begin(), sorted_idcs.end());
    for (std::size_t k = 1; k < sorted_idcs.size(); ++k) {
        assert(sorted_idcs[k] > sorted_idcs[k - 1]); // duplicate idcs
    }

    if (sorted_idcs.size() == num_uncoupled) {
        // we can just slide the whole tree through the twist and end up with a twist of the
        // coupled sector
        complex128 theta = symmetry->topological_twist(coupled);
        if (!overtwist) {
            theta = std::conj(theta);
        }
        return { { *this, theta } };
    }

    bool is_initial_range = true;
    for (std::size_t k = 0; k < sorted_idcs.size(); ++k) {
        if (sorted_idcs[k] != static_cast<int64>(k)) {
            is_initial_range = false;
            break;
        }
    }
    if (is_initial_range) {
        // we can slide a subtree through the twist and get a twist on an inner sector
        // note: have already excluded the special cases where this index would be out of bounds
        Sector const a = inner_sectors[sorted_idcs.back() - 1];
        complex128 theta = symmetry->topological_twist(a);
        if (!overtwist) {
            theta = std::conj(theta);
        }
        return { { *this, theta } };
    }

    // Not sure what the best strategy is in the general case.
    // Option A: we could do the twist on range(i, j) as:
    //           - twist on range(j)
    //           - inverse twist on range(i)
    //           - some extra braiding
    // Option B: break it down recursively
    //           - twist range(i, mid)
    //           - twist range(mid, j)
    //           - braid twice
    throw NotImplemented("FusionTree::twist");
}

fusion_trees::fusion_trees(Symmetry::Ptr symmetry,
                           SectorArray uncoupled,
                           Sector coupled,
                           std::optional<std::vector<std::uint8_t>> are_dual)
  : symmetry(std::move(symmetry))
  , coupled(coupled)
{
    // --- hints from Python fusion_trees.__init__ ---
    // OPTIMIZE demand SectorArray (not list) and skip?
    // ---
    // DOC: coupled = None means trivial sector (handled by caller / bindings if needed)
    assert(this->symmetry);
    if (uncoupled.size() == 0) {
        uncoupled = this->symmetry->empty_sector_array;
    }
    this->uncoupled = std::move(uncoupled);
    num_uncoupled = this->uncoupled.size();
    if (!are_dual.has_value()) {
        this->are_dual.assign(num_uncoupled, 0);
    } else {
        this->are_dual = std::move(*are_dual);
    }
    assert(this->are_dual.size() == num_uncoupled);
}

std::vector<FusionTree>
fusion_trees::all_trees() const
{
    std::vector<FusionTree> out;

    if (num_uncoupled == 0) {
        if (coupled == symmetry->trivial_sector) {
            out.push_back(FusionTree(symmetry,
                                     uncoupled,
                                     coupled,
                                     {},
                                     symmetry->empty_sector_array,
                                     std::vector<int64>{}));
        }
        return out;
    }

    if (num_uncoupled == 1) {
        if (uncoupled[0] == coupled) {
            out.push_back(FusionTree(symmetry,
                                     uncoupled,
                                     coupled,
                                     are_dual,
                                     symmetry->empty_sector_array,
                                     std::vector<int64>{}));
        }
        return out;
    }

    if (num_uncoupled == 2) {
        // OPTIMIZE does handling of multiplicities introduce significant overhead?
        //          could do a specialized version for multiplicity-free fusion
        int64 const n = symmetry->n_symbol(uncoupled[0], uncoupled[1], coupled);
        out.reserve(static_cast<std::size_t>(n));
        for (int64 mu = 0; mu < n; ++mu) {
            out.push_back(FusionTree(symmetry,
                                     uncoupled,
                                     coupled,
                                     are_dual,
                                     symmetry->empty_sector_array,
                                     std::vector<int64>{ mu }));
        }
        return out;
    }

    Sector const a1 = uncoupled[0];
    Sector const a2 = uncoupled[1];
    SectorArray const left_unc = uncoupled.slice(0, 2);
    std::vector<std::uint8_t> const left_dual = vector_slice(are_dual, 0, 2);

    SectorArray const fusion_bs = symmetry->fusion_outcomes(a1, a2);
    for (std::size_t ib = 0; ib < fusion_bs.size(); ++ib) {
        Sector const b = fusion_bs[ib];
        SectorArray const rest_unc =
          SectorArray::from_sector(b).concat(uncoupled.slice(2, num_uncoupled));
        std::vector<std::uint8_t> rest_dual = concat_vectors(
          std::span<std::uint8_t const>(std::vector<std::uint8_t>{ 0 }),
          std::span<std::uint8_t const>(vector_slice(are_dual, 2, are_dual.size())));
        // set multiplicity index to 0 for now. will adjust it later.
        FusionTree const left_tree(
          symmetry, left_unc, b, left_dual, symmetry->empty_sector_array, std::vector<int64>{ 0 });
        fusion_trees const rest(symmetry, rest_unc, coupled, rest_dual);
        for (FusionTree const& rest_tree : rest.all_trees()) {
            FusionTree tree = rest_tree.insert(left_tree);
            int64 const n_mu = symmetry->_n_symbol(a1, a2, b);
            for (int64 mu = 0; mu < n_mu; ++mu) {
                FusionTree res = tree.copy(true);
                res.multiplicities[0] = mu;
                out.push_back(std::move(res));
            }
        }
    }
    return out;
}

namespace {

/// Memo of ``fusion_trees::size``: the number of trees depends only on the symmetry, the uncoupled
/// sectors and the coupled sector. The computation recurses over all fusion outcomes of the first
/// two legs, so it is called very often with few distinct arguments. The weak pointer guards
/// against a symmetry object that was destroyed and whose address was reused.
struct TreeSizeEntry
{
    std::weak_ptr<Symmetry> symmetry;
    std::size_t size = 0;
};
using TreeSizeCache = ByteLRU<std::string, TreeSizeEntry>;

TreeSizeCache&
tree_size_cache()
{
    static TreeSizeCache cache([] { return megabytes_to_bytes(get_config().tree_size_memo_mb); },
                               [](std::string const& key, TreeSizeEntry const&) {
                                   constexpr std::size_t node_overhead =
                                     112; // list node + hash node + control block
                                   return 2 * (sizeof(std::string) + key.capacity()) +
                                          sizeof(TreeSizeEntry) + node_overhead;
                               });
    return cache;
}

void
tree_size_key_put_sector(std::string& k, Sector const& s)
{
    k.push_back(static_cast<char>(s.len()));
    for (std::size_t i = 0; i < s.len(); ++i) {
        k.append(reinterpret_cast<char const*>(&s.q[i]), sizeof(int16_t));
    }
}

} // namespace

CacheStats
tree_size_cache_stats()
{
    return tree_size_cache().stats();
}

void
clear_tree_size_cache()
{
    tree_size_cache().clear();
}

std::size_t
FusionTree::approx_bytes() const
{
    // every non-empty vector is a separate heap block with allocator overhead (header, rounding)
    constexpr std::size_t block_overhead = 16;
    auto const block = [](std::size_t bytes) { return bytes == 0 ? 0 : bytes + block_overhead; };
    return sizeof(FusionTree) + block(uncoupled.capacity() * sizeof(Sector)) +
           block(inner_sectors.capacity() * sizeof(Sector)) + block(are_dual.capacity()) +
           block(multiplicities.capacity() * sizeof(int64));
}

std::size_t
fusion_trees::size() const
{
    if (num_uncoupled == 0) {
        return coupled == symmetry->trivial_sector ? 1 : 0;
    }
    if (num_uncoupled == 1) {
        return uncoupled[0] == coupled ? 1 : 0;
    }
    if (num_uncoupled == 2) {
        return static_cast<std::size_t>(symmetry->n_symbol(uncoupled[0], uncoupled[1], coupled));
    }

    std::string memo_key;
    {
        auto const* symmetry_address = symmetry.get();
        memo_key.append(reinterpret_cast<char const*>(&symmetry_address),
                        sizeof(symmetry_address));
        for (std::size_t i = 0; i < num_uncoupled; ++i) {
            tree_size_key_put_sector(memo_key, uncoupled[i]);
        }
        tree_size_key_put_sector(memo_key, coupled);
        if (auto const hit = tree_size_cache().find(memo_key);
            hit && hit->symmetry.lock() == symmetry) {
            return hit->size;
        }
    }

    Sector const a1 = uncoupled[0];
    Sector const a2 = uncoupled[1];
    std::size_t count = 0;
    SectorArray const fusion_bs = symmetry->fusion_outcomes(a1, a2);
    for (std::size_t ib = 0; ib < fusion_bs.size(); ++ib) {
        Sector const b = fusion_bs[ib];
        SectorArray const rest_unc =
          SectorArray::from_sector(b).concat(uncoupled.slice(2, num_uncoupled));
        // Python ``len(fusion_trees(...))`` omits are_dual → defaults to False
        // no need to check if the fusion is allowed in n_symbol -> use _n_symbol
        std::size_t const num_subtrees = fusion_trees(symmetry, rest_unc, coupled).size();
        count += static_cast<std::size_t>(symmetry->_n_symbol(a1, a2, b)) * num_subtrees;
    }
    tree_size_cache().insert(std::move(memo_key), TreeSizeEntry{ symmetry, count });
    return count;
}

std::string
fusion_trees::str() const
{
    auto const signature =
      FusionTree::str_uncoupled_coupled(*symmetry, uncoupled, coupled, are_dual);
    return std::format("fusion_trees[{}]({})", symmetry->str(), signature);
}

std::string
fusion_trees::repr() const
{
    std::string unc = replace_all(sector_array_str(uncoupled), "\n", ",");
    std::ostringstream dual_ss;
    dual_ss << '[';
    for (std::size_t i = 0; i < are_dual.size(); ++i) {
        if (i > 0) {
            dual_ss << ' ';
        }
        dual_ss << static_cast<int>(are_dual[i]);
    }
    dual_ss << ']';
    return std::format("fusion_trees({}, {}, {}, {})",
                       symmetry->repr(),
                       unc,
                       symmetry->sector_str(coupled),
                       dual_ss.str());
}

std::size_t
fusion_trees::index(FusionTree const& tree) const
{
    // --- hints from Python fusion_trees.index ---
    // check compatibility first (same symmetry, same uncoupled, same coupled, same are_dual)
    // ---
    if (!symmetry->is_equivalent_to(*tree.symmetry)) {
        throw std::invalid_argument(
          std::format("Inconsistent symmetries, {} != {}", symmetry->str(), tree.symmetry->str()));
    }
    if (!(uncoupled == tree.uncoupled)) {
        throw std::invalid_argument("Inconsistent uncoupled sectors");
    }
    if (coupled != tree.coupled) {
        throw std::invalid_argument("Inconsistent coupled sector");
    }
    if (are_dual.size() != tree.are_dual.size() ||
        !std::equal(are_dual.begin(), are_dual.end(), tree.are_dual.begin())) {
        throw std::invalid_argument("Inconsistent dualities");
    }
    return compute_index(tree);
}

std::size_t
fusion_trees::compute_index(FusionTree const& tree) const
{
    if (num_uncoupled < 2) {
        if (num_uncoupled == 0 && coupled == symmetry->trivial_sector) {
            return 0;
        }
        if (num_uncoupled == 1 && uncoupled[0] == coupled) {
            return 0;
        }
        throw std::invalid_argument("Inconsistent coupled sector.");
    }

    std::size_t idx = 0;
    // product of all multiplicities to the left of left_sec in for loop below
    int64 left_multi = 1;
    // upper limit for the values multiplicities take at each vertex (of the tree)
    std::vector<int64> max_multis;
    max_multis.reserve(num_uncoupled > 0 ? num_uncoupled - 1 : 0);

    for (std::size_t i = 0; i + 2 < num_uncoupled; ++i) {
        // coupled sector is unique, no need to shift idx for target_sec == self.coupled
        Sector const target_sec = tree.inner_sectors[i];
        Sector const left_sec = (i == 0) ? uncoupled[i] : tree.inner_sectors[i - 1];
        bool sector_found = false;
        SectorArray const outcomes = symmetry->fusion_outcomes(left_sec, uncoupled[i + 1]);
        for (std::size_t io = 0; io < outcomes.size(); ++io) {
            Sector const fusion_sec = outcomes[io];
            int64 const multi = symmetry->_n_symbol(left_sec, uncoupled[i + 1], fusion_sec);
            if (fusion_sec == target_sec) {
                sector_found = true;
                left_multi *= multi;
                max_multis.push_back(multi);
                break;
            }
            SectorArray const rest_unc =
              SectorArray::from_sector(fusion_sec).concat(uncoupled.slice(i + 2, num_uncoupled));
            std::vector<std::uint8_t> rest_dual = concat_vectors(
              std::span<std::uint8_t const>(std::vector<std::uint8_t>{ 0 }),
              std::span<std::uint8_t const>(vector_slice(are_dual, i + 2, are_dual.size())));
            idx += static_cast<std::size_t>(left_multi * multi) *
                   fusion_trees(symmetry, rest_unc, coupled, rest_dual).size();
        }
        if (!sector_found) {
            throw std::invalid_argument("Inconsistent inner sector.");
        }
    }

    Sector const left_sec =
      (num_uncoupled == 2) ? uncoupled[0] : tree.inner_sectors[tree.inner_sectors.size() - 1];
    if (!symmetry->can_fuse_to(left_sec, uncoupled[num_uncoupled - 1], coupled)) {
        throw std::invalid_argument("Inconsistent inner sector.");
    }

    max_multis.push_back(symmetry->_n_symbol(left_sec, uncoupled[num_uncoupled - 1], coupled));
    if (tree.multiplicities.size() != max_multis.size()) {
        throw std::invalid_argument("Inconsistent multiplicity.");
    }
    for (std::size_t i = 0; i < tree.multiplicities.size(); ++i) {
        if (!(tree.multiplicities[i] < max_multis[i])) {
            throw std::invalid_argument("Inconsistent multiplicity.");
        }
    }

    // idx shift from multiplicities
    if (!symmetry->is_abelian()) {
        for (std::size_t i = 0; i < tree.multiplicities.size(); ++i) {
            int64 prod = 1;
            for (std::size_t j = 0; j < i; ++j) {
                prod *= max_multis[j];
            }
            idx += static_cast<std::size_t>(tree.multiplicities[i] * prod);
        }
    }
    return idx;
}

} // namespace lasyten

// =============================================================================
// ORPHANED PYTHON COMMENT HINTS (no matching C++ function body found)
// =============================================================================
// --- FusionTree._ascii_diagram ---
// We build a splitting tree (dagger=True), and if dagger=False, we mirror it at the end
// note: arrows are the same no matter if dagger or not
// step 1: build just the bare tree without inner sectors or multiplicity labels
// only remember where they would go
// positions of the ┴
// last line: pre_Z_uncoupled
// line -2: Z or vertical wires
// line -3: uncoupled
// {row: extra_str}
// one above the vertex
// finalize
// --- FusionTree._str_uncoupled_coupled ---
// before Zs
// after Zs
// =============================================================================
