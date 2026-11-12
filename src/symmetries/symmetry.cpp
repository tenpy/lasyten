#include <lasyten/symmetries/symmetry.h>

#include <lasyten/symmetries/fusion_symbol.h>
#include <lasyten/tools/warn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <hdf5_io/constants.h>
#include <hdf5_io/h5_ops.h>
#include <lasyten/tools/hdf5.h>
#include <lasyten/tools/hdf5_export.h>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace lasyten {

FusionStyle
Symmetry::max_fusion_style(std::vector<SymmetryFactor::Ptr> const& factors)
{
    if (factors.empty()) {
        return FusionStyle::single;
    }
    auto style = factors[0]->fusion_style;
    for (std::size_t i = 1; i < factors.size(); ++i) {
        if (factors[i]->fusion_style > style) {
            style = factors[i]->fusion_style;
        }
    }
    return style;
}

BraidingStyle
Symmetry::max_braiding_style(std::vector<SymmetryFactor::Ptr> const& factors)
{
    if (factors.empty()) {
        return BraidingStyle::bosonic;
    }
    auto style = factors[0]->braiding_style;
    for (std::size_t i = 1; i < factors.size(); ++i) {
        if (factors[i]->braiding_style > style) {
            style = factors[i]->braiding_style;
        }
    }
    return style;
}

Sector
Symmetry::concat_trivial_sectors(std::vector<SymmetryFactor::Ptr> const& factors)
{
    std::size_t len = 0;
    for (auto const& f : factors) {
        len += f->sector_ind_len;
    }
    if (len > max_sector_ind_len) {
        throw std::invalid_argument("Symmetry: product sector_ind_len exceeds max_sector_ind_len");
    }
    std::array<int16_t, max_sector_ind_len> buf{};
    std::size_t off = 0;
    for (auto const& f : factors) {
        for (std::uint8_t j = 0; j < f->sector_ind_len; ++j) {
            buf[off++] = f->trivial_sector.q[j];
        }
    }
    return Sector::from_span(std::span<const int16_t>(buf.data(), len));
}

float64
Symmetry::prod_num_sectors(std::vector<SymmetryFactor::Ptr> const& factors)
{
    float64 prod = 1.0;
    for (auto const& f : factors) {
        prod *= f->num_sectors;
        if (!std::isfinite(prod)) {
            return std::numeric_limits<float64>::infinity();
        }
    }
    return prod;
}

Symmetry::Symmetry(std::vector<SymmetryFactor::Ptr> factors_in)
  : BaseSymmetry(max_fusion_style(factors_in),
                 max_braiding_style(factors_in),
                 concat_trivial_sectors(factors_in),
                 prod_num_sectors(factors_in),
                 false,
                 true)
  , factors(std::move(factors_in))
{
    // Flatten nested Symmetry factors (should already be SymmetryFactor::Ptr only).
    for (auto const& f : factors) {
        if (!f) {
            throw std::invalid_argument("Symmetry: null factor");
        }
        if (dynamic_cast<Symmetry const*>(f.get()) != nullptr) {
            throw std::invalid_argument(
              "Symmetry: nested Symmetry factors must be flattened before construct");
        }
    }

    sector_slices.assign(1, 0);
    for (auto const& f : factors) {
        auto next = static_cast<std::uint8_t>(sector_slices.back() + f->sector_ind_len);
        sector_slices.push_back(next);
    }

    has_complex_topological_data = std::any_of(factors.begin(), factors.end(), [](auto const& f) {
        return f->has_complex_topological_data;
    });
    trivial_shift =
      std::all_of(factors.begin(), factors.end(), [](auto const& f) { return f->trivial_shift; });

    std::vector<Dtype> dtypes;
    dtypes.reserve(factors.size());
    bool any_none = false;
    for (auto const& f : factors) {
        // Python subclasses often set fusion_tensor_dtype as a class attribute; the C++
        // optional member may still be empty. Read via the Python object.
        py::object dt_py = py::cast(f).attr("fusion_tensor_dtype");
        if (dt_py.is_none()) {
            any_none = true;
            break;
        }
        dtypes.push_back(dt_py.cast<Dtype>());
    }
    if (any_none || factors.empty()) {
        fusion_tensor_dtype = std::nullopt;
    } else {
        fusion_tensor_dtype = dtype::common(dtypes);
    }

    // Multiple fermionic factors: warn via Python warnings module.
    int num_fermionic = 0;
    for (auto const& f : factors) {
        // Type check via Python isinstance once bindings exist; use group_name heuristic + RTTI
        // later.
        auto const& name = f->group_name;
        if (name.find("Fermion") != std::string::npos) {
            ++num_fermionic;
        }
    }
    if (num_fermionic > 1) {
        warn("Symmetry with multiple fermionic factors probably does not do what you "
             "expect. See docstring of FermionParity for details.");
    }
}

Sector
Symmetry::factor_sector(Sector const& a, std::size_t i) const
{
    auto const begin = sector_slices[i];
    auto const end = sector_slices[i + 1];
    return Sector::from_span(std::span<const int16_t>(a.q.data() + begin, end - begin));
}

SectorArray
Symmetry::factor_sectors(SectorArray const& a, std::size_t i) const
{
    auto const begin = sector_slices[i];
    auto const end = sector_slices[i + 1];
    auto const flen = static_cast<std::uint8_t>(end - begin);
    SectorArray out(a.size(), flen);
    for (std::size_t r = 0; r < a.size(); ++r) {
        out[r] = Sector::from_span(a[r].span().subspan(begin, flen));
    }
    return out;
}

std::size_t
Symmetry::factor_where(std::string const& descriptive_name) const
{
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (factors[i]->descriptive_name && *factors[i]->descriptive_name == descriptive_name) {
            return i;
        }
    }
    throw std::invalid_argument("Name not found: " + descriptive_name);
}

bool
Symmetry::has_factor(SymmetryFactor const& other) const
{
    for (auto const& f : factors) {
        if (f->equals(other)) {
            return true;
        }
    }
    return false;
}

bool
Symmetry::is_equivalent_to(Symmetry const& other, bool strict_ordering) const
{
    // --- hints from Python Symmetry.is_equivalent_to ---
    // no break occurred
    // if the loop terminates without returning False, every f1 found a match
    // ---
    if (num_factors() != other.num_factors()) {
        return false;
    }
    if (strict_ordering) {
        for (std::size_t i = 0; i < factors.size(); ++i) {
            if (!factors[i]->_is_equivalent_factor(*other.factors[i])) {
                return false;
            }
        }
        return true;
    }
    std::vector<bool> matched(other.factors.size(), false);
    for (auto const& f1 : factors) {
        bool found = false;
        for (std::size_t i = 0; i < other.factors.size(); ++i) {
            if (matched[i]) {
                continue;
            }
            if (f1->_is_equivalent_factor(*other.factors[i])) {
                matched[i] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

Symmetry::Ptr
Symmetry::as_Symmetry()
{
    return std::static_pointer_cast<Symmetry>(shared_from_this());
}

bool
Symmetry::is_valid_sector(Sector a) const
{
    if (a.len() != sector_ind_len) {
        return false;
    }
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (!factors[i]->is_valid_sector(factor_sector(a, i))) {
            return false;
        }
    }
    return true;
}

bool
Symmetry::are_valid_sectors(SectorArray const& sectors) const
{
    if (sectors.sector_ind_len() != sector_ind_len) {
        return false;
    }
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (!factors[i]->are_valid_sectors(factor_sectors(sectors, i))) {
            return false;
        }
    }
    return true;
}

SectorArray
Symmetry::fusion_outcomes(Sector a, Sector b) const
{
    // --- hints from Python Symmetry.fusion_outcomes ---
    // form an array of all combinations of the c_i
    // e.g. if we have 3 factors, we want
    // result[n1, n2, n3, :] = np.concatenate([c_1[n1, :], c_2[n2, :], c_3[n3, :]], axis=-1)
    // ---
    if (factors.empty()) {
        return SectorArray::empty(0);
    }

    std::vector<SectorArray> all_outcomes;
    std::vector<std::size_t> num_possibilities;
    all_outcomes.reserve(factors.size());
    num_possibilities.reserve(factors.size());

    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto c_i = factors[i]->fusion_outcomes(factor_sector(a, i), factor_sector(b, i));
        num_possibilities.push_back(c_i.size());
        all_outcomes.push_back(std::move(c_i));
    }

    std::size_t n_rows = 1;
    for (auto n : num_possibilities) {
        n_rows *= n;
    }
    SectorArray result(n_rows, sector_ind_len);
    for (std::size_t row = 0; row < n_rows; ++row) {
        Sector s = Sector::zeros(sector_ind_len);
        std::size_t tmp = row;
        // C-order: last factor index varies fastest.
        for (std::size_t fi = factors.size(); fi-- > 0;) {
            auto const ni = num_possibilities[fi];
            auto const idx = tmp % ni;
            tmp /= ni;
            auto const& c = all_outcomes[fi][idx];
            auto const begin = sector_slices[fi];
            for (std::uint8_t j = 0; j < c.len(); ++j) {
                s.q[begin + j] = c.q[j];
            }
        }
        result[row] = s;
    }
    return result;
}

SectorArray
Symmetry::fusion_outcomes_broadcast(SectorArray const& a, SectorArray const& b) const
{
    // --- hints from Python Symmetry.fusion_outcomes_broadcast ---
    // the c_i have the same first axis as a and b.
    // it remains to concatenate them along the last axis
    // ---
    if (!is_abelian()) {
        throw std::invalid_argument("fusion_outcomes_broadcast requires an abelian symmetry");
    }
    std::vector<SectorArray> components;
    components.reserve(factors.size());
    for (std::size_t i = 0; i < factors.size(); ++i) {
        components.push_back(
          factors[i]->fusion_outcomes_broadcast(factor_sectors(a, i), factor_sectors(b, i)));
    }
    // concatenate along last axis
    SectorArray out(a.size(), sector_ind_len);
    for (std::size_t r = 0; r < a.size(); ++r) {
        std::array<int16_t, max_sector_ind_len> buf{};
        std::size_t off = 0;
        for (std::size_t i = 0; i < factors.size(); ++i) {
            auto const flen = components[i].sector_ind_len();
            for (std::uint8_t j = 0; j < flen; ++j) {
                buf[off + j] = components[i][r][j];
            }
            off += flen;
        }
        out[r] = Sector::from_span(std::span<const int16_t>(buf.data(), sector_ind_len));
    }
    return out;
}

SectorArray
Symmetry::_multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    std::vector<SectorArray> components;
    components.reserve(factors.size());
    for (std::size_t i = 0; i < factors.size(); ++i) {
        std::vector<SectorArray> sectors_i;
        sectors_i.reserve(sectors.size());
        for (auto const& s : sectors) {
            sectors_i.push_back(factor_sectors(s, i));
        }
        components.push_back(factors[i]->_multiple_fusion_broadcast(sectors_i));
    }
    auto const n = components.empty() ? 0 : components[0].size();
    SectorArray out(n, sector_ind_len);
    for (std::size_t r = 0; r < n; ++r) {
        std::array<int16_t, max_sector_ind_len> buf{};
        std::size_t off = 0;
        for (std::size_t i = 0; i < factors.size(); ++i) {
            auto const flen = components[i].sector_ind_len();
            for (std::uint8_t j = 0; j < flen; ++j) {
                buf[off + j] = components[i][r][j];
            }
            off += flen;
        }
        out[r] = Sector::from_span(std::span<const int16_t>(buf.data(), sector_ind_len));
    }
    return out;
}

Sector
Symmetry::dual_sector(Sector a) const
{
    std::array<int16_t, max_sector_ind_len> buf{};
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto d = factors[i]->dual_sector(factor_sector(a, i));
        auto const begin = sector_slices[i];
        for (std::uint8_t j = 0; j < d.len(); ++j) {
            buf[begin + j] = d.q[j];
        }
    }
    return Sector::from_span(std::span<const int16_t>(buf.data(), a.len()));
}

SectorArray
Symmetry::dual_sectors(SectorArray const& sectors) const
{
    SectorArray res(sectors.size(), sectors.sector_ind_len());
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto d = factors[i]->dual_sectors(factor_sectors(sectors, i));
        auto const begin = sector_slices[i];
        auto const flen = d.sector_ind_len();
        for (std::size_t r = 0; r < sectors.size(); ++r) {
            for (std::uint8_t j = 0; j < flen; ++j) {
                res[r][begin + j] = d[r][j];
            }
        }
    }
    return res;
}

int64
Symmetry::_n_symbol(Sector a, Sector b, Sector c) const
{
    if (has_unique_fusion()) {
        return 1;
    }
    int64 res = 1;
    for (std::size_t i = 0; i < factors.size(); ++i) {
        res *=
          factors[i]->_n_symbol(factor_sector(a, i), factor_sector(b, i), factor_sector(c, i));
    }
    return res;
}

SectorArray
Symmetry::all_sectors() const
{
    // --- hints from Python Symmetry.all_sectors ---
    // construct like in fusion_outcomes
    // ---
    if (!std::isfinite(num_sectors)) {
        throw SymmetryError("symmetry has infinitely many sectors.");
    }
    if (factors.empty()) {
        return SectorArray::empty(0);
    }

    std::vector<SectorArray> factor_secs;
    std::vector<std::size_t> num_possibilities;
    factor_secs.reserve(factors.size());
    num_possibilities.reserve(factors.size());
    for (auto const& f : factors) {
        auto secs = f->all_sectors();
        num_possibilities.push_back(secs.size());
        factor_secs.push_back(std::move(secs));
    }

    std::size_t n_rows = 1;
    for (auto n : num_possibilities) {
        n_rows *= n;
    }
    SectorArray result(n_rows, sector_ind_len);
    for (std::size_t row = 0; row < n_rows; ++row) {
        Sector s = Sector::zeros(sector_ind_len);
        std::size_t tmp = row;
        for (std::size_t fi = factors.size(); fi-- > 0;) {
            auto const ni = num_possibilities[fi];
            auto const idx = tmp % ni;
            tmp /= ni;
            auto const& c = factor_secs[fi][idx];
            auto const begin = sector_slices[fi];
            for (std::uint8_t j = 0; j < c.len(); ++j) {
                s.q[begin + j] = c.q[j];
            }
        }
        result[row] = s;
    }
    return result;
}

int64
Symmetry::sector_dim(Sector a) const
{
    if (is_abelian()) {
        return 1;
    }
    int64 dim = 1;
    for (std::size_t i = 0; i < factors.size(); ++i) {
        dim *= factors[i]->sector_dim(factor_sector(a, i));
    }
    return dim;
}

std::vector<int64>
Symmetry::batch_sector_dim(SectorArray const& a) const
{
    if (is_abelian()) {
        return std::vector<int64>(a.size(), 1);
    }
    std::vector<int64> dims(a.size(), 1);
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto fi = factors[i]->batch_sector_dim(factor_sectors(a, i));
        for (std::size_t j = 0; j < dims.size(); ++j) {
            dims[j] *= fi[j];
        }
    }
    return dims;
}

std::vector<float64>
Symmetry::batch_qdim(SectorArray const& a) const
{
    if (is_abelian()) {
        return std::vector<float64>(a.size(), 1.0);
    }
    std::vector<float64> dims(a.size(), 1.0);
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto fi = factors[i]->batch_qdim(factor_sectors(a, i));
        for (std::size_t j = 0; j < dims.size(); ++j) {
            dims[j] *= fi[j];
        }
    }
    return dims;
}

float64
Symmetry::qdim(Sector a) const
{
    if (is_abelian()) {
        return 1.0;
    }
    float64 dim = 1.0;
    for (std::size_t i = 0; i < factors.size(); ++i) {
        dim *= factors[i]->qdim(factor_sector(a, i));
    }
    return dim;
}

std::string
Symmetry::sector_str(Sector a) const
{
    std::string out = "[";
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += factors[i]->sector_str(factor_sector(a, i));
    }
    out += "]";
    return out;
}

namespace {

std::string
trim_ws(std::string_view s)
{
    auto const start = s.find_first_not_of(" \t\n\r");
    if (start == std::string_view::npos) {
        return {};
    }
    auto const end = s.find_last_not_of(" \t\n\r");
    return std::string(s.substr(start, end - start + 1));
}

bool
matching_outer_brackets(std::string_view s)
{
    if (s.size() < 2 || s.front() != '[' || s.back() != ']') {
        return false;
    }
    int depth = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        char const c = s[i];
        if (c == '[') {
            ++depth;
        } else if (c == ']') {
            --depth;
            if (depth == 0) {
                return i + 1 == s.size();
            }
            if (depth < 0) {
                return false;
            }
        }
    }
    return false;
}

std::vector<std::string>
split_comma_top_level(std::string_view inner)
{
    std::vector<std::string> parts;
    int depth_bracket = 0;
    int depth_paren = 0;
    std::size_t start = 0;
    for (std::size_t i = 0; i < inner.size(); ++i) {
        char const c = inner[i];
        if (c == '[') {
            ++depth_bracket;
        } else if (c == ']') {
            --depth_bracket;
        } else if (c == '(') {
            ++depth_paren;
        } else if (c == ')') {
            --depth_paren;
        } else if (c == ',' && depth_bracket == 0 && depth_paren == 0) {
            parts.push_back(trim_ws(inner.substr(start, i - start)));
            start = i + 1;
        }
    }
    parts.push_back(trim_ws(inner.substr(start)));
    return parts;
}

} // namespace

Sector
Symmetry::get_sector(std::string const& s) const
{
    auto const t = trim_ws(s);
    if (t.empty()) {
        throw std::invalid_argument("get_sector: empty string");
    }
    if (factors.empty()) {
        throw std::invalid_argument("get_sector: symmetry has no factors");
    }

    auto concat_factor_sectors = [&](std::vector<Sector> const& secs) -> Sector {
        std::array<int16_t, max_sector_ind_len> buf{};
        for (std::size_t i = 0; i < factors.size(); ++i) {
            auto const begin = sector_slices[i];
            for (std::uint8_t j = 0; j < secs[i].len(); ++j) {
                buf[static_cast<std::size_t>(begin) + j] = secs[i].q[j];
            }
        }
        auto out = Sector::from_span(std::span<const int16_t>(buf.data(), sector_ind_len));
        if (!is_valid_sector(out)) {
            throw std::invalid_argument("get_sector: parsed sector is not valid");
        }
        return out;
    };

    auto try_factor_parse = [&](std::size_t i, std::string const& cand) -> std::optional<Sector> {
        try {
            return factors[i]->get_sector(cand);
        } catch (std::invalid_argument const&) {
            return std::nullopt;
        } catch (SymmetryError const&) {
            return std::nullopt;
        }
    };

    if (factors.size() == 1) {
        if (auto sec = try_factor_parse(0, t)) {
            return concat_factor_sectors({ *sec });
        }
        if (matching_outer_brackets(t)) {
            auto const inner = trim_ws(std::string_view(t).substr(1, t.size() - 2));
            if (auto sec = try_factor_parse(0, inner)) {
                return concat_factor_sectors({ *sec });
            }
        }
        throw std::invalid_argument(std::format("get_sector: cannot parse sector from '{}'", t));
    }

    if (!matching_outer_brackets(t)) {
        throw std::invalid_argument(std::format(
          "get_sector: multi-factor symmetry requires bracketed sector string, got '{}'", t));
    }
    auto const inner = trim_ws(std::string_view(t).substr(1, t.size() - 2));
    auto const parts = split_comma_top_level(inner);
    if (parts.size() != factors.size()) {
        throw std::invalid_argument(std::format(
          "get_sector: expected {} factor sector(s), got {}", factors.size(), parts.size()));
    }
    std::vector<Sector> secs;
    secs.reserve(factors.size());
    for (std::size_t i = 0; i < factors.size(); ++i) {
        secs.push_back(factors[i]->get_sector(parts[i]));
    }
    return concat_factor_sectors(secs);
}

FusionSymbol
Symmetry::_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    FusionSymbol res = FusionSymbol::one_4D();
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto Fi = factors[i]->_f_symbol(factor_sector(a, i),
                                        factor_sector(b, i),
                                        factor_sector(c, i),
                                        factor_sector(d, i),
                                        factor_sector(e, i),
                                        factor_sector(f, i));
        res = kron(res, Fi);
    }
    return res;
}

FusionSymbol
Symmetry::_r_symbol(Sector a, Sector b, Sector c) const
{
    FusionSymbol res = FusionSymbol::one_1D();
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto Ri =
          factors[i]->_r_symbol(factor_sector(a, i), factor_sector(b, i), factor_sector(c, i));
        res = kron(res, Ri);
    }
    return res;
}

FusionSymbol
Symmetry::_fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const
{
    if (!can_be_dropped()) {
        throw SymmetryError("fusion tensor can not be written as array for this symmetry");
    }
    FusionSymbol res = FusionSymbol::one_4D();
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto Xi = factors[i]->_fusion_tensor(
          factor_sector(a, i), factor_sector(b, i), factor_sector(c, i), Z_a, Z_b);
        res = kron(res, Xi);
    }
    return res;
}

FusionSymbol
Symmetry::swap_gate(Sector a, Sector b) const
{
    if (!can_be_dropped()) {
        throw SymmetryError("fusion tensor can not be written as array for this symmetry");
    }
    FusionSymbol res = FusionSymbol::one_4D();
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto Si = factors[i]->swap_gate(factor_sector(a, i), factor_sector(b, i));
        res = kron(res, Si);
    }
    return res;
}

FusionSymbol
Symmetry::Z_iso(Sector a) const
{
    if (!can_be_dropped()) {
        throw SymmetryError("Z iso can not be written as array for this symmetry");
    }
    FusionSymbol res = FusionSymbol::one_2D();
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto Zi = factors[i]->Z_iso(factor_sector(a, i));
        res = kron(res, Zi);
    }
    return res;
}

std::string
Symmetry::repr() const
{
    if (num_factors() == 0) {
        return "Symmetry([])";
    }
    if (num_factors() == 1) {
        return "Symmetry([" + factors[0]->repr() + "])";
    }
    std::string out = factors[0]->repr();
    for (std::size_t i = 1; i < factors.size(); ++i) {
        out += " * " + factors[i]->repr();
    }
    return out;
}

std::string
Symmetry::str() const
{
    if (num_factors() == 0) {
        return "Symmetry([])";
    }
    if (num_factors() == 1) {
        return "Symmetry([" + factors[0]->str() + "])";
    }
    std::string out = factors[0]->str();
    for (std::size_t i = 1; i < factors.size(); ++i) {
        out += " ⨉ " + factors[i]->str();
    }
    return out;
}

bool
Symmetry::equals(Symmetry const& other) const
{
    if (num_factors() != other.num_factors()) {
        return false;
    }
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (!factors[i]->equals(*other.factors[i])) {
            return false;
        }
    }
    return true;
}

Symmetry::Ptr
Symmetry::mul(SymmetryFactor::Ptr other) const
{
    auto out = factors;
    out.push_back(std::move(other));
    return std::make_shared<Symmetry>(std::move(out));
}

Symmetry::Ptr
Symmetry::mul(Symmetry const& other) const
{
    auto out = factors;
    out.insert(out.end(), other.factors.begin(), other.factors.end());
    return std::make_shared<Symmetry>(std::move(out));
}

void
Symmetry::save_hdf5(lasyten::hdf5::Saver& saver,
                    HighFive::Group& h5gr,
                    std::string const& subpath) const
{
    HighFive::Group factors_g;
    std::string factors_sub;
    saver.save_sequence_begin(subpath + "factors",
                              hdf5_io::REPR_LIST,
                              static_cast<std::int64_t>(factors.size()),
                              factors_g,
                              factors_sub);
    lasyten::hdf5::Saver factors_saver(factors_g);
    for (std::size_t i = 0; i < factors.size(); ++i) {
        hdf5_export::save_symmetry_factor(factors_saver, std::to_string(i), factors[i]);
    }
    {
        std::vector<std::int64_t> slices(sector_slices.begin(), sector_slices.end());
        hdf5_export::save_i64_vector(saver, subpath + "sector_slices", slices);
    }
    if (fusion_tensor_dtype.has_value()) {
        saver.save_int64(subpath + "fusion_tensor_dtype",
                         static_cast<std::int64_t>(*fusion_tensor_dtype));
    } else {
        saver.save_none(subpath + "fusion_tensor_dtype");
    }
    saver.save_int64(subpath + "fusion_style", static_cast<std::int64_t>(fusion_style));
    saver.save_int64(subpath + "braiding_style", static_cast<std::int64_t>(braiding_style));
    hdf5_export::save_sector(saver, subpath + "trivial_sector", trivial_sector);
    saver.save_float64(subpath + "num_sectors", num_sectors);
    saver.save_int64(subpath + "sector_ind_len", static_cast<std::int64_t>(sector_ind_len));
    hdf5_io::h5_set_attr(
      h5gr.getId(), "has_complex_topological_data", has_complex_topological_data);
}

Symmetry::Ptr
Symmetry::from_hdf5(lasyten::hdf5::Loader& loader,
                    HighFive::Group& h5gr,
                    std::string const& subpath)
{
    hid_t factors_id = loader.open(subpath + "factors");
    auto const n = hdf5_io::h5_get_attr_int64(factors_id, hdf5_io::ATTR_LEN).value_or(0);
    HighFive::Group factors_g = hdf5_io::group_from_hid(factors_id);
    lasyten::hdf5::Loader factors_loader(factors_g);
    std::vector<SymmetryFactor::Ptr> factors;
    factors.reserve(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        factors.push_back(hdf5_export::load_symmetry_factor(factors_loader, std::to_string(i)));
    }
    auto obj = std::make_shared<Symmetry>(std::move(factors));
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

} // namespace lasyten
