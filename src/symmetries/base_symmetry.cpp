#include <lasyten/symmetries/base_symmetry.h>

#include <lasyten/config.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace lasyten {

BaseSymmetry::BaseSymmetry(FusionStyle fusion_style_,
                           BraidingStyle braiding_style_,
                           Sector trivial_sector_,
                           float64 num_sectors_,
                           bool has_complex_topological_data_,
                           bool trivial_shift_)
  : fusion_style(fusion_style_)
  , braiding_style(braiding_style_)
  , trivial_sector(trivial_sector_)
  , num_sectors(num_sectors_)
  , sector_ind_len(trivial_sector_.len())
  , empty_sector_array(SectorArray::empty(trivial_sector_.len()))
  , has_complex_topological_data(has_complex_topological_data_)
  , trivial_shift(trivial_shift_)
{
    if (trivial_sector_.len() == 0) {
        throw std::invalid_argument("BaseSymmetry: invalid trivial_sector length");
    }
}

bool
BaseSymmetry::can_be_dropped() const
{
    // --- hints from Python BaseSymmetry.can_be_dropped ---
    // trivial braid -> can be dropped, clearly
    // symmetry braid -> we choose to allow it, but converting to/from numpy loses the braid
    // and makes swap gates necessary
    // general braid would break compatibility even with the tensor product, so we dont allow it
    // ---
    return has_symmetric_braid();
}

bool
BaseSymmetry::has_symmetric_braid() const
{
    return braiding_style <= BraidingStyle::fermionic;
}

bool
BaseSymmetry::has_trivial_braid() const
{
    return braiding_style == BraidingStyle::bosonic;
}

bool
BaseSymmetry::is_abelian() const
{
    return fusion_style == FusionStyle::single;
}

bool
BaseSymmetry::has_unique_fusion() const
{
    return fusion_style <= FusionStyle::multiple_unique;
}

FusionSymbol
BaseSymmetry::_fusion_tensor(Sector /*a*/, Sector /*b*/, Sector /*c*/, bool /*Z_a*/, bool /*Z_b*/)
  const
{
    if (!can_be_dropped()) {
        throw SymmetryError("fusion tensor can not be written as array for this symmetry");
    }
    throw std::runtime_error("BaseSymmetry::_fusion_tensor should be implemented by subclass");
}

FusionSymbol
BaseSymmetry::swap_gate(Sector /*a*/, Sector /*b*/) const
{
    if (!can_be_dropped()) {
        throw SymmetryError("braid can not be written as array for this symmetry");
    }
    throw std::runtime_error("BaseSymmetry::swap_gate should be implemented by subclass");
}

FusionSymbol
BaseSymmetry::Z_iso(Sector a) const
{
    // --- hints from Python BaseSymmetry.Z_iso ---
    // fallback implementation: solve [Jakob thesis, (5.84)] for Z_a
    // Note: leg order might be unintuitive at first!
    // [1] [2]     ;     [0]                 .--.  [0]
    // |   |      ;      |                  |  |   |
    // Y[0]Y      ;      Z   =   sqrt(d_a)  |  YYYYY   = sqrt(d_a) np.transpose(Y[0, :, :, 0])
    // |        ;      |                  |
    // [3]       ;     [1]                [1]
    // ---
    if (!can_be_dropped()) {
        throw SymmetryError("Z iso can not be written as array for this symmetry");
    }
    // fallback: sqrt(d_a) * conj(X)[0, :, :, 0].T
    auto X = fusion_tensor(a, dual_sector(a), trivial_sector).conj();
    // X has shape [μ, m_a, m_b, m_c]; take μ=0, m_c=0 → [m_a, m_b], then transpose
    auto const da = X.extent(1);
    auto const db = X.extent(2);
    FusionSymbol mat(2, FusionSymbol::Shape{ { da, db, 1, 1 } }, X.dtype());
    for (std::size_t i = 0; i < da; ++i) {
        for (std::size_t j = 0; j < db; ++j) {
            mat.set(i, j, X.get_complex(0, i, j, 0));
        }
    }
    return mat.transpose({ 1, 0, 2, 3 }) * sqrt_qdim(a);
}

SectorArray
BaseSymmetry::all_sectors() const
{
    if (!std::isfinite(num_sectors)) {
        throw SymmetryError("symmetry has infinitely many sectors.");
    }
    throw std::runtime_error("BaseSymmetry::all_sectors should be implemented in subclass");
}

int64
BaseSymmetry::n_symbol(Sector a, Sector b, Sector c) const
{
    if (!can_fuse_to(a, b, c)) {
        return 0;
    }
    return _n_symbol(a, b, c);
}

FusionSymbol
BaseSymmetry::f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    if (get_config().check_fusion) {
        bool ok = can_fuse_to(b, c, e) && can_fuse_to(a, e, d) && can_fuse_to(a, b, f) &&
                  can_fuse_to(f, c, d);
        if (!ok) {
            throw SymmetryError("Sectors are not consistent with fusion rules.");
        }
    }
    return _f_symbol(a, b, c, d, e, f);
}

FusionSymbol
BaseSymmetry::b_symbol(Sector a, Sector b, Sector c) const
{
    if (get_config().check_fusion) {
        if (!can_fuse_to(a, b, c)) {
            throw SymmetryError("Sectors are not consistent with fusion rules.");
        }
    }
    return _b_symbol(a, b, c);
}

FusionSymbol
BaseSymmetry::r_symbol(Sector a, Sector b, Sector c) const
{
    if (get_config().check_fusion) {
        if (!can_fuse_to(a, b, c)) {
            throw SymmetryError("Sectors are not consistent with fusion rules.");
        }
    }
    return _r_symbol(a, b, c);
}

FusionSymbol
BaseSymmetry::c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    if (get_config().check_fusion) {
        bool ok = can_fuse_to(a, b, e) && can_fuse_to(e, c, d) && can_fuse_to(a, c, f) &&
                  can_fuse_to(f, b, d);
        if (!ok) {
            throw SymmetryError("Sectors are not consistent with fusion rules.");
        }
    }
    return _c_symbol(a, b, c, d, e, f);
}

FusionSymbol
BaseSymmetry::fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const
{
    if (get_config().check_fusion) {
        if (!can_fuse_to(a, b, c)) {
            throw SymmetryError("Sectors are not consistent with fusion rules.");
        }
    }
    return _fusion_tensor(a, b, c, Z_a, Z_b);
}

bool
BaseSymmetry::are_valid_sectors(SectorArray const& sectors) const
{
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        if (!is_valid_sector(sectors[i])) {
            return false;
        }
    }
    return true;
}

SectorArray
BaseSymmetry::fusion_outcomes_broadcast(SectorArray const& a, SectorArray const& b) const
{
    // --- hints from Python BaseSymmetry.fusion_outcomes_broadcast ---
    // self.fusion_outcomes(s_a, s_b) is a 2D array with with shape [1, num_q]
    // stack the outcomes along the trivial first axis
    // ---
    if (!is_abelian()) {
        throw std::invalid_argument("fusion_outcomes_broadcast requires an abelian symmetry");
    }
    if (a.size() != b.size()) {
        throw std::invalid_argument("fusion_outcomes_broadcast: mismatched batch sizes");
    }
    if (a.sector_ind_len() != sector_ind_len || b.sector_ind_len() != sector_ind_len) {
        throw std::invalid_argument("fusion_outcomes_broadcast: mismatched sector_ind_len");
    }
    SectorArray out(a.size(), sector_ind_len);
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto outcomes = fusion_outcomes(a[i], b[i]);
        if (outcomes.size() != 1) {
            throw std::invalid_argument(
              "fusion_outcomes_broadcast: expected unique fusion outcome");
        }
        out[i] = outcomes[0];
    }
    return out;
}

Sector
BaseSymmetry::multiple_fusion(std::vector<Sector> const& sectors) const
{
    // --- hints from Python BaseSymmetry.multiple_fusion ---
    // OPTIMIZE ?
    // ---
    std::vector<SectorArray> as_arrays;
    as_arrays.reserve(sectors.size());
    for (auto const& s : sectors) {
        SectorArray row(1, s.len());
        row[0] = s;
        as_arrays.push_back(std::move(row));
    }
    return multiple_fusion_broadcast(as_arrays)[0];
}

SectorArray
BaseSymmetry::multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    if (sectors.empty()) {
        SectorArray out(1, sector_ind_len);
        out[0] = trivial_sector;
        return out;
    }
    if (sectors.size() == 1) {
        return sectors[0];
    }
    return _multiple_fusion_broadcast(sectors);
}

SectorArray
BaseSymmetry::_multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    assert(sectors.size() >= 2);
    SectorArray acc = sectors[0];
    for (std::size_t i = 1; i < sectors.size(); ++i) {
        acc = fusion_outcomes_broadcast(acc, sectors[i]);
    }
    return acc;
}

bool
BaseSymmetry::can_fuse_to(Sector a, Sector b, Sector c) const
{
    auto outcomes = fusion_outcomes(a, b);
    for (std::size_t i = 0; i < outcomes.size(); ++i) {
        if (outcomes[i] == c) {
            return true;
        }
    }
    return false;
}

int64
BaseSymmetry::sector_dim(Sector a) const
{
    if (!can_be_dropped()) {
        throw SymmetryError("sector_dim is not supported for this symmetry.");
    }
    // Note: Python incorrectly called qdim() without `a`; use qdim(a).
    return static_cast<int64>(std::llround(qdim(a)));
}

std::vector<int64>
BaseSymmetry::batch_sector_dim(SectorArray const& a) const
{
    std::vector<int64> out(a.size());
    if (is_abelian()) {
        std::fill(out.begin(), out.end(), 1);
        return out;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = sector_dim(a[i]);
    }
    return out;
}

std::vector<float64>
BaseSymmetry::batch_qdim(SectorArray const& a) const
{
    std::vector<float64> out(a.size());
    if (is_abelian()) {
        std::fill(out.begin(), out.end(), 1.0);
        return out;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = qdim(a[i]);
    }
    return out;
}

std::string
BaseSymmetry::sector_str(Sector a) const
{
    std::ostringstream oss;
    oss << '[';
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        if (i != 0) {
            oss << ' ';
        }
        oss << a[i];
    }
    oss << ']';
    return oss.str();
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

} // namespace

Sector
BaseSymmetry::get_sector(std::string const& s) const
{
    auto const t = trim_ws(s);
    if (t.empty()) {
        throw std::invalid_argument("get_sector: empty string");
    }
    if (!std::isfinite(num_sectors)) {
        throw std::invalid_argument(
          "get_sector: infinitely many sectors; this symmetry must implement get_sector");
    }
    auto const sectors = all_sectors();
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        if (sector_str(sectors[i]) == t) {
            return sectors[i];
        }
    }
    throw std::invalid_argument(std::format("get_sector: unknown sector '{}'", t));
}

SectorArray
BaseSymmetry::dual_sectors(SectorArray const& sectors) const
{
    SectorArray out(sectors.size(), sectors.sector_ind_len());
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        out[i] = dual_sector(sectors[i]);
    }
    return out;
}

int64
BaseSymmetry::frobenius_schur(Sector a) const
{
    auto F = _f_symbol(a, dual_sector(a), a, a, trivial_sector, trivial_sector);
    float64 re = F.get_complex(0, 0, 0, 0).real();
    return (re >= 0.0) ? 1 : -1;
}

float64
BaseSymmetry::qdim(Sector a) const
{
    auto F = _f_symbol(a, dual_sector(a), a, a, trivial_sector, trivial_sector);
    return 1.0 / std::abs(F.get_complex(0, 0, 0, 0));
}

float64
BaseSymmetry::sqrt_qdim(Sector a) const
{
    return std::sqrt(qdim(a));
}

float64
BaseSymmetry::inv_sqrt_qdim(Sector a) const
{
    return 1.0 / sqrt_qdim(a);
}

float64
BaseSymmetry::total_qdim() const
{
    auto sectors = all_sectors();
    float64 D2 = 0.0;
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        auto d = qdim(sectors[i]);
        D2 += d * d;
    }
    return std::sqrt(D2);
}

FusionSymbol
BaseSymmetry::_b_symbol(Sector a, Sector b, Sector c) const
{
    auto F = _f_symbol(a, b, dual_sector(b), a, trivial_sector, c).conj();
    // F[0, 0, :, :] * sqrt(d_b)
    return F.slice2d(0, 0) * sqrt_qdim(b);
}

FusionSymbol
BaseSymmetry::_c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    // --- hints from Python BaseSymmetry._c_symbol ---
    // axis [mu, nu, kap, lam] ; R symbols are diagonal
    // ---
    auto R1 = _r_symbol(e, c, d);
    auto F = _f_symbol(c, a, b, d, e, f);
    auto R2 = _r_symbol(a, c, f).conj();
    // R1[None, :, None, None] * F * conj(R2)[None, None, :, None]
    auto R1e = R1.reshaped(4, FusionSymbol::Shape{ { 1, R1.size(), 1, 1 } });
    auto R2e = R2.reshaped(4, FusionSymbol::Shape{ { 1, 1, R2.size(), 1 } });
    return R1e.multiply(F).multiply(R2e);
}

complex128
BaseSymmetry::topological_twist(Sector a) const
{
    // --- hints from Python BaseSymmetry.topological_twist ---
    // OPTIMIZE implement concrete formulae for anyons? or just cache?
    // sum_b sum_mu d_b / d_a * [R^aa_b]^mu_mu
    // must be +1 or -1
    // ---
    if (has_trivial_braid()) {
        return complex128{ +1.0, 0.0 };
    }
    complex128 res{ 0.0, 0.0 };
    auto outcomes = fusion_outcomes(a, a);
    for (std::size_t i = 0; i < outcomes.size(); ++i) {
        Sector b = outcomes[i];
        auto r = _r_symbol(a, a, b);
        res += qdim(b) * r.sum();
    }
    res /= qdim(a);
    if (has_symmetric_braid()) {
        float64 re = res.real();
        return complex128{ (re < 0.0) ? -1.0 : +1.0, 0.0 };
    }
    return res;
}

complex128
BaseSymmetry::s_matrix_element(Sector a, Sector b) const
{
    complex128 S{ 0.0, 0.0 };
    auto outcomes = fusion_outcomes(a, b);
    for (std::size_t i = 0; i < outcomes.size(); ++i) {
        Sector c = outcomes[i];
        S += static_cast<float64>(_n_symbol(a, b, c)) * qdim(c) * topological_twist(c);
    }
    S /= topological_twist(a) * topological_twist(b) * total_qdim();
    // real_if_close: if imag part tiny, drop it
    if (std::abs(S.imag()) < 1e-12) {
        return complex128{ S.real(), 0.0 };
    }
    return S;
}

FusionSymbol
BaseSymmetry::s_matrix() const
{
    auto sectors = all_sectors();
    auto const n = sectors.size();
    FusionSymbol S(2, FusionSymbol::Shape{ { n, n, 1, 1 } }, Dtype::Complex128);
    float64 D = total_qdim();
    std::vector<complex128> inv_twist(n);
    for (std::size_t i = 0; i < n; ++i) {
        inv_twist[i] = complex128{ 1.0, 0.0 } / topological_twist(sectors[i]);
    }
    for (std::size_t ia = 0; ia < n; ++ia) {
        Sector a = sectors[ia];
        for (std::size_t ib = 0; ib < n; ++ib) {
            Sector b = sectors[ib];
            complex128 Sab{ 0.0, 0.0 };
            auto outcomes = fusion_outcomes(a, b);
            for (std::size_t k = 0; k < outcomes.size(); ++k) {
                Sector c = outcomes[k];
                Sab += static_cast<float64>(_n_symbol(a, b, c)) * qdim(c) * topological_twist(c);
            }
            Sab *= inv_twist[ia] * inv_twist[ib] / D;
            if (std::abs(Sab.imag()) < 1e-12) {
                Sab = complex128{ Sab.real(), 0.0 };
            }
            S.set(ia, ib, Sab);
        }
    }
    return S;
}

} // namespace lasyten

// =============================================================================
// ORPHANED PYTHON COMMENT HINTS (no matching C++ function body found)
// =============================================================================
// --- Symmetry.__init__ ---
// avoid unnecessary nesting
// sanity check: multiple fermion symmetries probably dont do what you expect
// --- SymmetryFactor.__repr__ ---
// Convention: valid syntax for the constructor, i.e. "ClassName(..., name='...')"
// --- Group._fusion_tensor ---
// subclasses must implement. for groups it is always possible.
// =============================================================================
