#include <lasyten/symmetries/factors/fermion_number.h>

#include <lasyten/symmetries/topo_ones.h>

#include <cstdint>
#include <format>
#include <lasyten/tools/hdf5.h>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

int64_t
mod2(int16_t x)
{
    return static_cast<int64_t>(topo_ones::mod_n(static_cast<int32_t>(x), 2));
}

int64_t
fermion_sign(int16_t a, int16_t b)
{
    return 1 - 2 * mod2(a) * mod2(b);
}

} // namespace

FermionNumber::FermionNumber(std::optional<std::string> descriptive_name, bool trivial_shift)
  : SymmetryFactor(FusionStyle::single,
                   BraidingStyle::fermionic,
                   Sector{ 0 },
                   "FermionNumber",
                   std::numeric_limits<float64>::infinity(),
                   false,
                   std::move(descriptive_name),
                   trivial_shift)
{
    fusion_tensor_dtype = Dtype::Float64;
}

bool
FermionNumber::is_valid_sector(Sector a) const
{
    return a.len() == 1;
}

bool
FermionNumber::are_valid_sectors(SectorArray const& sectors) const
{
    return sectors.sector_ind_len() == 1;
}

SectorArray
FermionNumber::fusion_outcomes(Sector a, Sector b) const
{
    SectorArray aa(1, 1);
    SectorArray bb(1, 1);
    aa[0] = a;
    bb[0] = b;
    return fusion_outcomes_broadcast(aa, bb);
}

SectorArray
FermionNumber::fusion_outcomes_broadcast(SectorArray const& a, SectorArray const& b) const
{
    SectorArray out(a.size(), 1);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i][0] = static_cast<int16_t>(a[i][0] + b[i][0]);
    }
    return out;
}

SectorArray
FermionNumber::_multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    SectorArray out = sectors[0];
    for (std::size_t s = 1; s < sectors.size(); ++s) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i][0] = static_cast<int16_t>(out[i][0] + sectors[s][i][0]);
        }
    }
    return out;
}

int64
FermionNumber::sector_dim(Sector /*a*/) const
{
    return 1;
}

std::vector<int64>
FermionNumber::batch_sector_dim(SectorArray const& a) const
{
    return std::vector<int64>(a.size(), 1);
}

std::vector<float64>
FermionNumber::batch_qdim(SectorArray const& a) const
{
    return std::vector<float64>(a.size(), 1.0);
}

Sector
FermionNumber::get_sector(std::string const& s) const
{
    auto const start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        throw std::invalid_argument("get_sector: empty string");
    }
    auto const end = s.find_last_not_of(" \t\n\r");
    auto t = s.substr(start, end - start + 1);
    if (t.size() >= 2 && t.front() == '[' && t.back() == ']') {
        auto const inner = std::string_view(t).substr(1, t.size() - 2);
        auto const i0 = inner.find_first_not_of(" \t\n\r");
        if (i0 == std::string_view::npos) {
            throw std::invalid_argument(
              std::format("get_sector: cannot parse sector from '{}'", s));
        }
        auto const i1 = inner.find_last_not_of(" \t\n\r");
        t = std::string(inner.substr(i0, i1 - i0 + 1));
    }
    std::size_t idx = 0;
    long value = 0;
    try {
        value = std::stol(t, &idx, 10);
    } catch (std::exception const&) {
        throw std::invalid_argument(std::format("get_sector: cannot parse sector from '{}'", s));
    }
    if (idx != t.size() || value < static_cast<long>(std::numeric_limits<int16_t>::min()) ||
        value > static_cast<long>(std::numeric_limits<int16_t>::max())) {
        throw std::invalid_argument(std::format("get_sector: cannot parse sector from '{}'", s));
    }
    Sector const a{ static_cast<int16_t>(value) };
    if (!is_valid_sector(a)) {
        throw std::invalid_argument(std::format("get_sector: '{}' is not a valid sector", s));
    }
    return a;
}

bool
FermionNumber::_is_equivalent_factor(SymmetryFactor const& other) const
{
    return dynamic_cast<FermionNumber const*>(&other) != nullptr;
}

Sector
FermionNumber::dual_sector(Sector a) const
{
    return Sector{ static_cast<int16_t>(-a.q[0]) };
}

SectorArray
FermionNumber::dual_sectors(SectorArray const& sectors) const
{
    SectorArray out(sectors.size(), 1);
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        out[i][0] = static_cast<int16_t>(-sectors[i][0]);
    }
    return out;
}

int64
FermionNumber::_n_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return 1;
}

FusionSymbol
FermionNumber::_f_symbol(Sector /*a*/,
                         Sector /*b*/,
                         Sector /*c*/,
                         Sector /*d*/,
                         Sector /*e*/,
                         Sector /*f*/) const
{
    return topo_ones::one_4D();
}

int64
FermionNumber::frobenius_schur(Sector /*a*/) const
{
    return 1;
}

float64
FermionNumber::qdim(Sector /*a*/) const
{
    return 1.0;
}

float64
FermionNumber::sqrt_qdim(Sector /*a*/) const
{
    return 1.0;
}

float64
FermionNumber::inv_sqrt_qdim(Sector /*a*/) const
{
    return 1.0;
}

FusionSymbol
FermionNumber::_b_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    // --- hints from Python FermionNumber._b_symbol ---
    // sqrt(d_b) [F^{a b dual(b)}_a]^{111}_{c,mu,nu} = sqrt(1) * 1 = 1
    // ---
    return topo_ones::one_2D();
}

FusionSymbol
FermionNumber::_r_symbol(Sector a, Sector b, Sector /*c*/) const
{
    // --- hints from Python FermionNumber._r_symbol ---
    // if a and b are odd -1, otherwise +1
    // in the first (second) case above, we have ``a * b`` equal to 1 (0).
    // ---
    return FusionSymbol::scalar1d(static_cast<float64>(fermion_sign(a.q[0], b.q[0])));
}

FusionSymbol
FermionNumber::_c_symbol(Sector a, Sector /*b*/, Sector c, Sector /*d*/, Sector e, Sector /*f*/)
  const
{
    // --- hints from Python FermionNumber._c_symbol ---
    // F = 1  -->  C = R^{ec}_d conj(R)^{ca}_f
    // ---
    auto const C = fermion_sign(c.q[0], e.q[0]) * fermion_sign(a.q[0], c.q[0]);
    return FusionSymbol::full(
      4, FusionSymbol::Shape{ { 1, 1, 1, 1 } }, static_cast<float64>(C), Dtype::Float64);
}

FusionSymbol
FermionNumber::_fusion_tensor(Sector /*a*/, Sector /*b*/, Sector /*c*/, bool /*Z_a*/, bool /*Z_b*/)
  const
{
    return topo_ones::one_4D_float();
}

FusionSymbol
FermionNumber::swap_gate(Sector a, Sector b) const
{
    // --- hints from Python FermionNumber.swap_gate ---
    // if a and b are odd -1, otherwise +1
    // in the first (second) case above, we have ``a * b`` equal to 1 (0).
    // ---
    auto const sign = static_cast<float64>(fermion_sign(a.q[0], b.q[0]));
    return topo_ones::one_4D_float() * sign;
}

complex128
FermionNumber::topological_twist(Sector a) const
{
    // --- hints from Python FermionNumber.topological_twist ---
    // +1 for even parity, -1 for odd
    // ---
    auto const sign = static_cast<float64>(1 - 2 * mod2(a.q[0]));
    return complex128{ sign, 0.0 };
}

FusionSymbol
FermionNumber::Z_iso(Sector /*a*/) const
{
    return topo_ones::one_2D_float();
}

std::string
FermionNumber::repr() const
{
    if (!descriptive_name.has_value()) {
        return "FermionNumber()";
    }
    return std::string("FermionNumber(\"") + *descriptive_name + "\")";
}

FermionNumber::Ptr
FermionNumber::from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath)
{
    auto name = descriptive_name_from_hdf5_attrs(h5gr);
    bool trivial_shift = trivial_shift_from_hdf5(loader, subpath);
    auto obj = std::make_shared<FermionNumber>(name, trivial_shift);
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

} // namespace lasyten
