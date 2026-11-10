#pragma once

#include "../symmetry_factor.h"

#include <lasyten/tools/hdf5.h>
#include <optional>
#include <string>
#include <vector>

namespace lasyten {

/// Conserves a fermionic particle number (U(1)-like fusion, fermionic braid).
///
/// This is essentially U(1), but with a braid that encodes fermionic exchange statistics.
/// Allowed sectors are arrays with a single integer entry.
class FermionNumber : public SymmetryFactor
{
  public:
    using Ptr = std::shared_ptr<FermionNumber>;
    using CPtr = std::shared_ptr<const FermionNumber>;

    explicit FermionNumber(std::optional<std::string> descriptive_name = std::nullopt,
                           bool trivial_shift = true);
    ~FermionNumber() override = default;

    bool is_valid_sector(Sector a) const override;
    bool are_valid_sectors(SectorArray const& sectors) const override;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override;
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override;
    int64 sector_dim(Sector a) const override;
    std::vector<int64> batch_sector_dim(SectorArray const& a) const override;
    std::vector<float64> batch_qdim(SectorArray const& a) const override;
    Sector get_sector(std::string const& s) const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;
    Sector dual_sector(Sector a) const override;
    SectorArray dual_sectors(SectorArray const& sectors) const override;
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    int64 frobenius_schur(Sector a) const override;
    float64 qdim(Sector a) const override;
    float64 sqrt_qdim(Sector a) const override;
    float64 inv_sqrt_qdim(Sector a) const override;
    FusionSymbol _b_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override;
    FusionSymbol swap_gate(Sector a, Sector b) const override;
    complex128 topological_twist(Sector a) const override;
    FusionSymbol Z_iso(Sector a) const override;
    std::string repr() const override;

    static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath);
};

} // namespace lasyten
