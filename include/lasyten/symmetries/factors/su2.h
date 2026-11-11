#pragma once

#include "../group.h"

#include <lasyten/tools/hdf5.h>
#include <optional>
#include <string>
#include <vector>

namespace lasyten {

/// Test-only SU(2) symmetry factor (``_SU2``).
///
/// **Not for production.** Prefer :class:`SUN` with ``N=2`` for real SU(2).
/// This class exists for testing / symbol checks only; F / fusion / Z data still
/// come from ``lasyten.symmetries._su2data``.
///
/// Allowed sectors are 1D arrays ``[jj]`` of non-negative integers ``jj = 0, 1, 2, …``
/// which label the spin ``jj/2`` irrep of SU(2).
/// E.g. a spin-1/2 degree of freedom is represented by the sector ``[1]``.
class _SU2 : public Group
{
  public:
    using Ptr = std::shared_ptr<_SU2>;
    using CPtr = std::shared_ptr<const _SU2>;

    /// Convenience sector labels (``jj`` = ``2J``).
    static Sector const spin_zero;
    static Sector const spin_half;
    static Sector const spin_one;

    explicit _SU2(std::optional<std::string> descriptive_name = std::nullopt);
    ~_SU2() override = default;

    bool is_valid_sector(Sector a) const override;
    bool are_valid_sectors(SectorArray const& sectors) const override;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;
    bool can_fuse_to(Sector a, Sector b, Sector c) const override;
    int64 sector_dim(Sector a) const override;
    std::vector<int64> batch_sector_dim(SectorArray const& a) const override;
    std::string sector_str(Sector a) const override;
    Sector get_sector(std::string const& s) const override;
    std::string repr() const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;
    Sector dual_sector(Sector a) const override;
    SectorArray dual_sectors(SectorArray const& sectors) const override;
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    int64 frobenius_schur(Sector a) const override;
    float64 qdim(Sector a) const override;
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override;
    FusionSymbol Z_iso(Sector a) const override;

    static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath);
};

} // namespace lasyten
