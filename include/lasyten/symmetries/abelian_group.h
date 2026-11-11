#pragma once

#include "../block_backend/dtypes.h"
#include "group.h"

#include <optional>
#include <string>

namespace lasyten {

/// Base-class for abelian symmetry groups.
class AbelianGroup : public Group
{
  public:
    using Ptr = std::shared_ptr<AbelianGroup>;
    using CPtr = std::shared_ptr<const AbelianGroup>;

    AbelianGroup(Sector trivial_sector,
                 std::string group_name,
                 float64 num_sectors,
                 std::optional<std::string> descriptive_name = std::nullopt,
                 bool trivial_shift = true);
    ~AbelianGroup() override = default;

    /// sector_dim of every sector (row) in a
    std::string sector_str(Sector a) const override;
    /// Inverse of `sector_str` for abelian groups labelled by a single integer.
    Sector get_sector(std::string const& s) const override;
    int64 sector_dim(Sector a) const override;
    std::vector<int64> batch_sector_dim(SectorArray const& a) const override;
    /// Optimized version of self.n_symbol that assumes that c is a valid fusion outcome.
    ///
    /// If it is not, the results may be nonsensical. We do this for optimization purposes
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;
    /// Internal implementation of `f_symbol`. Can assume that inputs are valid.
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    /// The Frobenius Schur indicator of a sector.
    int64 frobenius_schur(Sector a) const override;
    /// The quantum dimension ``Tr(id_a)`` of a sector
    float64 qdim(Sector a) const override;
    /// The square root of the quantum dimension.
    float64 sqrt_qdim(Sector a) const override;
    /// The inverse square root of the quantum dimension.
    float64 inv_sqrt_qdim(Sector a) const override;
    /// Internal implementation of `b_symbol`. Can assume that inputs are valid.
    FusionSymbol _b_symbol(Sector a, Sector b, Sector c) const override;
    /// Internal implementation of `r_symbol`. Can assume that inputs are valid.
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override;
    /// Internal implementation of `c_symbol`. Can assume that inputs are valid.
    FusionSymbol _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    /// Internal implementation of `fusion_tensor`. Can assume that inputs are valid.
    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override;
    /// The Z isomorphism @f$ Z_{\bar{a}} : \bar{a}^* \to a @f$.
    ///
    /// The dual @f$ a^* @f$ of a sector @f$ a @f$ is another irreducible space.
    /// However, it may not be itself a sector. It must be isomorphic to one of the sector
    /// representatives though, which we call @f$ \bar{a} @f$.
    /// The Z isomorphism @f$ Z_a : a^* \to \bar{a} @f$ is that isomorphism.
    ///
    /// We return the matrix elements
    ///
    /// \f[
    ///     (Z_{\bar{a}})_{mn} = \langle m \vert Z_{\bar{a}}(\langle n \vert)
    /// \f]
    ///
    /// where @f$ m @f$ goes over a (dual) basis of @f$ \bar{a} @f$ and @f$ n @f$ over a basis of
    /// @f$ a @f$.
    ///
    /// @param a Note that this is the target sector of the map, not its subscript!
    /// @returns The matrix elements as a [d_a, d_a] numpy array.
    FusionSymbol Z_iso(Sector a) const override;
};

} // namespace lasyten
