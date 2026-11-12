#pragma once

#include "../block_backend/dtypes.h"
#include "symmetry_factor.h"

#include <cstdint>
#include <lasyten/tools/hdf5.h>
#include <optional>
#include <string>
#include <vector>

namespace lasyten {

/// Product symmetry: composition of one or more `SymmetryFactor`s.
///
/// Always use this product structure, even if there are no factors at all (trivial symmetry),
/// or just a single factor.
///
/// The prototypical example of a symmetry comes from the (representation of) a `Group`
/// and leads to conserved quantities. For a concrete example, we could have a `U1`
/// that represents the @f$ S^z @f$ conservation of a spin chain.
/// The framework of symmetries, however, is more general and extends to fermionic or anyonic
/// grading, see e.g. `FermionParity` or `FibonacciAnyonCategory`.
///
/// @param factors The factors that comprise this symmetry. If any are already `Symmetry`s,
///     the nesting is flattened, i.e. ``[*others, symm]`` is translated to
///     ``[*others, *symm.factors]``.
class Symmetry : public BaseSymmetry
{
  public:
    using Ptr = std::shared_ptr<Symmetry>;
    using CPtr = std::shared_ptr<const Symmetry>;

    /// Individual factors (no nesting: never contains another `Symmetry`).
    std::vector<SymmetryFactor::Ptr> factors;
    /// Cumulative factor offsets into a product sector; length ``num_factors + 1``.
    /// Slice ``sector_slices[i]:sector_slices[i+1]`` is the component for ``factors[i]``.
    std::vector<std::uint8_t> sector_slices;
    /// Dtype of fusion tensors, or nullopt if any factor lacks fusion tensors.
    std::optional<Dtype> fusion_tensor_dtype;

    /// Build from factors; nested `Symmetry` instances are flattened.
    explicit Symmetry(std::vector<SymmetryFactor::Ptr> factors);

    ~Symmetry() override = default;

    std::size_t num_factors() const { return factors.size(); }

    /// Return the index of the first factor with that name. Raises if not found.
    ///
    /// Index of the first factor with ``descriptive_name == name``; throws if missing.
    std::size_t factor_where(std::string const& descriptive_name) const;

    /// Whether ``other`` is among the factors (instance or type check via Python binding).
    bool has_factor(SymmetryFactor const& other) const;

    /// If two symmetries are equivalent.
    ///
    /// Equivalence ignores the `descriptive_name` of the factors.
    /// Ordering of the `factors` is also ignored, unless ``strict_ordering=True``.
    bool is_equivalent_to(Symmetry const& other, bool strict_ordering = false) const;

    /// Identity: already a product `Symmetry`.
    Ptr as_Symmetry() override;

    /// Check if `a` is a valid sector.
    ///
    /// For a `Symmetry`, the valid sectors are 1D integer arrays, which are "stacks" of
    /// valid sectors for each of the `factors`, see `sector_slices`.
    bool is_valid_sector(Sector a) const override;
    bool are_valid_sectors(SectorArray const& sectors) const override;
    /// Returns all outcomes for the fusion of sectors
    ///
    /// Each sector appears only once, regardless of its multiplicity (given by n_symbol) in the
    /// fusion
    SectorArray fusion_outcomes(Sector a, Sector b) const override;
    /// Allows optimized fusion in the case of FusionStyle.single.
    ///
    /// For two SectorArrays, return the element-wise fusion outcome of each pair of Sectors,
    /// which is a single unique Sector, as a new SectorArray.
    /// Subclasses may override this with more efficient implementations.
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override;
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override;

    /// The sector dual to a, such that N^{a,dual(a)}_u = 1.
    ///
    /// Note that the dual space @f$ a^\star @f$ to a sector @f$ a @f$ may not itself be one of
    /// the sectors, but it must be isomorphic to one of the sectors. This method returns that
    /// representative @f$ \bar{a} @f$ of the equivalence class.
    Sector dual_sector(Sector a) const override;
    /// dual_sector for multiple sectors
    ///
    /// subclasses my override this.
    SectorArray dual_sectors(SectorArray const& sectors) const override;
    /// Optimized version of self.n_symbol that assumes that c is a valid fusion outcome.
    ///
    /// If it is not, the results may be nonsensical. We do this for optimization purposes
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;
    /// Assume there are finitely many sectors, return all of them.
    ///
    /// @warning Do not perform inplace operations on the output. That may invalidate caches.
    SectorArray all_sectors() const override;

    /// The dimension of a sector, as an unstructured space (i.e. if we drop the symmetry).
    ///
    /// For bosonic braiding style, e.g. for group symmetries, this coincides with the quantum
    /// dimension computed by `qdim`.
    /// For other braiding styles,
    ///
    /// `swap_gate`
    ///     Similar method for braiding general spaces, not just single sectors.
    int64 sector_dim(Sector a) const override;
    /// sector_dim of every sector (row) in a
    std::vector<int64> batch_sector_dim(SectorArray const& a) const override;
    /// Quantum dimension of every sector (row) in `a`
    std::vector<float64> batch_qdim(SectorArray const& a) const override;
    /// The quantum dimension ``Tr(id_a)`` of a sector
    float64 qdim(Sector a) const override;
    /// Short and readable string for the sector. Is used in __str__ of symmetry-related objects.
    std::string sector_str(Sector a) const override;
    /// Inverse of `sector_str`.
    ///
    /// For a single-factor product, the outer product brackets may be omitted (e.g. ``"odd"``
    /// instead of ``"[odd]"``). Multi-factor symmetries require the full ``"[..., ...]"`` form.
    Sector get_sector(std::string const& s) const override;

    /// Internal implementation of `f_symbol`. Can assume that inputs are valid.
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    /// Internal implementation of `r_symbol`. Can assume that inputs are valid.
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override;
    /// Internal implementation of `fusion_tensor`. Can assume that inputs are valid.
    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override;
    /// The swap gate (numpy representation of the braid) of single sectors.
    ///
    ///     |   a   b
    ///     |   │   │
    ///     |   v   v
    ///     |    ╲ ╱
    ///     |     ╲          <-  overbraid == underbraid is assumed
    ///     |    ╱ ╲
    ///     |   v   v
    ///     |   │   │
    ///     |   b   a
    ///
    /// @returns A numpy representation of the above tensor with axes ``[b, a, b*, a*]``.
    FusionSymbol swap_gate(Sector a, Sector b) const override;
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

    std::string repr() const;
    std::string str() const;
    bool equals(Symmetry const& other) const;
    /// Product with another factor or product → new `Symmetry`.
    Ptr mul(SymmetryFactor::Ptr other) const;
    Ptr mul(Symmetry const& other) const;

    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;

    static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath);

    /// Slice of product sector belonging to factor ``i``.
    Sector factor_sector(Sector const& a, std::size_t i) const;
    SectorArray factor_sectors(SectorArray const& a, std::size_t i) const;

  private:
    static FusionStyle max_fusion_style(std::vector<SymmetryFactor::Ptr> const& factors);
    static BraidingStyle max_braiding_style(std::vector<SymmetryFactor::Ptr> const& factors);
    static Sector concat_trivial_sectors(std::vector<SymmetryFactor::Ptr> const& factors);
    static float64 prod_num_sectors(std::vector<SymmetryFactor::Ptr> const& factors);
};

} // namespace lasyten
