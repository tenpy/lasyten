#pragma once

#include "../lasyten.h"
#include "exceptions.h"
#include "fusion_symbol.h"
#include "sector.h"
#include "styles.h"

#include <memory>
#include <string>
#include <vector>

namespace lasyten {

class Symmetry; // product symmetry; defined in symmetry.h

/// Common method implementations for both `SymmetryFactor` and `Symmetry`.
///
/// This contains the fallback implementations of e.g. `qdim` in terms of F symbols.
class BaseSymmetry : public std::enable_shared_from_this<BaseSymmetry>
{
  public:
    using Ptr = std::shared_ptr<BaseSymmetry>;
    using CPtr = std::shared_ptr<const BaseSymmetry>;
    using SymmetryPtr = std::shared_ptr<Symmetry>;

    FusionStyle fusion_style;
    BraidingStyle braiding_style;
    Sector trivial_sector;
    /// Number of sectors; ``+inf`` if infinite (matches Python ``int | float``).
    float64 num_sectors;
    std::uint8_t sector_ind_len = 0;
    SectorArray empty_sector_array;
    bool has_complex_topological_data = false;
    bool trivial_shift = true;

    BaseSymmetry(FusionStyle fusion_style,
                 BraidingStyle braiding_style,
                 Sector trivial_sector,
                 float64 num_sectors,
                 bool has_complex_topological_data,
                 bool trivial_shift);
    virtual ~BaseSymmetry() = default;

    // --- properties (Python @property) ---

    /// If the symmetry supports converting tensors to/from numpy.
    virtual bool can_be_dropped() const;
    virtual bool has_symmetric_braid() const;
    virtual bool has_trivial_braid() const;
    /// If the symmetry is Abelian.
    ///
    /// An Abelian symmetry is characterized by `FusionStyle.single`, which implies that all
    /// sectors are one-dimensional.
    /// Note that this does *not* imply that it is a group, as the braiding may not be bosonic!
    virtual bool is_abelian() const;
    /// If N symbols are only 0 or 1.
    virtual bool has_unique_fusion() const;

    // --- abstract / must override ---

    /// The sector dual to `a`, such that @f$ N^{a,\mathrm{dual}(a)}_u = 1 @f$.
    ///
    /// Note that the dual space @f$ a^\star @f$ to a sector @f$ a @f$ may not itself be one of
    /// the sectors, but it must be isomorphic to one of the sectors. This method returns that
    /// representative @f$ \bar{a} @f$ of the equivalence class.
    virtual Sector dual_sector(Sector a) const = 0;
    /// Optimized version of `n_symbol` that assumes that `c` is a valid fusion outcome.
    ///
    /// If it is not, the results may be nonsensical. We do this for optimization purposes.
    virtual int64 _n_symbol(Sector a, Sector b, Sector c) const = 0;
    /// Internal implementation of `f_symbol`. Can assume that inputs are valid.
    virtual FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const = 0;
    /// Internal implementation of `r_symbol`. Can assume that inputs are valid.
    virtual FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const = 0;
    /// Wrap as a product `Symmetry` (identity if already a product).
    virtual SymmetryPtr as_Symmetry() = 0;
    /// Whether `a` is a valid sector of this symmetry
    virtual bool is_valid_sector(Sector a) const = 0;
    /// Returns all outcomes for the fusion of sectors
    ///
    /// Each sector appears only once, regardless of its multiplicity (given by n_symbol) in the
    /// fusion
    virtual SectorArray fusion_outcomes(Sector a, Sector b) const = 0;

    // --- defaults (may override) ---

    /// Internal implementation of `fusion_tensor`. Can assume that inputs are valid.
    ///
    /// Internal fusion tensor; inputs assumed valid.
    virtual FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const;
    /// The swap gate (numpy representation of the braid) of single sectors.
    virtual FusionSymbol swap_gate(Sector a, Sector b) const;
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
    virtual FusionSymbol Z_iso(Sector a) const;
    /// Assume there are finitely many sectors, return all of them.
    ///
    /// @warning Do not perform inplace operations on the output. That may invalidate caches.
    virtual SectorArray all_sectors() const;

    /// The N-symbol N^{ab}_c, i.e. how often c appears in the fusion of a and b.
    int64 n_symbol(Sector a, Sector b, Sector c) const;
    /// Coefficients @f$ [F^{abc}_d]^e_f @f$ related to recoupling of fusion.
    ///
    /// The F symbol relates the following two maps::
    ///
    ///     m1 := [a ⊗ b ⊗ c] --(1 ⊗ X_μ)--> [a ⊗ e] --(X_ν)--> d
    ///     m2 := [a ⊗ b ⊗ c] --(X_κ ⊗ 1)--> [f ⊗ c] --(X_λ)--> d
    ///
    /// Such that @f$ m_1 = \sum_{f\kappa\lambda} [F^{abc}_d]^{e\mu\nu}_{f\kappa\lambda} m_2 @f$.
    ///
    /// The F symbol is unitary as a matrix from indices @f$ (f\kappa\lambda) @f$
    /// to @f$ (e\mu\nu) @f$.
    ///
    /// @warning Do not perform inplace operations on the output. That may invalidate caches.
    ///
    /// @param a, b, c, d, e, f Sectors. Must be compatible with the fusion described above.
    /// @returns F : 4D array The F symbol as an array of the multiplicity indices [μ,ν,κ,λ]
    FusionSymbol f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const;
    /// Coefficients @f$ B^{ab}_c @f$ related to bending the right leg on a fusion tensor.
    ///
    /// The B symbol relates the following two maps::
    ///
    ///     m1 := a --(1 ⊗ η_b)--> [a ⊗ b ⊗ b^*] --(X_μ ⊗ 1)--> [c ⊗ b^*]
    ///     m2 := a --(Y_ν)--> [c ⊗ \\bar{b}] --(1 ⊗ Z_b^†)--> [c ⊗ b^*]
    ///
    /// such that @f$ m_1 = \sum_{\nu} [B^{ab}_c]^\mu_\nu m_2 @f$.
    ///
    /// The related A-symbol for bending left legs is not needed, since we always
    /// work with fusion trees in form
    ///
    /// @warning Do not perform inplace operations on the output. That may invalidate caches.
    ///
    /// @param a, b, c Sectors. Must be compatible with the fusion described above.
    /// @returns B : 2D array The B symbol as an array of the multiplicity indices [μ,ν]
    FusionSymbol b_symbol(Sector a, Sector b, Sector c) const;
    /// Coefficients @f$ R^{ab}_c @f$ related to braiding on a single fusion tensor.
    ///
    /// The R symbol relates the following two maps::
    ///
    ///     m1 := [b ⊗ a] --τ--> [a ⊗ b] --X_μ--> c
    ///     m2 := [b ⊗ a] --X_ν--> c
    ///
    /// such that @f$ m_1 = \sum_{\nu} [R^{ab}_c]^\mu_\nu m_2 @f$.
    ///
    /// We can use the unitary gauge freedom of the fusion tensors
    ///
    /// \f[
    ///     X_μ \mapsto \sum_ν U_{μ,ν} X_ν
    /// \f]
    ///
    /// to enforce that the R symbol is diagonal.
    ///
    /// @warning Do not perform inplace operations on the output. That may invalidate caches.
    ///
    /// @param a, b, c Sectors. Must be compatible with the fusion described above.
    /// @returns R : 1D array The diagonal entries of the R symbol as an array of the multiplicity
    /// index [μ].
    FusionSymbol r_symbol(Sector a, Sector b, Sector c) const;
    /// Coefficients @f$ [C^{abc}_d]^e_f @f$ related to braiding on a pair of fusion tensors.
    ///
    /// The C symbol relates the following two maps::
    ///
    ///     m1 := [a ⊗ c ⊗ b] --(1 ⊗ τ)--> [a ⊗ b ⊗ c] --(X_μ ⊗ 1)--> [e ⊗ c] --X_ν--> d
    ///     m2 := [a ⊗ c ⊗ b] --(X_κ ⊗ 1)--> [f ⊗ b] --X_λ--> d
    ///
    /// such that @f$ m_1 = \sum_{f\kappa\lambda} C^{e\mu\nu}_{f\kappa\lambda} m_2 @f$.
    ///
    /// @warning Do not perform inplace operations on the output. That may invalidate caches.
    ///
    /// @param a, b, c, d, e, f Sectors. Must be compatible with the fusion described above.
    /// @returns C : 4D array The C symbol as an array of the multiplicity indices [μ,ν,κ,λ]
    FusionSymbol c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const;
    /// Matrix elements of the fusion tensor @f$ X^{ab}_{c,\mu} @f$ for all @f$ \mu @f$.
    ///
    /// May not be well defined for anyons.
    ///
    /// @warning Do not perform inplace operations on the output. That may invalidate caches.
    ///
    /// @param a, b, c Sectors. Must be compatible with the fusion described above.
    /// @param Z_a If we should include a Z isomorphism @f$ Z_{\bar{a}} : \bar{a}^* -> a @f$ below
    /// the sector a. If so, the composite is a map from @f$ \bar{a}^* \otimes b \to c @f$.
    /// @param Z_b Analogously to `Z_a`.
    /// @returns X : 4D ndarray Axis [μ, m_a, m_b, m_c] where μ is the multiplicity index of the
    /// fusion tensor and m_a goes over a basis for sector a, etc.
    FusionSymbol fusion_tensor(Sector a,
                               Sector b,
                               Sector c,
                               bool Z_a = false,
                               bool Z_b = false) const;

    virtual bool are_valid_sectors(SectorArray const& sectors) const;
    /// Element-wise fusion for FusionStyle.single.
    ///
    /// Allows optimized fusion in the case of FusionStyle.single.
    ///
    /// For two SectorArrays, return the element-wise fusion outcome of each pair of Sectors,
    /// which is a single unique Sector, as a new SectorArray.
    /// Subclasses may override this with more efficient implementations.
    virtual SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                                  SectorArray const& b) const;
    Sector multiple_fusion(std::vector<Sector> const& sectors) const;
    /// Allows optimized fusion in the case of FusionStyle.single.
    ///
    /// It generalizes `fusion_outcomes_broadcast` to more than two fusion inputs.
    SectorArray multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const;
    /// Internal version of `multiple_fusion_broadcast`. May assume ``len(sectors) >= 2``.
    ///
    /// Internal; may assume ``sectors.size() >= 2``.
    virtual SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const;
    /// Whether c is a valid fusion outcome, i.e. if it appears in ``self.fusion_outcomes(a, b)``
    virtual bool can_fuse_to(Sector a, Sector b, Sector c) const;
    /// The dimension of a sector, as an unstructured space (i.e. if we drop the symmetry).
    ///
    /// Dimension as unstructured space (if symmetry can be dropped).
    ///
    /// For bosonic braiding style, e.g. for group symmetries, this coincides with the quantum
    /// dimension computed by `qdim`.
    /// For other braiding styles,
    virtual int64 sector_dim(Sector a) const;
    /// sector_dim of every sector (row) in a
    virtual std::vector<int64> batch_sector_dim(SectorArray const& a) const;
    /// Quantum dimension of every sector (row) in `a`
    virtual std::vector<float64> batch_qdim(SectorArray const& a) const;
    /// Short and readable string for the sector. Is used in __str__ of symmetry-related objects.
    virtual std::string sector_str(Sector a) const;
    /// Inverse of `sector_str`. Throws if `s` does not name a valid sector.
    ///
    /// This resolves display names such as ``"odd"`` / ``"vacuum"``, not site state labels
    /// like ``"up"`` / ``"empty"``.
    virtual Sector get_sector(std::string const& s) const;
    /// dual_sector for multiple sectors
    ///
    /// subclasses my override this.
    virtual SectorArray dual_sectors(SectorArray const& sectors) const;
    /// The Frobenius Schur indicator of a sector.
    virtual int64 frobenius_schur(Sector a) const;
    /// The quantum dimension ``Tr(id_a)`` of a sector
    virtual float64 qdim(Sector a) const;
    /// The square root of the quantum dimension.
    virtual float64 sqrt_qdim(Sector a) const;
    /// The inverse square root of the quantum dimension.
    virtual float64 inv_sqrt_qdim(Sector a) const;
    /// Total quantum dimension, @f$ D = \sqrt{\sum_a d_a^2} @f$.
    float64 total_qdim() const;
    /// Internal implementation of `b_symbol`. Can assume that inputs are valid.
    virtual FusionSymbol _b_symbol(Sector a, Sector b, Sector c) const;
    /// Internal implementation of `c_symbol`. Can assume that inputs are valid.
    virtual FusionSymbol _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const;
    /// The prefactor that relates the twist on a single sector to the identity.
    ///
    /// Graphically::
    ///
    ///     |   │   ╭─╮                |
    ///     |    ╲ ╱  │                |
    ///     |     ╱   │   =   theta_a  |
    ///     |    ╱ ╲  │                |
    ///     |   │   ╰─╯                |
    ///     |   a                      a
    ///
    /// Notes:
    ///
    /// For a twist with opposite chirality, the prefactor is conjugated.
    ///
    ///     |   │   ╭─╮                      |
    ///     |    ╲ ╱  │                      |
    ///     |     ╲   │   =   conj(theta_a)  |
    ///     |    ╱ ╲  │                      |
    ///     |   │   ╰─╯                      |
    ///     |   a                            a
    virtual complex128 topological_twist(Sector a) const;
    /// Single matrix-element of the S-matrix.
    ///
    /// s_matrix
    complex128 s_matrix_element(Sector a, Sector b) const;
    /// The modular S-matrix. Only defined for modular tensor categories.
    ///
    /// s_matrix_element
    FusionSymbol s_matrix() const;
};

} // namespace lasyten
