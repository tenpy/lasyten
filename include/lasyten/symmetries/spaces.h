#pragma once

#include "exceptions.h"
#include "fusion_symbol.h"
#include "sector.h"
#include "symmetry.h"
#include "trees.h"

#include <lasyten/backends/block_inds.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/tensors/forward_declare.h>
#include <lasyten/tensors/labels.h>

#include <array>
#include <cstdint>
#include <functional>
#include <lasyten/tools/hdf5.h>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

class TensorBackend;
class Space;
class ElementarySpace;

/// Common base of `Leg` and `Space`, providing ``shared_from_this``.
///
/// `ElementarySpace` inherits from both `Space` and `Leg`. They must
/// therefore share a single ``std::enable_shared_from_this`` base; two independent ones would
/// be ambiguous, such that ``shared_from_this()`` throws ``std::bad_weak_ptr``.
/// Not exposed to Python.
class LegOrSpace : public std::enable_shared_from_this<LegOrSpace>
{
  public:
    using Ptr = std::shared_ptr<LegOrSpace>;

    virtual ~LegOrSpace() = default;
};

/// Common base class for a single leg of a tensor.
///
/// A single leg on a tensor can either be an `ElementarySpace` or, e.g. as the result
/// of combining legs, a `LegPipe`.
class Leg : public virtual LegOrSpace
{
  public:
    using Ptr = std::shared_ptr<Leg>;
    using CPtr = std::shared_ptr<const Leg>;

    Symmetry::Ptr symmetry;
    /// Quantum dimension; integer if ``symmetry->can_be_dropped()``, otherwise may be non-integer.
    float64 dim = 0.;
    bool is_dual = false;

    Leg(Symmetry::Ptr symmetry,
        float64 dim,
        bool is_dual,
        std::optional<std::vector<int64>> basis_perm = std::nullopt);
    virtual ~Leg() = default;

    /// Perform sanity checks.
    virtual void test_sanity() const;

    /// Convert to (an appropriate subclass of) `Space` (MI hook; override `as_space_obj`).
    std::shared_ptr<Space> as_space() { return as_space_obj(); }

    /// Implement `as_space` for `Leg` subclasses.
    ///
    /// Named differently from `Space::as_Space` because `ElementarySpace` inherits both.
    virtual std::shared_ptr<Space> as_space_obj() = 0;

    /// Convert to an isomorphic `ElementarySpace`.
    virtual std::shared_ptr<ElementarySpace> as_ElementarySpace(bool is_dual = false);

    /// The dual leg (hook for MI; override `dual_leg`).
    Ptr dual() const { return dual_leg(); }

    /// Implement dual for `Leg` subclasses.
    virtual Ptr dual_leg() const = 0;

    virtual bool is_trivial() const = 0;

    [[nodiscard]] std::vector<int64> basis_perm() const;

    virtual void set_basis_perm(std::optional<std::vector<int64>> basis_perm);

    /// Inverse permutation of `basis_perm`.
    [[nodiscard]] std::vector<int64> inverse_basis_perm() const;

    virtual void set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm);

    /// Flatten until there are no more pipes.
    ///
    /// flat_spaces : Keeps `AbelianLegPipes` nested.
    virtual std::vector<Ptr> flat_legs();

    /// Flatten until we get spaces.
    ///
    /// flat_legs : Also flattens `AbelianLegPipes`.
    virtual std::vector<Ptr> flat_spaces();

    /// The number of `flat_legs`.
    virtual int64 num_flat_legs() const;

    virtual std::vector<int64> _flat_leg_permutation(int64 offset = 0) const;

    /// A single character arrow, for use in tensor diagrams
    ///
    /// Indicates (a) if the leg is a pipe and (b) for ElementarySpaces, the duality
    virtual std::string ascii_arrow() const;

    virtual bool operator==(Leg const& other) const = 0;

    /// Apply ``basis_perm`` (or its inverse) to ``arr`` (array or index / index array).
    py::object apply_basis_perm(py::object arr,
                                int64 axis = 0,
                                bool inverse = false,
                                bool pre_compose = false) const;

    [[nodiscard]] bool has_custom_basis_perm() const noexcept { return _basis_perm.has_value(); }

    /// ``shared_from_this()``, downcast to `Leg`.
    [[nodiscard]] Ptr shared_leg();

  protected:
    /// For subclasses that inherit `Leg` *virtually*.
    ///
    /// A virtual base is initialized by the constructor of the most derived class, which for
    /// pybind11 trampolines is generated from an inherited constructor and can not pass any
    /// arguments. Such subclasses therefore leave the `Leg` base default-constructed and
    /// call `init_leg` in their constructor body instead.
    Leg() = default;

    /// Set the state of the `Leg` base, see `Leg::Leg`.
    void init_leg(Symmetry::Ptr symmetry,
                  float64 dim,
                  bool is_dual,
                  std::optional<std::vector<int64>> basis_perm = std::nullopt);

    std::optional<std::vector<int64>> _basis_perm;
    std::optional<std::vector<int64>> _inverse_basis_perm;
};

using SectorMapFn = std::function<SectorArray(SectorArray const&)>;

/// Base class for symmetry spaces, see `ElementarySpace` for the standard case.
class Space : public virtual LegOrSpace
{
  public:
    using Ptr = std::shared_ptr<Space>;
    using CPtr = std::shared_ptr<const Space>;

    Symmetry::Ptr symmetry;
    SectorArray sector_decomposition;
    std::optional<std::string> sector_order;
    std::vector<int64> multiplicities;
    /// The number of unique sectors appearing in this space, regardless of their multiplicity.
    /// See :attr:`~lasyten.symmetries.spaces.Space.total_multiplicity` for the sum of the sector
    /// multiplicities.
    int64 num_sectors = 0;
    std::optional<std::vector<int64>> sector_dims;
    std::vector<float64> sector_qdims;
    std::optional<std::vector<std::array<int64, 2>>> slices;
    float64 dim = 0.;

    Space(Symmetry::Ptr symmetry,
          SectorArray sector_decomposition,
          std::optional<std::vector<int64>> multiplicities = std::nullopt,
          std::optional<std::string> sector_order = std::nullopt);
    virtual ~Space() = default;

    /// Convert to an isomorphic `ElementarySpace`.
    virtual void test_sanity() const;

    /// If the space is trivial, i.e. isomorphic to the one-dimensional trivial sector.
    ///
    /// A trivial space is one-dimensional and transforms trivially under a symmetry group.
    /// In category speak, it is (isomorphic to) the monoidal unit.
    Ptr dual() const { return dual_space(); }

    virtual Ptr dual_space() const = 0;

    virtual bool is_trivial() const;

    virtual bool operator==(Space const& other) const;

    [[nodiscard]] bool is_isomorphic_to(Space const& other) const;

    [[nodiscard]] bool is_subspace_of(Space const& other) const;

    virtual std::shared_ptr<ElementarySpace> as_ElementarySpace(bool is_dual = false);

    /// Change the symmetry by specifying how the sectors change.
    ///
    /// .. note ::
    ///     This interface assumes that a single sector of the old symmetry is mapped to a single
    ///     sector of the new symmetry, i.e. that the functor that we realize here preserves
    ///     simple objects. This does e.g. not cover the case of relaxing SU(2) to its U(1)
    ///     subgroup.
    ///
    /// @param symmetry The symmetry of the new space
    /// @param sector_map A map of sectors (2D int arrays), such that ``new_sectors =
    /// sector_map(old_sectors)``. The map is assumed to cooperate with duality, i.e. we assume
    /// without checking that ``symmetry.dual_sectors(sector_map(old_sectors))`` is the same as
    /// ``sector_map(old_symmetry.dual_sectors(old_sectors))``.
    /// @param injective If ``True``, the `sector_map` is assumed to be injective, i.e. produce a
    /// list of unique outputs, if the inputs are unique.
    /// @returns A space with the new symmetry. The order of the basis is preserved, but every
    /// basis element lives in a new sector, according to `sector_map`.
    virtual Ptr change_symmetry(Symmetry::Ptr symmetry,
                                SectorMapFn sector_map,
                                bool injective = false) = 0;

    /// Drop some or all symmetries.
    ///
    /// @param which If ``'all'`` (default) the entire symmetry is dropped and the result has
    /// ``no_symmetry``. An integer or list of integers indicates to drop the `factors` with those
    /// indices.
    virtual Ptr drop_symmetry(std::optional<std::vector<int64>> which = std::nullopt) = 0;

    /// Convert to (an appropriate subclass of) `Space`.
    Ptr as_Space();

    /// ``shared_from_this()``, downcast to `Space`.
    [[nodiscard]] Ptr shared_space();

    /// Find the index of a given sector in the `sector_decomposition`.
    ///
    /// @returns idx : int | None If the `sector` is found the `sector_decomposition`, its index
    /// there such that ``sector_decomposition[idx] == sector``. Otherwise ``None``.
    [[nodiscard]] std::optional<int64> sector_decomposition_where(Sector sector) const;

    /// The multiplicity of a given sector in the `sector_decomposition`.
    [[nodiscard]] int64 sector_multiplicity(Sector sector) const;

    /// The sum of the multiplicities of all sectors in this space.
    /// See :attr:`~lasyten.symmetries.spaces.Space.num_sectors` for the number of unique sectors.
    [[nodiscard]] int64 total_multiplicity() const;
};

/// A group of legs, i.e. resulting from `combine_legs`.
class LegPipe : public virtual Leg
{
  public:
    using Ptr = std::shared_ptr<LegPipe>;
    using CPtr = std::shared_ptr<const LegPipe>;

    std::vector<Leg::Ptr> legs;
    int64 num_legs = 0;
    bool combine_cstyle = true;

    explicit LegPipe(std::vector<Leg::Ptr> legs, bool is_dual = false, bool combine_cstyle = true);
    ~LegPipe() override = default;

    /// Perform sanity checks.
    void test_sanity() const override;

    std::shared_ptr<Space> as_space_obj() override;

    Leg::Ptr dual_leg() const override;

    bool is_trivial() const override;

    /// Flatten until there are no more pipes.
    ///
    /// flat_spaces : Keeps `AbelianLegPipes` nested.
    std::vector<Leg::Ptr> flat_legs() override;

    /// Flatten until we get spaces.
    ///
    /// flat_legs : Also flattens `AbelianLegPipes`.
    std::vector<Leg::Ptr> flat_spaces() override;

    /// The number of `flat_legs`.
    int64 num_flat_legs() const override;

    std::vector<int64> _flat_leg_permutation(int64 offset = 0) const override;

    void set_basis_perm(std::optional<std::vector<int64>> basis_perm) override;

    void set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm) override;

    std::string ascii_arrow() const override;

    bool operator==(Leg const& other) const override;

    [[nodiscard]] virtual bool is_abelian_leg_pipe() const { return false; }

    Leg::Ptr operator[](int64 idx) const;

    [[nodiscard]] std::string repr(bool show_symmetry = true, bool one_line = false) const;
};

/// A `Space` that is defined as (the dual of) a direct sum of sectors.
///
/// While every `Space` is isomorphic to a direct sum of sectors, an `ElementarySpace`
/// is by definition *equal* to such a direct sum, or to the dual of such a sum. We distinguish
/// "ket" spaces @f$ V_k := a_1 \oplus a_2 \oplus \dots \plus a_N @f$ with ``is_dual=False``
/// and "bra" spaces @f$ V_b := [b_1 \oplus b_2 \oplus \dots \plus b_N]^* @f$ with
/// ``is_dual=True``. The listed sectors, @f$ \{a_n\} @f$ for the ket space @f$ V_k @f$ and the
/// @f$ \{b_n\} @f$ for the bra space, are the `defining_sectors` of the space. For a ket
/// space, they coincide with the `sector_decomposition`, while for a bra space they are
/// mutually dual, since we have @f$ V_b \cong \bar{b}_1 \oplus \bar{b}_2 \oplus \dots \plus
/// \bar{b}_N @f$.
///
/// We impose a canonical order of sectors, such that the `defining_sectors` are sorted.
/// This in turn means that the `sector_order` is ``'sorted'`` for ket spaces and
/// ``'dual_sorted'`` for bra spaces.
///
/// If the symmetry `can_be_dropped`, there is a notion of a basis for the
/// spaces. We demand the basis to be compatible with the symmetry, i.e. each basis vector
/// needs to lie in one of the sectors of the symmetry. The *internal* basis order that results
/// from demanding that the sectors are contiguous and sorted may, however, not be the desired
/// basis order, e.g. for matrix representations.
///
/// Note that `Space::symmetry` / `Leg::symmetry` and `Space::dim` /
/// `Leg::dim` are separate members of the two bases. They are kept in sync; access them
/// through the `Space` base within this class.
class ElementarySpace
  : public Space
  , public virtual Leg
{
  public:
    using Ptr = std::shared_ptr<ElementarySpace>;
    using CPtr = std::shared_ptr<const ElementarySpace>;

    SectorArray defining_sectors;

    ElementarySpace(Symmetry::Ptr symmetry,
                    SectorArray defining_sectors,
                    std::optional<std::vector<int64>> multiplicities = std::nullopt,
                    bool is_dual = false,
                    std::optional<std::vector<int64>> basis_perm = std::nullopt);
    ~ElementarySpace() override = default;

    void test_sanity() const override;

    static Ptr from_basis(Symmetry::Ptr symmetry, SectorArray sectors_of_basis);

    static Ptr from_independent_symmetries(std::vector<Ptr> const& independent_descriptions);

    static Ptr from_largest_common_subspace(std::vector<Space::Ptr> const& spaces,
                                            bool is_dual = false);

    static Ptr from_null_space(Symmetry::Ptr symmetry, bool is_dual = false);

    static Ptr from_defining_sectors(
      Symmetry::Ptr symmetry,
      SectorArray defining_sectors,
      std::optional<std::vector<int64>> multiplicities = std::nullopt,
      bool is_dual = false,
      std::optional<std::vector<int64>> basis_perm = std::nullopt,
      bool unique_sectors = false,
      std::vector<std::size_t>* return_sorting_perm = nullptr);

    static Ptr from_sector_decomposition(
      Symmetry::Ptr symmetry,
      SectorArray sector_decomposition,
      std::optional<std::vector<int64>> multiplicities = std::nullopt,
      bool is_dual = false,
      std::optional<std::vector<int64>> basis_perm = std::nullopt,
      bool unique_sectors = false);

    static Ptr from_trivial_sector(int64 dim = 1,
                                   Symmetry::Ptr symmetry = nullptr,
                                   bool is_dual = false,
                                   std::optional<std::vector<int64>> basis_perm = std::nullopt);

    /// ``shared_from_this()``, downcast to `ElementarySpace`.
    ///
    /// Const, since the Python methods that "return self" are conceptually const.
    [[nodiscard]] Ptr shared_es() const;

    [[nodiscard]] SectorArray sectors_of_basis() const;

    [[nodiscard]] std::string repr(bool show_symmetry = true, bool one_line = false) const;

    bool operator==(Leg const& other) const override;
    bool operator==(Space const& other) const override;

    /// Common implementation of both ``operator==`` overloads.
    [[nodiscard]] bool equals_es(ElementarySpace const& other) const;

    ElementarySpace::Ptr as_ElementarySpace(bool is_dual = false) override;

    /// The ket space (``is_dual=False``) isomorphic or equal to self.
    [[nodiscard]] Ptr as_ket_space();
    [[nodiscard]] Ptr as_bra_space();

    Space::Ptr change_symmetry(Symmetry::Ptr symmetry,
                               SectorMapFn sector_map,
                               bool injective = false) override;

    /// Direct sum with `others`, always wrapped as a `DirectSumSpace` (even if `others`
    /// is empty and no `summand_labels` are given).
    [[nodiscard]] Ptr direct_sum(std::vector<Ptr> const& others,
                                 std::optional<LegLabelList> summand_labels = std::nullopt) const;

    Space::Ptr drop_symmetry(std::optional<std::vector<int64>> which = std::nullopt) override;

    Space::Ptr dual_space() const override;
    Leg::Ptr dual_leg() const override;

    [[nodiscard]] Ptr dual_es() const;

    std::pair<int64, int64> parse_index(int64 idx) const;

    Sector idx_to_sector(int64 idx) const;

    /// Take a "slice" of the leg, keeping only some of the basis states.
    ///
    /// Loses the product (pipe) structure and results in a plain `ElementarySpace`.
    ///
    /// @param blockmask For every basis state of self, in the public basis order, if it
    ///     should be kept (``True``) or discarded (``False``).
    [[nodiscard]] virtual Ptr take_slice(py::array blockmask) const;

    /// A space isomorphic to self with opposite ``is_dual`` attribute.
    ///
    /// Virtual so `AbelianLegPipe` can keep the pipe structure.
    [[nodiscard]] virtual Ptr with_opposite_duality() const;

    [[nodiscard]] Ptr with_is_dual(bool is_dual) const;

    std::shared_ptr<Space> as_space_obj() override;

    bool is_trivial() const override;

    std::string ascii_arrow() const override;

    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;

    static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath);

    /// If this is a `DirectSumSpace` (default: false).
    [[nodiscard]] virtual bool is_direct_sum_space() const { return false; }
};

/// An `ElementarySpace` that remembers an ordered direct-sum decomposition.
///
/// As an `ElementarySpace`, this is isomorphic to
/// ``⊕_i spaces[i]`` (sectors / multiplicities / optional ``basis_perm`` match
/// today's collapsing `ElementarySpace::direct_sum`). Unlike a plain
/// `ElementarySpace`, the ordered summands are retained so inclusions,
/// projections and unit vectors for individual summands can be built without
/// a dense computational basis.
///
/// Nested `DirectSumSpace` summands are flattened on construction (associativity
/// of ⊕). Operations that destroy the summand layout (e.g. `take_slice`) collapse
/// to a plain `ElementarySpace`, analogous to `AbelianLegPipe`.
class DirectSumSpace : public ElementarySpace
{
  public:
    using Ptr = std::shared_ptr<DirectSumSpace>;
    using CPtr = std::shared_ptr<const DirectSumSpace>;

    /// A summand reference: either its (possibly negative) index or its label.
    using SummandRef = std::variant<int64, std::string>;

    /// Ordered summands. Never contains nested `DirectSumSpace` instances.
    std::vector<ElementarySpace::Ptr> spaces;

    /// Optional label per summand (parallel to `spaces`; `nullopt` entries are unlabeled).
    LegLabelList summand_labels;

    explicit DirectSumSpace(std::vector<ElementarySpace::Ptr> spaces,
                            bool is_dual = false,
                            std::optional<LegLabelList> summand_labels = std::nullopt);
    ~DirectSumSpace() override = default;

    void test_sanity() const override;

    /// Build from an ordered list of summands (nested DSS flattened).
    ///
    /// If a summand is itself a `DirectSumSpace`, its own `summand_labels` are propagated
    /// to the flattened slots it expands into; in that case an explicit label for that slot
    /// in `summand_labels` must be `nullopt` (raises otherwise, since one label can not be
    /// attached to more than one flattened summand).
    [[nodiscard]] static Ptr from_spaces(
      std::vector<ElementarySpace::Ptr> spaces,
      bool is_dual = false,
      std::optional<LegLabelList> summand_labels = std::nullopt);

    /// Resolve a summand reference (index or label) to a normalized, non-negative index.
    [[nodiscard]] int64 get_summand_idx(SummandRef which) const;

    /// If a summand with the given label exists.
    [[nodiscard]] bool has_summand_label(std::string const& label) const;

    [[nodiscard]] bool is_direct_sum_space() const override { return true; }

    /// ``shared_from_this()`` downcast to `DirectSumSpace`.
    [[nodiscard]] Ptr shared_dss() const;

    /// Per-sector multiplicity cumsums of the summands into this fused space.
    ///
    /// ``result[sector_idx]`` has length ``spaces.size() + 1`` and is the cumsum
    /// (with leading zero) of each summand's multiplicity for
    /// ``sector_decomposition[sector_idx]``. Matches the slice tables used by
    /// `tensor_from_grid`.
    [[nodiscard]] std::vector<std::vector<int64>> mult_slices() const;

    /// Forget the summand structure; return a plain `ElementarySpace` with the
    /// same fused sectors / multiplicities / ``basis_perm``.
    [[nodiscard]] ElementarySpace::Ptr as_plain_ElementarySpace() const;

    std::shared_ptr<Space> as_space_obj() override;
    ElementarySpace::Ptr as_ElementarySpace(bool is_dual = false) override;

    Space::Ptr dual_space() const override;
    Leg::Ptr dual_leg() const override;
    [[nodiscard]] Ptr dual_dss() const;

    Space::Ptr change_symmetry(Symmetry::Ptr symmetry,
                               SectorMapFn sector_map,
                               bool injective = false) override;

    Space::Ptr drop_symmetry(std::optional<std::vector<int64>> which = std::nullopt) override;

    ElementarySpace::Ptr take_slice(py::array blockmask) const override;

    ElementarySpace::Ptr with_opposite_duality() const override;

    void set_basis_perm(std::optional<std::vector<int64>> basis_perm) override;
    void set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm) override;

    bool operator==(Leg const& other) const override;
    bool operator==(Space const& other) const override;

    /// Structural equality: matching duality and elementwise equal summands.
    [[nodiscard]] bool equals_dss(DirectSumSpace const& other) const;

    [[nodiscard]] std::string repr(bool show_symmetry = true, bool one_line = false) const;

    /// Projection Mask onto the union of selected summands.
    ///
    /// Kept multiplicity blocks appear in ascending summand-index order (duplicates ignored).
    /// Specify exactly one of ``keep_indices`` or ``discard_indices``; entries may be indices
    /// (negative indices count from the end) or summand labels. ``discard_indices`` keeps all
    /// summands not listed. The large leg is this space; the small leg is the plain
    /// `ElementarySpace` spanned by the kept multiplicities.
    ///
    /// For the labels, if the parent `DirectSumSpace` has a label `a`, usually it makes sense
    /// to set ``labels=[a, a*]``, or e.g. on a `wR` of an MPO, to use ``['wR', 'wL']``
    [[nodiscard]] MaskPtr projection_onto_summands(
      std::optional<std::vector<SummandRef>> keep_indices = std::nullopt,
      std::optional<std::vector<SummandRef>> discard_indices = std::nullopt,
      std::shared_ptr<TensorBackend> backend = nullptr,
      std::optional<LegLabelList> labels = std::nullopt,
      std::optional<std::string> device = std::nullopt) const;

    /// Inclusion Mask onto the union of selected summands (dagger of the projection).
    ///
    /// Specify exactly one of ``keep_indices`` or ``discard_indices``; entries may be indices
    /// (negative indices count from the end) or summand labels. ``discard_indices`` keeps all
    /// summands not listed.
    ///
    /// For the labels, if the parent `DirectSumSpace` has a label `a`, usually it makes sense
    /// to set ``labels=[a, a*]``, or e.g. on a `wR` of an MPO, to use ``['wR', 'wL']``
    [[nodiscard]] MaskPtr inclusion_of_summands(
      std::optional<std::vector<SummandRef>> keep_indices = std::nullopt,
      std::optional<std::vector<SummandRef>> discard_indices = std::nullopt,
      std::shared_ptr<TensorBackend> backend = nullptr,
      std::optional<LegLabelList> labels = std::nullopt,
      std::optional<std::string> device = std::nullopt) const;

    /// Projection Mask onto summand ``which`` (index or label).
    ///
    /// The large leg is this space; the small leg is isomorphic to the selected summand
    /// (built from the kept multiplicities). Negative indices count from the end.
    ///
    /// For the labels, if the parent `DirectSumSpace` has a label `a`, usually it makes sense
    /// to set ``labels=[a, a*]``, or e.g. on a `wR` of an MPO, to use ``['wR', 'wL']``
    [[nodiscard]] MaskPtr projection_onto_summand(
      SummandRef which,
      std::shared_ptr<TensorBackend> backend = nullptr,
      std::optional<LegLabelList> labels = std::nullopt,
      std::optional<std::string> device = std::nullopt) const;

    /// Inclusion Mask of summand ``which`` (index or label; dagger of the projection).
    ///
    /// For the labels, if the parent `DirectSumSpace` has a label `a`, usually it makes sense
    /// to set ``labels=[a, a*]``, or e.g. on a `wR` of an MPO, to use ``['wR', 'wL']``
    [[nodiscard]] MaskPtr inclusion_of_summand(
      SummandRef which,
      std::shared_ptr<TensorBackend> backend = nullptr,
      std::optional<LegLabelList> labels = std::nullopt,
      std::optional<std::string> device = std::nullopt) const;

    /// Unit vector selecting summand ``i``.
    ///
    /// Requires ``spaces[i]`` to be the one-dimensional trivial sector.
    /// Returns a rank-1 `SymmetricTensor` (codomain = this space, empty domain),
    /// obtained by converting the inclusion Mask and squeezing the trivial domain
    /// leg. Optional `labels` must have length 1 (the remaining leg).
    [[nodiscard]] SymmetricTensorPtr unit_vector_of_summand(
      int64 i,
      std::shared_ptr<TensorBackend> backend = nullptr,
      std::optional<LegLabelList> labels = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      std::optional<std::string> device = std::nullopt) const;

    // Unsupported ElementarySpace factories (raise TypeError).
    static Ptr from_basis(Symmetry::Ptr symmetry, SectorArray sectors_of_basis);
    static Ptr from_null_space(Symmetry::Ptr symmetry, bool is_dual = false);
    static Ptr from_defining_sectors(
      Symmetry::Ptr symmetry,
      SectorArray defining_sectors,
      std::optional<std::vector<int64>> multiplicities = std::nullopt,
      bool is_dual = false,
      std::optional<std::vector<int64>> basis_perm = std::nullopt,
      bool unique_sectors = false,
      std::vector<std::size_t>* return_sorting_perm = nullptr);
    static Ptr from_trivial_sector(int64 dim = 1,
                                   Symmetry::Ptr symmetry = nullptr,
                                   bool is_dual = false,
                                   std::optional<std::vector<int64>> basis_perm = std::nullopt);

    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;

    static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath);

  private:
    struct Prepared
    {
        std::vector<ElementarySpace::Ptr> spaces;
        Symmetry::Ptr symmetry;
        SectorArray defining_sectors;
        std::vector<int64> multiplicities;
        std::optional<std::vector<int64>> basis_perm;
        LegLabelList summand_labels;
    };

    static Prepared prepare(std::vector<ElementarySpace::Ptr> spaces,
                            bool is_dual,
                            std::optional<LegLabelList> summand_labels);

    DirectSumSpace(Prepared prepared, bool is_dual);

    /// Label -> index among `spaces` (built once at construction).
    std::unordered_map<std::string, int64> _summand_labelmap;
};

/// Half-open index range ``[start, stop)``, corresponding to a Python ``slice(start, stop)``.
struct IndexSlice
{
    int64 start = 0;
    int64 stop = 0;
};

/// One item of `TensorProduct::iter_uncoupled`.
struct UncoupledItem
{
    SectorArray uncoupled;
    std::vector<int64> multiplicities;
    std::optional<std::vector<IndexSlice>> slices;
};

/// One item of `TensorProduct::iter_forest_blocks`.
struct ForestBlockItem
{
    SectorArray uncoupled;
    IndexSlice slice;
    int64 coupled_idx = 0;
};

/// One item of `TensorProduct::iter_tree_blocks`.
struct TreeBlockItem
{
    FusionTree tree;
    IndexSlice slice;
    std::vector<int64> multiplicities;
    int64 coupled_idx = 0;
};

/// Represents a tensor product of `Space`\ s, e.g. the (co-)domain of a tensor.
class TensorProduct : public Space
{
  public:
    using Ptr = std::shared_ptr<TensorProduct>;
    using CPtr = std::shared_ptr<const TensorProduct>;

    /// Factors of the product: each is a single tensor leg (`ElementarySpace` or
    /// `LegPipe`).
    std::vector<Leg::Ptr> factors;
    int64 num_factors = 0;

    explicit TensorProduct(std::vector<Leg::Ptr> factors,
                           Symmetry::Ptr symmetry = nullptr,
                           std::optional<SectorArray> sector_decomposition = std::nullopt,
                           std::optional<std::vector<int64>> multiplicities = std::nullopt);
    ~TensorProduct() override = default;

    void test_sanity() const override;

    static Ptr from_partial_products(std::vector<Ptr> const& factors);

    Space::Ptr dual_space() const override;

    /// The size of a block (``coupled`` may be a sector index or a sector).
    [[nodiscard]] int64 block_size(std::variant<int64, Sector> coupled) const;

    Space::Ptr change_symmetry(Symmetry::Ptr symmetry,
                               SectorMapFn sector_map,
                               bool injective = false) override;

    Space::Ptr drop_symmetry(std::optional<std::vector<int64>> which = std::nullopt) override;

    [[nodiscard]] bool has_pipes() const;

    [[nodiscard]] std::vector<Leg::Ptr> flat_legs() const;

    [[nodiscard]] std::vector<Leg::Ptr> flat_spaces() const;

    [[nodiscard]] int64 num_flat_legs() const;

    [[nodiscard]] std::vector<std::vector<int64>> flat_legs_nesting() const;

    [[nodiscard]] std::vector<int64> flat_leg_idcs(int64 i) const;

    /// The size of a forest-block
    [[nodiscard]] int64 forest_block_size(SectorArray const& uncoupled, Sector coupled) const;

    /// The range of indices of a forest-block within its block, as a slice.
    [[nodiscard]] IndexSlice forest_block_slice(SectorArray const& uncoupled,
                                                Sector coupled) const;

    /// Insert a new space into the product at position `pos`.
    [[nodiscard]] Ptr insert_multiply(Leg::Ptr other, int64 pos) const;

    [[nodiscard]] std::vector<TreeBlockItem> iter_tree_blocks(SectorArray const& coupled) const;

    /// Iterate over forest blocks. Helper function for `FusionTreeBackend`.
    ///
    /// See `fusion_tree_backend__blocks` for definitions of blocks and forest blocks.
    ///
    /// Yields
    /// ------
    /// uncoupled : tuple of Sector
    ///     A tuple of uncoupled sectors that can fuse to a coupled sector ``coupled[i]``
    /// slc : slice
    ///     The slice of the tree-block associated with `tree` in its block.
    /// i : int
    ///     The index of the current coupled sector in `coupled`
    ///
    /// iter_tree_blocks
    /// iter_uncoupled
    [[nodiscard]] std::vector<ForestBlockItem> iter_forest_blocks(
      SectorArray const& coupled) const;

    /// Iterate over all combinations of sectors from the `flat_legs`.
    ///
    /// Yields
    /// ------
    /// uncoupled : 2D array of int
    ///     A combination of uncoupled sectors, where
    ///     ``uncoupled[i] == self.flat_legs[i].sector_decomposition[some_idx]``.
    /// multiplicities : 1D array of int
    ///     The corresponding multiplicities
    ///     ``multiplicities[i] == self.flat_legs[i].multiplicities[some_idx]``.
    /// slices : list of slice, optional
    ///     Only if ``yield_slices``, the corresponding entry of `slices`, as a slice.
    ///     I.e. ``slices[i] == slice(*self.flat_legs[i].slices[some_idx])``.
    ///
    /// Notes:
    ///
    /// For a TensorProduct of zero spaces, i.e. with ``num_factors == 0``,
    /// we *do* yield once, where the yielded arrays are empty (e.g. ``len(uncoupled) == 0``).
    [[nodiscard]] std::vector<UncoupledItem> iter_uncoupled(bool yield_slices = false) const;

    /// Add a new factor at the left / beginning of the spaces
    [[nodiscard]] Ptr left_multiply(Leg::Ptr other) const;

    /// A product of the same `factors` in a different order.
    [[nodiscard]] Ptr permuted(std::vector<int64> const& perm) const;

    /// Add a new factor at the right / end of the spaces
    [[nodiscard]] Ptr right_multiply(Leg::Ptr other) const;

    /// The size of a tree-block
    [[nodiscard]] int64 tree_block_size(SectorArray const& uncoupled) const;

    /// The range of indices of a tree-block within its block, as a slice.
    [[nodiscard]] IndexSlice tree_block_slice(FusionTree const& tree) const;

    bool operator==(Space const& other) const override;

    [[nodiscard]] Leg::Ptr operator[](int64 idx) const;

    [[nodiscard]] std::string repr(bool show_symmetry = true, bool one_line = false) const;

    [[nodiscard]] std::pair<SectorArray, std::vector<int64>> calc_sectors(
      std::vector<Leg::Ptr> const& factors) const;

    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;

    static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath);

  private:
    /// Arguments of the `Space` base, which is initialized before the constructor body.
    ///
    /// Python sets ``self.symmetry`` first and only then computes the sectors. In C++ the base
    /// must already know them, so they are computed by `prepare` and passed on to a
    /// delegated constructor.
    struct Prepared
    {
        Symmetry::Ptr symmetry;
        SectorArray sector_decomposition;
        std::vector<int64> multiplicities;
    };

    static Prepared prepare(std::vector<Leg::Ptr> const& factors,
                            Symmetry::Ptr symmetry,
                            std::optional<SectorArray> sector_decomposition,
                            std::optional<std::vector<int64>> multiplicities);

    TensorProduct(std::vector<Leg::Ptr> factors, Prepared prepared);
};

/// Special case of a `LegPipe` for abelian group symmetries.
///
/// Diamond MI: `LegPipe` + `ElementarySpace`, with a single virtual `Leg`.
class AbelianLegPipe
  : public LegPipe
  , public ElementarySpace
{
  public:
    using Ptr = std::shared_ptr<AbelianLegPipe>;
    using CPtr = std::shared_ptr<const AbelianLegPipe>;

    /// Strides for ``[leg.num_sectors for leg in legs]`` (C- or F-style).
    std::vector<int64> sector_strides;
    /// Permutation that sorts unsorted fusion outcomes (F-style combinations).
    std::vector<int64> fusion_outcomes_sort;
    /// Slice starts into the sorted fusion-outcome list; length ``num_sectors + 1``.
    std::vector<int64> block_ind_map_slices;
    /// Rows ``[b0, b1, i_0, ..., i_{n-1}, J]``; shape ``(M, 3 + num_legs)``.
    BlockInds block_ind_map;

    explicit AbelianLegPipe(std::vector<ElementarySpace::Ptr> legs,
                            bool is_dual = false,
                            bool combine_cstyle = true);
    ~AbelianLegPipe() override = default;

    /// Perform sanity checks.
    void test_sanity() const override;

    std::shared_ptr<Space> as_space_obj() override;

    ElementarySpace::Ptr as_ElementarySpace(bool is_dual = false) override;

    Space::Ptr dual_space() const override;
    Leg::Ptr dual_leg() const override;

    [[nodiscard]] Ptr dual_pipe() const;

    bool is_trivial() const override;

    std::vector<Leg::Ptr> flat_spaces() override;

    /// A filled arrow: unlike its two bases, a pipe that is also a space has its own symbol.
    ///
    /// Also resolves the ambiguity between `LegPipe::ascii_arrow` and
    /// `ElementarySpace::ascii_arrow`.
    std::string ascii_arrow() const override;

    [[nodiscard]] bool is_abelian_leg_pipe() const override { return true; }

    /// Create an AbelianLegPipe with multiple independent symmetries.
    ///
    /// @param independent_descriptions Each entry describes the resulting pipe in terms of *one*
    /// of the independent symmetries.
    static Ptr from_independent_symmetries(std::vector<Ptr> const& independent_descriptions);

    // Unsupported ElementarySpace factories (raise TypeError).
    /// Create an ElementarySpace by specifying the sector of every basis element.
    ///
    /// This requires that the symmetry `can_be_dropped`, such
    /// that there is a useful notion of a basis.
    ///
    /// .. note ::
    ///     Unlike `from_defining_sectors`, this method expects the same sector to be listed
    ///     multiple times, if the sector is multi-dimensional. The Hilbert Space of a
    ///     spin-one-half D.O.F. can e.g. be created as ``ElementarySpace.from_basis(su2,
    ///     [spin_half, spin_half])`` or as ``ElementarySpace.from_defining_sectors(su2,
    ///     [spin_half])``. In the former case we need to list the same sector both for the spin up
    ///     and spin down state.
    ///
    /// .. note ::
    ///     This classmethod always creates ket-spaces with ``is_dual=False``. This is to make
    ///     it unambiguous if `sectors_of_basis` refers to the `sector_decomposition` or the
    ///     `defining_sectors`, since they coincide for ket spaces.
    ///     Use `dual` or `as_bra_space` to create bra spaces.
    ///
    /// @param symmetry The symmetry associated with this space.
    /// @param sectors_of_basis Specifies the basis. ``sectors_of_basis[n]`` is the sector of the
    /// ``n``-th basis element. In particular, for a ``d`` dimensional sector, we expect an integer
    /// multiple of ``d`` occurrences. They need not be contiguous though. They will be grouped by
    /// order of appearance, such that they ``m``-th time a sector appears, that basis state is
    /// interpreted as the ``(m % d)``-th state of the multiplet. `sectors_of_basis`
    ///     Reproduces the `sectors_of_basis` parameter.
    /// from_defining_sectors
    ///     Similar to the constructor, but with fewer requirements.
    static Ptr from_basis(Symmetry::Ptr symmetry, SectorArray sectors_of_basis);
    /// The zero-dimensional space, i.e. the span of the empty set.
    static Ptr from_null_space(Symmetry::Ptr symmetry, bool is_dual = false);
    static Ptr from_defining_sectors(
      Symmetry::Ptr symmetry,
      SectorArray defining_sectors,
      std::optional<std::vector<int64>> multiplicities = std::nullopt,
      bool is_dual = false,
      std::optional<std::vector<int64>> basis_perm = std::nullopt,
      bool unique_sectors = false,
      std::vector<std::size_t>* return_sorting_perm = nullptr);
    /// Create an ElementarySpace that lives in the trivial sector (i.e. it is symmetric).
    ///
    /// @param dim The dimension of the space.
    /// @param symmetry The symmetry of the space. Defaults to ``no_symmetry``.
    /// @param is_dual If the space should be bra or a ket space.
    /// @param basis_perm Optional permutation of the public computational basis.
    static Ptr from_trivial_sector(int64 dim = 1,
                                   Symmetry::Ptr symmetry = nullptr,
                                   bool is_dual = false,
                                   std::optional<std::vector<int64>> basis_perm = std::nullopt);

    Space::Ptr change_symmetry(Symmetry::Ptr symmetry,
                               SectorMapFn sector_map,
                               bool injective = false) override;

    Space::Ptr drop_symmetry(std::optional<std::vector<int64>> which = std::nullopt) override;

    void set_basis_perm(std::optional<std::vector<int64>> basis_perm) override;

    void set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm) override;

    ElementarySpace::Ptr take_slice(py::array blockmask) const override;

    ElementarySpace::Ptr with_opposite_duality() const override;

    bool operator==(Leg const& other) const override;
    bool operator==(Space const& other) const override;

    /// String representation of this pipe.
    ///
    /// @param show_symmetry If the symmetry should be included in the string.
    /// @param one_line If the result should be a single line.
    [[nodiscard]] std::string repr(bool show_symmetry = true, bool one_line = false) const;

    /// The permutation of basis elements that is introduced by the fusion.
    ///
    /// ``_get_fusion_outcomes_perm`` in Python. Only depends on the `multiplicities`, which
    /// are passed explicitly since Python calls this before the `ElementarySpace` base is
    /// initialized.
    [[nodiscard]] std::vector<int64> get_fusion_outcomes_perm(
      std::vector<int64> const& multiplicities) const;

    /// The `legs`, downcast to `ElementarySpace`.
    [[nodiscard]] std::vector<ElementarySpace::Ptr> es_legs() const;

    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;

    /// Special case of a `LegPipe` for abelian group symmetries.
    ///
    /// This class essentially exists to allow specialized handling of combined legs in the
    /// `AbelianBackend`. For this backend, we want to treat combined legs, i.e. pipes, exactly
    /// the same as regular legs. This is why this class also inherits from `ElementarySpace`,
    /// which are the "uncombined" legs. Crucially, this allows the pipe to have
    /// `defining_sectors` for the `block_inds`
    /// to point to, to have a well-behaved `is_dual` attribute and to have a `basis_perm`,
    /// which can account for the basis permutation that is induced by going from sectors of the
    /// individual legs to a sorted list of coupled sectors on the pipe.
    ///
    /// Attributes:
    ///
    /// legs:
    ///     The individual legs that form this pipe, and that the pipe can be split into.
    ///     In particular, these are such that the pipe, as an `ElementarySpace`, is isomorphic
    ///     to their tensor product ``TensorProduct(legs)``, i.e. has the same
    ///     `sector_decomposition`.
    /// sector_strides : 1D numpy array of int
    ///     Strides for the shape ``[leg.num_sectors for leg in self.legs]``. Is either C-style or
    ///     F-style, depending on `combine_cstyle`. This allows one-to-one mapping between
    ///     multi-indices (one block_ind per space) to a single index.
    ///     Used in `combine_legs`.
    /// fusion_outcomes_sort : 1D numpy array of int
    ///     The permutation that sorts the list of fusion outcomes.
    ///     To calculate the `sector_decomposition` of the pipe, we go through all combinations
    ///     of sectors from the `legs` in F-style order, i.e. varying sectors from the first leg
    ///     the fastest. For each combination of sectors, we perform their fusion, which yields a
    ///     single sector in the abelian case assumed here. The resulting list of fused sectors is
    ///     in general neither sorted nor unique. This permutation (stable) sorts the resulting
    ///     list. We use F-style to match the sorting convention of `block_ind_map`.
    /// block_ind_map_slices : 1D numpy array of int
    ///     Slices for embedding the unique fused sectors in the sorted list of all fusion
    ///     outcomes. Shape is ``(K,)`` where ``K == pipe.num_sectors + 1``. Fusing all sectors
    ///     from the `sector_decomposition` of all legs and sorting the outcomes gives a list which
    ///     contains (in general) duplicates. The slice
    ///     ``block_ind_map_slices[n]:block_ind_map_slices[n + 1]`` within this sorted list
    ///     contains the same entry, namely ``pipe.sector_decomposition[n]``.
    ///     Used in @f$ AbelianBackend.split_legs @f$.
    /// block_ind_map : BlockInds
    ///     Map for the embedding of uncoupled to coupled indices, see notes of the Python class.
    ///     Shape is ``(M, N)`` where ``M`` is the number of combinations of sectors,
    ///     i.e. ``M == prod(leg.num_sectors for leg in legs)`` and ``N == 3 + len(legs)``.
    static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath);

  private:
    struct Prepared
    {
        std::vector<Leg::Ptr> legs;
        Symmetry::Ptr symmetry;
        SectorArray defining_sectors;
        std::vector<int64> multiplicities;
        std::optional<std::vector<int64>> basis_perm;
        std::vector<int64> sector_strides;
        std::vector<int64> fusion_outcomes_sort;
        std::vector<int64> block_ind_map_slices;
        BlockInds block_ind_map;
    };

    static Prepared prepare(std::vector<ElementarySpace::Ptr> const& legs,
                            bool is_dual,
                            bool combine_cstyle);

    AbelianLegPipe(Prepared prepared, bool is_dual, bool combine_cstyle);

    [[nodiscard]] static std::vector<int64> calc_basis_perm(
      std::vector<ElementarySpace::Ptr> const& legs,
      bool combine_cstyle,
      float64 dim,
      std::vector<int64> const& multiplicities,
      BlockInds const& block_ind_map);

    [[nodiscard]] static std::vector<int64> fusion_outcomes_perm(
      std::vector<ElementarySpace::Ptr> const& legs,
      bool combine_cstyle,
      float64 dim,
      std::vector<int64> const& multiplicities,
      BlockInds const& block_ind_map);
};

/// Convert a tensor leg to a `Space` (identity if it already is one).
[[nodiscard]] Space::Ptr as_space(Leg::Ptr const& leg);

/// The swap gate (numpy representation of the braid) between two legs.
///
/// Axes of the result are ``[W, V, W*, V*]``. Accepts plain `LegPipe`\ s (recursively)
/// as well as `ElementarySpace`\ s; an `AbelianLegPipe` is treated as an
/// `ElementarySpace` (matching Python ``isinstance``).
/// The swap gate (numpy representation of the braid).
///
///     |   V   W
///     |   │   │
///     |   v   v
///     |    ╲ ╱
///     |     ╲          <-  overbraid == underbraid is assumed
///     |    ╱ ╲
///     |   v   v
///     |   │   │
///     |   W   V
///
/// @returns
///
/// `swap_gate`
///     The swap gate for single sectors.
[[nodiscard]] FusionSymbol swap_gate(Leg::Ptr V, Leg::Ptr W);

/// The topological twist on a whole space, as a numpy matrix with axes ``[V, V*]``.
/// The topological twist on a whole space, as numpy representation.
///
/// @returns
///
/// `topological_twist`
///     The twist on a single sector, given in the form of a prefactor for the identity map.
[[nodiscard]] FusionSymbol twist_gate(Leg::Ptr V);

/// Diagonal of `twist_gate` (public basis order).
[[nodiscard]] FusionSymbol twist_gate_diag(Leg::Ptr V);

/// Leg permutation such that combining / splitting legs would be in C style.
[[nodiscard]] std::vector<int64> flat_leg_permutation(std::vector<Leg::Ptr> const& legs);

/// Sort sectors and merge duplicates; returns ``(sectors, multiplicities, perm)``.
[[nodiscard]] std::tuple<SectorArray, std::vector<int64>, std::vector<std::size_t>>
unique_sorted_sectors(SectorArray const& unsorted_sectors,
                      std::vector<int64> const& unsorted_multiplicities);

/// Lex-sort sectors, applying the same permutation to multiplicities.
[[nodiscard]] std::tuple<SectorArray, std::vector<int64>, std::vector<std::size_t>>
sort_sectors_public(SectorArray const& sectors, std::vector<int64> const& multiplicities);

/// Input parsing for `drop_symmetry`.
///
/// Returns ``(which_factors, remaining_symmetry)`` where ``which_factors`` is ``nullopt`` for
/// ``'all'``.
[[nodiscard]] std::pair<std::optional<std::vector<int64>>, Symmetry::Ptr>
parse_inputs_drop_symmetry_public(std::optional<std::vector<int64>> which, Symmetry::Ptr symmetry);

} // namespace lasyten
