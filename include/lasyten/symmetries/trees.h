#pragma once

#include "../block_backend/block_backend.h"
#include "../block_backend/dtypes.h"
#include "exceptions.h"
#include "sector.h"
#include "styles.h"
#include "symmetry.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace lasyten {

/// Linear combination of fusion trees: ``sum_i coeff_i * tree_i``.
using FusionTreeLinearCombination = std::map<class FusionTree, complex128>;

/// Linear combination of tree pairs ``(Y_i, X_i)`` (as in `bend_leg`).
using FusionTreePairLinearCombination =
  std::map<std::pair<class FusionTree, class FusionTree>, complex128>;

/// A fusion tree, which represents the map from uncoupled to coupled sectors.
///
/// Consider the following example tree::
///
///     FusionTree(
///         symmetry=symmetry,
///         coupled=coupled,
///         uncoupled=[a, b, c, d],
///         are_dual=[False, True, True, False],
///         inner_sectors=[x, y],
///         multiplicities=[i, j, k],
///     )
///
/// Graphically::
///
///     |    a     b     c     d     <- isomorphic to pre_Z_uncoupled
///     |    v     ^     ^     v        e.g. dual(b) iso to pre_Z_uncoupled[1]
///     |    │     Z     Z     │
///     |    v     v     v     v
///     |    a     b     c     d     <- uncoupled
///     |    ╰──i──╯     │     │
///     |      x│        │     │
///     |       ╰───j────╯     │
///     |          y│          │
///     |           ╰────k─────╯
///     |                │
///     |                coupled
class FusionTree
{
  public:
    Symmetry::Ptr symmetry;
    SectorArray uncoupled;
    std::size_t num_uncoupled = 0;
    std::size_t num_vertices = 0;
    std::size_t num_inner_edges = 0;
    Sector coupled;
    /// Length ``num_uncoupled``; ``1`` means a Z isomorphism above that uncoupled sector.
    std::vector<std::uint8_t> are_dual;
    SectorArray inner_sectors;
    std::vector<int64> multiplicities;
    FusionStyle fusion_style{};
    bool is_abelian = false;
    BraidingStyle braiding_style{};

    /// Estimated memory footprint in bytes (the object and the arrays it owns).
    [[nodiscard]] std::size_t approx_bytes() const;

    FusionTree(Symmetry::Ptr symmetry,
               SectorArray uncoupled,
               Sector coupled,
               std::vector<std::uint8_t> are_dual,
               SectorArray inner_sectors,
               std::optional<std::vector<int64>> multiplicities = std::nullopt);

    /// The empty tree with no uncoupled sectors.
    void test_sanity() const;

    /// Assume an abelian symmetry and build the unique tree with the given `uncoupled`.
    static FusionTree from_abelian_symmetry(Symmetry::Ptr symmetry,
                                            SectorArray const& uncoupled,
                                            std::vector<std::uint8_t> const& are_dual);

    /// The empty tree with no uncoupled sectors.
    static FusionTree from_empty(Symmetry::Ptr symmetry);

    /// A tree with a single uncoupled sector and no nodes.
    static FusionTree from_sector(Symmetry::Ptr symmetry, Sector sector, bool is_dual);

    /// The uncoupled sectors *above* any Z isomorphisms.
    [[nodiscard]] SectorArray pre_Z_uncoupled() const;

    [[nodiscard]] std::size_t hash() const;
    [[nodiscard]] bool operator==(FusionTree const& other) const;
    [[nodiscard]] bool operator<(FusionTree const& other) const;

    /// Visual representation of the tree as ASCII art.
    [[nodiscard]] std::string ascii_diagram(bool dagger = false) const;

    /// Helper function for string representation (also used by ``fusion_trees``).
    static std::string str_uncoupled_coupled(Symmetry const& symmetry,
                                             SectorArray const& uncoupled,
                                             Sector coupled,
                                             std::vector<std::uint8_t> const& are_dual);

    /// Bend a leg on a tree-pair, return the resulting linear combination of tree-pairs.
    ///
    /// Graphically::
    ///
    ///     |    bend_downward=True                    bend_downward=False
    ///     |
    ///     |   │   │   │   ╭────╮                    │   │   │   │    │
    ///     |   ┢━━━┷━━━┷━━━┷━┓  │                    ┢━━━┷━━━┷━━━┷━┓  │
    ///     |   ┡━━━━━━━━━━━━━┛  │                    ┡━━━━━━━━━━━━━┛  │
    ///     |   │                │                    │                │
    ///     |   ┢━━━━━━━━━━━━━┓  │                    ┢━━━━━━━━━━━━━┓  │
    ///     |   ┡━━━┯━━━┯━━━┯━┛  │                    ┡━━━┯━━━┯━━━┯━┛  │
    ///     |   │   │   │   │    │                    │   │   │   ╰────╯
    ///
    /// @param X, Y The original tree pair, such that we modify ``hconj(X) @ Y``. Note that `X` is
    /// a fusion tree that represents the splitting tree ``hconj(X)``.
    /// @param bend_downward Whether the rightmost leg of `Y` is bent down (``bend_downward ==
    /// True``) or the rightmost leg of ``hconj(X)`` is bent up (``bend_downward == False``).
    /// @param do_conj If ``True``, return the conjugate of the coefficients instead.
    /// @returns linear_combination : dict {FusionTree: complex} The bent tree pair is a linear
    /// combination ``bent = sum_i a_i hconj(Y_i) @ X_i`` of tree pairs (where ``Y_i`` is a fusion
    /// tree and thus ``hconj(Y_i)`` a splitting tree). The returned dictionary has entries
    /// ``linear_combination[Y_i, X_i] = a_i`` for the contributions to this linear combination
    /// (i.e. tree pairs for which the coefficient vanishes are omitted).
    static FusionTreePairLinearCombination bend_leg(FusionTree const& X,
                                                    FusionTree const& Y,
                                                    bool bend_downward,
                                                    bool do_conj = false);

    /// Braid a leg on a fusion tree, return the resulting linear combination of trees.
    ///
    /// Graphically::
    ///
    ///     |   overbraid:                  underbraid
    ///     |
    ///     |   │   │   │   │               │   │   │   │
    ///     |   │    ╲ ╱    │               │    ╲ ╱    │
    ///     |   │     ╱     │               │     ╲     │
    ///     |   │    ╱ ╲    │               │    ╱ ╲    │
    ///     |   │   j  j+1  │               │   j  j+1  │
    ///     |   ┢━━━┷━━━┷━━━┷━┓             ┢━━━┷━━━┷━━━┷━┓
    ///     |   ┡━━━━━━━━━━━━━┛             ┡━━━━━━━━━━━━━┛
    ///     |   │                           │
    ///
    /// @param j The index for the braid. We braid ``uncoupled[j]`` with ``uncoupled[j + 1]``.
    /// @param overbraid If we apply an overbraid or an underbraid (see graphic above).
    /// @param cutoff We skip contributions with a prefactor below this.
    /// @param do_conj If ``True``, return the conjugate of the coefficients instead.
    /// @returns linear_combination : dict {FusionTree: complex} The braided fusion tree is a
    /// linear combination ``braided_self = sum_i a_i X_i``. The returned dictionary has entries
    /// ``linear_combination[X_i] = a_i`` for the contributions to this linear combination (i.e.
    /// trees for which the coefficient vanishes may be omitted).
    ///
    /// @warning The diagram above is for FusionTrees, which this class implements.
    // Splitting trees are stored as daggers of fusion trees.
    /// The dagger of a braid is the opposite braid, and coefficients are conjugated::
    ///
    ///     dagger(braid(S)) = opposite_braid(dagger(S))
    ///
    /// Graphically for splitting trees::
    ///
    ///     |   overbraid:                  underbraid
    ///     |
    ///     |   │                           │
    ///     |   ┢━━━┷━━━┷━━━┷━┓             ┢━━━┷━━━┷━━━┷━┓
    ///     |   ┡━━━┯━━━┯━━━┯━┛             ┡━━━┯━━━┯━━━┯━┛
    ///     |   │   │   │   │               │   │   │   │
    ///     |   │    ╲ ╱    │               │    ╲ ╱    │
    ///     |   │     ╱     │               │     ╲     │
    ///     |   │    ╱ ╲    │               │    ╱ ╲    │
    ///     |   │   j  j+1  │               │   j  j+1  │
    ///
    /// So to apply an overbraid to the splitting tree ``hconj(X)``, call
    /// ``X.braid(j, overbraid=false, do_conj=true)``.
    [[nodiscard]] FusionTreeLinearCombination braid(int64 j,
                                                    bool overbraid,
                                                    float64 cutoff = 1e-16,
                                                    bool do_conj = false) const;

    /// For the ``n``-th fusion vertex, get the respective sectors.
    ///
    /// The sectors and multiplicity label around the ``n``-th vertex of the tree::
    ///
    ///         |   (n-1 higher vertices)      │
    ///         |                      │       │
    ///         |                      a       b
    ///         |                      ╰───µ───╯
    ///         |                          c
    ///         |                          │
    ///         |                          (possibly lower vertices)
    ///
    /// @returns The uncoupled sectors ``a``, ``b``, multiplicity ``µ``, and coupled sector ``c``.
    [[nodiscard]] std::tuple<Sector, Sector, int64, Sector> vertex_labels(int64 n) const;

    /// Update the multiplicity and the three sectors around the ``n``-th vertex.
    ///
    /// @param n The vertex.
    /// @param a, b, mu, c Three sectors and a multiplicity, like the returns of `vertex_labels`.
    /// ``None`` place-holders indicate to not update that value.
    /// @param copy If ``True``, we return a modified copy. If ``False``, we modify in place and
    /// return the modified instance.
    FusionTree modify_vertex_labels(int64 n,
                                    Sector a,
                                    Sector b,
                                    int64 mu,
                                    Sector c,
                                    bool copy = true);

    [[nodiscard]] std::string str() const;
    [[nodiscard]] std::string repr() const;

    /// Get the matrix elements of the map as a backend Block.
    ///
    /// If ``backend`` is null, uses `NumpyBlockBackend` on CPU.
    /// Optional Python ``TensorBackend`` may be passed via the pybind binding instead.
    ///
    /// @param backend The backend for the resulting block. By default, we return a numpy array.
    /// @param dtype The dtype for the resulting block. By default, inferred from the symmetry
    /// @param understood_braiding For symmetries with non-trivial (but symmetric) braiding, e.g.
    /// fermions, the resulting dense block does no longer capture the braiding statistics
    /// correctly. This means that `permute_legs` is not consistently reproduced by e.g.
    /// ``numpy.transpose`` on the dense block representation. Permuting its legs would require
    /// e.g. explicit swap gates. When using the result, special care needs to be taken regarding
    /// the leg order. To avoid this pitfall, we raise an error by default. Set this flag to
    /// ``True`` to disable the error. It is then your responsibility to take care of leg orders
    /// and braids. See `swap_gate_numpy` for manipulations on these dense blocks.
    /// @returns The matrix elements with axes ``[m_a1, m_a2, ..., m_aJ, m_c]``.
    [[nodiscard]] BlockBackend::BlockPtr to_dense_block(BlockBackend* backend = nullptr,
                                                        std::optional<Dtype> dtype = std::nullopt,
                                                        bool understood_braiding = false) const;

    /// Return a shallow (or deep) copy.
    [[nodiscard]] FusionTree copy(bool deep = true) const;

    /// A new tree, from adding a new fusion node at the bottom, below the coupled sector.
    ///
    /// Graphically::
    ///
    ///     |               │
    ///     |              (Z)
    ///     |               v
    ///     |   (self)     new_uncoupled
    ///     |       │       │
    ///     |       ╰───µ───╯
    ///     |           │
    ///     |          new_coupled

    [[nodiscard]] FusionTree extended(Sector new_uncoupled,
                                      int64 mu,
                                      Sector new_coupled,
                                      bool is_dual) const;

    /// Insert a tree `t2` above the first uncoupled sector.
    ///
    /// insert_at
    ///     Inserting at general position
    /// split
    ///     Split into two separate fusion trees.
    [[nodiscard]] FusionTree insert(FusionTree const& t2) const;

    /// Insert a tree `t2` above the `n`-th uncoupled sector.
    ///
    /// The result is (in general) not a canonical tree.
    /// We transform it to canonical form via a series of F moves.
    /// This yields the result as a linear combination of canonical trees.
    /// We return a dictionary, with those trees as keys and the prefactors as values.
    ///
    /// @param n The position to insert at. `t2` is inserted above ``t1.uncoupled[n]``. We must
    /// have have ``self.are_dual[n] is False``, as we can not have a Z between trees.
    /// @param t2 The fusion tree to insert
    /// @param eps F symbols whose absolute values are smaller than this number are treated as
    /// zero.
    /// @returns coefficients : dict Trees and coefficients that form the composite map as a linear
    /// combination. Abusing notation (``FusionTree`` instances can not actually be scaled or
    /// added), this means ``map = sum(c * t for t, c in coefficient.items())``.
    ///
    /// insert
    ///     The same insertion, but restricted to ``n=0``, and returns that tree directly, no dict.
    /// split
    ///     Split into two separate fusion trees.
    [[nodiscard]] FusionTreeLinearCombination insert_at(int64 n,
                                                        FusionTree const& t2,
                                                        float64 eps = 1.0e-14) const;

    /// Outer product with another tree.
    ///
    /// Fuse with `right_tree` at the coupled sector (-> new coupled sectors are all sectors that
    /// are allowed fusion channels of the coupled sectors).
    ///
    /// @param right_tree Tree to be combined with at the coupled sector from the right.
    /// @param eps F symbols whose absolute values are smaller than this number are treated as
    /// zero.
    /// @returns linear_combination : dict {FusionTree: complex} Result expressed as linear
    /// combination of fusion trees in the canonical basis with the corresponding coefficients.
    ///
    /// insert_at
    ///     Similar insertion, but the tree is inserted above of an uncoupled sector rather than
    ///     fused with the coupled sector.
    [[nodiscard]] FusionTreeLinearCombination outer(FusionTree const& right_tree,
                                                    float64 eps = 1.0e-14) const;

    /// Split into two separate fusion trees.
    ///
    /// @param n Where to split. Must fulfill ``2 <= n < self.num_uncoupled``.
    /// @returns t1 : `FusionTree` The part that fuses the ``uncoupled_sectors[:n]`` to
    /// ``inner_sectors[n - 2]`` t2 : `FusionTree` The part that fuses ``inner_sectors[n - 2]`` and
    /// ``uncoupled_sectors[n:]`` to ``coupled``.
    ///
    /// insert
    [[nodiscard]] std::pair<FusionTree, FusionTree> split(int64 n) const;

    /// Split off the bottom vertex.
    ///
    /// Graphically::
    ///
    ///     |   a b x y z           a  b  x  y     z
    ///     |   │ │ │ │ │           │  │  │  │     │
    ///     |   (self_tree)    =    (rest_tree)    │
    ///     |       │                    │         │
    ///     |       │                    │         │
    ///     |       c                    ╰────µ────╯
    ///     |                                 │
    ///     |                                 c

    [[nodiscard]] std::tuple<FusionTree, Sector, int64, Sector> split_bottom_vertex() const;

    /// Twist some legs above a tree, return the resulting linear combination of trees.
    ///
    /// @param idcs Which uncoupled legs to twist
    /// @param overtwist The chirality of the twist. If the loop is to the right of the wires, an
    /// overtwist is such that the free end is on top. See notes below.
    /// @returns linear_combination : dict {FusionTree: complex} The composite object of tree and
    /// twist is a linear combination ``twisted_self = sum_i a_i X_i``. The returned dictionary has
    /// entries ``linear_combination[X_i] = a_i`` for the contributions to this linear combination
    /// (i.e. trees for which the coefficient vanishes may be omitted).
    ///
    /// Notes:
    ///
    /// See the following graphical examples for braid chiralities::
    ///
    ///     |   idcs = [-1]                    idcs = [-1]
    ///     |   overtwist = True               overtwist = False
    ///     |
    ///     |   │   │   │   │                  │   │   │   │
    ///     |   │   │   │   │   ╭─╮            │   │   │   │   ╭─╮
    ///     |   │   │   │    ╲ ╱  │            │   │   │    ╲ ╱  │
    ///     |   │   │   │     ╱   │            │   │   │     ╲   │
    ///     |   │   │   │    ╱ ╲  │            │   │   │    ╱ ╲  │
    ///     |   ┢━━━┷━━━┷━━━┷━┓ ╰─╯            ┢━━━┷━━━┷━━━┷━┓ ╰─╯
    ///     |   ┡━━━━━━━━━━━━━┛                ┡━━━━━━━━━━━━━┛
    ///     |   │                              │
    ///
    /// For multiple legs (``len(idcs) > 1``), we twist the together, e.g. here for
    /// ``idcs=[-2, -1]`` and ``overtwist=True``::
    ///
    ///     |   │   │   │   │   ╭──────╮
    ///     |   │   │    ╲   ╲ ╱       │
    ///     |   │   │     ╲   ╱   ╭─╮  │
    ///     |   │   │      ╲ ╱ ╲ ╱  │  │
    ///     |   │   │       ╱   ╱   │  │
    ///     |   │   │      ╱ ╲ ╱ ╲  │  │
    ///     |   │   │     ╱   ╱   ╰─╯  │
    ///     |   │   │    ╱   ╱ ╲       │
    ///     |   ┢━━━┷━━━┷━━━┷━┓ ╰──────╯
    ///     |   ┡━━━━━━━━━━━━━┛
    ///     |   │
    [[nodiscard]] FusionTreeLinearCombination twist(std::vector<int64> const& idcs,
                                                    bool overtwist) const;

  private:
    /// The `ascii_diagram` as a 2D array of single Unicode characters (cols × rows).
    /// Each cell is a UTF-8 string of one codepoint (box-drawing / sector labels).
    [[nodiscard]] std::vector<std::vector<std::string>> ascii_diagram_chars(
      bool dagger,
      int uncoupled_padding = 2,
      int inner_sector_padding = 0) const;
};

/// Iterable over all `FusionTree`\ s with given uncoupled and coupled sectors.
///
/// Efficient ``len`` and `index` avoid generating all intermediate trees.
///
/// TODO elaborate on canonical order of trees -> reference in module level docstring.
class fusion_trees
{
  public:
    Symmetry::Ptr symmetry;
    SectorArray uncoupled;
    std::size_t num_uncoupled = 0;
    Sector coupled;
    std::vector<std::uint8_t> are_dual;

    fusion_trees(Symmetry::Ptr symmetry,
                 SectorArray uncoupled,
                 Sector coupled,
                 std::optional<std::vector<std::uint8_t>> are_dual = std::nullopt);

    /// Materialize all trees (used by Python ``__iter__``).
    [[nodiscard]] std::vector<FusionTree> all_trees() const;

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::string str() const;
    [[nodiscard]] std::string repr() const;

    /// The index of a given tree in the iterator.
    [[nodiscard]] std::size_t index(FusionTree const& tree) const;

  private:
    [[nodiscard]] std::size_t compute_index(FusionTree const& tree) const;
};

} // namespace lasyten

template<>
struct std::hash<lasyten::FusionTree>
{
    std::size_t operator()(lasyten::FusionTree const& t) const noexcept { return t.hash(); }
};

/// Hash for tree-pair keys used by `lasyten::SparseMappingFusionTreePair`.
template<>
struct std::hash<std::pair<lasyten::FusionTree, lasyten::FusionTree>>
{
    std::size_t operator()(
      std::pair<lasyten::FusionTree, lasyten::FusionTree> const& p) const noexcept
    {
        // Boost-style hash_combine of the two tree hashes.
        std::size_t seed = std::hash<lasyten::FusionTree>{}(p.first);
        std::size_t h2 = std::hash<lasyten::FusionTree>{}(p.second);
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
