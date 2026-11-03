#pragma once

#include <lasyten/backends/fusion_tree_backend.h>
#include <lasyten/symmetries/exceptions.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/trees.h>
#include <lasyten/tools/mappings.h>

#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

/// Instruction to braid two neighboring legs.
///
/// Notes:
///
/// Examples for over-braids::
///
///     |    │    ╲ ╱    │                      │   │   │   │
///     |    │     ╱     │                     ┏┷━━━┷━━━┷━━━┷┓
///     |    │    ╱ ╲    │                     ┃             ┃
///     |   ┏┷━━━┷━━━┷━━━┷┓                    ┗━━┯━━━┯━━━┯━━┛
///     |   ┃             ┃         OR             ╲ ╱    │
///     |   ┗━━┯━━━┯━━━┯━━┛                         ╱     │
///     |      │   │   │                           ╱ ╲    │
///
/// Examples for under-braids::
///
///     |    │    ╲ ╱    │                      │   │   │   │
///     |    │     ╲     │                     ┏┷━━━┷━━━┷━━━┷┓
///     |    │    ╱ ╲    │                     ┃             ┃
///     |   ┏┷━━━┷━━━┷━━━┷┓                    ┗━━┯━━━┯━━━┯━━┛
///     |   ┃             ┃         OR             ╲ ╱    │
///     |   ┗━━┯━━━┯━━━┯━━┛                         ╲     │
///     |      │   │   │                           ╱ ╲    │
struct BraidInstruction
{
    /// If the braid is in the codomain, otherwise in the domain.
    bool codomain = false;
    /// Which leg of the (co-)domain braids.
    /// We braid ``(co)domain[idx]`` with ``(co)domain[idx + 1]``.
    int64 idx = 0;
    /// Chirality of the braid. An overbraid is a braid where the leg that goes
    /// from bottom left to top right is on top, see notes below.
    bool overbraid = false;

    bool operator==(BraidInstruction const&) const = default;
};

/// Instruction to bend the rightmost leg of the codomain down (of the domain up).
struct BendInstruction
{
    bool bend_down = false;

    bool operator==(BendInstruction const&) const = default;
};

/// Instruction to apply a twist on one or more contiguous legs.
///
/// Notes:
///
/// Let us first illustrate how the chirality is given by `overtwist`.
/// For simplicity, we always show ``idcs=[-1]``.
/// Example for over-twists::
///
///     |    │   │   │   │   ╭─╮             │   │   │   │
///     |    │   │   │    ╲ ╱  │            ┏┷━━━┷━━━┷━━━┷┓
///     |    │   │   │     ╱   │            ┃             ┃
///     |    │   │   │    ╱ ╲  │            ┗━━┯━━━┯━━━┯━━┛╭─╮
///     |   ┏┷━━━┷━━━┷━━━┷┓  ╰─╯               │   │    ╲ ╱  │
///     |   ┃             ┃         OR         │   │     ╱   │
///     |   ┗━━┯━━━┯━━━┯━━┛                    │   │    ╱ ╲  │
///     |      │   │   │                       │   │   │   ╰─╯
///
/// Examples for under-twists::
///
///     |    │   │   │   │   ╭─╮             │   │   │   │
///     |    │   │   │    ╲ ╱  │            ┏┷━━━┷━━━┷━━━┷┓
///     |    │   │   │     ╲   │            ┃             ┃
///     |    │   │   │    ╱ ╲  │            ┗━━┯━━━┯━━━┯━━┛╭─╮
///     |   ┏┷━━━┷━━━┷━━━┷┓  ╰─╯               │   │    ╲ ╱  │
///     |   ┃             ┃         OR         │   │     ╲   │
///     |   ┗━━┯━━━┯━━━┯━━┛                    │   │    ╱ ╲  │
///     |      │   │   │                       │   │   │   ╰─╯
///
/// For multiple legs (``len(idcs) > 1``), we twist them together, e.g.::
///
///     |
///     |    │   │   │   │   ╭──────╮
///     |    │   │    ╲   ╲ ╱       │
///     |    │   │     ╲   ╱   ╭─╮  │
///     |    │   │      ╲ ╱ ╲ ╱  │  │
///     |    │   │       ╱   ╱   │  │
///     |    │   │      ╱ ╲ ╱ ╲  │  │
///     |    │   │     ╱   ╱   ╰─╯  │
///     |    │   │    ╱   ╱ ╲       │
///     |   ┏┷━━━┷━━━┷━━━┷┓  ╰──────╯
///     |   ┃             ┃
///     |   ┗━━┯━━━┯━━━┯━━┛
///     |      │   │   │
struct TwistInstruction
{
    /// If the twist is in the codomain, otherwise in the domain.
    bool codomain = false;
    /// Which legs of the (co-)domain are twisted; we twist ``(co)domain[idcs]``.
    /// Must be contiguous.
    std::vector<int64> idcs;
    /// Chirality of the twist.
    // An overtwist (undertwist) has an overbraid (underbraid) at the center, and a cup and cap.
    bool overtwist = false;

    bool operator==(TwistInstruction const&) const = default;
};

using Instruction = std::variant<BraidInstruction, BendInstruction, TwistInstruction>;

/// Parse a Python list of bound instruction objects into C++ ``Instruction`` values.
[[nodiscard]] std::vector<Instruction> instructions_from_python(py::object instructions);

/// Convert linear combinations from fusion-tree operations to sparse mapping rows.
[[nodiscard]] SparseMappingFusionTree::Inner to_inner(FusionTreeLinearCombination const& lc);

[[nodiscard]] SparseMappingFusionTreePair::Inner to_inner_pair(
  FusionTreePairLinearCombination const& lc);

using FusionTreeMappingVariant = std::variant<IdentityMappingFusionTree, SparseMappingFusionTree>;

/// Symbolic representation of a map on tensors, defined by the action on tree pairs.
class TensorMapping
{
  public:
    bool is_real;

    explicit TensorMapping(bool is_real_)
      : is_real(is_real_)
    {
    }

    virtual ~TensorMapping() = default;

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_instruction(
      Instruction const& instruction,
      bool instruction_is_real,
      std::optional<float64> prune_tol = 1e-15) const;

    [[nodiscard]] virtual std::unique_ptr<TensorMapping> pre_compose_bend_instruction(
      BendInstruction const& instruction,
      bool instruction_is_real) const = 0;

    [[nodiscard]] virtual std::unique_ptr<TensorMapping> pre_compose_braid_instruction(
      BraidInstruction const& instruction,
      bool instruction_is_real) const = 0;

    [[nodiscard]] virtual std::unique_ptr<TensorMapping> pre_compose_twist_instruction(
      TwistInstruction const& instruction,
      bool instruction_is_real) const = 0;

    /// Remove small contributions with ``abs(coefficient) < tol`` in-place.
    virtual void prune(float64 tol = 1e-15) = 0;

    /// Transform a tensor by applying the mapping to its tree-pairs. See class docstring.
    ///
    /// @param data The data of the input tensor.
    /// @param codomain, domain The (co)domain of the input tensor.
    /// @param new_codomain, new_domain The (co)domain of the output tensor.
    /// @param codomain_idcs, domain_idcs The permutations such that ``new_(co)domain[i] =
    /// old_legs[(co)domain_idcs[i]]``. This permutation acts on the uncoupled multiplicity
    /// indices.
    /// @param block_backend The block backend used to build the transformed tree-pair data.
    [[nodiscard]] virtual FusionTreeData::Ptr transform_tensor(
      FusionTreeData const& data,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      TensorProduct::Ptr new_codomain,
      TensorProduct::Ptr new_domain,
      std::vector<int64> const& codomain_idcs,
      std::vector<int64> const& domain_idcs,
      std::shared_ptr<BlockBackend> block_backend) const = 0;
};

/// A `TensorMapping`, defined at the level of tree-pairs, i.e. the general case.
///
/// We store the component ``f_{JI} = <X_J @ Y_J | f(X_I @ Y_I)>``,
/// which represents ``X_I @ Y_I \\mapsto f_{JI} X_J @ Y_J`` as ``mapping[I][J] = f_{JI}``.
/// In practice, the keys are ``I = (X_I, Y_I)`` tuples of two FusionTrees.
class TreePairMapping : public TensorMapping
{
  public:
    SparseMappingFusionTreePair mapping;

    TreePairMapping(SparseMappingFusionTreePair mapping_, bool is_real_);

    /// Estimated memory footprint in bytes.
    [[nodiscard]] std::size_t approx_bytes() const
    {
        return sizeof(*this) + mapping.approx_bytes();
    }

    /// The identity mapping.
    ///
    /// @param codomain, domain The codomain and domain that determine the possible fusion and
    /// splitting trees.
    /// @param block_inds Same format and meaning as the `block_inds`. If given, we only initialize
    /// those components ``X_I @ Y_I -> X_I @ Y_I`` where the coupled sector of the tree-pair is
    /// pointed to by a row in the `block_inds`, i.e. if we have ``coupled ==
    /// codomain.sector_decomposition[block_inds[some_idx, 0]]``.
    [[nodiscard]] static std::unique_ptr<TreePairMapping> from_identity(
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      py::object block_inds = py::none());

    [[nodiscard]] static std::unique_ptr<TreePairMapping> from_instructions(
      std::vector<Instruction> const& instructions,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      py::object block_inds = py::none());

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_bend_instruction(
      BendInstruction const& instruction,
      bool instruction_is_real) const override;

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_braid_instruction(
      BraidInstruction const& instruction,
      bool instruction_is_real) const override;

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_twist_instruction(
      TwistInstruction const& instruction,
      bool instruction_is_real) const override;

    void prune(float64 tol = 1e-15) override;

    [[nodiscard]] FusionTreeData::Ptr transform_tensor(
      FusionTreeData const& data,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      TensorProduct::Ptr new_codomain,
      TensorProduct::Ptr new_domain,
      std::vector<int64> const& codomain_idcs,
      std::vector<int64> const& domain_idcs,
      std::shared_ptr<BlockBackend> block_backend) const override;

  private:
    [[nodiscard]] TreePairMapping pre_compose_fusion_tree_mapping(
      SparseMappingFusionTree const& tree_mapping,
      bool instruction_is_real) const;

    [[nodiscard]] TreePairMapping pre_compose_splitting_tree_mapping(
      SparseMappingFusionTree const& tree_mapping,
      bool instruction_is_real) const;
};

/// A `TensorMapping` that factorizes into maps on single trees.
///
/// In particular, the action of the mapping on a tree pair factorizes as::
///
///     f(X @ Y) = g(X) @ h(Y)
///
/// and we store the component ``X \\mapsto g_{X2, X} X2`` as
/// ``g_{X2, X} = splitting_tree_mapping[X2][X] = <X2 | X>`` and similarly
/// ``h_{Y2, Y} = fusion_tree_mapping[Y2][Y] = <Y2 | Y>`` for ``Y \\mapsto h_{Y2, Y} Y2``.
/// Note that ``g`` contains the coefficients in a linear combination of splitting trees,
/// which are conjugated compared to the analogous linear combination of fusion trees.
class FactorizedTreeMapping : public TensorMapping
{
  public:
    FusionTreeMappingVariant splitting_tree_mapping;
    FusionTreeMappingVariant fusion_tree_mapping;

    /// Estimated memory footprint in bytes.
    [[nodiscard]] std::size_t approx_bytes() const
    {
        auto const bytes = [](auto const& m) { return m.approx_bytes(); };
        return sizeof(*this) + std::visit(bytes, splitting_tree_mapping) +
               std::visit(bytes, fusion_tree_mapping);
    }

    FactorizedTreeMapping(FusionTreeMappingVariant splitting_tree_mapping_,
                          FusionTreeMappingVariant fusion_tree_mapping_,
                          bool is_real_);

    [[nodiscard]] static std::unique_ptr<FactorizedTreeMapping> from_identity(
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      py::object block_inds = py::none());

    [[nodiscard]] static std::unique_ptr<FactorizedTreeMapping> from_instructions(
      std::vector<Instruction> const& instructions,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      py::object block_inds = py::none());

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_bend_instruction(
      BendInstruction const& instruction,
      bool instruction_is_real) const override;

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_braid_instruction(
      BraidInstruction const& instruction,
      bool instruction_is_real) const override;

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_twist_instruction(
      TwistInstruction const& instruction,
      bool instruction_is_real) const override;

    void prune(float64 tol = 1e-15) override;

    [[nodiscard]] FusionTreeData::Ptr transform_tensor(
      FusionTreeData const& data,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      TensorProduct::Ptr new_codomain,
      TensorProduct::Ptr new_domain,
      std::vector<int64> const& codomain_idcs,
      std::vector<int64> const& domain_idcs,
      std::shared_ptr<BlockBackend> block_backend) const override;

  private:
    [[nodiscard]] std::pair<BlockBackend::BlockPtr, bool> transform_splitting_trees(
      BlockBackend::BlockPtr const& old_block,
      BlockBackend::BlockPtr const& out,
      Sector coupled,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr new_codomain,
      std::vector<int64> const& tree_block_axes_1,
      std::shared_ptr<BlockBackend> block_backend) const;

    [[nodiscard]] std::pair<BlockBackend::BlockPtr, bool> transform_fusion_trees(
      BlockBackend::BlockPtr const& old_block,
      BlockBackend::BlockPtr const& out,
      Sector coupled,
      TensorProduct::Ptr domain,
      TensorProduct::Ptr new_domain,
      std::vector<int64> const& tree_block_axes_2,
      std::shared_ptr<BlockBackend> block_backend) const;
};

} // namespace lasyten
