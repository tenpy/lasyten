#include <lasyten/backends/fusion_tree_mapping.h>

#include <lasyten/block_backend/dtypes.h>
#include <lasyten/tools.h>

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <unordered_set>

namespace lasyten {

namespace {

BlockBackend::AxisSlice
axis_slice_from_index_slice(IndexSlice slc)
{
    return BlockBackend::AxisSlice{ slc.start, slc.stop, std::nullopt };
}

BlockBackend::BlockPtr
b_get(BlockBackend::BlockPtr const& b, std::initializer_list<BlockBackend::BlockIndex> key)
{
    return b->get_item(std::span<BlockBackend::BlockIndex const>(key.begin(), key.end()));
}

void
b_set(BlockBackend::BlockPtr const& b,
      std::initializer_list<BlockBackend::BlockIndex> key,
      BlockBackend::BlockPtr const& v)
{
    b->set_item(std::span<BlockBackend::BlockIndex const>(key.begin(), key.end()), *v);
}

void
collect_tree_pair_keys(TensorProduct::Ptr codomain,
                       TensorProduct::Ptr domain,
                       std::optional<BlockInds> const& block_inds,
                       std::vector<std::pair<FusionTree, FusionTree>>& keys)
{
    auto process = [&](int64 i) {
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(i)];
        SectorArray coupled_arr = SectorArray::repeat(coupled, 1);
        for (auto const& xb : codomain->iter_tree_blocks(coupled_arr)) {
            for (auto const& yb : domain->iter_tree_blocks(coupled_arr)) {
                keys.emplace_back(xb.tree, yb.tree);
            }
        }
    };

    if (!block_inds.has_value()) {
        SectorArray::iter_common_sorted(
          codomain->sector_decomposition,
          domain->sector_decomposition,
          true,
          true,
          [&](std::ptrdiff_t i, std::ptrdiff_t) { process(static_cast<int64>(i)); });
    } else {
        BlockInds const& bi = *block_inds;
        for (std::size_t row = 0; row < bi.nrows(); ++row) {
            process(bi(row, 0));
        }
    }
}

void
collect_splitting_and_fusion_trees(TensorProduct::Ptr codomain,
                                   TensorProduct::Ptr domain,
                                   std::optional<BlockInds> const& block_inds,
                                   std::vector<FusionTree>& splitting_trees,
                                   std::vector<FusionTree>& fusion_trees)
{
    auto process = [&](int64 i) {
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(i)];
        SectorArray coupled_arr = SectorArray::repeat(coupled, 1);
        for (auto const& xb : codomain->iter_tree_blocks(coupled_arr)) {
            splitting_trees.push_back(xb.tree);
        }
        for (auto const& yb : domain->iter_tree_blocks(coupled_arr)) {
            fusion_trees.push_back(yb.tree);
        }
    };

    if (!block_inds.has_value()) {
        SectorArray::iter_common_sorted(
          codomain->sector_decomposition,
          domain->sector_decomposition,
          true,
          true,
          [&](std::ptrdiff_t i, std::ptrdiff_t) { process(static_cast<int64>(i)); });
    } else {
        BlockInds const& bi = *block_inds;
        for (std::size_t row = 0; row < bi.nrows(); ++row) {
            process(bi(row, 0));
        }
    }
}

FusionTreeMappingVariant
pre_compose_tree_mapping(FusionTreeMappingVariant const& self,
                         SparseMappingFusionTree const& other)
{
    return std::visit(
      [&](auto const& m) -> FusionTreeMappingVariant { return m.pre_compose(other); }, self);
}

void
prune_tree_mapping(FusionTreeMappingVariant& m, float64 tol)
{
    std::visit([&](auto& x) { x.prune(tol); }, m);
}

std::unordered_set<FusionTree>
tree_mapping_nonzero_rows(FusionTreeMappingVariant const& m)
{
    return std::visit([](auto const& x) { return x.nonzero_rows(); }, m);
}

} // namespace

SparseMappingFusionTree::Inner
to_inner(FusionTreeLinearCombination const& lc)
{
    SparseMappingFusionTree::Inner inner;
    inner.reserve(lc.size());
    for (auto const& [t, c] : lc) {
        inner[t] = c;
    }
    return inner;
}

SparseMappingFusionTreePair::Inner
to_inner_pair(FusionTreePairLinearCombination const& lc)
{
    SparseMappingFusionTreePair::Inner inner;
    inner.reserve(lc.size());
    for (auto const& [p, c] : lc) {
        inner[p] = c;
    }
    return inner;
}

#if LASYTEN_HAS_PYTHON
std::vector<Instruction>
instructions_from_python(py::object instructions)
{
    std::vector<Instruction> out;
    if (!py::isinstance<py::list>(instructions) && !py::isinstance<py::tuple>(instructions)) {
        throw py::type_error("instructions must be a list or tuple");
    }
    for (py::handle h : instructions) {
        if (py::isinstance<BraidInstruction>(h)) {
            out.push_back(py::cast<BraidInstruction>(h));
        } else if (py::isinstance<BendInstruction>(h)) {
            out.push_back(py::cast<BendInstruction>(h));
        } else if (py::isinstance<TwistInstruction>(h)) {
            out.push_back(py::cast<TwistInstruction>(h));
        } else {
            throw py::type_error("instruction entries must be BraidInstruction, BendInstruction, "
                                 "or TwistInstruction");
        }
    }
    return out;
}
#endif // LASYTEN_HAS_PYTHON

std::unique_ptr<TensorMapping>
TensorMapping::pre_compose_instruction(Instruction const& instruction,
                                       bool instruction_is_real,
                                       std::optional<float64> prune_tol) const
{
    // --- hints from Python TensorMapping.pre_compose_instruction ---
    // this should never happen
    // ---
    std::unique_ptr<TensorMapping> res;
    std::visit(
      [&](auto const& inst) {
          using T = std::decay_t<decltype(inst)>;
          if constexpr (std::is_same_v<T, BendInstruction>) {
              res = pre_compose_bend_instruction(inst, instruction_is_real);
          } else if constexpr (std::is_same_v<T, BraidInstruction>) {
              res = pre_compose_braid_instruction(inst, instruction_is_real);
          } else if constexpr (std::is_same_v<T, TwistInstruction>) {
              res = pre_compose_twist_instruction(inst, instruction_is_real);
          }
      },
      instruction);
    if (prune_tol.has_value()) {
        res->prune(*prune_tol);
    }
    return res;
}

TreePairMapping::TreePairMapping(SparseMappingFusionTreePair mapping_, bool is_real_)
  : TensorMapping(is_real_)
  , mapping(std::move(mapping_))
{
}

std::unique_ptr<TreePairMapping>
TreePairMapping::from_identity(TensorProduct::Ptr codomain,
                               TensorProduct::Ptr domain,
                               std::optional<BlockInds> block_inds)
{
    std::vector<std::pair<FusionTree, FusionTree>> keys;
    collect_tree_pair_keys(codomain, domain, block_inds, keys);
    return std::make_unique<TreePairMapping>(SparseMappingFusionTreePair::from_identity(keys),
                                             true);
}

std::unique_ptr<TreePairMapping>
TreePairMapping::from_instructions(std::vector<Instruction> const& instructions,
                                   TensorProduct::Ptr codomain,
                                   TensorProduct::Ptr domain,
                                   std::optional<BlockInds> block_inds)
{
    auto res = from_identity(codomain, domain, block_inds);
    bool const instruction_is_real = !codomain->symmetry->has_complex_topological_data;
    std::unique_ptr<TensorMapping> mapped = std::move(res);
    for (Instruction const& inst : instructions) {
        mapped = mapped->pre_compose_instruction(inst, instruction_is_real);
    }
    return std::unique_ptr<TreePairMapping>(dynamic_cast<TreePairMapping*>(mapped.release()));
}

std::unique_ptr<TensorMapping>
TreePairMapping::pre_compose_bend_instruction(BendInstruction const& instruction,
                                              bool instruction_is_real) const
{
    // --- hints from Python TreePairMapping.pre_compose_bend_instruction ---
    // to pre-compose the bend_mapping, we only need to compute the ``bend_mapping[j][i]``
    // for those ``j`` for which an entry ``self.mapping[k][j]`` exists.
    // ---
    SparseMappingFusionTreePair bend_mapping;
    for (auto const& key : mapping.nonzero_rows()) {
        auto const& [X, Y] = key;
        bend_mapping.data[key] = to_inner_pair(FusionTree::bend_leg(X, Y, instruction.bend_down));
    }
    return std::make_unique<TreePairMapping>(
      TreePairMapping(mapping.pre_compose(bend_mapping), is_real && instruction_is_real));
}

TreePairMapping
TreePairMapping::pre_compose_fusion_tree_mapping(SparseMappingFusionTree const& tree_mapping,
                                                 bool instruction_is_real) const
{
    SparseMappingFusionTreePair res;
    for (auto const& [k, self_k] : mapping.data) {
        auto& res_k = res.data[k];
        for (auto const& [pair_j, self_jk] : self_k) {
            auto const& [X, Y_j] = pair_j;
            auto it = tree_mapping.data.find(Y_j);
            if (it == tree_mapping.data.end()) {
                continue;
            }
            for (auto const& [Y_i, other_ij] : it->second) {
                auto i = std::make_pair(X, Y_i);
                res_k[i] += other_ij * self_jk;
            }
        }
    }
    return TreePairMapping(std::move(res), is_real && instruction_is_real);
}

TreePairMapping
TreePairMapping::pre_compose_splitting_tree_mapping(SparseMappingFusionTree const& tree_mapping,
                                                    bool instruction_is_real) const
{
    SparseMappingFusionTreePair res;
    for (auto const& [k, self_k] : mapping.data) {
        auto& res_k = res.data[k];
        for (auto const& [pair_j, self_jk] : self_k) {
            auto const& [X_j, Y] = pair_j;
            auto it = tree_mapping.data.find(X_j);
            if (it == tree_mapping.data.end()) {
                continue;
            }
            for (auto const& [X_i, other_ij] : it->second) {
                auto i = std::make_pair(X_i, Y);
                res_k[i] += other_ij * self_jk;
            }
        }
    }
    return TreePairMapping(std::move(res), is_real && instruction_is_real);
}

std::unique_ptr<TensorMapping>
TreePairMapping::pre_compose_braid_instruction(BraidInstruction const& instruction,
                                               bool instruction_is_real) const
{
    // --- hints from Python TreePairMapping.pre_compose_braid_instruction ---
    // the splitting tree in the codomain is represented by a FusionTree and::
    // res_fusion_tree = dagger(res_splitting_tree)
    // = dagger(braid(splitting_tree))
    // = opposite_braid(dagger(splitting_tree))
    // = opposite_braid(fusion_tree)
    // additionally, since we represent t = dagger(t_fusion), coefficients get a conj
    // a t + b t2 = dagger(conj(a) t_fusion + conj(b) t2_fusion)
    // ---
    SparseMappingFusionTree braid_mapping;
    if (instruction.codomain) {
        std::set<FusionTree> trees;
        for (auto const& key : mapping.nonzero_rows()) {
            trees.insert(key.first);
        }
        for (FusionTree const& X : trees) {
            braid_mapping.data[X] =
              to_inner(X.braid(instruction.idx, !instruction.overbraid, 1e-16, true));
        }
        return std::make_unique<TreePairMapping>(
          pre_compose_splitting_tree_mapping(braid_mapping, instruction_is_real));
    }
    std::set<FusionTree> trees;
    for (auto const& key : mapping.nonzero_rows()) {
        trees.insert(key.second);
    }
    for (FusionTree const& Y : trees) {
        braid_mapping.data[Y] = to_inner(Y.braid(instruction.idx, instruction.overbraid));
    }
    return std::make_unique<TreePairMapping>(
      pre_compose_fusion_tree_mapping(braid_mapping, instruction_is_real));
}

std::unique_ptr<TensorMapping>
TreePairMapping::pre_compose_twist_instruction(TwistInstruction const& instruction,
                                               bool instruction_is_real) const
{
    // --- hints from Python TreePairMapping.pre_compose_twist_instruction ---
    // because this is a splitting tree, we need to do the opposite twist to its
    // fusiontree representative, giving us one conj.
    // then, we need to conj the resulting coefficient, cancelling that conj again.
    // ---
    SparseMappingFusionTree twist_mapping;
    if (instruction.codomain) {
        std::set<FusionTree> trees;
        for (auto const& key : mapping.nonzero_rows()) {
            trees.insert(key.first);
        }
        for (FusionTree const& X : trees) {
            twist_mapping.data[X] = to_inner(X.twist(instruction.idcs, instruction.overtwist));
        }
        return std::make_unique<TreePairMapping>(
          pre_compose_splitting_tree_mapping(twist_mapping, instruction_is_real));
    }
    std::set<FusionTree> trees;
    for (auto const& key : mapping.nonzero_rows()) {
        trees.insert(key.second);
    }
    for (FusionTree const& Y : trees) {
        twist_mapping.data[Y] = to_inner(Y.twist(instruction.idcs, instruction.overtwist));
    }
    return std::make_unique<TreePairMapping>(
      pre_compose_fusion_tree_mapping(twist_mapping, instruction_is_real));
}

void
TreePairMapping::prune(float64 tol)
{
    mapping.prune(tol);
}

FusionTreeData::Ptr
TreePairMapping::transform_tensor(FusionTreeData const& data,
                                  TensorProduct::Ptr codomain,
                                  TensorProduct::Ptr domain,
                                  TensorProduct::Ptr new_codomain,
                                  TensorProduct::Ptr new_domain,
                                  std::vector<int64> const& codomain_idcs,
                                  std::vector<int64> const& domain_idcs,
                                  std::shared_ptr<BlockBackend> block_backend) const
{
    // --- hints from Python TreePairMapping.transform_tensor ---
    // f(T)_{Jm} = sum_I f_{JI} T_{Im} = sum_I mapping[I][J] T_{Im}
    // note: we first add all contributions to the new tree block, and do the axes
    // permutation only once to the result
    // ie old block is not set / is zero
    // OPTIMIZE cache these?
    // from the iterator, we get mults1, mults2 in the new axis order, but wee need
    // them in the old order. OPTIMIZE can we do better than this??
    // 0   1      J-1  J   J+1      J+K-1
    // tree_block [m1, m2, ..., mJ, n1, n2, ..., nK]
    // ---
    int64 const J = codomain->num_flat_legs();
    int64 const K = domain->num_flat_legs();
    int64 const N = J + K;

    std::vector<int64> tree_block_axes_1;
    tree_block_axes_1.reserve(codomain_idcs.size());
    for (int64 i : codomain_idcs) {
        tree_block_axes_1.push_back(i < J ? i : (N - 1) + (J - i));
    }
    std::vector<int64> tree_block_axes_2;
    tree_block_axes_2.reserve(domain_idcs.size());
    for (int64 i : domain_idcs) {
        tree_block_axes_2.push_back(i < J ? i : (N - 1) + (J - i));
    }

    std::vector<int64> leg_perm;
    leg_perm.insert(leg_perm.end(), codomain_idcs.begin(), codomain_idcs.end());
    std::vector<int64> domain_idcs_rev = domain_idcs;
    std::reverse(domain_idcs_rev.begin(), domain_idcs_rev.end());
    leg_perm.insert(leg_perm.end(), domain_idcs_rev.begin(), domain_idcs_rev.end());
    auto inv_leg_perm = inverse_permutation(leg_perm);

    Dtype dtype = data.dtype;
    if (dtype::is_real(dtype) && !is_real) {
        dtype = dtype::to_complex(dtype);
    }

    std::vector<std::vector<int64>> block_inds_rows;
    std::vector<BlockBackend::BlockPtr> blocks;

    SectorArray::iter_common_sorted(
      new_codomain->sector_decomposition,
      new_domain->sector_decomposition,
      true,
      true,
      [&](std::ptrdiff_t ii, std::ptrdiff_t jj) {
          int64 i = static_cast<int64>(ii);
          int64 j = static_cast<int64>(jj);
          Sector coupled = new_codomain->sector_decomposition[static_cast<std::size_t>(i)];
          SectorArray coupled_arr = SectorArray::repeat(coupled, 1);

          auto shape = std::make_pair(new_codomain->block_size(i), new_domain->block_size(j));
          auto block = block_backend->zeros({ shape.first, shape.second }, dtype, data.device);
          bool is_zero_block = true;

          for (auto const& xb : new_codomain->iter_tree_blocks(coupled_arr)) {
              for (auto const& yb : new_domain->iter_tree_blocks(coupled_arr)) {
                  BlockBackend::BlockPtr tree_block;
                  for (auto const& [pair_I, self_I] : mapping.data) {
                      auto it = self_I.find(std::make_pair(xb.tree, yb.tree));
                      if (it == self_I.end()) {
                          continue;
                      }
                      auto which_block = data.block_ind_from_coupled(pair_I.first.coupled, domain);
                      if (!which_block.has_value()) {
                          continue;
                      }
                      auto old_block = data.blocks[static_cast<std::size_t>(*which_block)];
                      auto i1 = codomain->tree_block_slice(pair_I.first);
                      auto i2 = domain->tree_block_slice(pair_I.second);
                      auto sub = b_get(
                        old_block,
                        { axis_slice_from_index_slice(i1), axis_slice_from_index_slice(i2) });
                      auto add_block =
                        block_backend->mul(block_backend->as_scalar(it->second, dtype), sub);
                      if (!tree_block) {
                          tree_block = add_block;
                      } else {
                          tree_block = (*tree_block) + (*add_block);
                      }
                  }
                  if (!tree_block) {
                      continue;
                  }
                  is_zero_block = false;

                  std::vector<int64> leg_mults;
                  leg_mults.insert(
                    leg_mults.end(), xb.multiplicities.begin(), xb.multiplicities.end());
                  leg_mults.insert(
                    leg_mults.end(), yb.multiplicities.rbegin(), yb.multiplicities.rend());
                  std::vector<int64> old_mults;
                  old_mults.reserve(inv_leg_perm.size());
                  for (int64 idx : inv_leg_perm) {
                      old_mults.push_back(leg_mults[static_cast<std::size_t>(idx)]);
                  }

                  std::vector<int64> old_mults_cod(
                    old_mults.begin(), old_mults.begin() + static_cast<std::size_t>(J));
                  std::vector<int64> old_mults_dom(old_mults.begin() + static_cast<std::size_t>(J),
                                                   old_mults.end());
                  std::reverse(old_mults_dom.begin(), old_mults_dom.end());

                  auto permuted = block_backend->permute_combined_matrix(tree_block,
                                                                         old_mults_cod,
                                                                         tree_block_axes_1,
                                                                         old_mults_dom,
                                                                         tree_block_axes_2);
                  b_set(block,
                        { axis_slice_from_index_slice(xb.slice),
                          axis_slice_from_index_slice(yb.slice) },
                        permuted);
              }
          }
          if (is_zero_block) {
              return;
          }
          block_inds_rows.push_back({ i, j });
          blocks.push_back(block);
      });

    BlockInds block_inds =
      block_inds_rows.empty() ? BlockInds::zeros(0, 2) : BlockInds::from_rows(block_inds_rows);
    return std::make_shared<FusionTreeData>(
      std::move(block_inds), std::move(blocks), dtype, data.device, true);
}

FactorizedTreeMapping::FactorizedTreeMapping(FusionTreeMappingVariant splitting_tree_mapping_,
                                             FusionTreeMappingVariant fusion_tree_mapping_,
                                             bool is_real_)
  : TensorMapping(is_real_)
  , splitting_tree_mapping(std::move(splitting_tree_mapping_))
  , fusion_tree_mapping(std::move(fusion_tree_mapping_))
{
}

std::unique_ptr<FactorizedTreeMapping>
FactorizedTreeMapping::from_identity(TensorProduct::Ptr codomain,
                                     TensorProduct::Ptr domain,
                                     std::optional<BlockInds> block_inds)
{
    std::vector<FusionTree> splitting_trees;
    std::vector<FusionTree> fusion_trees;
    collect_splitting_and_fusion_trees(
      codomain, domain, block_inds, splitting_trees, fusion_trees);
    return std::make_unique<FactorizedTreeMapping>(
      IdentityMappingFusionTree(splitting_trees), IdentityMappingFusionTree(fusion_trees), true);
}

std::unique_ptr<FactorizedTreeMapping>
FactorizedTreeMapping::from_instructions(std::vector<Instruction> const& instructions,
                                         TensorProduct::Ptr codomain,
                                         TensorProduct::Ptr domain,
                                         std::optional<BlockInds> block_inds)
{
    auto res = from_identity(codomain, domain, block_inds);
    bool const instruction_is_real = !codomain->symmetry->has_complex_topological_data;
    std::unique_ptr<TensorMapping> mapped = std::move(res);
    for (Instruction const& inst : instructions) {
        mapped = mapped->pre_compose_instruction(inst, instruction_is_real);
    }
    return std::unique_ptr<FactorizedTreeMapping>(
      dynamic_cast<FactorizedTreeMapping*>(mapped.release()));
}

std::unique_ptr<TensorMapping>
FactorizedTreeMapping::pre_compose_bend_instruction(BendInstruction const& /*instruction*/,
                                                    bool /*instruction_is_real*/) const
{
    throw std::invalid_argument("FactorizedTreeMapping is incompatible with BendInstruction");
}

std::unique_ptr<TensorMapping>
FactorizedTreeMapping::pre_compose_braid_instruction(BraidInstruction const& instruction,
                                                     bool instruction_is_real) const
{
    // --- hints from Python FactorizedTreeMapping.pre_compose_braid_instruction ---
    // (see notes in TreePairMapping.pre_compose_braid_instruction)
    // ---
    SparseMappingFusionTree braid_mapping;
    FusionTreeMappingVariant splitting = splitting_tree_mapping;
    FusionTreeMappingVariant fusion = fusion_tree_mapping;
    if (instruction.codomain) {
        for (FusionTree const& X : tree_mapping_nonzero_rows(splitting_tree_mapping)) {
            braid_mapping.data[X] =
              to_inner(X.braid(instruction.idx, !instruction.overbraid, 1e-16, true));
        }
        splitting = pre_compose_tree_mapping(splitting_tree_mapping, braid_mapping);
    } else {
        for (FusionTree const& Y : tree_mapping_nonzero_rows(fusion_tree_mapping)) {
            braid_mapping.data[Y] = to_inner(Y.braid(instruction.idx, instruction.overbraid));
        }
        fusion = pre_compose_tree_mapping(fusion_tree_mapping, braid_mapping);
    }
    return std::make_unique<FactorizedTreeMapping>(
      std::move(splitting), std::move(fusion), is_real && instruction_is_real);
}

std::unique_ptr<TensorMapping>
FactorizedTreeMapping::pre_compose_twist_instruction(TwistInstruction const& instruction,
                                                     bool instruction_is_real) const
{
    // --- hints from Python FactorizedTreeMapping.pre_compose_twist_instruction ---
    // because this is a splitting tree, we need to do the opposite twist to its
    // fusiontree representative, giving us one conj.
    // then, we need to conj the resulting coefficient, cancelling that conj again.
    // ---
    SparseMappingFusionTree twist_mapping;
    FusionTreeMappingVariant splitting = splitting_tree_mapping;
    FusionTreeMappingVariant fusion = fusion_tree_mapping;
    if (instruction.codomain) {
        for (FusionTree const& X : tree_mapping_nonzero_rows(splitting_tree_mapping)) {
            twist_mapping.data[X] = to_inner(X.twist(instruction.idcs, instruction.overtwist));
        }
        splitting = pre_compose_tree_mapping(splitting_tree_mapping, twist_mapping);
    } else {
        for (FusionTree const& Y : tree_mapping_nonzero_rows(fusion_tree_mapping)) {
            twist_mapping.data[Y] = to_inner(Y.twist(instruction.idcs, instruction.overtwist));
        }
        fusion = pre_compose_tree_mapping(fusion_tree_mapping, twist_mapping);
    }
    return std::make_unique<FactorizedTreeMapping>(
      std::move(splitting), std::move(fusion), is_real && instruction_is_real);
}

void
FactorizedTreeMapping::prune(float64 tol)
{
    prune_tree_mapping(splitting_tree_mapping, tol);
    prune_tree_mapping(fusion_tree_mapping, tol);
}

std::pair<BlockBackend::BlockPtr, bool>
FactorizedTreeMapping::transform_splitting_trees(BlockBackend::BlockPtr const& old_block,
                                                 BlockBackend::BlockPtr const& out,
                                                 Sector coupled,
                                                 TensorProduct::Ptr codomain,
                                                 TensorProduct::Ptr new_codomain,
                                                 std::vector<int64> const& tree_block_axes_1,
                                                 std::shared_ptr<BlockBackend> block_backend) const
{
    if (std::holds_alternative<IdentityMappingFusionTree>(splitting_tree_mapping)) {
        return { old_block, false };
    }
    auto const& sparse = std::get<SparseMappingFusionTree>(splitting_tree_mapping);
    bool is_zero = true;
    SectorArray coupled_arr = SectorArray::repeat(coupled, 1);
    for (auto const& x2 : new_codomain->iter_tree_blocks(coupled_arr)) {
        BlockBackend::BlockPtr tree_row;
        for (auto const& [X, self_X] : sparse.data) {
            auto it = self_X.find(x2.tree);
            if (it == self_X.end()) {
                continue;
            }
            auto i1 = codomain->tree_block_slice(X);
            auto part = b_get(old_block,
                              { axis_slice_from_index_slice(i1), BlockBackend::AxisSlice::all() });
            auto scaled = block_backend->mul(
              block_backend->as_scalar(it->second, block_backend->get_dtype(old_block)), part);
            if (!tree_row) {
                tree_row = scaled;
            } else {
                tree_row = (*tree_row) + (*scaled);
            }
        }
        if (!tree_row) {
            continue;
        }
        is_zero = false;
        std::vector<int64> mults_old_order;
        mults_old_order.reserve(tree_block_axes_1.size());
        auto inv_axes = inverse_permutation(tree_block_axes_1);
        for (int64 idx : inv_axes) {
            mults_old_order.push_back(x2.multiplicities[static_cast<std::size_t>(idx)]);
        }
        auto permuted =
          block_backend->permute_combined_idx(tree_row, 0, mults_old_order, tree_block_axes_1);
        b_set(out,
              { axis_slice_from_index_slice(x2.slice), BlockBackend::AxisSlice::all() },
              permuted);
    }
    return { out, is_zero };
}

std::pair<BlockBackend::BlockPtr, bool>
FactorizedTreeMapping::transform_fusion_trees(BlockBackend::BlockPtr const& old_block,
                                              BlockBackend::BlockPtr const& out,
                                              Sector coupled,
                                              TensorProduct::Ptr domain,
                                              TensorProduct::Ptr new_domain,
                                              std::vector<int64> const& tree_block_axes_2,
                                              std::shared_ptr<BlockBackend> block_backend) const
{
    if (std::holds_alternative<IdentityMappingFusionTree>(fusion_tree_mapping)) {
        return { old_block, false };
    }
    auto const& sparse = std::get<SparseMappingFusionTree>(fusion_tree_mapping);
    bool is_zero_block = true;
    SectorArray coupled_arr = SectorArray::repeat(coupled, 1);
    for (auto const& y2 : new_domain->iter_tree_blocks(coupled_arr)) {
        BlockBackend::BlockPtr tree_col;
        bool is_zero_tree_col = true;
        for (auto const& [Y, self_Y] : sparse.data) {
            auto it = self_Y.find(y2.tree);
            if (it == self_Y.end()) {
                continue;
            }
            auto i2 = domain->tree_block_slice(Y);
            auto part = b_get(old_block,
                              { BlockBackend::AxisSlice::all(), axis_slice_from_index_slice(i2) });
            auto scaled = block_backend->mul(
              block_backend->as_scalar(it->second, block_backend->get_dtype(old_block)), part);
            if (is_zero_tree_col) {
                is_zero_tree_col = false;
                tree_col = scaled;
            } else {
                tree_col = (*tree_col) + (*scaled);
            }
        }
        if (is_zero_tree_col) {
            continue;
        }
        is_zero_block = false;
        std::vector<int64> mults_old_order;
        mults_old_order.reserve(tree_block_axes_2.size());
        auto inv_axes = inverse_permutation(tree_block_axes_2);
        for (int64 idx : inv_axes) {
            mults_old_order.push_back(y2.multiplicities[static_cast<std::size_t>(idx)]);
        }
        auto permuted =
          block_backend->permute_combined_idx(tree_col, 1, mults_old_order, tree_block_axes_2);
        b_set(out,
              { BlockBackend::AxisSlice::all(), axis_slice_from_index_slice(y2.slice) },
              permuted);
    }
    return { out, is_zero_block };
}

FusionTreeData::Ptr
FactorizedTreeMapping::transform_tensor(FusionTreeData const& data,
                                        TensorProduct::Ptr codomain,
                                        TensorProduct::Ptr domain,
                                        TensorProduct::Ptr new_codomain,
                                        TensorProduct::Ptr new_domain,
                                        std::vector<int64> const& codomain_idcs,
                                        std::vector<int64> const& domain_idcs,
                                        std::shared_ptr<BlockBackend> block_backend) const
{
    int64 const J = codomain->num_flat_legs();
    int64 const K = domain->num_flat_legs();
    int64 const N = J + K;

    std::vector<int64> tree_block_axes_2;
    tree_block_axes_2.reserve(domain_idcs.size());
    for (int64 i : domain_idcs) {
        tree_block_axes_2.push_back((N - 1) - i);
    }

    Dtype dtype = data.dtype;
    if (dtype::is_real(dtype) && !is_real) {
        dtype = dtype::to_complex(dtype);
    }

    std::vector<std::vector<int64>> block_inds_rows;
    std::vector<BlockBackend::BlockPtr> blocks;

    SectorArray::iter_common_sorted(
      new_codomain->sector_decomposition,
      new_domain->sector_decomposition,
      true,
      true,
      [&](std::ptrdiff_t ii, std::ptrdiff_t jj) {
          int64 i = static_cast<int64>(ii);
          int64 j = static_cast<int64>(jj);
          Sector coupled = new_codomain->sector_decomposition[static_cast<std::size_t>(i)];

          auto which_block = data.block_ind_from_coupled(coupled, domain);
          if (!which_block.has_value()) {
              return;
          }
          auto old_block = data.blocks[static_cast<std::size_t>(*which_block)];
          auto shape = std::make_pair(new_codomain->multiplicities[static_cast<std::size_t>(i)],
                                      new_domain->multiplicities[static_cast<std::size_t>(j)]);

          auto tmp_block = block_backend->zeros({ shape.first, shape.second }, dtype, data.device);
          auto [after_split, split_zero] = transform_splitting_trees(
            old_block, tmp_block, coupled, codomain, new_codomain, codomain_idcs, block_backend);
          if (split_zero) {
              return;
          }

          auto block = block_backend->zeros({ shape.first, shape.second }, dtype, data.device);
          auto [final_block, fusion_zero] = transform_fusion_trees(
            after_split, block, coupled, domain, new_domain, tree_block_axes_2, block_backend);
          if (fusion_zero) {
              return;
          }

          block_inds_rows.push_back({ i, j });
          blocks.push_back(final_block);
      });

    BlockInds block_inds =
      block_inds_rows.empty() ? BlockInds::zeros(0, 2) : BlockInds::from_rows(block_inds_rows);
    return std::make_shared<FusionTreeData>(
      std::move(block_inds), std::move(blocks), dtype, data.device, true);
}

} // namespace lasyten
