#pragma once

#include <lasyten/backends/block_inds.h>
#include <lasyten/backends/tensor_backend.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/trees.h>

#include <lasyten/tools/hdf5.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace lasyten {

/// Data stored in a Tensor for `FusionTreeBackend`.
///
/// Attributes:
///
/// block_inds : BlockInds
///     Indices that specify the coupled sectors of the non-zero blocks.
///     Shape ``(N, 2)``. ``block_inds[n] == [i, j]`` indicates that the coupled sector for
///     ``blocks[n]`` is given by ``tensor.codomain.sector_decomposition[i] == coupled ==
///     tensor.domain.sector_decomposition[j]``.
/// blocks : list of 2D Block
///     The nonzero blocks, ``blocks[n]`` corresponding to ``coupled_sectors[n]``.
/// dtype : Dtype
///     The dtype of the tensor (and of the `blocks`).
/// device : str
///     The device on which the blocks are currently stored.
///     We currently only support tensors which have all blocks on a single device.
///     Should be the device returned by `as_device`.
/// is_sorted : bool
///     If ``False`` (default), we permute `blocks` and `block_inds` according to
///     ``np.lexsort(block_inds.T)``.
///     If ``True``, we assume they are sorted *without* checking.
class FusionTreeData : public TensorBackend::Data
{
  public:
    using Ptr = std::shared_ptr<FusionTreeData>;
    using CPtr = std::shared_ptr<const FusionTreeData>;

    BlockInds block_inds;
    std::vector<BlockBackend::BlockPtr> blocks;
    Dtype dtype;
    std::string device;

    FusionTreeData(BlockInds block_inds,
                   std::vector<BlockBackend::BlockPtr> blocks,
                   Dtype dtype,
                   std::string device,
                   bool is_sorted = false);

    /// Return `ind` such that ``blocks[ind]`` is associated with the `coupled` sector.
    ///
    /// This is such that ``domain.sector_decomposition[block_inds[res][1]] == coupled``.
    ///
    /// Note: we use the domain (and not the codomain), since only the `block_inds[:, 1]`
    /// are sorted.
    [[nodiscard]] std::optional<int64> block_ind_from_coupled(Sector coupled,
                                                              TensorProduct::Ptr domain) const;

    /// Return `ind` such that ``block_inds[ind, 1] == domain_sector_ind``.
    ///
    /// Note: we use the domain (and not the codomain), since only the `block_inds[:, 1]`
    /// are sorted.
    [[nodiscard]] std::optional<int64> block_ind_from_domain_sector_ind(
      int64 domain_sector_ind) const;

    /// Discard blocks whose norm is below the threshold `eps`.
    void discard_zero_blocks(std::shared_ptr<BlockBackend> backend, float64 eps);

    void save_hdf5(lasyten::hdf5::Saver& saver, HighFive::Group& h5gr, std::string subpath) const;

    static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string subpath);
};

/// A backend based on fusion trees.
///
/// .. _fusion_tree_backend__blocks:
///
/// Blocks
/// ------
/// A block :math:`T_c` of a symmetric tensor is associated with a coupled sector :math:`c` and is
/// a matrix. It has the following indices
///
///     [ [ T_c ]_{a_1, ..., a_J, 𝛼}^{b_1, ..., b_K, β} ]_{m_1, ..., m_J}^{n_1, ..., n_K}
///
/// Where :math:`a_j` is an uncoupled sector ``a_j = codomain[j].sector_decomposition[i_j]`` of the
/// a space in the codomain, and ``0 <= m_j < codomain[j].multiplicities[i_j]`` is an associated
/// multiplicity index, and :math:`𝛼` labels a fusion tree ``(a_1, ..., a_J) -> c``.
/// Similarly, :math:`b_k` are uncoupled sectors ``b_k = domain[k].sector_decomposition[i_k]``, and
/// ``0 <= n_k < domain[k].multiplicities[i_k]`` is a multiplicity index and :math:`β` labels
/// a fusion tree ``(b_1, ..., b_K) -> c``.
///
/// We call ``T_c`` a *block* and  ``[ T_c ]_{a_1, ..., a_J, 𝛼}^{b_1, ..., b_K, β}`` a *tree
/// block*. We group the tree blocks with the same uncoupled sectors to a *forest block*
/// ``[ T_c ]_{a_1, ..., a_J}^{b_1, ..., b_K}``.
///
/// The blocks parametrize a tensor as::
///
///     |                                                                        W1         WK
///     |                                                                         │    │    │
///     |                                                                      n1 ▽    ▽    ▽ nK
///     |                                                                      b1 ↑    ↓    ↑ bK
///     |                                                                         │    Z    │
///     |   W1     WK                                                          b1 ↑    ↑    ↑ bK
///     |    ↑  ↓  ↑                                                             ┏┷━━━━┷━━━━┷┓
///     |    │  │  │                                                             ┃     β     ┃
///     |   ┏┷━━┷━━┷┓                              ┌ ┌     ┐b1..bK,β ┐n1..nK     ┗━━━━━┯━━━━━┛
///     |   ┃   T   ┃   =   sum     sum     sum    │ │ T_c │         │                 │ c
///     |   ┗┯━━┯━━┯┛      b1..bK  a1..aJ    c     └ └     ┘a1..aJ,𝛼 ┘m1..mJ     ┏━━━━━┷━━━━━┓
///     |    │  │  │       n1..nK  m1..mJ   𝛼 β                                  ┃     𝛼     ┃
///     |    ↓  ↓  ↑                                                             ┗┯━━━━┯━━━━┯┛
///     |   V1     VJ                                                          a1 ↑    ↑    ↑ aJ
///     |                                                                         Z    Z    │
///     |                                                                 bar(a1) ↓    ↓    ↑ aJ
///     |                                                                      m1 △    △    △ mK
///     |                                                                         │    │    │
///     |                                                                        V1         VJ
///
/// And we store the blocks as matrices, with combined multi-indices::
///
///     |   ┌ ┌     ┐b1..bK,β ┐n1..nK
///     |   │ │ T_c │         │         =   blocks[c_idx][M, N]
///     |   └ └     ┘a1..aJ,𝛼 ┘m1..mJ
///
/// where ``c = codomain.sector_decomposition[block_inds[c_idx, 0]]`` and
/// ``M = stridify(a1, ..., aJ, 𝛼, m1, ..., mJ)``, i.e. such that ``mJ`` changes the fastest when
/// ``M`` is increased, and analogously ``N = stridify(b1, ..., bK, β, n1, ..., nK)``.
/// See the following methods for the respective slices / strides of the indices ``M, N``::
///
///     - :meth:`TensorProduct.forest_block_size`
///     - :meth:`TensorProduct.forest_block_slice`
///     - :meth:`TensorProduct.forest_tree_size`
///     - :meth:`TensorProduct.forest_tree_slice`
///
/// Visually, the blocks have the following structure::
///
///     |         --------------------------> (b1...bK)
///     |          ----->β ------->β ------>β
///     |    |    ┏━━━┯━━━┳━┯━┯━┯━┯━┳━━┯━━┯━━┓
///     |    | |  ┃   │   ┃ │ │ │ │ ┃  │  │  ┃
///     |    | |  ┠───┼───╂─┼─┼─┼─┼─╂──┼──┼──┃
///     |    | |  ┃   │   ┃ │ │ │ │ ┃  │  │  ┃
///     |    | v  ┠───┼───╂─┼─┼─┼─┼─╂──┼──┼──┃
///     |    | 𝛼  ┃   │   ┃ │ │ │ │ ┃  │  │  ┃
///     |    |    ┣━━━┿━━━╋━┿━┿━┿━┿━╋━━┿━━┿━━┫
///     |    | |  ┃   │   ┃ │ │ │ │ ┃  │  │  ┃
///     |    | v  ┠───┼───╂─┼─┼─┼─┼─╂──┼──┼──┃
///     |    | 𝛼  ┃   │   ┃ │ │ │ │ ┃  │  │  ┃
///     |    V    ┗━━━┷━━━┻━┷━┷━┷━┷━┻━━┷━━┷━━┛
///     |  (a1..aJ)
///
///
/// Data is `FusionTreeData` (coupled-sector ``block_inds`` + forest blocks).
class FusionTreeBackend : public TensorBackend
{
  public:
    using Ptr = std::shared_ptr<FusionTreeBackend>;
    using CPtr = std::shared_ptr<const FusionTreeBackend>;

    /// Wrap ``FusionTreeData`` as abstract ``DataPtr``.
    static DataPtr wrap(FusionTreeData::Ptr d);

    /// Unwrap ``DataPtr`` to ``FusionTreeData``; throws if wrong type or null.
    static FusionTreeData::Ptr unwrap(DataPtr d);

    /// Read ``tensor.data`` as ``FusionTreeData``.
    static FusionTreeData::Ptr data_from_tensor(TensorCPtr tensor);

    explicit FusionTreeBackend(std::shared_ptr<BlockBackend> block_backend);
    ~FusionTreeBackend() override = default;

    bool can_decompose_tensors() const override { return true; }
    bool is_correct_data_type(DataCPtr data) const override
    {
        return dynamic_cast<FusionTreeData const*>(data.get()) != nullptr;
    }

    void test_tensor_sanity(TensorCPtr a, bool is_diagonal) override;
    void test_mask_sanity(MaskCPtr a) override;

    DataPtr act_block_diagonal_square_matrix(
      SymmetricTensorCPtr a,
      BlockUnaryFn block_method,
      std::optional<DtypeMapFn> dtype_map = std::nullopt) override;

    DataPtr add_trivial_leg(TensorCPtr a,
                            int64 legs_pos,
                            bool add_to_domain,
                            int64 co_domain_pos,
                            TensorProduct::Ptr new_codomain,
                            TensorProduct::Ptr new_domain) override;

    bool almost_equal(TensorCPtr a, TensorCPtr b, float64 rtol, float64 atol) override;

    DataPtr apply_mask_to_DiagonalTensor(DiagonalTensorCPtr tensor, MaskCPtr mask) override;

    DataPtr combine_legs(TensorCPtr tensor,
                         std::vector<std::vector<int64>> leg_idcs_combine,
                         std::vector<Leg::Ptr> pipes,
                         TensorProduct::Ptr new_codomain,
                         TensorProduct::Ptr new_domain) override;

    DataPtr compose(SymmetricTensorCPtr a, SymmetricTensorCPtr b) override;

    DataPtr copy_data(TensorCPtr a, std::optional<std::string> device = std::nullopt) override;

    DataPtr dagger(TensorCPtr a) override;

    BlockBackend::Scalar data_item(DataPtr a) override;

    bool diagonal_all(DiagonalTensorCPtr a) override;

    bool diagonal_any(DiagonalTensorCPtr a) override;

    DataPtr diagonal_elementwise_binary(DiagonalTensorCPtr a,
                                        DiagonalTensorCPtr b,
                                        BlockBinaryFn func,
                                        bool partial_zero_is_zero) override;

    DataPtr diagonal_elementwise_unary(DiagonalTensorCPtr a,
                                       BlockUnaryFn func,
                                       bool maps_zero_to_zero) override;

    DataPtr diagonal_from_block(BlockBackend::BlockPtr a,
                                TensorProduct::Ptr co_domain,
                                float64 tol) override;

    DataPtr diagonal_from_sector_block_func(SectorBlockFactoryFn func,
                                            TensorProduct::Ptr co_domain) override;

    DataPtr diagonal_tensor_from_full_tensor(SymmetricTensorCPtr a,
                                             std::optional<float64> tol = 1e-12) override;

    BlockBackend::Scalar diagonal_tensor_trace_full(DiagonalTensorCPtr a) override;

    BlockBackend::BlockPtr diagonal_tensor_to_block(DiagonalTensorCPtr a) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> diagonal_to_mask(DiagonalTensorCPtr tens) override;

    std::tuple<Space::Ptr, DataPtr> diagonal_transpose(DiagonalTensorCPtr tens) override;

    std::tuple<DataPtr, DataPtr, ElementarySpace::Ptr> eigh(
      SymmetricTensorCPtr a,
      bool new_leg_dual,
      std::optional<std::string> sort = std::nullopt) override;

    std::tuple<DataPtr, DataPtr, ElementarySpace::Ptr> eig(
      SymmetricTensorCPtr a,
      bool new_leg_dual,
      std::optional<std::string> sort = std::nullopt) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> eigvalsh(
      SymmetricTensorCPtr a,
      bool new_leg_dual,
      std::optional<std::string> sort = std::nullopt) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> eigvals(
      SymmetricTensorCPtr a,
      bool new_leg_dual,
      std::optional<std::string> sort = std::nullopt) override;

    DataPtr eye_data(TensorProduct::Ptr co_domain, Dtype dtype, std::string device) override;

    DataPtr from_dense_block(BlockBackend::BlockPtr a,
                             TensorProduct::Ptr codomain,
                             TensorProduct::Ptr domain,
                             float64 tol) override;

    DataPtr from_dense_block_trivial_sector(BlockBackend::BlockPtr block, Space::Ptr leg) override;

    DataPtr from_grid(std::vector<std::vector<py::object>> grid,
                      TensorProduct::Ptr new_codomain,
                      TensorProduct::Ptr new_domain,
                      std::vector<std::vector<int64>> left_mult_slices,
                      std::vector<std::vector<int64>> right_mult_slices,
                      Dtype dtype,
                      std::string device) override;

    DataPtr from_random_normal(TensorProduct::Ptr codomain,
                               TensorProduct::Ptr domain,
                               float64 sigma,
                               Dtype dtype,
                               std::string device) override;

    DataPtr from_sector_block_func(SectorBlockFactoryFn func,
                                   TensorProduct::Ptr codomain,
                                   TensorProduct::Ptr domain) override;

    DataPtr from_tree_pairs(
      std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      Dtype dtype,
      std::string device) override;

    DataPtr full_data_from_diagonal_tensor(DiagonalTensorCPtr a) override;

    DataPtr full_data_from_mask(MaskCPtr a, Dtype dtype) override;

    std::string get_device_from_data(DataPtr a) override;

    Dtype get_dtype_from_data(DataPtr a) override;

    BlockBackend::Scalar get_element(SymmetricTensorCPtr a, std::vector<int64> idcs) override;

    BlockBackend::Scalar get_element_diagonal(DiagonalTensorCPtr a, int64 idx) override;

    BlockBackend::Scalar get_element_mask(MaskCPtr a, std::vector<int64> idcs) override;

    BlockBackend::Scalar inner(SymmetricTensorCPtr a,
                               SymmetricTensorCPtr b,
                               bool do_dagger) override;

    DataPtr inv_part_from_dense_block_single_sector(BlockBackend::BlockPtr vector,
                                                    Space::Ptr space,
                                                    ElementarySpace::Ptr charge_leg) override;

    BlockBackend::BlockPtr inv_part_to_dense_block_single_sector(
      SymmetricTensorCPtr tensor) override;

    DataPtr linear_combination(BlockBackend::Scalar a,
                               TensorCPtr v,
                               BlockBackend::Scalar b,
                               TensorCPtr w) override;

    std::tuple<DataPtr, DataPtr> lq(SymmetricTensorCPtr a,
                                    TensorProduct::Ptr new_co_domain) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> mask_binary_operand(MaskCPtr mask1,
                                                                  MaskCPtr mask2,
                                                                  BlockBinaryFn func) override;

    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
    mask_contract_large_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx) override;

    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
    mask_contract_small_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx) override;

    DataPtr mask_dagger(MaskCPtr mask) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> mask_from_block(BlockBackend::BlockPtr a,
                                                              Space::Ptr large_leg) override;

    BlockBackend::BlockPtr mask_to_block(MaskCPtr a) override;

    DataPtr mask_to_diagonal(MaskCPtr a, Dtype dtype) override;

    std::tuple<Space::Ptr, Space::Ptr, DataPtr> mask_transpose(MaskCPtr tens) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> mask_unary_operand(MaskCPtr mask,
                                                                 BlockUnaryFn func) override;

    DataPtr move_to_device(TensorCPtr a, std::string device) override;

    DataPtr mul(BlockBackend::Scalar a, TensorCPtr b) override;

    BlockBackend::Scalar norm(TensorCPtr a) override;

    DataPtr outer(SymmetricTensorCPtr a, SymmetricTensorCPtr b) override;

    DataPtr partial_compose(SymmetricTensorCPtr a,
                            SymmetricTensorCPtr b,
                            int64 a_first_leg,
                            TensorProduct::Ptr new_codomain,
                            TensorProduct::Ptr new_domain) override;

    /// Perform an arbitrary number of traces. Pairs are converted to leg idcs.
    ///
    /// Returns ``data, codomain, domain``.
    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr> partial_trace(
      SymmetricTensorCPtr tensor,
      std::vector<std::pair<int64, int64>> pairs,
      std::vector<std::optional<int64>> levels) override;

    DataPtr permute_legs(TensorCPtr a,
                         std::vector<int64> codomain_idcs,
                         std::vector<int64> domain_idcs,
                         TensorProduct::Ptr new_codomain,
                         TensorProduct::Ptr new_domain,
                         bool mixes_codomain_domain,
                         std::vector<std::optional<int64>> levels,
                         std::vector<std::optional<bool>> bend_right) override;

    std::tuple<DataPtr, DataPtr> qr(SymmetricTensorCPtr a,
                                    TensorProduct::Ptr new_co_domain) override;

    BlockBackend::Scalar reduce_DiagonalTensor(DiagonalTensorCPtr tensor,
                                               BlockToScalarFn block_func,
                                               ScalarReduceFn func) override;

    DataPtr scale_axis(TensorCPtr a, DiagonalTensorCPtr b, int64 leg) override;

    DataPtr split_legs(TensorCPtr a,
                       std::vector<int64> leg_idcs,
                       TensorProduct::Ptr new_codomain,
                       TensorProduct::Ptr new_domain) override;

    DataPtr squeeze_legs(TensorCPtr a, std::vector<int64> idcs) override;

    bool supports_symmetry(Symmetry::Ptr symmetry) override;

    std::tuple<DataPtr, DataPtr, DataPtr> svd(SymmetricTensorCPtr a,
                                              TensorProduct::Ptr new_co_domain,
                                              std::optional<std::string> algorithm) override;

    py::object state_tensor_product(BlockBackend::BlockPtr state1,
                                    BlockBackend::BlockPtr state2,
                                    Leg::Ptr pipe) override;

    DataPtr to_block_backend(DataPtr data,
                             std::shared_ptr<BlockBackend> block_backend,
                             std::optional<Dtype> dtype = std::nullopt,
                             std::optional<std::string> device = std::nullopt) override;

    BlockBackend::BlockPtr to_dense_block(TensorCPtr a) override;

    BlockBackend::BlockPtr to_dense_block_trivial_sector(TensorCPtr tensor) override;

    DataPtr to_dtype(TensorCPtr a, Dtype dtype) override;

    BlockBackend::Scalar trace_full(SymmetricTensorCPtr a,
                                    std::vector<int64> idcs1,
                                    std::vector<int64> idcs2) override;

    std::tuple<DataPtr, ElementarySpace::Ptr, float64, float64> truncate_singular_values(
      DiagonalTensorCPtr S,
      std::optional<int64> chi_max,
      int64 chi_min,
      float64 degeneracy_tol,
      float64 trunc_cut,
      std::optional<float64> svd_min,
      bool minimize_error = true) override;

    DataPtr zero_data(TensorProduct::Ptr codomain,
                      TensorProduct::Ptr domain,
                      Dtype dtype,
                      std::string device,
                      bool all_blocks = false) override;

    DataPtr zero_diagonal_data(TensorProduct::Ptr co_domain,
                               Dtype dtype,
                               std::string device) override;

    DataPtr zero_mask_data(Space::Ptr large_leg, std::string device) override;

    /// Apply a sequence of braid/bend/twist instructions (used by `permute_legs`).
    DataPtr apply_instructions(TensorCPtr tensor,
                               py::object instructions,
                               std::vector<int64> codomain_idcs,
                               std::vector<int64> domain_idcs,
                               TensorProduct::Ptr new_codomain,
                               TensorProduct::Ptr new_domain,
                               bool mixes_codomain_domain);

  private:
    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr> _mask_contract(TensorCPtr tensor,
                                                                               MaskCPtr mask,
                                                                               int64 leg_idx,
                                                                               bool large_leg);

    /// Helper for `to_dense_block` — contribution of one forest block.
    std::tuple<BlockBackend::BlockPtr, int64, int64> _get_forest_block_contribution(
      BlockBackend::BlockPtr block,
      Symmetry::Ptr sym,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      Sector coupled,
      py::object a_sectors,
      py::object b_sectors,
      std::vector<int64> a_dims,
      std::vector<int64> b_dims,
      int64 tree_block_width,
      int64 tree_block_height,
      int64 i1_init,
      int64 i2_init,
      std::vector<int64> m_mults,
      std::vector<int64> n_mults,
      Dtype dtype) const;

    /// Helper for `from_dense_block` — accumulate one forest block into ``block``.
    /// Returns ``(num_alpha_trees, num_beta_trees)``.
    std::tuple<int64, int64> _add_forest_block_entries(BlockBackend::BlockPtr block,
                                                       BlockBackend::BlockPtr entries,
                                                       Symmetry::Ptr sym,
                                                       TensorProduct::Ptr codomain,
                                                       TensorProduct::Ptr domain,
                                                       Sector coupled,
                                                       float64 dim_c,
                                                       py::object a_sectors,
                                                       py::object b_sectors,
                                                       std::vector<int64> a_dims,
                                                       std::vector<int64> b_dims,
                                                       int64 tree_block_width,
                                                       int64 tree_block_height,
                                                       int64 i1_init,
                                                       int64 i2_init,
                                                       std::vector<int64> m_mults,
                                                       std::vector<int64> n_mults) const;
};

} // namespace lasyten
