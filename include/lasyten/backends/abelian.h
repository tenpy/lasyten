#pragma once

#include <lasyten/backends/block_inds.h>
#include <lasyten/backends/tensor_backend.h>
#include <lasyten/block_backend/block_backend.h>
#include <lasyten/block_backend/dtypes.h>
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

/// Data stored in a Tensor for `AbelianBackend`.
///
/// The `block_inds` can be visualized as follows::
///
///     |           ---- codomain ---->  <--- domain ----
///     |
///     |      |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |    b |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |    l |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |    o |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |    c |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |    k |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |    s |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |      |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |      v
///
/// Attributes:
///
/// dtype : Dtype
///     The dtype of the data
/// device : str
///     The device on which the blocks are currently stored.
///     We currently only support tensors which have all blocks on a single device.
///     Should be the device returned by `as_device`.
/// blocks : list of block
///     A list of blocks containing the actual entries of the tensor.
///     Leg order is ``[*codomain, *reversed(domain()]``, like ``Tensor.legs``.
/// block_inds : BlockInds
///     A 2D array of positive integers with shape (len(blocks), num_legs).
///     The block `blocks[n]` belongs to the `block_inds[n, m]`-th sector of ``leg``,
///     that is to ``leg.sector_decomposition[block_inds[n, m]]``, where::
///
///         leg == (codomain.spaces[m] if m < len(codomain) else domain.spaces[-1 - m])
///             == tensor.get_leg_co_domain(m)
///
///     Thus, the columns of `block_inds` follow the same ordering convention as `legs`.
///     By convention, we store `blocks` and `block_inds` such that ``np.lexsort(block_inds.T)``
///     is sorted.
///
/// @param dtype, device, blocks, block_inds like attributes above, but not necessarily sorted
/// @param is_sorted If ``False`` (default), we permute `blocks` and `block_inds` according to
/// ``np.lexsort(block_inds.T)``. If ``True``, we assume they are sorted *without* checking.
class AbelianBackendData : public TensorBackend::Data
{
  public:
    using Ptr = std::shared_ptr<AbelianBackendData>;
    using CPtr = std::shared_ptr<const AbelianBackendData>;

    Dtype dtype;
    std::string device;
    std::vector<BlockBackend::BlockPtr> blocks;
    BlockInds block_inds;

    /// Construct data. If ``is_sorted`` is false, permute ``blocks`` / ``block_inds``
    /// according to ``np.lexsort(block_inds.T)``.
    AbelianBackendData(Dtype dtype,
                       std::string device,
                       std::vector<BlockBackend::BlockPtr> blocks,
                       BlockInds block_inds,
                       bool is_sorted = false);

    ~AbelianBackendData() override = default;

    /// Return the index ``n`` of the block which matches ``block_inds``,
    /// i.e. such that ``all(self.block_inds[n, :] == block_inds)``.
    /// Return ``nullopt`` if no such ``n`` exists.
    std::optional<int64> get_block_num(BlockInds const& block_inds) const;

    /// Get the block at given block indices, or ``nullptr`` if none exists.
    BlockBackend::BlockPtr get_block(BlockInds const& block_inds) const;

    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;

    static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath);
};

/// Charge-allowed block index combinations for ``codomain`` / ``domain``, lexsorted.
BlockInds valid_block_inds(TensorProduct::Ptr codomain, TensorProduct::Ptr domain);

/// Backend for Abelian group symmetries.
///
/// Notes:
///
/// The data stored for the various tensor classes defined in ``lasyten.tensors`` is::
///
///     - ``SymmetricTensor``:
///         An ``AbelianBackendData`` instance whose blocks have as many axes as the tensor has
///         legs.
///
///     - ``DiagonalTensor`` :
///         An ``AbelianBackendData`` instance whose blocks have only a single axis.
///         This is the diagonal of the corresponding 2D block in a ``Tensor``.
///
///     - ``Mask`` :
///         An ``AbelianBackendData`` instance whose blocks have only a single axis and bool
///         values.
class AbelianBackend : public TensorBackend
{
  public:
    using Ptr = std::shared_ptr<AbelianBackend>;
    using CPtr = std::shared_ptr<const AbelianBackend>;

    /// Wrap ``AbelianBackendData`` as abstract ``DataPtr``.
    static DataPtr wrap(AbelianBackendData::Ptr d);

    /// Unwrap ``DataPtr`` to ``AbelianBackendData``; throws if wrong type or null.
    static AbelianBackendData::Ptr unwrap(DataPtr d);

    /// Read ``tensor.data`` as ``AbelianBackendData``.
    static AbelianBackendData::Ptr data_from_tensor(TensorCPtr tensor);

    explicit AbelianBackend(std::shared_ptr<BlockBackend> block_backend);
    ~AbelianBackend() override = default;

    bool is_correct_data_type(DataCPtr data) const override
    {
        return dynamic_cast<AbelianBackendData const*>(data.get()) != nullptr;
    }

    void test_tensor_sanity(TensorCPtr a, bool is_diagonal) override;
    void test_mask_sanity(MaskCPtr a) override;

    Leg::Ptr make_pipe(std::vector<Leg::Ptr> legs, bool is_dual, Leg::Ptr pipe = nullptr) override;

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

    std::tuple<DataPtr, DataPtr> lq(SymmetricTensorCPtr tensor,
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

    /// Map incoming multi-leg block indices through a pipe ``block_ind_map``.
    ///
    /// Needed for `combine_legs`.
    ///
    /// @param pipe The pipe which indices are to be mapped
    /// @param incoming_block_inds Rows are block indices @f$ (i_1, i_2, ... i_{nlegs}) @f$ for
    /// incoming legs.
    /// @returns block_inds: 1D array For each row j of `incoming_block_inds` an index `J` such
    /// that ``pipe.block_ind_map[J, 2:-1] == block_inds[j]``.
    BlockInds leg_pipe_map_incoming_block_inds(AbelianLegPipe const& pipe,
                                               BlockInds const& incoming_block_inds) const;

  private:
    DataPtr _compose_worker(SymmetricTensorCPtr a, SymmetricTensorCPtr b);
    DataPtr _compose_no_contraction(SymmetricTensorCPtr a, SymmetricTensorCPtr b);
    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr> _mask_contract(TensorCPtr tensor,
                                                                               MaskCPtr mask,
                                                                               int64 leg_idx,
                                                                               bool large_leg);
};

} // namespace lasyten
