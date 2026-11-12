#pragma once

#include <lasyten/block_backend/block_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/lasyten.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/symmetry.h>
#include <lasyten/symmetries/trees.h>
#include <lasyten/tensors/forward_declare.h>

#include <functional>
#include <lasyten/tools/hdf5.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace lasyten {

/// Callbacks that act on backend blocks / scalars. Python callables are wrapped in pybind.
using BlockUnaryFn = std::function<BlockBackend::BlockPtr(BlockBackend::BlockPtr const&)>;
using BlockBinaryFn = std::function<BlockBackend::BlockPtr(BlockBackend::BlockPtr const&,
                                                           BlockBackend::BlockPtr const&)>;
using BlockFactoryFn = std::function<BlockBackend::BlockPtr(std::vector<int64> const& shape)>;
using SectorBlockFactoryFn =
  std::function<BlockBackend::BlockPtr(std::vector<int64> const& shape, Sector const& coupled)>;
using BlockToScalarFn = std::function<BlockBackend::Scalar(BlockBackend::BlockPtr const&)>;
using ScalarReduceFn =
  std::function<BlockBackend::Scalar(std::vector<BlockBackend::Scalar> const&)>;
using DtypeMapFn = std::function<Dtype(Dtype)>;

/// Abstract base class for tensor-backends.
///
/// A backends implements functions that act on tensors.
/// We abstract two separate concepts for a backend.
/// There is a block backend, that abstracts what the numerical data format (numpy array,
/// torch Tensor, CUDA tensor, ...) is and a tensor-backend that abstracts how block-sparse
/// structures that arise from symmetries are accounted for.
///
/// A tensor backend has a the `block_backend` as an attribute and can call its functions
/// to operate on blocks. This allows the tensor backend to be agnostic of the details of these
/// blocks.
class TensorBackend : public std::enable_shared_from_this<TensorBackend>
{
  public:
    using Ptr = std::shared_ptr<TensorBackend>;
    using CPtr = std::shared_ptr<const TensorBackend>;

    /// Backend-specific payload stored on a tensor (except symmetry data on legs).
    /// Concrete backends subclass this (or wrap a `BlockBackend::Block`).
    class Data : public std::enable_shared_from_this<Data>
    {
      public:
        using Ptr = std::shared_ptr<Data>;
        using CPtr = std::shared_ptr<const Data>;

        virtual ~Data() = default;
    };

    using DataPtr = Data::Ptr;
    using DataCPtr = Data::CPtr;

    std::shared_ptr<BlockBackend> block_backend;

    explicit TensorBackend(std::shared_ptr<BlockBackend> block_backend);
    virtual ~TensorBackend() = default;

    virtual std::string __repr__() const;
    virtual std::string __str__() const;

    /// Semantic equality: same backend class and equivalent `block_backend`.
    [[nodiscard]] bool operator==(TensorBackend const& other) const;
    bool operator!=(TensorBackend const& other) const { return !(*this == other); }

    /// If decompositions (SVD, QR, EIGH, ...) can operate on many-leg tensors.
    /// Otherwise legs must be combined first. Default: ``false``.
    [[nodiscard]] virtual bool can_decompose_tensors() const { return false; }

    /// Return true if ``data`` is this backend's payload type.
    [[nodiscard]] virtual bool is_correct_data_type(DataCPtr data) const = 0;

    /// Convert tensor to a python scalar.
    ///
    /// Assumes that tensor is a scalar (i.e. has only one entry).
    BlockBackend::Scalar item(TensorCPtr a);

    /// Called as part of `test_sanity`.
    ///
    /// Perform sanity checks on the ``a.data``, and possibly additional backend-specific checks
    /// of the tensor.
    virtual void test_tensor_sanity(TensorCPtr a, bool is_diagonal);

    virtual void test_mask_sanity(MaskCPtr a);

    /// Make a pipe *of the appropriate type* for `combine_legs`.
    ///
    /// If `pipe` is given, try to return it if suitable.
    /// May return a plain `LegPipe` or an `AbelianLegPipe` (both report `is_pipe()`).
    virtual Leg::Ptr make_pipe(std::vector<Leg::Ptr> legs, bool is_dual, Leg::Ptr pipe = nullptr);

    /// Apply functions like exp() and log() on a (square) block-diagonal `a`.
    ///
    /// Assumes the block_method returns blocks on the same device.
    ///
    /// @param a The tensor to act on. Can assume ``a.codomain == a.domain``.
    /// @param block_method A function with signature ``block_method(a: Block) -> Block`` acting on
    /// backend-blocks.
    /// @param dtype_map Specify how the result dtype depends on the input dtype. ``None`` means
    /// unchanged. This is needed in abelian and fusion-tree backends, in case there are 0 blocks.
    virtual DataPtr act_block_diagonal_square_matrix(
      SymmetricTensorCPtr a,
      BlockUnaryFn block_method,
      std::optional<DtypeMapFn> dtype_map = std::nullopt) = 0;

    /// Add a trivial leg to a tensor.
    ///
    /// A trivial leg is one-dimensional and consists only of the trivial sector of the symmetry.
    ///
    /// The position of the new leg can be specified in three mutually exclusive ways. If the
    /// positional argument `leg_pos` is used, ``result.legs[leg_pos]`` will be the trivial leg.
    /// In most cases that unambiguously assigns it to either the domain or the codomain. If
    /// ambiguous (``if legs_pos == num_codomain_legs``), it is added to the codomain.
    /// Alternatively, it can be added to the codomain at ``codomain[codomain_pos]`` or to the
    /// domain at ``domain_pos``. Note the implications for the ``is_dual`` argument! Per default,
    /// we use ``0``, i.e. add at ``legs[0]`` / ``codomain[0]``.
    ///
    /// The `label` is the label for the new leg. `is_dual` chooses if we add a dual (bra-like) or
    /// ket-like leg. Note that if `leg_pos` is given, we have
    /// ``result.legs[leg_pos].is_dual == is_dual``, but if `domain_pos` is given, we have
    /// ``result.domain[domain_pos].is_dual == is_dual``, which are mutually opposite.
    ///
    /// This backend method receives the already resolved placement as `add_to_domain` and
    /// `co_domain_pos`, together with the resulting (co)domains.
    ///
    /// @param a The tensor to add a leg to. Since `DiagonalTensor` and `Mask` do not support
    /// adding legs, they will be converted to `SymmetricTensor` first.
    /// @param legs_pos Position of the new leg in `a.legs`.
    /// @param add_to_domain If true, add the leg to the domain, otherwise to the codomain.
    /// @param co_domain_pos Position of the new leg in that (co)domain.
    /// @param new_codomain, new_domain The (co)domain of the result.
    virtual DataPtr add_trivial_leg(TensorCPtr a,
                                    int64 legs_pos,
                                    bool add_to_domain,
                                    int64 co_domain_pos,
                                    TensorProduct::Ptr new_codomain,
                                    TensorProduct::Ptr new_domain) = 0;

    /// Checks if two tensors are equal up to numerical tolerance.
    ///
    /// We compare the blocks, i.e. the free parameters of the tensors.
    /// The tensors count as almost equal if all block-entries, i.e. all their free parameters
    /// individually fulfill ``abs(a1 - a2) <= atol + rtol * abs(a1)``.
    /// Note that this is a basis-dependent and backend-dependent notion of distance, which does
    /// not come from a norm in the strict mathematical sense.
    ///
    /// If `allow_different_types` is ``True``, we convert types, e.g. via `as_SymmetricTensor`
    /// to allow comparison. If ``False``, we raise on mismatching types.
    ///
    /// @param a, b The tensors to compare.
    /// @param atol, rtol Absolute and relative tolerance, see above.
    ///
    /// Notes:
    ///
    /// Unlike numpy, our definition is symmetric under exchanging.
    ///
    /// planar_almost_equal
    ///     Comparison between two tensors with a possible planar permutation between them.
    virtual bool almost_equal(TensorCPtr a, TensorCPtr b, float64 rtol, float64 atol) = 0;

    virtual DataPtr apply_mask_to_DiagonalTensor(DiagonalTensorCPtr tensor, MaskCPtr mask) = 0;

    /// Implementation of `combine_legs`.
    ///
    /// Assumptions:
    ///
    /// - Legs have been permuted, such that each group of legs to be combined appears contiguously
    ///   and either entirely in the codomain or entirely in the domain
    ///
    /// `new_codomain_combine` is a list of tuples ``(positions, combined)``, where positions are
    /// all the codomain-indices which should be combined and ``combined`` is the resulting
    /// `LegPipe`, i.e. ``combined == LegPipe([tensor.codomain[n] for n in positions])``.
    /// `new_domain_combine` is similar as `new_codomain_combine` but for the domain. Note that
    /// ``positions`` are domain-indices, i.e ``n = positions[i]`` refers to ``tensor.domain[n]``,
    /// *not* ``tensor.legs[n]`` !
    ///
    /// @param tensor The tensor to modify
    /// @param leg_idcs_combine A list of groups. Each group a list of integer leg indices, to be
    /// combined. Must be in ascending order.
    /// @param pipes The resulting pipes. Same length and order as `leg_idcs_combine`. In the
    /// domain, this is the product space as it will appear in the domain, not in legs.
    /// @param new_codomain, new_domain The codomain and domain of the resulting tensor
    virtual DataPtr combine_legs(TensorCPtr tensor,
                                 std::vector<std::vector<int64>> leg_idcs_combine,
                                 std::vector<Leg::Ptr> pipes,
                                 TensorProduct::Ptr new_codomain,
                                 TensorProduct::Ptr new_domain) = 0;

    /// Assumes ``a.domain == b.codomain`` and performs contraction over those legs.
    ///
    /// Assumes there is at least one open leg, i.e. the codomain of `a` and the domain of `b` are
    /// not both empty. Assumes both input tensors are on the same device.
    virtual DataPtr compose(SymmetricTensorCPtr a, SymmetricTensorCPtr b) = 0;

    /// Return a copy.
    ///
    /// The main requirement is that future in-place operations on the output data do not affect
    /// the input data
    ///
    /// @param a The tensor to copy
    /// @param device The device for the result. Per default (or if ``None``), use the same device
    /// as `a`. move_to_device
    virtual DataPtr copy_data(TensorCPtr a, std::optional<std::string> device = std::nullopt) = 0;

    /// The hermitian conjugate tensor, a.k.a the dagger of a tensor.
    ///
    /// For a tensor with one leg each in (co-)domain (i.e. a matrix), this coincides with
    /// the hermitian conjugate matrix @f$ (M^\dagger)_{i,j} = \bar{M}_{j, i} @f$ .
    /// For a tensor ``A: W -> V`` the dagger is a map ``dagger(A): V -> W``.
    /// Graphically::
    ///
    ///     |          e   d             a   b   c
    ///     |          │   │             │   │   │
    ///     |       ┏━━┷━━━┷━━┓         ┏┷━━━┷━━━┷┓
    ///     |       ┃    A    ┃         ┃dagger(A)┃
    ///     |       ┗┯━━━┯━━━┯┛         ┗━━┯━━━┯━━┛
    ///     |        │   │   │             │   │
    ///     |        a   b   c             e   d
    ///
    /// Where ``a, b, c, d, e`` denote the legs in to (co-)domain.
    ///
    /// @returns The hermitian conjugate tensor. Its legs and labels are::
    ///
    ///     dagger(A).codomain == A.domain
    ///     dagger(A).domain == A.codomain
    ///     dagger(A).legs == [leg.dual for leg in reversed(A.legs)]
    ///     dagger(A).labels == [_dual_leg_label(l) for l in reversed(A.labels)]
    ///
    /// Note that the resulting `legs` only depend on the input `legs`, not
    /// on their bipartition into domain and codomain.
    /// For labels, we toggle a duality marker, i.e. if ``A.labels == ['a', 'b', 'c', 'd*',
    /// 'e*']``, then ``dagger(A).labels == ['e', 'd', 'c*', 'b*','a*']``.
    virtual DataPtr dagger(TensorCPtr a) = 0;

    /// Assumes that data is a scalar (as defined in tensors.is_scalar).
    ///
    /// Return that scalar as a Scalar.
    virtual BlockBackend::Scalar data_item(DataPtr a) = 0;

    /// Assumes a boolean DiagonalTensor. If all entries are True.
    virtual bool diagonal_all(DiagonalTensorCPtr a) = 0;

    /// Assumes a boolean DiagonalTensor. If any entry is True.
    virtual bool diagonal_any(DiagonalTensorCPtr a) = 0;

    /// Return a modified copy of the data, resulting from applying an elementwise function.
    ///
    /// Apply a function ``func(a_block: Block, b_block: Block, **kwargs) -> Block`` to all
    /// pairs of elements.
    /// Input tensors are both DiagonalTensor and have equal legs.
    /// ``partial_zero_is_zero=True`` promises that ``func(any_block, zero_block) == zero_block``,
    /// and similarly for the second argument.
    ///
    /// Assumes both tensors are on the same device.
    virtual DataPtr diagonal_elementwise_binary(DiagonalTensorCPtr a,
                                                DiagonalTensorCPtr b,
                                                BlockBinaryFn func,
                                                bool partial_zero_is_zero) = 0;

    /// Return a modified copy of the data, resulting from applying an elementwise function.
    ///
    /// Apply ``func(block: Block, **kwargs) -> Block`` to all elements of a diagonal tensor.
    /// ``maps_zero_to_zero=True`` promises that ``func(zero_block) == zero_block``.
    virtual DataPtr diagonal_elementwise_unary(DiagonalTensorCPtr a,
                                               BlockUnaryFn func,
                                               bool maps_zero_to_zero) = 0;

    /// The DiagonalData from a 1D block in *internal* basis order.
    virtual DataPtr diagonal_from_block(BlockBackend::BlockPtr a,
                                        TensorProduct::Ptr co_domain,
                                        float64 tol) = 0;

    /// Generate diagonal data from a function.
    ///
    /// Signature is ``func(shape: tuple[int], coupled: Sector) -> Block``.
    /// Assumes all generated blocks are on the same device.
    virtual DataPtr diagonal_from_sector_block_func(SectorBlockFactoryFn func,
                                                    TensorProduct::Ptr co_domain) = 0;

    /// Get the DiagonalData corresponding to a tensor with two legs.
    ///
    /// Can assume that domain and codomain consist of the same single leg.
    virtual DataPtr diagonal_tensor_from_full_tensor(SymmetricTensorCPtr a,
                                                     std::optional<float64> tol = 1e-12) = 0;

    virtual BlockBackend::Scalar diagonal_tensor_trace_full(DiagonalTensorCPtr a) = 0;

    /// Forget about symmetry structure and convert to a single 1D block.
    ///
    /// This is the diagonal of the respective non-symmetric 2D tensor.
    /// In the *internal* basis order of the leg.
    virtual BlockBackend::BlockPtr diagonal_tensor_to_block(DiagonalTensorCPtr a) = 0;

    /// Convert a DiagonalTensor to a Mask.
    ///
    /// May assume that dtype is bool.
    /// Returns ``mask_data, small_leg``.
    virtual std::tuple<DataPtr, ElementarySpace::Ptr> diagonal_to_mask(
      DiagonalTensorCPtr tens) = 0;

    /// Transpose a diagonal tensor. Also return the new leg ``tens.leg.dual``
    virtual std::tuple<Space::Ptr, DataPtr> diagonal_transpose(DiagonalTensorCPtr tens) = 0;

    /// Eigenvalue decomposition of a hermitian tensor
    ///
    /// Note that this does *not* guarantee to return the duality given by `new_leg_dual`.
    /// In particular, for the abelian backend, the duality is fixed.
    ///
    /// @param a The input tensor. Assumed to be hermitian without checking!
    /// @param new_leg_dual If the new leg should be dual or not.
    /// @param sort How the eigenvalues are sorted *within* each charge block. See `argsort` for
    /// details.
    /// @returns Data for the `DiagonalTensor` of eigenvalues v_data Data for the `Tensor` of
    /// eigenvectors new_leg The new leg.
    virtual std::tuple<DataPtr, DataPtr, ElementarySpace::Ptr> eigh(
      SymmetricTensorCPtr a,
      bool new_leg_dual,
      std::optional<std::string> sort = std::nullopt) = 0;

    /// Eigenvalue decomposition of a general (not necessarily hermitian) tensor.
    ///
    /// Same as `eigh`, but uses a general eigensolver. Eigenvalues and eigenvectors are complex.
    ///
    /// @param a The input tensor. Must have matching domain and codomain.
    /// @param new_leg_dual If the new leg should be dual or not.
    /// @param sort How the eigenvalues are sorted *within* each charge block. See `argsort` for
    /// details.
    /// @returns Data for the `DiagonalTensor` of eigenvalues, data for the `Tensor` of
    /// eigenvectors, and the new leg.
    virtual std::tuple<DataPtr, DataPtr, ElementarySpace::Ptr> eig(
      SymmetricTensorCPtr a,
      bool new_leg_dual,
      std::optional<std::string> sort = std::nullopt) = 0;

    /// Eigenvalues of a hermitian tensor, without eigenvectors.
    ///
    /// @param a The input tensor. Assumed to be hermitian without checking!
    /// @param new_leg_dual If the new leg should be dual or not.
    /// @param sort How the eigenvalues are sorted *within* each charge block. See `argsort` for
    /// details.
    /// @returns Data for the `DiagonalTensor` of eigenvalues and the new leg.
    virtual std::tuple<DataPtr, ElementarySpace::Ptr> eigvalsh(
      SymmetricTensorCPtr a,
      bool new_leg_dual,
      std::optional<std::string> sort = std::nullopt) = 0;

    /// Eigenvalues of a general tensor, without eigenvectors.
    ///
    /// @param a The input tensor. Must have matching domain and codomain.
    /// @param new_leg_dual If the new leg should be dual or not.
    /// @param sort How the eigenvalues are sorted *within* each charge block. See `argsort` for
    /// details.
    /// @returns Data for the `DiagonalTensor` of (generally complex) eigenvalues and the new leg.
    virtual std::tuple<DataPtr, ElementarySpace::Ptr> eigvals(
      SymmetricTensorCPtr a,
      bool new_leg_dual,
      std::optional<std::string> sort = std::nullopt) = 0;

    /// Data for `eye`.
    ///
    /// Data for :meth:``SymmetricTensor.eye``.
    ///
    /// The result has legs ``first_legs + [l.dual for l in reversed(firs_legs)]``.
    virtual DataPtr eye_data(TensorProduct::Ptr co_domain, Dtype dtype, std::string device) = 0;

    /// Convert a dense block to the data for a symmetric tensor.
    ///
    /// Block is in the *internal* basis order of the respective legs and the leg order is
    /// ``[*codomain, *reversed(domain)]``.
    ///
    /// If the block is not symmetric, measured by ``allclose(a, projected, atol, rtol)``,
    /// where ``projected`` is `a` projected to the space of symmetric tensors, raise a
    /// ``ValueError``.
    virtual DataPtr from_dense_block(BlockBackend::BlockPtr a,
                                     TensorProduct::Ptr codomain,
                                     TensorProduct::Ptr domain,
                                     float64 tol) = 0;

    /// Data of a single-leg `Tensor` from the *part of* the coefficients in the trivial sector.
    ///
    /// Is given in the *internal* basis order.
    virtual DataPtr from_dense_block_trivial_sector(BlockBackend::BlockPtr block,
                                                    Space::Ptr leg) = 0;

    /// Data from a grid of tensors.
    ///
    /// @param grid Contains the tensors from which a single tensor is constructed. `None` entries
    /// are interpreted as tensors with all blocks equal to zero.
    /// @param new_codomain Codomain of the resulting tensor after stacking the tensors in the
    /// grid.
    /// @param new_domain Domain of the resulting tensor after stacking the tensors in the grid.
    /// @param left_mult_slices Multiplicity slices for each sector for the stacking in the
    /// codomain. That is, ``slice(left_mult_slices[sector_idx][i], left_mult_slices[sector_idx][i
    /// + 1])`` is the slice that is contributed from the tensors in the `i`th column to the sector
    /// ``new_codomain[0].sector_decomposition[sector_idx]`` of the leg ``new_codomain[0]``.
    /// @param right_mult_slices Multiplicity slices for each sector for the stacking in the
    /// domain. That is, ``slice(right_mult_slices[sector_idx][i], right_mult_slices[sector_idx][i
    /// + 1])`` is the slice that is contributed from the tensors in the `i`th row to the sector
    /// ``new_domain[-1].sector_decomposition[sector_idx]`` of the leg ``new_domain[-1]``.
    /// @param dtype The new dtype of the block.
    /// @param device The device for the block.
    virtual DataPtr from_grid(std::vector<std::vector<py::object>> grid,
                              TensorProduct::Ptr new_codomain,
                              TensorProduct::Ptr new_domain,
                              std::vector<std::vector<int64>> left_mult_slices,
                              std::vector<std::vector<int64>> right_mult_slices,
                              Dtype dtype,
                              std::string device) = 0;

    virtual DataPtr from_random_normal(TensorProduct::Ptr codomain,
                                       TensorProduct::Ptr domain,
                                       float64 sigma,
                                       Dtype dtype,
                                       std::string device) = 0;

    /// Generate tensor data from a function-
    ///
    /// Signature is ``func(shape: tuple[int], coupled: Sector) -> Block``.
    /// Assumes all generated blocks are on the same device.
    virtual DataPtr from_sector_block_func(SectorBlockFactoryFn func,
                                           TensorProduct::Ptr codomain,
                                           TensorProduct::Ptr domain) = 0;

    /// Compute the data for `from_tree_pairs`.
    virtual DataPtr from_tree_pairs(
      std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      Dtype dtype,
      std::string device) = 0;

    virtual DataPtr full_data_from_diagonal_tensor(DiagonalTensorCPtr a) = 0;

    /// May assume that the mask is a projection.
    virtual DataPtr full_data_from_mask(MaskCPtr a, Dtype dtype) = 0;

    /// Extract the device from the data object
    virtual std::string get_device_from_data(DataPtr a) = 0;

    virtual Dtype get_dtype_from_data(DataPtr a) = 0;

    /// Get a single scalar element from a tensor.
    ///
    /// Should be equivalent to ``a.to_numpy()[tuple(idcs)].item()``.
    ///
    /// @param a The tensor.
    /// @param idcs The indices. Checks have already been performed, i.e. we may assume that -
    /// len(idcs) == a.num_legs - 0 <= idx < leg.dim
    virtual BlockBackend::Scalar get_element(SymmetricTensorCPtr a, std::vector<int64> idcs) = 0;

    /// Get a single scalar element from a diagonal tensor.
    ///
    /// Should be equivalent to ``a.to_numpy()[idx, idx].item()`` or
    /// ``a.diagonal_as_numpy()[idx].item()``.
    ///
    /// @param a The diagonal tensor.
    /// @param idx The index for both legs. Checks have already been performed, i.e. we may assume
    /// that ``0 <= idx < leg.dim``
    virtual BlockBackend::Scalar get_element_diagonal(DiagonalTensorCPtr a, int64 idx) = 0;

    /// Get a single scalar element from a mask.
    ///
    /// Get a single scalar element from a diagonal tensor.
    ///
    /// Should be equivalent to ``a.to_numpy()[tuple(idcs)].item()``.
    ///
    /// @param a The mask.
    /// @param idcs The indices. Checks have already been performed, i.e. we may assume that -
    /// len(idcs) == a.num_legs == 2 - 0 <= idx < leg.dim
    virtual BlockBackend::Scalar get_element_mask(MaskCPtr a, std::vector<int64> idcs) = 0;

    /// tensors.inner on SymmetricTensors
    virtual BlockBackend::Scalar inner(SymmetricTensorCPtr a,
                                       SymmetricTensorCPtr b,
                                       bool do_dagger) = 0;

    /// Data for the invariant part used in ChargedTensor.from_dense_block_single_sector
    ///
    /// The vector is given in the *internal* basis order of `spaces`.
    virtual DataPtr inv_part_from_dense_block_single_sector(BlockBackend::BlockPtr vector,
                                                            Space::Ptr space,
                                                            ElementarySpace::Ptr charge_leg) = 0;

    /// Inverse of inv_part_from_dense_block_single_sector
    ///
    /// In the *internal* basis order of `spaces`.
    virtual BlockBackend::BlockPtr inv_part_to_dense_block_single_sector(
      SymmetricTensorCPtr tensor) = 0;

    /// Form the linear combinations ``a * v + b * w``.
    ///
    /// Assumes `v` and `w` are on the same device.
    virtual DataPtr linear_combination(BlockBackend::Scalar a,
                                       TensorCPtr v,
                                       BlockBackend::Scalar b,
                                       TensorCPtr w) = 0;

    /// The LQ decomposition of a tensor.
    ///
    /// A tensor decomposition (see `decompositions`) ``tensor ~ L @ Q`` with the following
    /// properties:
    ///
    /// - ``L`` has a lower triangular structure *in the coupled basis*.
    /// - ``Q`` is an isometry: ``dagger(Q) @ Q ~ eye``.
    ///
    /// Graphically::
    ///
    ///     |                                 │   │   │   │
    ///     |                                ┏┷━━━┷━━━┷━━━┷┓
    ///     |        │   │   │   │           ┃      Q      ┃
    ///     |       ┏┷━━━┷━━━┷━━━┷┓          ┗━━━━━━┯━━━━━━┛
    ///     |       ┃   tensor    ┃    ==           │
    ///     |       ┗━━┯━━━┯━━━┯━━┛          ┏━━━━━━┷━━━━━━┓
    ///     |          │   │   │             ┃      L      ┃
    ///     |                                ┗━━┯━━━┯━━━┯━━┛
    ///     |                                   │   │   │
    ///
    /// We always compute the "reduced", a.k.a. "economic" version.
    /// To group the legs differently, use `permute_legs` or `combine_to_matrix` first.
    ///
    /// Labels for the new legs can be given as two labels ``[a, b]`` s.t. ``L.labels[-1] == a``
    /// and ``Q.labels[0] == b``. A single label ``a`` is equivalent to ``[a, a*]``.
    /// `new_leg_dual` chooses if the new leg should be a ket space (``False``) or bra space
    /// (``True``). `charge_leg_top` fixes whether the charge leg of a decomposed `ChargedTensor`
    /// should end up in the top tensor ``Q`` (``True``) or the bottom tensor ``L`` (``False``).
    /// The corresponding tensor is then also a `ChargedTensor`. Is ignored if the input tensor is
    /// not a `ChargedTensor`.
    ///
    /// @param tensor The tensor to decompose.
    /// @param new_co_domain The (co)domain of the new connecting leg.
    virtual std::tuple<DataPtr, DataPtr> lq(SymmetricTensorCPtr tensor,
                                            TensorProduct::Ptr new_co_domain) = 0;

    /// Elementwise binary function acting on two masks.
    ///
    /// May assume that both masks are a projection (from large to small leg)
    /// and that the large legs match.
    ///
    /// Assumes that `mask1` and `mask2` are on the same device.
    ///
    /// returns ``mask_data, new_small_leg``
    virtual std::tuple<DataPtr, ElementarySpace::Ptr> mask_binary_operand(MaskCPtr mask1,
                                                                          MaskCPtr mask2,
                                                                          BlockBinaryFn func) = 0;

    /// Contraction with the large leg of a Mask.
    ///
    /// Implementation of `_compose_with_Mask` in the case where
    /// the large leg of the mask is contracted.
    /// Note that the mask may be a projection to be applied to the codomain or an inclusion
    /// to be contracted on the domain.
    virtual std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
    mask_contract_large_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx) = 0;

    /// Contraction with the small leg of a Mask.
    ///
    /// Implementation of `_compose_with_Mask` in the case where
    /// the small leg of the mask is contracted.
    /// Note that the mask may be an inclusion to be applied to the codomain or a projection
    /// to be contracted on the domain.
    virtual std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
    mask_contract_small_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx) = 0;

    virtual DataPtr mask_dagger(MaskCPtr mask) = 0;

    /// Data for a *projection* Mask, and the resulting small leg, from a 1D block.
    virtual std::tuple<DataPtr, ElementarySpace::Ptr> mask_from_block(BlockBackend::BlockPtr a,
                                                                      Space::Ptr large_leg) = 0;

    /// As a block of the large_leg, in *internal* basis order.
    virtual BlockBackend::BlockPtr mask_to_block(MaskCPtr a) = 0;

    virtual DataPtr mask_to_diagonal(MaskCPtr a, Dtype dtype) = 0;

    /// Transpose a mask. Also return the new ``space_in`` and ``space_out``.
    ///
    /// Those spaces are the duals of the respective other in the old mask.
    virtual std::tuple<Space::Ptr, Space::Ptr, DataPtr> mask_transpose(MaskCPtr tens) = 0;

    /// Elementwise function acting on a mask.
    ///
    /// May assume that mask is a projection (from large to small leg).
    /// Returns ``mask_data, new_small_leg``
    virtual std::tuple<DataPtr, ElementarySpace::Ptr> mask_unary_operand(MaskCPtr mask,
                                                                         BlockUnaryFn func) = 0;

    /// Move tensor to a given device.
    ///
    /// The result is *not* guaranteed to be a copy. In particular, if `a` already is on the
    /// target device, it is returned without modification.
    ///
    /// copy_data
    virtual DataPtr move_to_device(TensorCPtr a, std::string device) = 0;

    virtual DataPtr mul(BlockBackend::Scalar a, TensorCPtr b) = 0;

    /// Norm of a tensor. order has already been parsed and is a number
    virtual BlockBackend::Scalar norm(TensorCPtr a) = 0;

    /// Form the outer product, or tensor product of maps.
    ///
    /// Assumes that `a` and `b` are on the same device.
    virtual DataPtr outer(SymmetricTensorCPtr a, SymmetricTensorCPtr b) = 0;

    /// Contract the codomain (domain) of `b` with the a part of the domain (codomain) of `a`.
    ///
    /// Assumes that there is at least one open leg in the domain (codomain) of the resulting
    /// tensor. Assumes both input tensors are on the same device.
    virtual DataPtr partial_compose(SymmetricTensorCPtr a,
                                    SymmetricTensorCPtr b,
                                    int64 a_first_leg,
                                    TensorProduct::Ptr new_codomain,
                                    TensorProduct::Ptr new_domain) = 0;

    /// Perform an arbitrary number of traces. Pairs are converted to leg idcs.
    ///
    /// Returns ``data, codomain, domain``.
    virtual std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr> partial_trace(
      SymmetricTensorCPtr tensor,
      std::vector<std::pair<int64, int64>> pairs,
      std::vector<std::optional<int64>> levels) = 0;

    /// Permute legs on the tensors.
    ///
    /// @param a The tensor to act on.
    /// @param codomain_idcs, domain_idcs Which of the legs should end up in the (co-)domain. All
    /// are leg indices (``0 <= i < a.num_legs``).
    /// @param new_codomain, new_domain The (co)domain of the result.
    /// @param mixes_codomain_domain If any leg moves from the codomain to the domain or vv during
    /// the permutation.
    /// @param levels The levels. Must support comparison with ``<`` or be ``None``, meaning
    /// unspecified.
    /// @param bend_right For each leg, whether it bends to the left or right of the tensor.
    /// ``None`` is allowed as a placeholder, only if that leg does not bend at all. Note that
    /// non-bending legs do not necessarily have a ``None`` entry, however.
    /// @returns data: The data for the permuted tensor, or ``None`` if `levels` are required but
    /// were not specified. codomain, domain The (co-)domain of the new tensor.
    virtual DataPtr permute_legs(TensorCPtr a,
                                 std::vector<int64> codomain_idcs,
                                 std::vector<int64> domain_idcs,
                                 TensorProduct::Ptr new_codomain,
                                 TensorProduct::Ptr new_domain,
                                 bool mixes_codomain_domain,
                                 std::vector<std::optional<int64>> levels,
                                 std::vector<std::optional<bool>> bend_right) = 0;

    /// Perform a QR decomposition.
    ///
    /// With ``a == Q @ R``
    /// ``Q.domain == a.domain``, ``Q.codomain == new_codomain``
    /// ``R.domain == new_codomain``, ``R.codomain == a.codomain``
    virtual std::tuple<DataPtr, DataPtr> qr(SymmetricTensorCPtr a,
                                            TensorProduct::Ptr new_co_domain) = 0;

    /// Reduce a diagonal tensor to a single number.
    ///
    /// Used e.g. to implement ``DiagonalTensor.max``.
    /// ``block_func(block: Block) -> Scalar`` realizes that reduction on blocks,
    /// ``func(numbers: Sequence[Scalar]) -> Scalar`` for numbers.
    virtual BlockBackend::Scalar reduce_DiagonalTensor(DiagonalTensorCPtr tensor,
                                                       BlockToScalarFn block_func,
                                                       ScalarReduceFn func) = 0;

    /// Scale axis ``leg`` of ``a`` with ``b``.
    ///
    /// Can assume ``a.get_leg_co_domain(leg) == b.leg``.
    /// Assumes that `a` and `b` are on the same device.
    virtual DataPtr scale_axis(TensorCPtr a, DiagonalTensorCPtr b, int64 leg) = 0;

    /// Split (multiple) product space legs.
    ///
    /// @param a The tensor to split legs on.
    /// @param leg_idcs List of leg-indices, fulfilling ``0 <= i < a.num_legs``, to split. Must be
    /// in ascending order.
    /// @param new_codomain, new_domain The new (co-)domain, after splitting. Has same sectors and
    /// multiplicities.
    virtual DataPtr split_legs(TensorCPtr a,
                               std::vector<int64> leg_idcs,
                               TensorProduct::Ptr new_codomain,
                               TensorProduct::Ptr new_domain) = 0;

    /// Assume the legs at given indices are trivial and get rid of them
    virtual DataPtr squeeze_legs(TensorCPtr a, std::vector<int64> idcs) = 0;

    virtual bool supports_symmetry(Symmetry::Ptr symmetry) = 0;

    /// The singular value decomposition (SVD) of a tensor.
    ///
    /// A tensor decomposition (see `decompositions`) ``tensor ~ U @ S @ Vh`` with the following
    /// properties:
    ///
    /// - ``Vh`` and ``U`` are isometries: ``dagger(U) @ U ~ eye ~ Vh @ dagger(Vh)``.
    /// - ``S`` is a `DiagonalTensor` with real, non-negative entries.
    /// - If `tensor` is a matrix (i.e. if it has exactly one leg each in domain and codomain), it
    ///   reproduces the usual matrix SVD.
    ///
    /// .. note ::
    ///     The basis for the newly generated leg is chosen arbitrarily, and in particular, unlike,
    ///     e.g., `svd` it is not guaranteed that ``S.diag_numpy`` is sorted.
    ///
    /// Graphically::
    ///
    ///     |                                 │   │   │   │
    ///     |                                ┏┷━━━┷━━━┷━━━┷┓
    ///     |                                ┃      Vh     ┃
    ///     |        │   │   │   │           ┗━━━━━━┯━━━━━━┛
    ///     |       ┏┷━━━┷━━━┷━━━┷┓               ┏━┷━┓
    ///     |       ┃   tensor    ┃    ==         ┃ S ┃
    ///     |       ┗━━┯━━━┯━━━┯━━┛               ┗━┯━┛
    ///     |          │   │   │             ┏━━━━━━┷━━━━━━┓
    ///     |                                ┃      U      ┃
    ///     |                                ┗━━┯━━━┯━━━┯━━┛
    ///     |                                   │   │   │
    ///
    /// We always compute the "reduced", a.k.a. "economic" version of SVD, where the isometries are
    /// (in general) not full unitaries.
    ///
    /// To group the legs differently, use `permute_legs` or `combine_to_matrix` first.
    ///
    /// Labels for the new legs can be specified in the following three ways: Four labels
    /// ``[a, b, c, d]`` result in ``U.labels[-1] == a``, ``S.labels == [b, c]`` and
    /// ``Vh.labels[0] == d``. Two labels ``[a, b]`` are equivalent to ``[a, b, a, b]``. A single
    /// label ``a`` is equivalent to ``[a, a*, a, a*]``. The new legs are unlabelled by default.
    /// `new_leg_dual` chooses if the new leg should be a ket space (``False``) or bra space
    /// (``True``). `charge_leg_top` fixes whether the charge leg of a decomposed `ChargedTensor`
    /// should end up in the top tensor ``Vh`` (``True``) or the bottom tensor ``U`` (``False``).
    /// The corresponding tensor is then also a `ChargedTensor`. Is ignored if the input tensor is
    /// not a `ChargedTensor`.
    ///
    /// @param a The tensor to decompose.
    /// @param new_co_domain The (co)domain of the new connecting legs.
    /// @param algorithm The algorithm (a.k.a. "driver") for the block-wise svd. Choices are
    /// backend-specific. See `possible_svd_algorithms`.
    /// @returns U: SymmetricTensor | ChargedTensor S: DiagonalTensor Vh: SymmetricTensor |
    /// ChargedTensor
    virtual std::tuple<DataPtr, DataPtr, DataPtr> svd(SymmetricTensorCPtr a,
                                                      TensorProduct::Ptr new_co_domain,
                                                      std::optional<std::string> algorithm) = 0;

    /// TODO clearly define what this should do in tensors.py first!
    ///
    /// In particular regarding basis orders.
    virtual py::object state_tensor_product(BlockBackend::BlockPtr state1,
                                            BlockBackend::BlockPtr state2,
                                            Leg::Ptr pipe) = 0;

    virtual DataPtr to_block_backend(DataPtr data,
                                     std::shared_ptr<BlockBackend> block_backend,
                                     std::optional<Dtype> dtype = std::nullopt,
                                     std::optional<std::string> device = std::nullopt) = 0;

    /// Forget about symmetry structure and convert to a single block.
    ///
    /// Return a block in the *internal* basis order of the respective legs,
    /// with leg order ``[*codomain, *reversed(domain)]``.
    virtual BlockBackend::BlockPtr to_dense_block(TensorCPtr a) = 0;

    /// Single-leg tensor to the *part of* the coefficients in the trivial sector.
    ///
    /// In *internal* basis order.
    virtual BlockBackend::BlockPtr to_dense_block_trivial_sector(TensorCPtr tensor) = 0;

    /// Cast to given dtype. No copy if already has dtype.
    virtual DataPtr to_dtype(TensorCPtr a, Dtype dtype) = 0;

    virtual BlockBackend::Scalar trace_full(SymmetricTensorCPtr a,
                                            std::vector<int64> idcs1,
                                            std::vector<int64> idcs2) = 0;

    /// Implementation of `truncate_singular_values`.
    ///
    /// @returns Data for the mask new_leg : ElementarySpace The new leg after truncation, i.e. the
    /// small leg of the mask err : float The truncation error ``norm(S_discard) == norm(S -
    /// S_keep)``. new_norm The norm ``norm(S_keep)`` of the approximation.
    virtual std::tuple<DataPtr, ElementarySpace::Ptr, float64, float64> truncate_singular_values(
      DiagonalTensorCPtr S,
      std::optional<int64> chi_max,
      int64 chi_min,
      float64 degeneracy_tol,
      float64 trunc_cut,
      std::optional<float64> svd_min,
      bool minimize_error = true) = 0;

    /// Helper function for `truncate_singular_values`.
    ///
    /// @param S A numpy array of singular values S[i]
    /// @param qdims A numpy array of the quantum dimensions. ``None`` means all qdims are one.
    /// @param chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min, minimize_error Constraints for
    /// truncation. See `truncate_singular_values`.
    /// @returns mask : 1D numpy array of bool A boolean mask, indicating that ``S[mask]``
    /// should be kept; err : float The truncation error ``norm(S_discard) == norm(S - S_keep)``.
    /// new_norm The norm ``norm(S_keep)`` of the approximation.
    std::tuple<py::array, float64, float64> _truncate_singular_values_selection(
      py::array S,
      py::object qdims,
      std::optional<int64> chi_max,
      int64 chi_min,
      float64 degeneracy_tol,
      float64 trunc_cut,
      std::optional<float64> svd_min,
      bool minimize_error = true);

    /// Data for a zero tensor.
    ///
    /// @param codomain, domain The (co)domain of the tensor.
    /// @param dtype The dtype of the entries.
    /// @param device The device of the tensor.
    /// @param all_blocks Some specific backends can omit zero blocks ("sparsity"). By default
    /// (``False``), omit them if possible. If ``True``, force all blocks to be created, with zero
    /// entries.
    virtual DataPtr zero_data(TensorProduct::Ptr codomain,
                              TensorProduct::Ptr domain,
                              Dtype dtype,
                              std::string device,
                              bool all_blocks = false) = 0;

    virtual DataPtr zero_diagonal_data(TensorProduct::Ptr co_domain,
                                       Dtype dtype,
                                       std::string device) = 0;

    virtual DataPtr zero_mask_data(Space::Ptr large_leg, std::string device) = 0;

    /// If the Tensor is comprised of real numbers.
    ///
    /// Complex numbers with small or zero imaginary part still cause a `False` return.
    virtual bool is_real(TensorCPtr a);

    virtual void save_hdf5(lasyten::hdf5::Saver& saver,
                           HighFive::Group& h5gr,
                           std::string subpath);

    /// Reconstruct a concrete backend. Bound as a Python classmethod so `cls` is the saved type.
    static Ptr from_hdf5(py::object cls,
                         lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string subpath);
};

/// The conventional order of legs: ``[*codomain.factors, *reversed(domain.factors)]``.
/// The conventional order of legs.
std::vector<Leg::Ptr> conventional_leg_order(TensorProduct::Ptr codomain,
                                             TensorProduct::Ptr domain);

/// Overload accepting a tensor-like object with ``.codomain`` / ``.domain`` attributes.
std::vector<Leg::Ptr> conventional_leg_order(py::object tensor_or_codomain,
                                             py::object domain = py::none());

std::vector<Leg::Ptr> conventional_leg_order(TensorCPtr tensor);

/// If the given objects have the same backend, return it. Raise otherwise.
/// If the given object have the same backend, return it. Raise otherwise.
TensorBackend::Ptr get_same_backend(const std::vector<py::object>& objs,
                                    std::string error_msg = "Incompatible backends.");

TensorBackend::Ptr get_same_backend(const std::vector<TensorCPtr>& objs,
                                    std::string error_msg = "Incompatible backends.");

} // namespace lasyten
