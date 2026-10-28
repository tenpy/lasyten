#pragma once

#include <lasyten/backends/abelian.h>
#include <lasyten/backends/fusion_tree_backend.h>
#include <lasyten/backends/no_symmetry.h>
#include <lasyten/backends/tensor_backend.h>
#include <lasyten/block_backend/block_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/symmetries/sector.h>
#include <lasyten/symmetries/trees.h>
#include <lasyten/tensors/tensor.h>

#include <lasyten/tools/hdf5.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lasyten {

/// A tensor that is symmetric, i.e. invariant under the symmetry.
///
/// .. note ::
///     The constructor is not particularly user friendly.
///     Consider using the various classmethods instead.
///
/// @param codomain The codomain.
/// @param domain The domain. ``None`` (the default) is equivalent to ``[]``, i.e. no legs in the
/// domain.
/// @param backend The backend of the tensor.
/// @param labels Specify the labels for the legs. Can either give two lists, one for the codomain,
/// one for the domain. Or a single flat list for all legs in the order of the `legs`, such that
/// ``[codomain_labels, domain_labels]`` is equivalent to ``[*codomain_legs,
/// *reversed(domain_legs)]``.
/// @param dtype The dtype of tensor entries.
///
/// Attributes:
///
/// data:
///     Backend-specific data structure that contains the numerical data, i.e. the free parameters
///     of tensors with the given symmetry.
class SymmetricTensor : public Tensor
{
  public:
    using Ptr = std::shared_ptr<SymmetricTensor>;
    using CPtr = std::shared_ptr<const SymmetricTensor>;

    /// Backend-specific free parameters of tensors with the given symmetry.
    TensorBackend::DataPtr data;

    /// When true, labels may contain a charge-leg marker ``!`` (used only as the
    /// `ChargedTensor` invariant part). Normal SymmetricTensors must leave this false.
    bool allow_charge_leg_label = false;

    /// ``check_complex_dtype`` must be false when constructing a `DiagonalTensor`
    /// subclass: virtual ``verify_dtype`` does not dispatch to the derived override while the
    /// base constructor runs, and diagonal tensors intentionally allow real dtypes.
    SymmetricTensor(TensorBackend::DataPtr data,
                    TensorProduct::Ptr codomain,
                    TensorProduct::Ptr domain,
                    TensorBackend::Ptr backend,
                    Symmetry::Ptr symmetry,
                    LegLabelList labels,
                    bool check_complex_dtype = true);

    ~SymmetricTensor() override = default;

    /// Perform sanity checks.
    void test_sanity() const override;

    virtual void verify_dtype() const;

    [[nodiscard]] std::string ascii_diagram_type_name() const override;
    [[nodiscard]] std::string class_name() const override;

    [[nodiscard]] static std::optional<Dtype> _parse_default_dtype(std::optional<Dtype> dtype,
                                                                   Symmetry::Ptr const& symmetry);

    // --- factories ---

    /// Initialize a `SymmetricTensor` by generating its blocks from a function.
    ///
    /// Here "the blocks of a tensor" are the backend-specific blocks that contain the free
    /// parameters of the tensor in the `data`. The concrete meaning of these blocks depends
    /// on the backend.
    ///
    /// `func` has two possible signatures. If `shape_kw` is given, we expect
    /// ``func(*, shape_kw: tuple[int, ...], **kwargs) -> BlockLike``. Otherwise
    /// ``func(shape: tuple[int, ...], **kwargs) -> BlockLike``. ``shape`` is the shape of the
    /// block to generate and `func_kwargs` are passed as ``kwargs``. The output is converted to
    /// backend-specific blocks via ``backend.as_block``. In particular, it may be modified
    /// in-place after that. If `shape_kw` is given, the shape is passed to `func` as a kwarg with
    /// this keyword.
    ///
    /// @param func The block factory, see above.
    /// @param codomain, domain, backend, labels Arguments for constructor of `SymmetricTensor`.
    /// @param dtype If given, the resulting blocks from `func` are converted to this dtype.
    /// @param device If given, the resulting blocks are moved to that device. Per default, if
    /// `func` returns backend-specific blocks, their device is used and otherwise the default
    /// device of the backend. from_sector_block_func
    ///     Allows the `func` to take the current coupled sectors as an argument.
    [[nodiscard]] static Ptr from_block_func(BlockFactoryFn func,
                                             TensorProduct::Ptr codomain,
                                             TensorProduct::Ptr domain = nullptr,
                                             TensorBackend::Ptr backend = nullptr,
                                             std::optional<LegLabelList> labels = std::nullopt,
                                             std::optional<Dtype> dtype = std::nullopt,
                                             std::optional<std::string> device = std::nullopt);

    /// Convert a dense block of the backend to a SymmetricTensor.
    [[nodiscard]] static Ptr from_dense_block(BlockBackend::BlockPtr block,
                                              TensorProduct::Ptr codomain,
                                              TensorProduct::Ptr domain = nullptr,
                                              TensorBackend::Ptr backend = nullptr,
                                              std::optional<LegLabelList> labels = std::nullopt,
                                              std::optional<Dtype> dtype = std::nullopt,
                                              std::optional<std::string> device = std::nullopt,
                                              float64 tol = 1e-6,
                                              bool understood_braiding = false);

    /// Inverse of to_dense_block_trivial_sector.
    [[nodiscard]] static Ptr from_dense_block_trivial_sector(
      BlockBackend::BlockPtr vector,
      Leg::Ptr space,
      TensorBackend::Ptr backend = nullptr,
      std::optional<std::string> device = std::nullopt,
      std::string label = "");

    /// The identity map as a SymmetricTensor.
    ///
    /// @param co_domain The domain *and* codomain of the resulting tensor.
    /// @param labels Can either specify the labels for all legs of the resulting tensor, like in
    /// the constructor of `SymmetricTensor`. Alternatively, can give labels only for the codomain
    /// (one list), and the domain labels are constructed as their dual labels i.e. ``'p' <->
    /// 'p*'``.
    /// @param backend The backend of the tensor.
    /// @param dtype The dtype of the tensor.
    /// @param device The device of the tensor. If ``None``, use the `default_device` of the block
    /// backend.
    [[nodiscard]] static Ptr from_eye(TensorProduct::Ptr co_domain,
                                      TensorBackend::Ptr backend = nullptr,
                                      std::optional<LegLabelList> labels = std::nullopt,
                                      Dtype dtype = Dtype::Complex128,
                                      std::optional<std::string> device = std::nullopt);

    /// Generate a sample from the normal distribution.
    [[nodiscard]] static Ptr from_random_normal(TensorProduct::Ptr codomain,
                                                TensorProduct::Ptr domain = nullptr,
                                                TensorCPtr mean = nullptr,
                                                float64 sigma = 1.0,
                                                TensorBackend::Ptr backend = nullptr,
                                                std::optional<LegLabelList> labels = std::nullopt,
                                                std::optional<Dtype> dtype = Dtype::Complex128,
                                                std::optional<std::string> device = std::nullopt);

    /// Generate a tensor with uniformly random block-entries.
    ///
    /// The block entries, i.e. the free parameters of the tensor are drawn independently and
    /// uniformly. If dtype is a real type, they are drawn from [-1, 1], if it is complex, real and
    /// imaginary part are drawn independently from [-1, 1].
    ///
    /// .. note ::
    ///     This is not a well defined probability distribution on the space of symmetric tensors,
    ///     since the meaning of the uniformly drawn numbers depends on both the choice of the
    ///     basis and on the backend.
    ///
    /// @param codomain, domain, backend, labels Arguments, like for constructor of
    /// `SymmetricTensor`.
    /// @param dtype The dtype for the tensor.
    /// @param device The device of the tensor. If omitted, use the default device of the backend.
    [[nodiscard]] static Ptr from_random_uniform(TensorProduct::Ptr codomain,
                                                 TensorProduct::Ptr domain = nullptr,
                                                 TensorBackend::Ptr backend = nullptr,
                                                 std::optional<LegLabelList> labels = std::nullopt,
                                                 Dtype dtype = Dtype::Complex128,
                                                 std::optional<std::string> device = std::nullopt);

    /// Initialize a `SymmetricTensor` by generating its blocks from a function.
    ///
    /// Here "the blocks of a tensor" are the backend-specific blocks that contain the free
    /// parameters of the tensor in the `data`. The concrete meaning of these blocks depends
    /// on the backend.
    ///
    /// Unlike `from_block_func`, this classmethod supports a `func` that takes the current
    /// coupled sector as an argument. The tensor, as a map from its domain to its codomain is
    /// block-diagonal in the coupled sectors, i.e. in the ``domain.sector_decomposition``.
    /// Thus, the free parameters of a tensor are associated with one block of this structure,
    /// and thus with a given coupled sector. A value of ``coupled`` indicates that the generated
    /// block is (part of) the components that maps from ``coupled`` in the domain to ``coupled``
    /// in the codomain.
    ///
    /// `func` has signature ``func(shape: tuple[int, ...], coupled: Sector, **kwargs) ->
    /// BlockLike``. ``shape`` is the shape of the block to be generated, ``coupled`` is the
    /// current coupled sector and `func_kwargs` are passed as ``kwargs``. The output is converted
    /// to backend-specific blocks via ``backend.block_backend.as_block``. If `shape_kw` is given,
    /// the shape is passed to `func` as a kwarg with this keyword.
    ///
    /// @param func The block factory, see above.
    /// @param codomain, domain, backend, labels Arguments, like for constructor of
    /// `SymmetricTensor`.
    /// @param dtype If given, the resulting blocks from `func` are converted to this dtype.
    /// @param device If given, the resulting blocks are moved to that device. Per default, if
    /// `func` returns backend-specific blocks, their device is used and otherwise the default
    /// device of the backend. from_block_func
    [[nodiscard]] static Ptr from_sector_block_func(
      SectorBlockFactoryFn func,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain = nullptr,
      TensorBackend::Ptr backend = nullptr,
      std::optional<LegLabelList> labels = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      std::optional<std::string> device = std::nullopt);

    /// A tensor that projects onto a given coupled sector of it domain.
    [[nodiscard]] static Ptr from_sector_projection(
      TensorProduct::Ptr co_domain,
      Sector sector,
      TensorBackend::Ptr backend = nullptr,
      std::optional<LegLabelList> labels = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      std::optional<std::string> device = std::nullopt);

    /// Create a tensor from a linear combination of fusion-tree splitting-tree pairs.
    ///
    /// @param trees Specifies the linear combination that defines the resulting tensor. Each entry
    /// of the dict, ``{(X, Y): coeffs}`` represents several contributions to the linear
    /// combination, one per entry of the block ``coeffs``. The contribution with prefactor
    /// ``coeffs[n1, ..., nJ, mK, ..., m1]`` (note the axis order!) consists of the following steps
    /// as a map from domain to codomain::  1. Project each leg ``k`` of the domain to a single
    /// sector, where the sector is given by ``Y.uncoupled[k]`` and the degeneracy index by ``mk``
    /// (an index to the array ``coeffs``).  2. Apply the fusion tree ``Y``.  3. Apply the
    /// splitting tree ``X``.  4. Apply inclusions on each leg ``j`` of the codomain, where the
    /// sector is given by ``X.uncoupled[j]`` and the degeneracy index by ``nj`` (an index to the
    /// array ``coeffs``).
    /// @param codomain, domain, backend, labels Arguments, like for constructor of
    /// `SymmetricTensor`.
    /// @param dtype The dtype of the resulting tensor.
    /// @param device The device of the tensor. If omitted, use the default device of the backend.
    [[nodiscard]] static Ptr from_tree_pairs(py::object trees,
                                             TensorProduct::Ptr codomain,
                                             TensorProduct::Ptr domain = nullptr,
                                             TensorBackend::Ptr backend = nullptr,
                                             std::optional<LegLabelList> labels = std::nullopt,
                                             std::optional<Dtype> dtype = std::nullopt,
                                             std::optional<std::string> device = std::nullopt);

    /// A zero tensor.
    ///
    /// @param codomain, domain, backend, labels Arguments, like for constructor of
    /// `SymmetricTensor`.
    /// @param dtype The dtype for the entries.
    /// @param device The device of the tensor. If ``None``, use the `default_device` of the block
    /// backend.
    [[nodiscard]] static Ptr from_zero(TensorProduct::Ptr codomain,
                                       TensorProduct::Ptr domain = nullptr,
                                       TensorBackend::Ptr backend = nullptr,
                                       std::optional<LegLabelList> labels = std::nullopt,
                                       Dtype dtype = Dtype::Complex128,
                                       std::optional<std::string> device = std::nullopt);

    /// Import SymmetricTensor from hdf5
    [[nodiscard]] static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                                       HighFive::Group& h5gr,
                                       std::string const& subpath);

    /// Export SymmetricTensor to hdf5 such that it can be re-imported with from_hdf5
    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;

    // --- Tensor overrides ---

    /// Convert to a tensor of the given dtype on the same device.
    ///
    /// @param dtype The dtype of the result.
    [[nodiscard]] Tensor::Ptr as_dtype(Dtype dtype) override;

    /// Convert to a `SymmetricTensor`, if possible.
    ///
    /// @param guarantee_copy If already a SymmetricTensor, we do *not* make a copy by default. Set
    /// this flag to ``True`` to guarantee a copy.
    /// @param warning If given, and if the conversion is non-trivial (i.e. if it was not already a
    /// SymmetricTensor to begin with), a warning with this text is issued.
    [[nodiscard]] SymmetricTensorPtr as_SymmetricTensor(
      bool guarantee_copy = false,
      std::optional<std::string> warning = std::nullopt) override;

    /// Copy the tensor.
    ///
    /// @param deep If the copy should be deep. A shallow copy is a new instance with the same
    /// data.
    /// @param device The device for the result. Per default, use the same device as `self`.
    /// @param dtype The dtype of the result. Per default, use the same dtype as `self`.
    [[nodiscard]] Tensor::Ptr copy(bool deep = true,
                                   std::optional<std::string> device = std::nullopt,
                                   std::optional<Dtype> dtype = std::nullopt) override;

    /// The diagonal part as a `DiagonalTensor`.
    ///
    /// @param check_offdiagonal If we should check that the off-diagonal parts vanish.
    [[nodiscard]] DiagonalTensorPtr diagonal(bool check_offdiagonal = false) const;

    /// Implementation of `__getitem__`.
    ///
    /// Can assume we have one non-negative integer index per leg.
    [[nodiscard]] BlockBackend::Scalar _get_item(std::vector<int64> const& idx) override;

    /// Move tensor to a given device, *in place*.
    void move_to_device(std::string device) override;

    /// Convert to a tensor with a different backend.
    ///
    /// @param backend The backend of the result.
    /// @param dtype The dtype of the result. Per default, use the same dtype as `self`.
    /// @param device The device for the result. Per default, use the same device as `self`.
    [[nodiscard]] Tensor::Ptr to_backend(
      TensorBackend::Ptr backend,
      std::optional<Dtype> dtype = std::nullopt,
      std::optional<std::string> device = std::nullopt) override;

    /// Convert to a dense block of the backend, if possible.
    ///
    /// This corresponds to "forgetting" the symmetry structure and is only possible if the
    /// symmetry `can_be_dropped`.
    /// The result is a backend-specific block, e.g. a numpy array if the block backend is a
    /// `NumpyBlockBackend` or a torch Tensor if the backend is a `TorchBlockBackend`.
    ///
    /// @param leg_order If given, the leg of the resulting block are permuted to match this leg
    /// order.
    /// @param dtype If given, the result is converted to this dtype. Per default it has the
    /// `dtype` of the tensor.
    /// @param understood_braiding For symmetries with non-trivial (but symmetric) braiding, e.g.
    /// fermions, the resulting dense block does no longer capture the braiding statistics
    /// correctly. This means that `permute_legs` is not consistently reproduced by e.g.
    /// ``numpy.transpose`` on the dense block representation. Permuting its legs would require
    /// e.g. explicit swap gates. When using the result, special care needs to be taken regarding
    /// the leg order. To avoid this pitfall, we raise an error by default. Set this flag to
    /// ``True`` to disable the error. It is then your responsibility to take care of leg orders
    /// and braids. See `swap_gate_numpy` for manipulations on these dense blocks.
    [[nodiscard]] BlockBackend::BlockPtr to_dense_block(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      bool understood_braiding = false) override;

    /// Assumes self is a single-leg tensor and returns its components in the trivial sector.
    ///
    /// from_dense_block_trivial_sector
    [[nodiscard]] BlockBackend::BlockPtr to_dense_block_trivial_sector() const;
};

} // namespace lasyten
