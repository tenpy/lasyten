#pragma once

#include <lasyten/tensors/symmetric_tensor.h>

#include <lasyten/tools/hdf5.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

class Mask;

/// Special case of a `SymmetricTensor` that is diagonal in the computational basis.
///
/// The domain and codomain of a diagonal tensor are the same and consist of a single leg::
///
///     |        │
///     |      ┏━┷━┓
///     |      ┃ D ┃
///     |      ┗━┯━┛
///     |        │
///
/// A diagonal tensor then is a map that is a multiple of the identity on each sector of the leg,
/// i.e. it is given by @f$ \bigoplus_a \lambda_a \eye_a @f$, where the sum goes over sectors
/// @f$ a @f$ of the `leg` @f$ V = \bigoplus_a a @f$.
///
/// This is the natural type e.g. for singular values or eigenvalues and allows
/// elementwise operations (for example `complex_conj`, `sqrt`, `exp`).
///
/// If a function can be defined as a power series in ``D`` and ``D.hc``, its action can be
/// achieved by applying that power series to the diagonal elements individually.
///
/// @param data The numerical data ("free parameters") comprising the tensor. type is
/// backend-specific
/// @param leg The single leg in both the domain and codomain
/// @param backend The backend of the tensor.
/// @param labels Specify the labels for the legs. Can either give two lists, one for the codomain,
/// one for the domain. Or a single flat list for all legs in the order of the `legs`, such that
/// ``[codomain_labels, domain_labels]`` is equivalent to ``[*codomain_legs,
/// *reversed(domain_legs)]``.
class DiagonalTensor : public SymmetricTensor
{
  public:
    using Ptr = std::shared_ptr<DiagonalTensor>;
    using CPtr = std::shared_ptr<const DiagonalTensor>;

    /// Empty — bool dtype is allowed for diagonal tensors (Python ``_forbidden_dtypes = []``).
    static std::vector<Dtype> _forbidden_dtypes;

    DiagonalTensor(TensorBackend::DataPtr data,
                   Space::Ptr leg,
                   TensorBackend::Ptr backend,
                   Symmetry::Ptr symmetry,
                   LegLabelList labels);

    ~DiagonalTensor() override = default;

    [[nodiscard]] std::vector<Dtype> const& forbidden_dtypes() const override;

    /// Perform sanity checks.
    void test_sanity() const override;
    void verify_dtype() const override;

    [[nodiscard]] std::string ascii_diagram_type_name() const override;
    [[nodiscard]] std::string class_name() const override;

    /// Return the single space that makes up to domain and codomain.
    [[nodiscard]] Space::Ptr leg() const;

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
    /// @param leg, backend, labels Arguments for constructor of `DiagonalTensor`.
    /// @param dtype If given, the resulting blocks from `func` are converted to this dtype.
    /// @param device If given, the resulting blocks are moved to that device. Per default, if
    /// `func` returns backend-specific blocks, their device is used and otherwise the default
    /// device of the backend. from_sector_block_func
    ///     Allows the `func` to take the current coupled sectors as an argument.
    [[nodiscard]] static Ptr from_block_func(BlockFactoryFn func,
                                             Space::Ptr leg,
                                             TensorBackend::Ptr backend = nullptr,
                                             std::optional<LegLabelList> labels = std::nullopt,
                                             std::optional<Dtype> dtype = std::nullopt,
                                             std::optional<std::string> device = std::nullopt);

    /// Convert a dense block of the backend to a DiagonalTensor.
    [[nodiscard]] static Ptr from_dense_block(BlockBackend::BlockPtr block,
                                              Space::Ptr leg,
                                              TensorBackend::Ptr backend = nullptr,
                                              std::optional<LegLabelList> labels = std::nullopt,
                                              std::optional<Dtype> dtype = std::nullopt,
                                              float64 tol = 1e-6,
                                              std::optional<std::string> device = std::nullopt,
                                              bool understood_braiding = false);

    /// Convert a dense 1D block containing the diagonal entries to a DiagonalTensor.
    ///
    /// @param diag The diagonal entries as a backend-specific block or some data that can be
    /// converted using `as_block`. This includes e.g. nested python iterables or numpy arrays.
    /// @param leg, backend, labels Arguments for constructor of `DiagonalTensor`.
    /// @param dtype If given, `diag` is converted to this dtype.
    /// @param device The device of the tensor. If omitted, use the default device of the backend.
    /// @param tol Tolerance for converting / validating the diagonal entries.
    /// diagonal_as_block, diagonal_as_numpy
    ///     Inverse methods that recover the `diag` entries.
    [[nodiscard]] static Ptr from_diag_block(BlockBackend::BlockPtr diag,
                                             Space::Ptr leg,
                                             TensorBackend::Ptr backend = nullptr,
                                             std::optional<LegLabelList> labels = std::nullopt,
                                             std::optional<Dtype> dtype = std::nullopt,
                                             std::optional<std::string> device = std::nullopt,
                                             float64 tol = 1e-6);

    /// The identity map as a DiagonalTensor.
    ///
    /// @param leg, backend, labels Arguments for constructor of `DiagonalTensor`.
    /// @param dtype The dtype for the entries.
    /// @param device The device of the tensor. If omitted, use the default device of the backend.
    [[nodiscard]] static Ptr from_eye(Space::Ptr leg,
                                      TensorBackend::Ptr backend = nullptr,
                                      std::optional<LegLabelList> labels = std::nullopt,
                                      Dtype dtype = Dtype::Float64,
                                      std::optional<std::string> device = std::nullopt);

    /// Generate a sample from the complex normal distribution.
    ///
    /// The probability density is
    ///
    /// \f[
    ///     p(T) \propto \mathrm{exp}\left[
    ///         \frac{1}{2 \sigma^2} \mathrm{Tr} (T - \mathtt{mean}) (T - \mathtt{mean})^\dagger
    ///     \right]
    /// \f]
    ///
    /// @param leg, backend, labels Arguments for constructor of `DiagonalTensor`.
    /// @param mean The mean of the distribution. ``None`` is equivalent to zero mean.
    /// @param sigma The standard deviation of the distribution
    /// @param dtype The dtype for the entries.
    /// @param device The device of the tensor. If omitted, use the default device of the backend.
    [[nodiscard]] static Ptr from_random_normal(Space::Ptr leg = nullptr,
                                                TensorCPtr mean = nullptr,
                                                float64 sigma = 1.0,
                                                TensorBackend::Ptr backend = nullptr,
                                                std::optional<LegLabelList> labels = std::nullopt,
                                                Dtype dtype = Dtype::Complex128,
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
    /// @param leg, backend, labels Arguments for constructor of `DiagonalTensor`.
    /// @param dtype The dtype for the entries.
    /// @param device The device of the tensor. If omitted, use the default device of the backend.
    [[nodiscard]] static Ptr from_random_uniform(Space::Ptr leg,
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
    /// @param leg, backend, labels Arguments, like for constructor of `DiagonalTensor`.
    /// @param dtype If given, the resulting blocks from `func` are converted to this dtype.
    /// @param device If given, the resulting blocks are moved to that device. Per default, if
    /// `func` returns backend-specific blocks, their device is used and otherwise the default
    /// device of the backend. from_block_func
    [[nodiscard]] static Ptr from_sector_block_func(
      SectorBlockFactoryFn func,
      Space::Ptr leg,
      TensorBackend::Ptr backend = nullptr,
      std::optional<LegLabelList> labels = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      std::optional<std::string> device = std::nullopt);

    /// Create DiagonalTensor from a Tensor.
    ///
    /// @param tens Must have exactly two legs. Its diagonal entries ``tens[i, i]`` are used.
    /// @param tol Tolerance for checking if the `tens` is actually diagonal, in the sense that any
    /// "off-diagonal" free parameters that should vanish are smaller than this by magnitude. Set
    /// to ``None`` to disable the check.
    [[nodiscard]] static Ptr from_tensor(SymmetricTensorCPtr tens,
                                         std::optional<float64> tol = 1e-12);

    /// A zero tensor.
    ///
    /// @param leg, backend, labels Arguments for constructor of `DiagonalTensor`.
    /// @param dtype The dtype for the entries.
    /// @param device The device of the tensor. If ``None``, use the `default_device` of the block
    /// backend.
    [[nodiscard]] static Ptr from_zero(Space::Ptr leg,
                                       TensorBackend::Ptr backend = nullptr,
                                       std::optional<LegLabelList> labels = std::nullopt,
                                       Dtype dtype = Dtype::Complex128,
                                       std::optional<std::string> device = std::nullopt);

    /// Import DiagonalTensor from hdf5
    [[nodiscard]] static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                                       HighFive::Group& h5gr,
                                       std::string const& subpath);

    /// Export DiagonalTensor to hdf5 such that it can be re-imported with from_hdf5
    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;

    // --- Tensor / SymmetricTensor overrides ---

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
    [[nodiscard]] virtual DiagonalTensorPtr diagonal(bool check_offdiagonal = false) const;

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

    // --- Diagonal-specific API ---

    [[nodiscard]] virtual Ptr as_DiagonalTensor(bool guarantee_copy = false,
                                                std::optional<std::string> warning = std::nullopt);

    [[nodiscard]] virtual BlockBackend::BlockPtr diagonal_as_block(
      std::optional<Dtype> dtype = std::nullopt);

    [[nodiscard]] virtual py::array diagonal_as_numpy(py::object numpy_dtype = py::none());

    [[nodiscard]] virtual Ptr elementwise_almost_equal(DiagonalTensorCPtr other,
                                                       float64 rtol = 1e-5,
                                                       float64 atol = 1e-8);

    /// An elementwise function acting on a diagonal tensor.
    ///
    /// Applies ``func(self_block: Block, **func_kwargs) -> Block`` elementwise.
    /// Set ``maps_zero_to_zero=True`` to promise that ``func(0) == 0``.
    [[nodiscard]] virtual Ptr _elementwise_unary(BlockUnaryFn func,
                                                 bool maps_zero_to_zero = false);

    /// An elementwise function acting on two diagonal tensors.
    ///
    /// Applies ``func(self_block: Block, other_block: Block, **func_kwargs) -> Block``
    /// elementwise. Set ``partial_zero_is_zero=True`` to promise that ``func(0, any) == 0 ==
    /// func(any, 0)``.
    [[nodiscard]] virtual Ptr _elementwise_binary(DiagonalTensorCPtr other,
                                                  BlockBinaryFn func,
                                                  bool partial_zero_is_zero = false);

    /// Common implementation for the binary dunder methods ``__mul__`` etc.
    ///
    /// If `return_NotImplemented` is set, `NotImplemented` should be returned on a non-scalar
    /// and non-`Tensor` `other`.
    ///
    /// @param other Either a number or a DiagonalTensor.
    /// @param func The function with signature ``func(self_block: Block, other_block: Block) ->
    /// Block`` Scalars get passed the (0D) block representation of the scalar.
    /// @param operand A string representation of the operand, used in error messages
    /// @param right If this is the "right" version, i.e. ``func(other, self)``.
    [[nodiscard]] virtual Ptr _binary_operand(BlockBackend::Scalar other,
                                              BlockBinaryFn func,
                                              std::string const& operand,
                                              bool right = false);

    [[nodiscard]] virtual Ptr _binary_operand(DiagonalTensorCPtr other,
                                              BlockBinaryFn func,
                                              std::string const& operand,
                                              bool right = false);

    /// For a bool dtype, if all values are True. Raises for other dtypes.
    [[nodiscard]] virtual bool all() const;
    /// For a bool dtype, if any value is True. Raises for other dtypes.
    [[nodiscard]] virtual bool any() const;

    [[nodiscard]] virtual BlockBackend::Scalar max() const;
    [[nodiscard]] virtual BlockBackend::Scalar min() const;
    /// Identify a single sector that attains the minimum value.
    ///
    /// Unlike `argmin`, this works even if the symmetry cannot be dropped. The mask's small leg
    /// has total multiplicity one. On ties between sectors, the first sector in decomposition
    /// order is selected; within that sector, the first minimum in its block is selected.
    ///
    /// @param s If given, only the diagonal block of this charge sector is considered.
    /// @returns A mask projecting onto that single sector and the minimal value.
    ///
    /// argmin
    ///     Returns the index in the public computational basis instead, and requires that the
    ///     symmetry can be dropped.
    [[nodiscard]] virtual std::pair<std::shared_ptr<Mask>, BlockBackend::Scalar> sector_argmin(
      std::optional<Sector> s = std::nullopt) const;

    /// Index ``i0`` in the public computational basis of the minimum diagonal entry.
    ///
    /// Defined for real dtypes only. On ties, the first occurrence (in public basis
    /// order) is returned. If the minimum occurs within a sector of dimension greater than one,
    /// the first basis state within that multiplet is selected arbitrarily. Satisfies
    /// ``self[i0, i0] == self.min()`` when ``s`` is omitted and the symmetry can be dropped.
    ///
    /// sector_argmin
    ///     Also works when the symmetry cannot be dropped, and returns a projection Mask onto one
    ///     minimizing basis state together with the minimum value.
    ///
    /// @param s If given, only the diagonal block of this charge sector is considered. The
    /// returned index is still in the public basis of the full leg.
    [[nodiscard]] virtual int64 argmin(std::optional<Sector> s = std::nullopt) const;

    [[nodiscard]] virtual Ptr abs() const;
};

/// Special case of a `DiagonalTensor` that is exactly the identity map on its leg.
class Identity : public DiagonalTensor
{
  public:
    using Ptr = std::shared_ptr<Identity>;
    using CPtr = std::shared_ptr<const Identity>;

    Identity(Space::Ptr leg,
             TensorBackend::Ptr backend,
             Symmetry::Ptr symmetry,
             LegLabelList labels,
             Dtype dtype,
             std::string device);

    ~Identity() override = default;

    /// Perform sanity checks.
    void test_sanity() const override;

    [[nodiscard]] std::string class_name() const override;

    // Unsupported factories (TypeError in Python)
    static void unsupported_factory(char const* name);

    /// The identity map as a DiagonalTensor.
    ///
    /// @param leg, backend, labels Arguments for constructor of `DiagonalTensor`.
    /// @param dtype The dtype for the entries.
    /// @param device The device of the tensor. If omitted, use the default device of the backend.
    [[nodiscard]] static Ptr from_eye(Space::Ptr leg,
                                      TensorBackend::Ptr backend = nullptr,
                                      std::optional<LegLabelList> labels = std::nullopt,
                                      Dtype dtype = Dtype::Float64,
                                      std::optional<std::string> device = std::nullopt);

    /// Import Identity from hdf5
    [[nodiscard]] static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                                       HighFive::Group& h5gr,
                                       std::string const& subpath);

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

    [[nodiscard]] DiagonalTensor::Ptr as_DiagonalTensor(
      bool guarantee_copy = false,
      std::optional<std::string> warning = std::nullopt) override;

    [[nodiscard]] DiagonalTensor::Ptr _binary_operand(BlockBackend::Scalar other,
                                                      BlockBinaryFn func,
                                                      std::string const& operand,
                                                      bool right = false) override;

    [[nodiscard]] DiagonalTensor::Ptr _binary_operand(DiagonalTensorCPtr other,
                                                      BlockBinaryFn func,
                                                      std::string const& operand,
                                                      bool right = false) override;

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
    [[nodiscard]] DiagonalTensorPtr diagonal(bool check_offdiagonal = false) const override;

    [[nodiscard]] BlockBackend::BlockPtr diagonal_as_block(
      std::optional<Dtype> dtype = std::nullopt) override;

    [[nodiscard]] py::array diagonal_as_numpy(py::object numpy_dtype = py::none()) override;

    [[nodiscard]] DiagonalTensor::Ptr elementwise_almost_equal(DiagonalTensorCPtr other,
                                                               float64 rtol = 1e-5,
                                                               float64 atol = 1e-8) override;

    [[nodiscard]] DiagonalTensor::Ptr _elementwise_unary(BlockUnaryFn func,
                                                         bool maps_zero_to_zero = false) override;

    [[nodiscard]] DiagonalTensor::Ptr _elementwise_binary(
      DiagonalTensorCPtr other,
      BlockBinaryFn func,
      bool partial_zero_is_zero = false) override;

    /// Implementation of `__getitem__`.
    ///
    /// Can assume we have one non-negative integer index per leg.
    [[nodiscard]] BlockBackend::Scalar _get_item(std::vector<int64> const& idx) override;

    /// For a bool dtype, if all values are True. Raises for other dtypes.
    [[nodiscard]] bool all() const override;
    /// For a bool dtype, if any value is True. Raises for other dtypes.
    [[nodiscard]] bool any() const override;

    [[nodiscard]] BlockBackend::Scalar max() const override;
    [[nodiscard]] BlockBackend::Scalar min() const override;
    /// ``argmin`` is not supported for `Identity`.
    [[nodiscard]] int64 argmin(std::optional<Sector> s = std::nullopt) const override;

    [[nodiscard]] DiagonalTensor::Ptr abs() const override;

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
};

} // namespace lasyten
