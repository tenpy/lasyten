#pragma once

#include <lasyten/tensors/diagonal_tensor.h>

#include <lasyten/tools/hdf5.h>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lasyten {

/// A boolean mask that can be used to project or enlarge a leg.
///
/// Masks come in two versions: projections and inclusions. A projection Mask has a single leg, the
/// `large_leg` in its domain and maps it to a single leg, the `small_leg` in the
/// codomain. An inclusion Mask is the dagger of this projection Mask and maps from the small leg
/// in the domain to the large leg in the codomain::
///
///     |         ║                 │
///     |      ┏━━┷━━┓           ┏━━┷━━┓
///     |      ┃ M_p ┃    OR     ┃ M_i ┃
///     |      ┗━━┯━━┛           ┗━━┯━━┛
///     |         │                 ║
///
/// A Mask places restrictions on the basis order of the respective legs. For a projection Mask,
/// the kept basis elements from the large leg need to appear in their original order in the small
/// leg. Analogously, for an inclusion, the basis elements from the small leg need to be embedded
/// into the large leg in their original order. This restricts
/// the `basis_perm` of the legs, see notes below.
/// Most classmethods that are used to build Masks take care of this for you.
///
/// Attributes:
///
/// is_projection: bool
///     If the Mask is a projection or inclusion map (see class docstring above).
///
/// @param data The numerical data (i.e. boolean flags) comprising the mask. type is
/// backend-specific. Should have boolean dtype.
/// @param space_in The single space of the domain. This is the large leg for projections or the
/// small leg for inclusions.
/// @param space_out The single space of the codomain This is the small leg for projections or the
/// large leg for inclusions.
/// @param is_projection If this Mask is a projection (from large to small) map. Otherwise it is in
/// inclusion map (from small to large). Required if ``space_in == space_out``, since it is
/// ambiguous in that case.
/// @param backend The backend of the tensor.
/// @param labels Specify the labels for the legs. Can either give two lists, one for the codomain,
/// one for the domain. Or a single flat list for all legs in the order of the `legs`, such that
/// ``[codomain_labels, domain_labels]`` is equivalent to ``[*codomain_labels,
/// *reversed(domain_labels)]``.
///
/// Notes:
///
/// The `basis_perm` of the legs is constrained by the
/// requirements of the Mask, and in particular *depending on the data* as follows;
/// The following explanation is intuitive only for a projection Mask but also applies to
/// inclusions. Taking the ordered set of basis elements, permuting it by the large legs basis
/// perm, then discarding some of them according to the mask data, and finally permuting the
/// remaining elements back by the (inverse) small leg perm should result in a basis of the small
/// leg, where the relative ordering of elements is preserved.
///
/// In code, this means ::
///
///     ranks =
///     self.large_leg.basis_perm[mask_in_internal_basis][self.small_leg.inverse_basis_perm]
///
/// In particular, the basis permutation of the small leg is uniquely determined by the
/// permutation of the large leg and the mask data.
///
/// Consider the following valid example, assuming for simplicity only one one-dim. sector ::
///
///     large_leg_perm = [2, 4, 0, 1, 3]
///     mask_in_internal_basis = [True, True, False, True, False]
///     # mask_in_public_basis = [False, True, True, False, True]
///     small_leg_perm = [1, 2, 0]
///     small_leg_perm_inv = [2, 0, 1]
///
/// Which maps an ordered basis as follows ::
///
///     {e0, e1, e2, e3, e4}
///     ---large_leg_perm--> {e2, e4, e0, e1, e3}
///     ---mask_in_internal_basis--> {e2, e4, e1}
///     ---small_leg_perm_inv--> {e1, e2, e4}
///
/// Such that the result is ordered.
class Mask : public Tensor
{
  public:
    using Ptr = std::shared_ptr<Mask>;
    using CPtr = std::shared_ptr<const Mask>;

    /// Float / complex dtypes are forbidden (bool only). Matches Python ``_forbidden_dtypes``.
    static std::vector<Dtype> _forbidden_dtypes;

    bool is_projection = true;
    TensorBackend::DataPtr data;

    Mask(TensorBackend::DataPtr data,
         Space::Ptr space_in,
         Space::Ptr space_out,
         bool is_projection,
         TensorBackend::Ptr backend,
         Symmetry::Ptr symmetry,
         LegLabelList labels,
         std::string device);

    ~Mask() override = default;

    [[nodiscard]] std::vector<Dtype> const& forbidden_dtypes() const override;

    /// Perform sanity checks.
    void test_sanity() const override;

    [[nodiscard]] std::string ascii_diagram_type_name() const override;
    [[nodiscard]] std::string class_name() const override;

    /// The large leg (domain for projection, codomain for inclusion).
    [[nodiscard]] ElementarySpace::Ptr large_leg() const;

    /// The small leg (codomain for projection, domain for inclusion).
    [[nodiscard]] ElementarySpace::Ptr small_leg() const;

    // --- factories ---

    /// The identity map as a Mask, i.e. the mask that keeps all states and discards none.
    ///
    /// @param leg The single leg for the Mask, equal to both its small and large leg.
    /// @param is_projection, backend, labels Arguments, like for constructor of `Mask`.
    /// @param device The device of the tensor. If omitted, use the default device of the backend.
    /// from_zero
    ///     The projection Mask that discards all states and keeps none.
    [[nodiscard]] static Ptr from_eye(Space::Ptr leg,
                                      bool is_projection = true,
                                      TensorBackend::Ptr backend = nullptr,
                                      std::optional<LegLabelList> labels = std::nullopt,
                                      std::optional<std::string> device = std::nullopt);

    /// Create a projection Mask from a boolean block.
    ///
    /// To get the related inclusion Mask, use `dagger`.
    ///
    /// The small leg of the projection is fully determined by the large leg and by the boolean
    /// data. In particular, its basis permutation is such that the kept basis elements from the
    /// large leg appear in order.
    ///
    /// @param block_mask A boolean Block indicating for each basis element of the public basis, if
    /// it is kept.
    /// @param large_leg The large leg, in the domain of the projection
    /// @param backend, labels Arguments, like for the constructor
    /// @param device The device of the tensor. If omitted, use the default device of the backend.
    [[nodiscard]] static Ptr from_block_mask(BlockBackend::BlockPtr block_mask,
                                             Space::Ptr large_leg,
                                             TensorBackend::Ptr backend = nullptr,
                                             std::optional<LegLabelList> labels = std::nullopt,
                                             std::optional<std::string> device = std::nullopt);

    /// Create a projection Mask from a boolean DiagonalTensor.
    ///
    /// The resulting mask keeps exactly those basis elements for which the entry of `diag` is
    /// ``True``. To get the related inclusion Mask, use the `dagger`.
    ///
    /// The small leg of the projection is fully determined by the large leg and by `diag`.
    /// In particular, its basis permutation is such that those basis elements from the large leg
    /// that are kept appear in order.
    [[nodiscard]] static Ptr from_DiagonalTensor(DiagonalTensorCPtr diag);

    /// Create a projection Mask from the indices that are kept.
    ///
    /// To get the related inclusion Mask, use `dagger`.
    ///
    /// The small leg of the projection is fully determined by the large leg and by the `indices`.
    /// In particular, its basis permutation is such that those basis elements from the large leg
    /// that are kept appear in order.
    ///
    /// @param indices Valid index/indices for a 1D numpy array. The elements of the public basis
    /// of `large_leg` with these indices are kept by the projection.
    /// @param large_leg, backend, labels Same as for `__init__`.
    /// @param device The device of the tensor. If omitted, use the default device of the backend.
    [[nodiscard]] static Ptr from_indices(py::object indices,
                                          Space::Ptr large_leg,
                                          TensorBackend::Ptr backend = nullptr,
                                          std::optional<LegLabelList> labels = std::nullopt,
                                          std::optional<std::string> device = std::nullopt);

    /// Create a random projection Mask.
    ///
    /// To get the related inclusion Mask, use `dagger`.
    ///
    /// @param large_leg The large leg, in the domain of the projection
    /// @param small_leg The small leg. If given, must be a subspace of the `large_leg` with
    /// compatible basis order (see notes in class docstring of `Mask`). If ``None``, a small leg
    /// is randomly generated, according to `p_keep` and `min_keep`.
    /// @param backend, labels Arguments, like for the constructor
    /// @param p_keep If `small_leg` is not given, the probability that any single sector is kept.
    /// Is ignored if `small_leg` is given, since it determines the number of kept sectors.
    /// @param min_keep If `small_leg` is not given, the minimum number of sectors kept. Is ignored
    /// of `small_leg` is given.
    /// @param device The device of the tensor. If omitted, use the default device of the backend.
    /// @param np_random Optional numpy random generator used when sampling the mask.
    [[nodiscard]] static Ptr from_random(Space::Ptr large_leg,
                                         Space::Ptr small_leg = nullptr,
                                         TensorBackend::Ptr backend = nullptr,
                                         float64 p_keep = 0.5,
                                         int64 min_keep = 0,
                                         std::optional<LegLabelList> labels = std::nullopt,
                                         std::optional<std::string> device = std::nullopt,
                                         py::object np_random = py::none());

    /// The zero projection Mask, that discards all states and keeps none.
    ///
    /// To get the related inclusion Mask, use `dagger`.
    ///
    /// @param large_leg The large leg, in the domain of the projection
    /// @param backend, labels Arguments, like for the constructor
    /// @param device The device of the tensor. If ``None``, use the `default_device` of the block
    /// backend. from_eye
    ///     The projection (or inclusion) Mask that keeps all states
    [[nodiscard]] static Ptr from_zero(Space::Ptr large_leg,
                                       TensorBackend::Ptr backend = nullptr,
                                       std::optional<LegLabelList> labels = std::nullopt,
                                       std::optional<std::string> device = std::nullopt);

    /// Import Mask from hdf5
    [[nodiscard]] static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                                       HighFive::Group& h5gr,
                                       std::string const& subpath);

    /// Export Mask to hdf5 such that it can be re-imported with from_hdf5
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

    /// Like `as_SymmetricTensor`, with an explicit result dtype (Python Mask API).
    [[nodiscard]] SymmetricTensorPtr as_SymmetricTensor(bool guarantee_copy,
                                                        std::optional<std::string> warning,
                                                        Dtype dtype);

    /// Copy the tensor.
    ///
    /// @param deep If the copy should be deep. A shallow copy is a new instance with the same
    /// data.
    /// @param device The device for the result. Per default, use the same device as `self`.
    /// @param dtype The dtype of the result. Per default, use the same dtype as `self`.
    [[nodiscard]] Tensor::Ptr copy(bool deep = true,
                                   std::optional<std::string> device = std::nullopt,
                                   std::optional<Dtype> dtype = std::nullopt) override;

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
    [[nodiscard]] Tensor::Ptr dagger() const override;

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

    /// Convert to a numpy array
    [[nodiscard]] py::array to_numpy(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      py::object numpy_dtype = py::none(),
      bool understood_braiding = false);

    // --- Mask-specific API ---

    [[nodiscard]] DiagonalTensor::Ptr as_DiagonalTensor(Dtype dtype = Dtype::Complex128);

    [[nodiscard]] BlockBackend::BlockPtr as_block_mask();

    [[nodiscard]] py::array as_numpy_mask();

    /// If the mask keeps all basis elements
    [[nodiscard]] bool all() const;

    /// If the mask keeps any basis elements
    [[nodiscard]] bool any() const;

    /// Alias for `orthogonal_complement`
    [[nodiscard]] Ptr logical_not();

    /// The "opposite" Mask, that keeps exactly what self discards and vv.
    [[nodiscard]] Ptr orthogonal_complement();

    /// Utility for binary boolean ops (``&``, ``|``, ``^``, ``==``, ``!=``).
    [[nodiscard]] Ptr _binary_operand(bool other, BlockBinaryFn func, std::string const& operand);

    [[nodiscard]] Ptr _binary_operand(MaskCPtr other,
                                      BlockBinaryFn func,
                                      std::string const& operand);

    [[nodiscard]] Ptr _unary_operand(BlockUnaryFn func);
};

} // namespace lasyten
