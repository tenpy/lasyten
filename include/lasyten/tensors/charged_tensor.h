#pragma once

#include <lasyten/tensors/symmetric_tensor.h>

#include <lasyten/tools/hdf5.h>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lasyten {

/// Tensors which are not symmetric, but carry a well defined charge.
///
/// The main component is an invariant part, which is a `SymmetricTensor` that has an additional
/// charge leg (label ``"!"``) as ``domain.spaces[0]``. A particular state (i.e. a vector) on that
/// extra leg is specified as `charged_state`. It is (generally) not symmetric, and thus this
/// state is not a "tensor". The composite object of invariant part and this `charged_state` then
/// has a well-defined transformation behavior under the action of the symmetry group; unlike a
/// `SymmetricTensor`, which is invariant under the action, it transforms under the group
/// representation associated with the sectors of the additional leg.
///
/// To hide legs from algorithms without specifying a charged state, use `HiddenLegTensor`.
///
/// @param invariant_part The symmetry-invariant part. the charge leg is the its
/// ``domain.spaces[0]``.
/// @param charged_state A backend-specific block of shape ``(charge_leg.dim,)``, which specifies
/// a state on the charge leg. Must not be ``None``.
class ChargedTensor : public Tensor
{
  public:
    using Ptr = std::shared_ptr<ChargedTensor>;
    using CPtr = std::shared_ptr<const ChargedTensor>;

    /// Canonical label for the charge leg on the invariant part.
    static constexpr char const* _CHARGE_LEG_LABEL = "!";

    SymmetricTensor::Ptr invariant_part;
    /// Non-null block of shape ``(charge_leg.dim,)``.
    BlockBackend::BlockPtr charged_state;
    /// Usually an `ElementarySpace`; may be a `LegPipe` after
    /// `from_two_charge_legs` / ``combine_legs``.
    Leg::Ptr charge_leg;

    ChargedTensor(SymmetricTensor::Ptr invariant_part, BlockBackend::BlockPtr charged_state);

    ~ChargedTensor() override = default;

    /// Perform sanity checks.
    void test_sanity() const override;

    [[nodiscard]] std::string ascii_diagram_type_name() const override;
    [[nodiscard]] std::string class_name() const override;

    // --- helpers ---

    /// Helper function to build the domain of the invariant part.
    ///
    /// @param domain The domain of the ChargedTensor
    /// @param charge Specification for the charge_leg, either as a space or a single sector
    /// @returns inv_domain: TensorProduct The domain of the invariant part charge_leg: Space The
    /// charge_leg of the resulting ChargedTensor
    [[nodiscard]] static std::tuple<TensorProduct::Ptr, Space::Ptr> _parse_inv_domain(
      TensorProduct::Ptr domain,
      std::variant<ElementarySpace::Ptr, Sector> charge);

    /// Utility like `_init_parse_labels`, but also returns invariant part labels.
    [[nodiscard]] static std::tuple<LegLabelList, LegLabelList> _parse_inv_labels(
      std::optional<LegLabelList> labels,
      TensorProduct::Ptr const& codomain,
      TensorProduct::Ptr const& domain);

    /// If the `ChargedTensor` concept is well defined for the `symmetry`.
    [[nodiscard]] static bool supports_symmetry(Symmetry::Ptr const& symmetry);

    // --- factories ---

    /// Create a charged tensor with inv_part from `SymmetricTensor::from_block_func`.
    [[nodiscard]] static Ptr from_block_func(BlockFactoryFn func,
                                             std::variant<ElementarySpace::Ptr, Sector> charge,
                                             TensorProduct::Ptr codomain,
                                             TensorProduct::Ptr domain = nullptr,
                                             BlockBackend::BlockPtr charged_state = nullptr,
                                             TensorBackend::Ptr backend = nullptr,
                                             std::optional<LegLabelList> labels = std::nullopt,
                                             std::optional<Dtype> dtype = std::nullopt,
                                             std::optional<std::string> device = std::nullopt);

    /// Convert a dense block to a ChargedTensor, if possible.
    [[nodiscard]] static Ptr from_dense_block(
      BlockBackend::BlockPtr block,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain = nullptr,
      std::optional<std::variant<ElementarySpace::Ptr, Sector>> charge = std::nullopt,
      TensorBackend::Ptr backend = nullptr,
      std::optional<LegLabelList> labels = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      std::optional<std::string> device = std::nullopt,
      float64 tol = 1e-6,
      bool understood_braiding = false);

    /// Given a `vector` in single `space`, represent the components in a single given `sector`.
    [[nodiscard]] static Ptr from_dense_block_single_sector(
      BlockBackend::BlockPtr vector,
      Leg::Ptr space,
      Sector sector,
      TensorBackend::Ptr backend = nullptr,
      std::optional<std::string> label = std::nullopt,
      std::optional<std::string> device = std::nullopt);

    /// Like constructor, but deals with the case where invariant_part has only one leg.
    ///
    /// In that case, we return a scalar if the charged_state is specified and raise otherwise.
    [[nodiscard]] static std::variant<Ptr, BlockBackend::Scalar> from_invariant_part(
      SymmetricTensor::Ptr invariant_part,
      BlockBackend::BlockPtr charged_state = nullptr);

    /// Create a charged tensor from an invariant part with two charged legs.
    [[nodiscard]] static std::variant<Ptr, BlockBackend::Scalar> from_two_charge_legs(
      SymmetricTensor::Ptr invariant_part,
      BlockBackend::BlockPtr state1 = nullptr,
      BlockBackend::BlockPtr state2 = nullptr);

    /// A zero charged tensor.
    [[nodiscard]] static Ptr from_zero(TensorProduct::Ptr codomain,
                                       TensorProduct::Ptr domain,
                                       std::variant<ElementarySpace::Ptr, Sector> charge,
                                       BlockBackend::BlockPtr charged_state = nullptr,
                                       TensorBackend::Ptr backend = nullptr,
                                       std::optional<LegLabelList> labels = std::nullopt,
                                       Dtype dtype = Dtype::Complex128,
                                       std::optional<std::string> device = std::nullopt);

    /// Import ChargedTensor from hdf5
    [[nodiscard]] static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                                       HighFive::Group& h5gr,
                                       std::string const& subpath);

    /// Export ChargedTensor to hdf5 such that it can be re-imported with from_hdf5
    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;

    // --- Tensor overrides ---

    /// Convert to a tensor of the given dtype on the same device.
    ///
    /// @param dtype The dtype of the result.
    [[nodiscard]] Tensor::Ptr as_dtype(Dtype dtype) override;

    /// Convert to symmetric tensor, if possible.
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

    [[nodiscard]] std::vector<std::string> _repr_header_lines(
      std::string const& indent,
      bool use_symm_str = false) const override;

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

    /// Return the components associated with a single sector.
    ///
    /// Assumes a single-leg tensor living in a single sector and returns its components within
    /// that sector.
    [[nodiscard]] BlockBackend::BlockPtr to_dense_block_single_sector();
};

} // namespace lasyten
