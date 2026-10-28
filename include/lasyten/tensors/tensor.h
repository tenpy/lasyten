#pragma once

#include <lasyten/backends/backend_factory.h>
#include <lasyten/backends/tensor_backend.h>
#include <lasyten/block_backend/block_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/config.h>
#include <lasyten/symmetries/exceptions.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/symmetry.h>
#include <lasyten/tensors/forward_declare.h>
#include <lasyten/tensors/labels.h>
#include <lasyten/tensors/vector_like.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

/// Common base class for tensors.
///
/// TODO elaborate
///
/// The legs of the tensor (spaces of the domain or codomain) can be referred to either via
/// string labels (see `tensor_leg_labels` and the `labels` attribute) or via integer
/// positional indices. Both allow you to be ignorant of the distinction between domain and
/// codomain (see `tensors_as_maps`). For the integer indices, we refer to the position of a given
/// legs in the `legs`. E.g. if ``codomain == [V, W, Z]`` and ``domain == [X, Y]``, we have ``legs
/// == [V, W, Z, Y.dual, X.dual]`` and indices ``1`` and ``-4`` both refer to the
/// ``W`` leg in the codomain, while indices ``3`` and ``-2`` both refer to the ``X`` leg in the
/// domain. Graphically, the leg indices are arranged as follows::
///
///     |      11  10   9   8   7   6
///     |      ┏┷━━━┷━━━┷━━━┷━━━┷━━━┷┓
///     |      ┃          T          ┃
///     |      ┗┯━━━┯━━━┯━━━┯━━━┯━━━┯┛
///     |       0   1   2   3   4   5
///
/// A similar graphical representation is available as `ascii_diagram` and can be
/// printed to stdout using `dbg`.
///
/// Attributes:
///
/// codomain, domain : TensorProduct
///     The domain and codomain of the tensor. See also `legs` and `tensors_as_maps`.
/// backend : TensorBackend
///     The backend of the tensor.
/// symmetry : Symmetry
///     The symmetry of the tensor.
/// num_legs : int
///     The total number of legs in the domain and codomain.
/// dtype : Dtype
///     The dtype of tensor entries. Note that a real dtype does not necessarily imply that
///     the result of `to_dense_block` is real.
/// shape: tuple of int
///     The dimension of each of the `legs`.
class Tensor
  : public VectorLike
  , public std::enable_shared_from_this<Tensor>
{
  public:
    using Ptr = std::shared_ptr<Tensor>;
    using CPtr = std::shared_ptr<const Tensor>;

    /// Dtypes rejected by `test_sanity` (Python ``_forbidden_dtypes``).
    static std::vector<Dtype> _forbidden_dtypes;

    /// Subclass-overridable forbidden dtypes (e.g. DiagonalTensor allows bool).
    [[nodiscard]] virtual std::vector<Dtype> const& forbidden_dtypes() const;

    /// Leg labels (composition; empty string = unset).
    LegLabels labels;
    /// Number of legs (== `labels.size()`). Kept in sync with `labels`.
    int64 num_legs = 0;

    TensorProduct::Ptr codomain;
    TensorProduct::Ptr domain;
    TensorBackend::Ptr backend;
    Symmetry::Ptr symmetry;
    Dtype dtype = Dtype::Float64;
    std::string device;
    /// Dimension of each of the `legs` (codomain dims, then reversed domain dims).
    std::vector<float64> shape;

    /// Construct from already-parsed C++ inputs.
    Tensor(TensorProduct::Ptr codomain,
           TensorProduct::Ptr domain,
           TensorBackend::Ptr backend,
           Symmetry::Ptr symmetry,
           LegLabelList labels,
           Dtype dtype,
           std::string device);

    /// Construct from `parse_tensor_init` (py-object subclass factories).
    struct InitParsed
    {
        TensorProduct::Ptr codomain;
        TensorProduct::Ptr domain;
        TensorBackend::Ptr backend;
        Symmetry::Ptr symmetry;
        LegLabelList labels;
    };
    Tensor(InitParsed init, Dtype dtype, std::string device);

    ~Tensor() override = default;

    Tensor(Tensor const&) = delete;
    Tensor& operator=(Tensor const&) = delete;
    Tensor(Tensor&&) = delete;
    Tensor& operator=(Tensor&&) = delete;

    /// Common input parsing for ``__init__`` methods of tensor classes.
    ///
    /// Also checks if they are compatible. Sequence-of-spaces conversion is done in pybind
    /// (or `parse_tensor_init_args`).
    ///
    /// @returns codomain, domain The (co)domain, converted to `TensorProduct` if needed.
    ///     backend The given backend, or the default compatible with `symmetry`.
    ///     symmetry The symmetry of the domain and codomain.
    static std::tuple<TensorProduct::Ptr, TensorProduct::Ptr, TensorBackend::Ptr, Symmetry::Ptr>
    _init_parse_args(TensorProduct::Ptr codomain,
                     TensorProduct::Ptr domain,
                     TensorBackend::Ptr backend);

    /// Parse the various allowed input formats for labels to the format of `labels`.
    ///
    /// Also supports a special case for input formats of endomorphisms (maps where domain
    /// and codomain coincide), where a flat list of labels for the codomain can be given,
    /// and the domain labels are auto-filled with the respective dual labels.
    static LegLabelList _init_parse_labels(std::optional<LegLabelList> labels,
                                           TensorProduct::Ptr const& codomain,
                                           TensorProduct::Ptr const& domain,
                                           bool is_endomorphism = false);

    /// Perform sanity checks.
    virtual void test_sanity() const;

    /// An ascii representation of the tensor.
    ///
    /// It shows the type, leg labels, leg dimensions and leg arrows.
    ///
    /// Consider the following example::
    ///
    ///     |     123   123   132   123
    ///     |       ^     v     v     ^
    ///     |       a     b     c     d
    ///     |   ┏━━━┷━━━━━┷━━━━━┷━━━━━┷━━━┓
    ///     |   ┃          TEXT           ┃
    ///     |   ┗┯━━━━━┯━━━━━┯━━━━━┯━━━━━┯┛
    ///     |    i     h     g     f     e
    ///     |    ^     v     ^     ^     v
    ///     |   42   777    11     2     3
    [[nodiscard]] virtual std::string ascii_diagram() const;

    /// Convert to a tensor of the given dtype on the same device.
    ///
    /// @param dtype The dtype of the result.
    [[nodiscard]] virtual Ptr as_dtype(Dtype dtype) = 0;

    /// Convert to a `SymmetricTensor`, if possible.
    ///
    /// @param guarantee_copy If already a SymmetricTensor, we do *not* make a copy by default. Set
    /// this flag to ``True`` to guarantee a copy.
    /// @param warning If given, and if the conversion is non-trivial (i.e. if it was not already a
    /// SymmetricTensor to begin with), a warning with this text is issued.
    [[nodiscard]] virtual SymmetricTensorPtr as_SymmetricTensor(
      bool guarantee_copy = false,
      std::optional<std::string> warning = std::nullopt) = 0;

    /// Copy the tensor.
    ///
    /// @param deep If the copy should be deep. A shallow copy is a new instance with the same
    /// data.
    /// @param device The device for the result. Per default, use the same device as `self`.
    /// @param dtype The dtype of the result. Per default, use the same dtype as `self`.
    [[nodiscard]] virtual Ptr copy(bool deep = true,
                                   std::optional<std::string> device = std::nullopt,
                                   std::optional<Dtype> dtype = std::nullopt) = 0;

    /// Convert to a tensor with a different backend.
    ///
    /// @param backend The backend of the result.
    /// @param dtype The dtype of the result. Per default, use the same dtype as `self`.
    /// @param device The device for the result. Per default, use the same device as `self`.
    [[nodiscard]] virtual Ptr to_backend(TensorBackend::Ptr backend,
                                         std::optional<Dtype> dtype = std::nullopt,
                                         std::optional<std::string> device = std::nullopt) = 0;

    /// Convert to a dense block of the backend, if possible.
    ///
    /// This corresponds to "forgetting" the symmetry structure and is only possible if the
    /// symmetry `can_be_dropped`.
    /// The result is a backend-specific block, e.g. a numpy array if the block backend is a
    /// `NumpyBlockBackend` or a torch Tensor if the backend is a `TorchBlockBackend`.
    ///
    /// @param leg_order If given, the leg of the resulting block are permuted to match this leg
    /// order. For a `HiddenLegTensor`, it may contain only the public legs; the hidden legs are
    /// then the last legs of the block, in their original order.
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
    [[nodiscard]] virtual BlockBackend::BlockPtr to_dense_block(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      bool understood_braiding = false) = 0;

    /// The labels that refer to legs in the codomain.
    [[nodiscard]] LegLabelList codomain_labels() const;

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
    [[nodiscard]] virtual Ptr dagger() const;

    /// The labels that refer to legs in the domain.
    [[nodiscard]] LegLabelList domain_labels() const;

    /// If any of the legs is a pipe
    [[nodiscard]] bool has_pipes() const;

    /// The `dagger`
    [[nodiscard]] Ptr hc() const { return dagger(); }

    /// All legs of the tensor.
    ///
    /// These the spaces of the codomain, followed by the duals of the domain spaces
    /// *in reverse order*.
    /// If we permute all legs to the codomain, we would get these spaces, i.e.::
    ///
    ///     tensor.legs == tensor.permute_legs(codomain=range(tensor.num_legs)).codomain.spaces
    ///
    /// See `tensors_as_maps`.
    [[nodiscard]] std::vector<Leg::Ptr> legs() const;

    /// Move tensor to a given device, *in place*.
    virtual void move_to_device(std::string device) = 0;

    /// How many of the legs are in the codomain. See `tensors_as_maps`.
    [[nodiscard]] int64 num_codomain_legs() const;

    /// How many of the legs are in the domain. See `tensors_as_maps`.
    [[nodiscard]] int64 num_domain_legs() const;

    /// Number of flat legs in the codomain.
    [[nodiscard]] int64 num_codomain_flat_legs() const;

    /// Number of flat legs in the domain.
    [[nodiscard]] int64 num_domain_flat_legs() const;

    /// Total number of flat legs of self.
    [[nodiscard]] int64 num_flat_legs() const;

    /// The number of free parameters for the given legs.
    ///
    /// This is the dimension of the space of symmetry-preserving tensors with the given legs.
    [[nodiscard]] int64 num_parameters() const;

    /// The number of entries of a dense block representation of self.
    ///
    /// This is only defined if ``self.symmetry.can_be_dropped``.
    /// In that case, it is the number of entries of `to_dense_block`.
    [[nodiscard]] int64 size() const;

    /// The `transpose`.
    [[nodiscard]] virtual Ptr T() const;

    /// Implementation of `__getitem__`.
    ///
    /// Can assume we have one non-negative integer index per leg.
    [[nodiscard]] virtual BlockBackend::Scalar _get_item(std::vector<int64> const& idx) = 0;

    /// Parse a leg index or a leg label.
    ///
    /// @param idx An index referring to one of the `legs` *or* a label.
    /// @returns in_domain: bool If the leg is in the domain. co_domain_idx: int The index of the
    /// leg in the (co-)domain legs_idx: int The index of the leg in `legs`. Same as input ``idx``,
    /// except it is guaranteed to be in ``range(num_legs)``.
    [[nodiscard]] Leg::Ptr _as_codomain_leg(std::variant<int64, std::string> idx) const;

    /// Return the leg, as if it was moved to the domain.
    ///
    /// May be a `LegPipe` (not only a `Space`).
    [[nodiscard]] Leg::Ptr _as_domain_leg(std::variant<int64, std::string> idx) const;

    /// Print `ascii_diagram` to stdout.
    void dbg() const;

    /// Parse a leg index or a leg label.
    ///
    /// Returns ``(in_domain, co_domain_idx, legs_idx)``.
    [[nodiscard]] std::tuple<bool, int64, int64> _parse_leg_idx(
      std::variant<int64, std::string> which_leg) const;

    [[nodiscard]] virtual std::vector<std::string> _repr_header_lines(
      std::string const& indent,
      bool use_symm_str = false) const;

    /// Basically ``self.legs[which_leg]``, but allows labels and multiple indices.
    [[nodiscard]] Leg::Ptr get_leg(std::variant<int64, std::string> which_leg) const;
    [[nodiscard]] std::vector<Leg::Ptr> get_leg(
      std::vector<std::variant<int64, std::string>> const& which_legs) const;

    /// Get the specified leg from the domain or codomain.
    ///
    /// May be a `LegPipe` (not only a `Space`).
    ///
    /// This is the same as `get_leg` if the leg is in the codomain, and the respective
    /// dual if the leg is in the domain.
    [[nodiscard]] Leg::Ptr get_leg_co_domain(std::variant<int64, std::string> which_leg) const;
    [[nodiscard]] std::vector<Leg::Ptr> get_leg_co_domain(
      std::vector<std::variant<int64, std::string>> const& which_legs) const;

    /// Convert to a numpy array.
    [[nodiscard]] py::array to_numpy(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      py::object numpy_dtype = py::none(),
      bool understood_braiding = false);

    [[nodiscard]] virtual std::string __repr__() const;
    [[nodiscard]] virtual std::string __str__() const;

    /// Subclass short name for `ascii_diagram` (``Symm``, ``Diag``, …).
    [[nodiscard]] virtual std::string ascii_diagram_type_name() const;

    /// Python ``type(self).__name__`` for ``__repr__`` / ``__str__``.
    [[nodiscard]] virtual std::string class_name() const;

    // VectorLike
    [[nodiscard]] VectorLike::Ptr clone() const override;
    [[nodiscard]] Dtype vector_dtype() const override;
    [[nodiscard]] std::string vector_device() const override;
    [[nodiscard]] TensorBackend::Ptr vector_backend() const override;
    [[nodiscard]] BlockBackend::Scalar vector_norm() const override;
    [[nodiscard]] BlockBackend::Scalar vector_inner(VectorLike::CPtr other,
                                                    bool do_dagger = true) const override;
    [[nodiscard]] VectorLike::Ptr scaled(BlockBackend::Scalar const& a) const override;
    [[nodiscard]] VectorLike::Ptr axpy(BlockBackend::Scalar const& a,
                                       VectorLike::CPtr other) const override;
    [[nodiscard]] bool compatible_with(VectorLike::CPtr other) const override;
};

/// Convert a `TensorProduct`, a sequence of legs, or ``None`` (empty product).
/// Used by pybind and remaining py-object factory overloads.
TensorProduct::Ptr tensor_product_from_python(py::object obj, Symmetry::Ptr symmetry = nullptr);

/// Python-flexible (co)domain parsing, then `Tensor::_init_parse_args`.
std::tuple<TensorProduct::Ptr, TensorProduct::Ptr, TensorBackend::Ptr, Symmetry::Ptr>
parse_tensor_init_args(py::object codomain, py::object domain, TensorBackend::Ptr backend);

/// Python-flexible label parsing (``None``, nested lists, endomorphism shorthand).
LegLabelList parse_tensor_init_labels(py::object labels,
                                      TensorProduct::Ptr const& codomain,
                                      TensorProduct::Ptr const& domain,
                                      bool is_endomorphism = false);

/// Parse (co)domain, backend, and labels together for py-object subclass constructors.
Tensor::InitParsed parse_tensor_init(py::object codomain,
                                     py::object domain,
                                     TensorBackend::Ptr backend,
                                     py::object labels,
                                     bool is_endomorphism = false);

} // namespace lasyten
