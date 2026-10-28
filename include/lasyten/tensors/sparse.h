#pragma once

#include <lasyten/backends/tensor_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/lasyten.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/tensors/direct_sum.h>
#include <lasyten/tensors/forward_declare.h>
#include <lasyten/tensors/vector_like.h>

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lasyten {

using VectorLabels = std::optional<LegLabelList>;
inline constexpr float64 kGramSchmidtDefaultRcond = 1.0e-14;

bool same_legs(std::vector<Leg::Ptr> const& legs1, std::vector<Leg::Ptr> const& legs2);

/// Base class for a linear operator acting on lasyten tensors.
///
/// Attributes:
///
/// vector_legs : list of Space
///     The legs of tensors that this operator can act on.
/// vector_labels : list of str, or unset
///     Labels of the vectors that this operator can act on (empty string = unset label),
///     or unset for the whole list.
/// dtype : Dtype
///     The dtype of a full representation of the operator
/// acts_on : list of str
///     Labels of the state on which the operator can act. NB: Class attribute.
class LinearOperator
{
  public:
    using Ptr = std::shared_ptr<LinearOperator>;
    using CPtr = std::shared_ptr<const LinearOperator>;

    std::vector<Leg::Ptr> vector_legs;
    VectorLabels vector_labels;
    Dtype dtype = Dtype::Float64;

    static py::object& acts_on;

    LinearOperator(std::vector<Leg::Ptr> vector_legs = {},
                   Dtype dtype = Dtype::Float64,
                   VectorLabels vector_labels = std::nullopt);
    virtual ~LinearOperator() = default;

    /// Apply the linear operator to a "vector".
    ///
    /// We consider as vectors all `VectorLike` objects, including
    /// `Tensor` (any rank) and `DirectSum`.
    /// The result of `matvec` must live in the same vector space as `vec`.
    [[nodiscard]] virtual VectorLike::Ptr matvec(VectorLike::CPtr vec) = 0;
    /// Compute a full tensor representation of the linear operator.
    ///
    /// @returns A tensor `t` with ``2 * N`` legs ``[a1, a2, ..., aN, aN*, ..., a2*, a1*]``, where
    /// ``[a1, a2, ..., aN]`` are the legs of the vectors this operator acts on. S.t.
    /// ``self.matvec(vec)`` is equivalent to ``tdot(t, vec, [N, ..., 2*N-1], [N-1,...,0])``.
    [[nodiscard]] virtual TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) = 0;
    /// The tensor representation of self, reshaped to a matrix.
    [[nodiscard]] TensorPtr to_matrix(TensorBackend::Ptr backend = nullptr);
    /// Return the hermitian conjugate operator.
    ///
    /// If `self` is hermitian, subclasses *can* choose to implement this to define
    /// the adjoint operator of `self` to be `self`.
    [[nodiscard]] virtual Ptr adjoint();
};

/// Linear operator defined by a two-leg tensor with contractible legs.
///
/// The matvec is defined by contracting one of the two legs of this tensor with the vector.
/// This class is effectively a thin wrapper around tensors that allows them to be used as inputs
/// for sparse linear algebra routines, such as lanczos.
///
/// @param tensor The tensor that is contracted with the vector on matvec
/// @param which_leg Which leg of `tensor` is to be contracted on matvec
class TensorLinearOperator : public LinearOperator
{
  public:
    using Ptr = std::shared_ptr<TensorLinearOperator>;

    SymmetricTensorPtr tensor;
    int64 which_leg = 1;
    int64 other_leg = 0;

    TensorLinearOperator(SymmetricTensorPtr tensor,
                         std::variant<int64, std::string> which_leg = int64(-1));

    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
    [[nodiscard]] LinearOperator::Ptr adjoint() override;
};

/// Base class for wrapping around another `LinearOperator`.
///
/// The wrapped operator is stored as `original_operator`.
/// Use `unwrapped` to recover the innermost operator.
///
/// @warning If there are multiple levels of wrapping operators, the order might be critical
///     to get correct results; e.g. `ProjectedLinearOperator` needs to be the outer-most
///     wrapper to produce correct results and/or be efficient.
///
/// @param original_operator The original operator implementing the `matvec`.
class LinearOperatorWrapper : public LinearOperator
{
  public:
    using Ptr = std::shared_ptr<LinearOperatorWrapper>;

    LinearOperator::Ptr original_operator;

    explicit LinearOperatorWrapper(LinearOperator::Ptr original_operator);

    /// Return the original `LinearOperator`
    ///
    /// By default, unwrapping is done recursively, such that the result is *not* a
    /// `LinearOperatorWrapper`.
    [[nodiscard]] LinearOperator::Ptr unwrapped(bool recursive = true) const;
    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
    [[nodiscard]] LinearOperator::Ptr adjoint() override;
};

/// The sum of multiple operators.
class SumLinearOperator : public LinearOperatorWrapper
{
  public:
    using Ptr = std::shared_ptr<SumLinearOperator>;

    std::vector<LinearOperator::Ptr> more_operators;

    SumLinearOperator(LinearOperator::Ptr original_operator,
                      std::vector<LinearOperator::Ptr> more_operators = {});

    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
    [[nodiscard]] LinearOperator::Ptr adjoint() override;
};

/// A shifted operator, i.e. ``original_operator + shift * identity``.
///
/// This can be useful e.g. for better Lanczos convergence.
class ShiftedLinearOperator : public LinearOperatorWrapper
{
  public:
    using Ptr = std::shared_ptr<ShiftedLinearOperator>;

    complex128 shift = complex128(0.);

    ShiftedLinearOperator(LinearOperator::Ptr original_operator, complex128 shift);

    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
    [[nodiscard]] LinearOperator::Ptr adjoint() override;
};

/// Projected version ``P H P + penalty * (1 - P)`` of an original operator ``H``.
///
/// The projector ``P = 1 - sum_o |o> <o|`` is given in terms of a set `ortho_vecs` of vectors
/// ``|o>``.
///
/// The result is that all vectors from the subspace spanned by the `ortho_vecs` are eigenvectors
/// with eigenvalue `penalty`, while the eigensystem in the "rest" (i.e. in the orthogonal
/// complement to that subspace) remains unchanged.
///
/// This can be used to exclude the `ortho_vecs` from extremal eigensolvers, i.e. to find
/// the extremal eigenvectors among those that are orthogonal to the `ortho_vecs`.
/// In previous versions of tenpy, this behavior was achieved by an argument called
/// `orthogonal_to`. If this is done, at least for krylov-based eigensolvers such as lanczos, the
/// penalty should be chosen such that the `ortho_vecs` are somewhere in the bulk of the spectrum.
/// This is because lanczos has best convergence for the extremal eigenvalues and we want to
/// converge the solutions well, not the `ortho_vecs`. E.g. for a typical Hamiltonian with a
/// spectrum symmetric around zero, ``project_operator=True`` and ``penalty=None`` shifts the
/// `ortho_vecs` to eigenvalue zero, thus fulfilling this criterion. However, for operators with
/// e.g. strictly positive spectrum, this prescription might fail.
///
/// @param original_operator The original operator, denoted ``H`` in the summary above.
/// @param ortho_vecs The list of vectors spanning the projected space. They need not be
/// orthonormal, as Gram-Schmidt is performed on them explicitly.
/// @param project_operator If False (True per default), the projection of the operator ``H -> P H
/// P`` is skipped and ``H + penalty * (1 - P)`` is represented instead.
/// @param penalty See summary above. Defaults to ``None``, which is equivalent to ``0.``.
class ProjectedLinearOperator : public LinearOperatorWrapper
{
  public:
    using Ptr = std::shared_ptr<ProjectedLinearOperator>;

    std::vector<VectorLike::Ptr> ortho_vecs;
    bool project_operator = true;
    std::optional<complex128> penalty = std::nullopt;

    ProjectedLinearOperator(LinearOperator::Ptr original_operator,
                            std::vector<VectorLike::Ptr> ortho_vecs,
                            bool project_operator = true,
                            std::optional<complex128> penalty = std::nullopt);

    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
    [[nodiscard]] LinearOperator::Ptr adjoint() override;
};

/// Block-diagonal operator acting componentwise on a `DirectSum`.
class DirectSumLinearOperator : public LinearOperator
{
  public:
    using Ptr = std::shared_ptr<DirectSumLinearOperator>;

    std::vector<LinearOperator::Ptr> operators;

    explicit DirectSumLinearOperator(std::vector<LinearOperator::Ptr> operators);

    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
    [[nodiscard]] LinearOperator::Ptr adjoint() override;
};

/// Gram-Schmidt orthonormalization of a list of vectors.
///
/// @param vecs The list of vectors to be orthogonalized. All must be mutually compatible.
/// @param rcond Vectors of ``norm < rcond`` (after projecting out previous vectors) are discarded.
/// @returns A list of orthonormal vectors which span the same space as `vecs`.
std::vector<VectorLike::Ptr> gram_schmidt(std::vector<VectorLike::Ptr> const& vecs,
                                          float64 rcond = kGramSchmidtDefaultRcond);

} // namespace lasyten
