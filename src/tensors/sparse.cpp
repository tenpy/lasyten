#include <lasyten/tensors/sparse.h>

#include <lasyten/block_backend/dtypes.h>
#include <lasyten/tensors/constructors.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tensors/ops_legs.h>
#include <lasyten/tensors/symmetric_tensor.h>

#include <complex>
#include <stdexcept>
#include <utility>

namespace lasyten {

namespace {

constexpr int64 kMaxUnwrapDepth = 10000;

} // namespace

py::object& LinearOperator::acts_on = leak_py_object(py::object(py::none()));

bool
same_legs(std::vector<Leg::Ptr> const& legs1, std::vector<Leg::Ptr> const& legs2)
{
    if (legs1.size() != legs2.size()) {
        return false;
    }
    for (std::size_t i = 0; i < legs1.size(); ++i) {
        if (!(*legs1[i] == *legs2[i])) {
            return false;
        }
    }
    return true;
}

LinearOperator::LinearOperator(std::vector<Leg::Ptr> vector_legs,
                               Dtype dtype,
                               VectorLabels vector_labels)
  : vector_legs(std::move(vector_legs))
  , vector_labels(std::move(vector_labels))
  , dtype(dtype)
{
}

TensorPtr
LinearOperator::to_matrix(TensorBackend::Ptr backend)
{
    auto tens = to_tensor(std::move(backend));
    auto N = static_cast<int64>(vector_legs.size());
    std::vector<LegRef> codomain;
    std::vector<LegRef> domain;
    codomain.reserve(static_cast<std::size_t>(N));
    domain.reserve(static_cast<std::size_t>(N));
    for (int64 i = 0; i < N; ++i) {
        codomain.emplace_back(i);
        domain.emplace_back(i + N);
    }
    return combine_to_matrix(tens, codomain, domain);
}

LinearOperator::Ptr
LinearOperator::adjoint()
{
    throw std::runtime_error("No adjoint defined");
}

TensorLinearOperator::TensorLinearOperator(SymmetricTensorPtr tensor,
                                           std::variant<int64, std::string> which_leg_ref)
  : LinearOperator({}, Dtype::Float64, std::nullopt)
  , tensor(std::move(tensor))
{
    if (!this->tensor) {
        throw std::invalid_argument("tensor must not be null");
    }
    if (this->tensor->num_legs != 2) {
        throw std::invalid_argument("Expected a two-leg tensor");
    }

    this->which_leg =
      std::visit([&](auto const& key) { return this->tensor->labels.index(key); }, which_leg_ref);
    this->other_leg = 1 - this->which_leg;

    auto legs = this->tensor->legs();
    auto expected = legs[other_leg]->dual();
    if (!(*legs[which_leg] == *expected)) {
        throw std::invalid_argument("Expected contractible legs");
    }

    this->vector_legs = { legs[other_leg] };
    auto labels = this->tensor->labels.as_list();
    this->vector_labels = VectorLabels{ LegLabelList{ labels[other_leg] } };
    this->dtype = this->tensor->dtype;
}

VectorLike::Ptr
TensorLinearOperator::matvec(VectorLike::CPtr vec)
{
    auto tvec = std::dynamic_pointer_cast<Tensor const>(vec);
    if (!tvec) {
        throw std::invalid_argument("TensorLinearOperator.matvec expects a Tensor input");
    }
    if (tvec->num_legs != 1) {
        throw std::invalid_argument("TensorLinearOperator.matvec expects a single-leg tensor");
    }
    auto res = tdot(tensor, tvec, { LegRef(which_leg) }, { LegRef(0) });
    if (std::holds_alternative<BlockBackend::Scalar>(res)) {
        throw std::runtime_error("TensorLinearOperator.matvec unexpectedly returned a scalar");
    }
    return std::get<TensorPtr>(std::move(res));
}

TensorPtr
TensorLinearOperator::to_tensor(TensorBackend::Ptr)
{
    if (which_leg == 1) {
        return tensor;
    }
    return permute_legs(
      tensor, std::vector<LegRef>{ LegRef(other_leg) }, std::vector<LegRef>{ LegRef(which_leg) });
}

LinearOperator::Ptr
TensorLinearOperator::adjoint()
{
    auto dag = dagger(tensor);
    auto dag_sym = std::dynamic_pointer_cast<SymmetricTensor>(dag);
    if (!dag_sym) {
        throw std::runtime_error("TensorLinearOperator.adjoint expected SymmetricTensor dagger");
    }
    return std::make_shared<TensorLinearOperator>(std::move(dag_sym), which_leg);
}

LinearOperatorWrapper::LinearOperatorWrapper(LinearOperator::Ptr original_operator)
  : LinearOperator(original_operator ? original_operator->vector_legs : std::vector<Leg::Ptr>{},
                   original_operator ? original_operator->dtype : Dtype::Float64,
                   original_operator ? original_operator->vector_labels : VectorLabels{})
  , original_operator(std::move(original_operator))
{
    if (!this->original_operator) {
        throw std::invalid_argument("original_operator must not be null");
    }
}

LinearOperator::Ptr
LinearOperatorWrapper::unwrapped(bool recursive) const
{
    auto parent = original_operator;
    if (!recursive) {
        return parent;
    }
    for (int64 n = 0; n < kMaxUnwrapDepth; ++n) {
        auto as_wrap = std::dynamic_pointer_cast<LinearOperatorWrapper>(parent);
        if (!as_wrap) {
            return parent;
        }
        parent = as_wrap->original_operator;
    }
    throw std::runtime_error("maximum recursion depth for unwrapping reached");
}

VectorLike::Ptr
LinearOperatorWrapper::matvec(VectorLike::CPtr vec)
{
    return original_operator->matvec(std::move(vec));
}

TensorPtr
LinearOperatorWrapper::to_tensor(TensorBackend::Ptr backend)
{
    return original_operator->to_tensor(std::move(backend));
}

LinearOperator::Ptr
LinearOperatorWrapper::adjoint()
{
    return std::make_shared<LinearOperatorWrapper>(original_operator->adjoint());
}

SumLinearOperator::SumLinearOperator(LinearOperator::Ptr original_operator,
                                     std::vector<LinearOperator::Ptr> more_operators)
  : LinearOperatorWrapper(std::move(original_operator))
  , more_operators(std::move(more_operators))
{
    std::vector<Dtype> dtypes{ this->original_operator->dtype };
    dtypes.reserve(this->more_operators.size() + 1);
    for (auto const& op : this->more_operators) {
        if (!same_legs(op->vector_legs, this->original_operator->vector_legs)) {
            throw std::invalid_argument(
              "All operators in SumLinearOperator must act on same legs");
        }
        dtypes.push_back(op->dtype);
    }
    this->dtype = dtype::common(dtypes);
}

VectorLike::Ptr
SumLinearOperator::matvec(VectorLike::CPtr vec)
{
    auto res = original_operator->matvec(vec);
    auto one = res->vector_backend()->block_backend->as_scalar(1.0);
    for (auto const& op : more_operators) {
        auto term = op->matvec(vec);
        res = term->axpy(one, res);
    }
    return res;
}

TensorPtr
SumLinearOperator::to_tensor(TensorBackend::Ptr backend)
{
    auto res = original_operator->to_tensor(backend);
    auto one = res->backend->block_backend->as_scalar(1.0);
    for (auto const& op : more_operators) {
        res = linear_combination(one, TensorCPtr(op->to_tensor(backend)), one, TensorCPtr(res));
    }
    return res;
}

LinearOperator::Ptr
SumLinearOperator::adjoint()
{
    std::vector<LinearOperator::Ptr> others;
    others.reserve(more_operators.size());
    for (auto const& op : more_operators) {
        others.push_back(op->adjoint());
    }
    return std::make_shared<SumLinearOperator>(original_operator->adjoint(), std::move(others));
}

ShiftedLinearOperator::ShiftedLinearOperator(LinearOperator::Ptr original_operator,
                                             complex128 shift)
  : LinearOperatorWrapper(std::move(original_operator))
  , shift(std::move(shift))
{
    if (dtype::is_real(this->dtype) && std::imag(this->shift) != 0.0) {
        this->dtype = dtype::to_complex(this->dtype);
    }
}

VectorLike::Ptr
ShiftedLinearOperator::matvec(VectorLike::CPtr vec)
{
    auto res = original_operator->matvec(vec);
    auto s = vec->vector_backend()->block_backend->as_scalar(shift);
    return vec->axpy(s, res);
}

TensorPtr
ShiftedLinearOperator::to_tensor(TensorBackend::Ptr backend)
{
    auto res = original_operator->to_tensor(backend);
    auto sym = vector_legs.empty() ? nullptr : vector_legs[0]->symmetry;
    auto tp = std::make_shared<TensorProduct>(vector_legs, sym);
    auto identity =
      SymmetricTensor::from_eye(tp, res->backend, vector_labels, res->dtype, res->device);
    auto one = res->backend->block_backend->as_scalar(1.0);
    auto s = res->backend->block_backend->as_scalar(shift);
    return linear_combination(one, TensorCPtr(res), s, TensorCPtr(identity));
}

LinearOperator::Ptr
ShiftedLinearOperator::adjoint()
{
    return std::make_shared<ShiftedLinearOperator>(original_operator->adjoint(), std::conj(shift));
}

ProjectedLinearOperator::ProjectedLinearOperator(LinearOperator::Ptr original_operator,
                                                 std::vector<VectorLike::Ptr> ortho_vecs,
                                                 bool project_operator,
                                                 std::optional<complex128> penalty)
  : LinearOperatorWrapper(std::move(original_operator))
  , ortho_vecs(std::move(ortho_vecs))
  , project_operator(project_operator)
  , penalty(std::move(penalty))
{
    if (!this->ortho_vecs.empty()) {
        for (auto const& v : this->ortho_vecs) {
            if (!v->compatible_with(this->ortho_vecs[0])) {
                throw std::invalid_argument("All ortho_vecs must be mutually compatible");
            }
        }
    }
}

VectorLike::Ptr
ProjectedLinearOperator::matvec(VectorLike::CPtr vec)
{
    auto res = vec->clone();
    std::vector<BlockBackend::Scalar> coeffs;
    coeffs.reserve(ortho_vecs.size());

    if (project_operator) {
        for (auto const& o : ortho_vecs) {
            auto c = o->vector_inner(res);
            coeffs.push_back(c);
            res = o->axpy(-c, res);
        }
    } else {
        for (auto const& o : ortho_vecs) {
            coeffs.push_back(o->vector_inner(res));
        }
    }

    res = original_operator->matvec(res);
    if (project_operator) {
        for (auto const& o : ortho_vecs) {
            auto c = o->vector_inner(res);
            res = o->axpy(-c, res);
        }
    }
    if (penalty.has_value()) {
        for (std::size_t i = 0; i < ortho_vecs.size(); ++i) {
            auto p = res->vector_backend()->block_backend->as_scalar(*penalty);
            res = ortho_vecs[i]->axpy(p * coeffs[i], res);
        }
    }
    return res;
}

TensorPtr
ProjectedLinearOperator::to_tensor(TensorBackend::Ptr)
{
    throw std::runtime_error("ProjectedLinearOperator.to_tensor not implemented");
}

LinearOperator::Ptr
ProjectedLinearOperator::adjoint()
{
    std::optional<complex128> p = std::nullopt;
    if (penalty.has_value()) {
        p = std::conj(*penalty);
    }
    return std::make_shared<ProjectedLinearOperator>(
      original_operator->adjoint(), ortho_vecs, project_operator, p);
}

DirectSumLinearOperator::DirectSumLinearOperator(std::vector<LinearOperator::Ptr> operators)
  : LinearOperator({}, Dtype::Float64, std::nullopt)
  , operators(std::move(operators))
{
    if (this->operators.empty()) {
        throw std::invalid_argument("DirectSumLinearOperator needs at least one operator");
    }
    for (auto const& op : this->operators) {
        if (!op) {
            throw std::invalid_argument(
              "DirectSumLinearOperator operators must not contain null pointers");
        }
    }
    for (auto const& op : this->operators) {
        if (!same_legs(op->vector_legs, this->operators.front()->vector_legs)) {
            throw std::invalid_argument(
              "All operators in DirectSumLinearOperator must act on same legs");
        }
    }

    std::vector<Dtype> dtypes;
    dtypes.reserve(this->operators.size());
    for (auto const& op : this->operators) {
        dtypes.push_back(op->dtype);
    }

    this->vector_legs = this->operators.front()->vector_legs;
    this->vector_labels = this->operators.front()->vector_labels;
    this->dtype = dtype::common(dtypes);
}

VectorLike::Ptr
DirectSumLinearOperator::matvec(VectorLike::CPtr vec)
{
    auto dvec = std::dynamic_pointer_cast<DirectSum const>(vec);
    if (!dvec) {
        throw std::invalid_argument("DirectSumLinearOperator.matvec expects a DirectSum input");
    }
    if (dvec->size() != operators.size()) {
        throw std::invalid_argument(
          "DirectSumLinearOperator.matvec expects matching component count");
    }

    std::vector<TensorPtr> res;
    res.reserve(operators.size());
    auto const& comps = dvec->components();
    for (std::size_t i = 0; i < operators.size(); ++i) {
        auto term = operators[i]->matvec(comps[i]);
        auto tterm = std::dynamic_pointer_cast<Tensor>(term);
        if (!tterm) {
            throw std::runtime_error("DirectSumLinearOperator.matvec expected Tensor results");
        }
        res.push_back(std::move(tterm));
    }
    return std::make_shared<DirectSum>(std::move(res));
}

TensorPtr
DirectSumLinearOperator::to_tensor(TensorBackend::Ptr)
{
    throw NotImplemented("DirectSumLinearOperator has no single-tensor representation");
}

LinearOperator::Ptr
DirectSumLinearOperator::adjoint()
{
    std::vector<LinearOperator::Ptr> others;
    others.reserve(operators.size());
    for (auto const& op : operators) {
        others.push_back(op->adjoint());
    }
    return std::make_shared<DirectSumLinearOperator>(std::move(others));
}

std::vector<VectorLike::Ptr>
gram_schmidt(std::vector<VectorLike::Ptr> const& vecs, float64 rcond)
{
    std::vector<VectorLike::Ptr> res;
    for (auto const& in_vec : vecs) {
        auto vec = in_vec->clone();
        for (auto const& other : res) {
            auto ov = other->vector_inner(vec);
            vec = other->axpy(-ov, vec);
        }
        auto n = vec->vector_norm();
        if ((n > rcond).as_bool()) {
            res.push_back(vec->scaled(1.0 / n));
        }
    }
    return res;
}

} // namespace lasyten
