#include <lasyten/tensors/sparse.h>

#include <lasyten/tensors/symmetric_tensor.h>
#include <lasyten/tensors/tensor.h>

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"

#include "docstrings/tensors/sparse.h"

#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

std::optional<LegLabelList>
optional_labels(py::object labels)
{
    if (labels.is_none()) {
        return std::nullopt;
    }
    return labels.cast<LegLabelList>();
}

std::vector<std::string>
to_leg_labels(py::object maybe_labels)
{
    std::vector<std::string> out;
    for (auto item : maybe_labels) {
        if (item.is_none()) {
            out.push_back("");
        } else {
            out.push_back(item.cast<std::string>());
        }
    }
    return out;
}

std::optional<LegLabelList>
optional_vector_labels_from_py(py::object labels)
{
    if (labels.is_none()) {
        return std::nullopt;
    }
    return LegLabelList(to_leg_labels(labels));
}

class PyLinearOperatorAdapter : public LinearOperator
{
  public:
    py::object original;

    explicit PyLinearOperatorAdapter(py::object op)
      : LinearOperator(op.attr("vector_legs").cast<std::vector<Leg::Ptr>>(),
                       op.attr("dtype").cast<Dtype>(),
                       optional_vector_labels_from_py(op.attr("vector_labels")))
      , original(std::move(op))
    {
    }

    VectorLike::Ptr matvec(VectorLike::CPtr vec) override
    {
        auto res = original.attr("matvec")(py::cast(vec));
        return res.cast<VectorLike::Ptr>();
    }

    TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override
    {
        py::object arg = backend ? py::cast(backend) : py::none();
        auto res = original.attr("to_tensor")(arg);
        return res.cast<TensorPtr>();
    }

    LinearOperator::Ptr adjoint() override;
};

LinearOperator::Ptr
wrap_linear_operator(py::object op)
{
    try {
        return op.cast<LinearOperator::Ptr>();
    } catch (py::cast_error const&) {
        return std::make_shared<PyLinearOperatorAdapter>(std::move(op));
    }
}

LinearOperator::Ptr
PyLinearOperatorAdapter::adjoint()
{
    auto res = original.attr("adjoint")();
    return wrap_linear_operator(std::move(res));
}

class PyLinearOperator
  : public LinearOperator
  , public py::trampoline_self_life_support
{
  public:
    using LinearOperator::LinearOperator;

    VectorLike::Ptr matvec(VectorLike::CPtr vec) override
    {
        PYBIND11_OVERRIDE_PURE(VectorLike::Ptr, LinearOperator, matvec, vec);
    }

    TensorPtr to_tensor(TensorBackend::Ptr backend) override
    {
        PYBIND11_OVERRIDE_PURE(TensorPtr, LinearOperator, to_tensor, backend);
    }

    LinearOperator::Ptr adjoint() override
    {
        PYBIND11_OVERRIDE(LinearOperator::Ptr, LinearOperator, adjoint);
    }
};

} // namespace

void
bind_tensors_sparse(py::module_& m)
{
    py::class_<LinearOperator, PyLinearOperator, py::smart_holder> linear_operator(
      m, "LinearOperator");
    linear_operator.doc() = DOC(lasyten, LinearOperator);
    linear_operator.attr("acts_on") = LinearOperator::acts_on;

    linear_operator
      .def(py::init<std::vector<Leg::Ptr>, Dtype, std::optional<LegLabelList>>(),
           py::arg("vector_legs") = std::vector<Leg::Ptr>{},
           py::arg("dtype") = Dtype::Float64,
           py::arg("vector_labels") = py::none())
      .def_readwrite("vector_legs", &LinearOperator::vector_legs)
      .def_property(
        "vector_labels",
        [](LinearOperator const& self) -> py::object {
            if (!self.vector_labels.has_value()) {
                return py::none();
            }
            return py::cast(*self.vector_labels);
        },
        [](LinearOperator& self, py::object labels) {
            self.vector_labels = optional_labels(labels);
        })
      .def_readwrite("dtype", &LinearOperator::dtype)
      .def("matvec", &LinearOperator::matvec, py::arg("vec"), DOC(lasyten, LinearOperator, matvec))
      .def("to_tensor",
           &LinearOperator::to_tensor,
           py::arg("backend") = nullptr,
           DOC(lasyten, LinearOperator, to_tensor))
      .def("to_matrix",
           &LinearOperator::to_matrix,
           py::arg("backend") = nullptr,
           DOC(lasyten, LinearOperator, to_matrix))
      .def("adjoint", &LinearOperator::adjoint, DOC(lasyten, LinearOperator, adjoint));

    auto tensor_linear_operator_cls =
      py::class_<TensorLinearOperator, LinearOperator, py::smart_holder>(m,
                                                                         "TensorLinearOperator");
    tensor_linear_operator_cls.doc() = DOC(lasyten, TensorLinearOperator);
    tensor_linear_operator_cls
      .def(py::init<SymmetricTensorPtr, std::variant<int64, std::string>>(),
           py::arg("tensor"),
           py::arg("which_leg") = -1)
      .def_readwrite("tensor", &TensorLinearOperator::tensor)
      .def_readwrite("which_leg", &TensorLinearOperator::which_leg)
      .def_readwrite("other_leg", &TensorLinearOperator::other_leg)
      .def("matvec", &TensorLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &TensorLinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &TensorLinearOperator::adjoint);

    auto linear_operator_wrapper_cls =
      py::class_<LinearOperatorWrapper, LinearOperator, py::smart_holder>(m,
                                                                          "LinearOperatorWrapper");
    linear_operator_wrapper_cls.doc() = DOC(lasyten, LinearOperatorWrapper);
    linear_operator_wrapper_cls.def(py::init<LinearOperator::Ptr>(), py::arg("original_operator"))
      .def_readwrite("original_operator", &LinearOperatorWrapper::original_operator)
      .def("unwrapped",
           &LinearOperatorWrapper::unwrapped,
           py::arg("recursive") = true,
           DOC(lasyten, LinearOperatorWrapper, unwrapped))
      .def("__getattr__",
           [](LinearOperatorWrapper const& self, std::string const& name) {
               return py::getattr(py::cast(self.original_operator), name.c_str());
           })
      .def("matvec", &LinearOperatorWrapper::matvec, py::arg("vec"))
      .def("to_tensor", &LinearOperatorWrapper::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &LinearOperatorWrapper::adjoint);

    auto sum_linear_operator_cls =
      py::class_<SumLinearOperator, LinearOperatorWrapper, py::smart_holder>(m,
                                                                             "SumLinearOperator");
    sum_linear_operator_cls.doc() = DOC(lasyten, SumLinearOperator);
    sum_linear_operator_cls
      .def(py::init([](LinearOperator::Ptr original_operator, py::args more_ops) {
               std::vector<LinearOperator::Ptr> more;
               more.reserve(more_ops.size());
               for (auto item : more_ops) {
                   more.push_back(item.cast<LinearOperator::Ptr>());
               }
               return std::make_shared<SumLinearOperator>(std::move(original_operator),
                                                          std::move(more));
           }),
           py::arg("original_operator"))
      .def_readwrite("more_operators", &SumLinearOperator::more_operators)
      .def("matvec", &SumLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &SumLinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &SumLinearOperator::adjoint);

    auto shifted_linear_operator_cls =
      py::class_<ShiftedLinearOperator, LinearOperatorWrapper, py::smart_holder>(
        m, "ShiftedLinearOperator");
    shifted_linear_operator_cls.doc() = DOC(lasyten, ShiftedLinearOperator);
    shifted_linear_operator_cls
      .def(py::init([](py::object original_operator, py::object shift) {
               return std::make_shared<ShiftedLinearOperator>(
                 wrap_linear_operator(std::move(original_operator)), py::cast<complex128>(shift));
           }),
           py::arg("original_operator"),
           py::arg("shift"))
      .def_readwrite("shift", &ShiftedLinearOperator::shift)
      .def("matvec", &ShiftedLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &ShiftedLinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &ShiftedLinearOperator::adjoint);

    auto projected_linear_operator_cls =
      py::class_<ProjectedLinearOperator, LinearOperatorWrapper, py::smart_holder>(
        m, "ProjectedLinearOperator");
    projected_linear_operator_cls.doc() = DOC(lasyten, ProjectedLinearOperator);
    projected_linear_operator_cls
      .def(py::init([](py::object original_operator,
                       std::vector<VectorLike::Ptr> ortho_vecs,
                       bool project_operator,
                       std::optional<complex128> penalty) {
               return std::make_shared<ProjectedLinearOperator>(
                 wrap_linear_operator(std::move(original_operator)),
                 std::move(ortho_vecs),
                 project_operator,
                 penalty);
           }),
           py::arg("original_operator"),
           py::arg("ortho_vecs"),
           py::arg("project_operator") = true,
           py::arg("penalty") = py::none())
      .def_readwrite("ortho_vecs", &ProjectedLinearOperator::ortho_vecs)
      .def_readwrite("project_operator", &ProjectedLinearOperator::project_operator)
      .def_property(
        "penalty",
        [](ProjectedLinearOperator const& self) -> py::object {
            if (!self.penalty.has_value()) {
                return py::none();
            }
            return py::cast(*self.penalty);
        },
        [](ProjectedLinearOperator& self, py::object value) {
            if (value.is_none()) {
                self.penalty = std::nullopt;
                return;
            }
            self.penalty = py::cast<complex128>(value);
        })
      .def("matvec", &ProjectedLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &ProjectedLinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &ProjectedLinearOperator::adjoint);

    auto direct_sum_linear_operator_cls =
      py::class_<DirectSumLinearOperator, LinearOperator, py::smart_holder>(
        m, "DirectSumLinearOperator");
    direct_sum_linear_operator_cls.doc() = DOC(lasyten, DirectSumLinearOperator);
    direct_sum_linear_operator_cls
      .def(py::init([](std::vector<LinearOperator::Ptr> operators) {
               return std::make_shared<DirectSumLinearOperator>(std::move(operators));
           }),
           py::arg("operators"))
      .def("matvec", &DirectSumLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &DirectSumLinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &DirectSumLinearOperator::adjoint);

    m.def("gram_schmidt",
          &gram_schmidt,
          py::arg("vecs"),
          py::arg("rcond") = kGramSchmidtDefaultRcond,
          DOC(lasyten, gram_schmidt));
}

} // namespace lasyten
