#include <lasyten/tensors/sparse.h>

#include <lasyten/tensors/symmetric_tensor.h>
#include <lasyten/tensors/tensor.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"

#include "docstrings/tensors/sparse.h"

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

/// nanobind trampoline: lets Python subclasses (e.g. in tenpy) override the virtual methods.
/// This is one of only two trampolines kept after the nanobind port (the other is
/// ``PyPlanarLinearOperator``).
class PyLinearOperator : public LinearOperator
{
  public:
    NB_TRAMPOLINE(LinearOperator, 3);

    VectorLike::Ptr matvec(VectorLike::CPtr vec) override { NB_OVERRIDE_PURE(matvec, vec); }

    TensorPtr to_tensor(TensorBackend::Ptr backend) override
    {
        NB_OVERRIDE_PURE(to_tensor, backend);
    }

    LinearOperator::Ptr adjoint() override { NB_OVERRIDE(adjoint); }
};

} // namespace

void
bind_tensors_sparse(nb::module_& m)
{
    nb::class_<LinearOperator, PyLinearOperator> linear_operator(m, "LinearOperator");
    linear_operator.doc() = DOC(lasyten, LinearOperator);
    linear_operator.attr("acts_on") = LinearOperator::acts_on;

    linear_operator
      .def(lasyten::ctor<decltype(linear_operator),
                         std::vector<Leg::Ptr>,
                         Dtype,
                         std::optional<LegLabelList>>(),
           nb::arg("vector_legs") = std::vector<Leg::Ptr>{},
           nb::arg("dtype") = Dtype::Float64,
           nb::arg("vector_labels").none() = nb::none())
      .def_rw("vector_legs", &LinearOperator::vector_legs)
      .def_prop_rw(
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
      .def_rw("dtype", &LinearOperator::dtype)
      .def("matvec", &LinearOperator::matvec, nb::arg("vec"), DOC(lasyten, LinearOperator, matvec))
      .def("to_tensor",
           &LinearOperator::to_tensor,
           nb::arg("backend").none() = nullptr,
           DOC(lasyten, LinearOperator, to_tensor))
      .def("to_matrix",
           &LinearOperator::to_matrix,
           nb::arg("backend").none() = nullptr,
           DOC(lasyten, LinearOperator, to_matrix))
      .def("adjoint", &LinearOperator::adjoint, DOC(lasyten, LinearOperator, adjoint));

    auto tensor_linear_operator_cls =
      nb::class_<TensorLinearOperator, LinearOperator>(m, "TensorLinearOperator");
    tensor_linear_operator_cls.doc() = DOC(lasyten, TensorLinearOperator);
    tensor_linear_operator_cls
      .def(lasyten::ctor<decltype(tensor_linear_operator_cls),
                         SymmetricTensorPtr,
                         std::variant<int64, std::string>>(),
           nb::arg("tensor"),
           nb::arg("which_leg") = -1)
      .def_rw("tensor", &TensorLinearOperator::tensor)
      .def_rw("which_leg", &TensorLinearOperator::which_leg)
      .def_rw("other_leg", &TensorLinearOperator::other_leg)
      .def("matvec", &TensorLinearOperator::matvec, nb::arg("vec"))
      .def("to_tensor", &TensorLinearOperator::to_tensor, nb::arg("backend").none() = nullptr)
      .def("adjoint", &TensorLinearOperator::adjoint);

    auto linear_operator_wrapper_cls =
      nb::class_<LinearOperatorWrapper, LinearOperator>(m, "LinearOperatorWrapper");
    linear_operator_wrapper_cls.doc() = DOC(lasyten, LinearOperatorWrapper);
    linear_operator_wrapper_cls
      .def(lasyten::ctor<decltype(linear_operator_wrapper_cls), LinearOperator::Ptr>(),
           nb::arg("original_operator"))
      .def_rw("original_operator", &LinearOperatorWrapper::original_operator)
      .def("unwrapped",
           &LinearOperatorWrapper::unwrapped,
           nb::arg("recursive") = true,
           DOC(lasyten, LinearOperatorWrapper, unwrapped))
      .def("__getattr__",
           [](LinearOperatorWrapper const& self, std::string const& name) {
               return py::getattr(py::cast(self.original_operator), name.c_str());
           })
      .def("matvec", &LinearOperatorWrapper::matvec, nb::arg("vec"))
      .def("to_tensor", &LinearOperatorWrapper::to_tensor, nb::arg("backend").none() = nullptr)
      .def("adjoint", &LinearOperatorWrapper::adjoint);

    auto sum_linear_operator_cls =
      nb::class_<SumLinearOperator, LinearOperatorWrapper>(m, "SumLinearOperator");
    sum_linear_operator_cls.doc() = DOC(lasyten, SumLinearOperator);
    sum_linear_operator_cls
      .def(nb::new_([](LinearOperator::Ptr original_operator, nb::args more_ops_nb) {
               auto more_ops = pyargs(more_ops_nb);
               std::vector<LinearOperator::Ptr> more;
               more.reserve(more_ops.size());
               for (auto item : more_ops) {
                   more.push_back(item.cast<LinearOperator::Ptr>());
               }
               return std::make_shared<SumLinearOperator>(std::move(original_operator),
                                                          std::move(more));
           }),
           nb::arg("original_operator"),
           nb::arg("more_operators"))
      .def_rw("more_operators", &SumLinearOperator::more_operators)
      .def("matvec", &SumLinearOperator::matvec, nb::arg("vec"))
      .def("to_tensor", &SumLinearOperator::to_tensor, nb::arg("backend").none() = nullptr)
      .def("adjoint", &SumLinearOperator::adjoint);

    auto shifted_linear_operator_cls =
      nb::class_<ShiftedLinearOperator, LinearOperatorWrapper>(m, "ShiftedLinearOperator");
    shifted_linear_operator_cls.doc() = DOC(lasyten, ShiftedLinearOperator);
    shifted_linear_operator_cls
      .def(nb::new_([](py::object original_operator, py::object shift) {
               return std::make_shared<ShiftedLinearOperator>(
                 wrap_linear_operator(std::move(original_operator)), py::cast<complex128>(shift));
           }),
           nb::arg("original_operator"),
           nb::arg("shift"))
      .def_rw("shift", &ShiftedLinearOperator::shift)
      .def("matvec", &ShiftedLinearOperator::matvec, nb::arg("vec"))
      .def("to_tensor", &ShiftedLinearOperator::to_tensor, nb::arg("backend").none() = nullptr)
      .def("adjoint", &ShiftedLinearOperator::adjoint);

    auto projected_linear_operator_cls =
      nb::class_<ProjectedLinearOperator, LinearOperatorWrapper>(m, "ProjectedLinearOperator");
    projected_linear_operator_cls.doc() = DOC(lasyten, ProjectedLinearOperator);
    projected_linear_operator_cls
      .def(nb::new_([](py::object original_operator,
                       std::vector<VectorLike::Ptr> ortho_vecs,
                       bool project_operator,
                       std::optional<complex128> penalty) {
               return std::make_shared<ProjectedLinearOperator>(
                 wrap_linear_operator(std::move(original_operator)),
                 std::move(ortho_vecs),
                 project_operator,
                 penalty);
           }),
           nb::arg("original_operator"),
           nb::arg("ortho_vecs"),
           nb::arg("project_operator") = true,
           nb::arg("penalty").none() = nb::none())
      .def_rw("ortho_vecs", &ProjectedLinearOperator::ortho_vecs)
      .def_rw("project_operator", &ProjectedLinearOperator::project_operator)
      .def_prop_rw(
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
      .def("matvec", &ProjectedLinearOperator::matvec, nb::arg("vec"))
      .def("to_tensor", &ProjectedLinearOperator::to_tensor, nb::arg("backend").none() = nullptr)
      .def("adjoint", &ProjectedLinearOperator::adjoint);

    auto direct_sum_linear_operator_cls =
      nb::class_<DirectSumLinearOperator, LinearOperator>(m, "DirectSumLinearOperator");
    direct_sum_linear_operator_cls.doc() = DOC(lasyten, DirectSumLinearOperator);
    direct_sum_linear_operator_cls
      .def(nb::new_([](std::vector<LinearOperator::Ptr> operators) {
               return std::make_shared<DirectSumLinearOperator>(std::move(operators));
           }),
           nb::arg("operators"))
      .def("matvec", &DirectSumLinearOperator::matvec, nb::arg("vec"))
      .def("to_tensor", &DirectSumLinearOperator::to_tensor, nb::arg("backend").none() = nullptr)
      .def("adjoint", &DirectSumLinearOperator::adjoint);

    m.def("gram_schmidt",
          &gram_schmidt,
          nb::arg("vecs"),
          nb::arg("rcond") = kGramSchmidtDefaultRcond,
          DOC(lasyten, gram_schmidt));
}

} // namespace lasyten
