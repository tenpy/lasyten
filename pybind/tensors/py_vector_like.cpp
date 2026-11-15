#include <lasyten/tensors/direct_sum.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tensors/tensor.h>
#include <lasyten/tensors/vector_like.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"

#include "docstrings/tensors/vector_like.h"

#include <optional>
#include <string>

namespace lasyten {

namespace {

BlockBackend::Scalar
as_scalar_for(VectorLike const& vec, py::object value)
{
    return py::cast<BlockBackend::Scalar>(
      py::cast(vec.vector_backend()->block_backend).attr("as_scalar")(value));
}

py::object
py_cast_vector_like(VectorLike::Ptr p)
{
    if (!p) {
        return py::none();
    }
    if (auto t = std::dynamic_pointer_cast<Tensor>(p)) {
        return py::cast(std::move(t));
    }
    if (auto ds = std::dynamic_pointer_cast<DirectSum>(p)) {
        return py::cast(std::move(ds));
    }
    return py::cast(std::move(p));
}

std::optional<nb::class_<VectorLike>> g_vector_like;

} // namespace

void
declare_tensors_vector_like(nb::module_& m)
{
    g_vector_like = nb::class_<VectorLike>(m, "VectorLike");
    (*g_vector_like).doc() = DOC(lasyten, VectorLike);
}

void
bind_tensors_vector_like(nb::module_& m)
{
    auto& cls = *g_vector_like;

    cls.attr("__array_ufunc__") = py::none();
    cls.def(lasyten::ctor<decltype(cls)>());

    cls.def(
      "copy",
      [](VectorLike const& self) { return py_cast_vector_like(self.clone()); },
      DOC(lasyten, VectorLike, clone));
    cls.def_prop_ro("dtype", &VectorLike::vector_dtype, DOC(lasyten, VectorLike, vector_dtype));
    cls.def_prop_ro("device", &VectorLike::vector_device, DOC(lasyten, VectorLike, vector_device));
    cls.def_prop_ro(
      "backend", &VectorLike::vector_backend, DOC(lasyten, VectorLike, vector_backend));
    cls.def("compatible_with",
            &VectorLike::compatible_with,
            nb::arg("other"),
            DOC(lasyten, VectorLike, compatible_with));
    cls.def(
      "scaled",
      [](VectorLike const& self, py::object a) {
          return py_cast_vector_like(self.scaled(as_scalar_for(self, a)));
      },
      nb::arg("a"),
      DOC(lasyten, VectorLike, scaled));
    cls.def(
      "axpy",
      [](VectorLike const& self, py::object a, VectorLike::CPtr other) {
          return py_cast_vector_like(self.axpy(as_scalar_for(self, a), std::move(other)));
      },
      nb::arg("a"),
      nb::arg("other"),
      DOC(lasyten, VectorLike, axpy));

    cls
      .def(
        "__add__",
        [](VectorLike::Ptr self, VectorLike::CPtr other) {
            auto one = self->vector_backend()->block_backend->as_scalar(1.0);
            return py_cast_vector_like(self->axpy(one, std::move(other)));
        },
        nb::is_operator())
      .def(
        "__sub__",
        [](VectorLike::Ptr self, VectorLike::CPtr other) {
            auto minus_one = self->vector_backend()->block_backend->as_scalar(-1.0);
            return py_cast_vector_like(other->axpy(minus_one, std::move(self)));
        },
        nb::is_operator())
      .def(
        "__mul__",
        [](VectorLike::Ptr self, py::object other) {
            return py_cast_vector_like(self->scaled(as_scalar_for(*self, other)));
        },
        nb::is_operator())
      .def(
        "__rmul__",
        [](VectorLike::Ptr self, py::object other) {
            return py_cast_vector_like(self->scaled(as_scalar_for(*self, other)));
        },
        nb::is_operator())
      .def(
        "__truediv__",
        [](VectorLike::Ptr self, py::object other) {
            py::object inv;
            try {
                inv = py::float_(1.0) / other;
            } catch (py::error_already_set&) {
                throw py::value_error("VectorLike can only be divided by invertible scalars.");
            }
            return py_cast_vector_like(self->scaled(as_scalar_for(*self, inv)));
        },
        nb::is_operator())
      .def(
        "__neg__",
        [](VectorLike::Ptr self) {
            auto minus_one = self->vector_backend()->block_backend->as_scalar(-1.0);
            return py_cast_vector_like(self->scaled(minus_one));
        },
        nb::is_operator())
      .def("__pos__", [](VectorLike::Ptr self) { return self; });
}

} // namespace lasyten
