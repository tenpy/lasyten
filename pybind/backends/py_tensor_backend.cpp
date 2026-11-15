#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "../tensors/py_callbacks.hpp"
#include "docstrings/backends/tensor_backend.h"
#include "py_trampolines.hpp"

#include <lasyten/backends/tensor_backend.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/symmetric_tensor.h>

#include "tools/hdf5_bind.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lasyten {

namespace {

std::optional<nb::class_<TensorBackend::Data>> g_tensor_backend_data;
std::optional<nb::class_<TensorBackend>> g_tensor_backend;

} // namespace

void
declare_tensor_backend(nb::module_& m)
{
    g_tensor_backend_data = nb::class_<TensorBackend::Data>(m, "TensorBackendData");
    (*g_tensor_backend_data).doc() =
      "Backend-specific payload stored on a tensor (except symmetry data on legs).";

    g_tensor_backend = nb::class_<TensorBackend>(m, "TensorBackend");
    (*g_tensor_backend).doc() = DOC(lasyten, TensorBackend);
}

void
bind_tensor_backend(nb::module_& m)
{
    auto& data_cls = *g_tensor_backend_data;
    auto& tensor_backend = *g_tensor_backend;

    tensor_backend
      .def(lasyten::ctor<decltype(tensor_backend), std::shared_ptr<BlockBackend>>(),
           nb::arg("block_backend"))
      .def_prop_ro("can_decompose_tensors", &TensorBackend::can_decompose_tensors)
      .def_rw("block_backend", &TensorBackend::block_backend);

    tensor_backend //  methods
      .def("__repr__", &TensorBackend::__repr__)
      .def("__str__", &TensorBackend::__str__)
      .def("__eq__",
           [](TensorBackend const& self, py::object other) {
               if (!py::isinstance<TensorBackend>(other)) {
                   return false;
               }
               return self == py::cast<TensorBackend const&>(other);
           })
      .def("item", &TensorBackend::item, nb::arg("a"), DOC(lasyten, TensorBackend, item))
      .def("test_tensor_sanity",
           &TensorBackend::test_tensor_sanity,
           nb::arg("a"),
           nb::arg("is_diagonal"),
           DOC(lasyten, TensorBackend, test_tensor_sanity))
      .def("test_mask_sanity", &TensorBackend::test_mask_sanity, nb::arg("a"))
      .def("make_pipe",
           &TensorBackend::make_pipe,
           nb::arg("legs"),
           nb::arg("is_dual"),
           nb::arg("pipe").none() = nb::none(),
           DOC(lasyten, TensorBackend, make_pipe))
      .def(
        "act_block_diagonal_square_matrix",
        [](TensorBackend& self,
           SymmetricTensorCPtr a,
           py::function block_method,
           py::object dtype_map) {
            return self.act_block_diagonal_square_matrix(
              a, block_unary_from_python(block_method), dtype_map_from_python(dtype_map));
        },
        nb::arg("a"),
        nb::arg("block_method"),
        nb::arg("dtype_map"),
        DOC(lasyten, TensorBackend, act_block_diagonal_square_matrix))
      .def("add_trivial_leg",
           &TensorBackend::add_trivial_leg,
           nb::arg("a"),
           nb::arg("legs_pos"),
           nb::arg("add_to_domain"),
           nb::arg("co_domain_pos"),
           nb::arg("new_codomain"),
           nb::arg("new_domain"),
           DOC(lasyten, TensorBackend, add_trivial_leg))
      .def("almost_equal",
           &TensorBackend::almost_equal,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("rtol"),
           nb::arg("atol"),
           DOC(lasyten, TensorBackend, almost_equal))
      .def("apply_mask_to_DiagonalTensor",
           &TensorBackend::apply_mask_to_DiagonalTensor,
           nb::arg("tensor"),
           nb::arg("mask"))
      .def("combine_legs",
           &TensorBackend::combine_legs,
           nb::arg("tensor"),
           nb::arg("leg_idcs_combine"),
           nb::arg("pipes"),
           nb::arg("new_codomain"),
           nb::arg("new_domain"),
           DOC(lasyten, TensorBackend, combine_legs))
      .def("compose",
           &TensorBackend::compose,
           nb::arg("a"),
           nb::arg("b"),
           DOC(lasyten, TensorBackend, compose))
      .def("copy_data",
           &TensorBackend::copy_data,
           nb::arg("a"),
           nb::arg("device").none() = nb::none(),
           DOC(lasyten, TensorBackend, copy_data))
      .def("dagger", &TensorBackend::dagger, nb::arg("a"), DOC(lasyten, TensorBackend, dagger))
      .def("data_item",
           &TensorBackend::data_item,
           nb::arg("a"),
           DOC(lasyten, TensorBackend, data_item))
      .def("diagonal_all",
           &TensorBackend::diagonal_all,
           nb::arg("a"),
           DOC(lasyten, TensorBackend, diagonal_all))
      .def("diagonal_any",
           &TensorBackend::diagonal_any,
           nb::arg("a"),
           DOC(lasyten, TensorBackend, diagonal_any))
      .def(
        "diagonal_elementwise_binary",
        [](TensorBackend& self,
           DiagonalTensorCPtr a,
           DiagonalTensorCPtr b,
           py::function func,
           py::dict func_kwargs,
           bool partial_zero_is_zero) {
            return self.diagonal_elementwise_binary(
              a, b, block_binary_from_python(func, func_kwargs), partial_zero_is_zero);
        },
        nb::arg("a"),
        nb::arg("b"),
        nb::arg("func"),
        nb::arg("func_kwargs"),
        nb::arg("partial_zero_is_zero"),
        DOC(lasyten, TensorBackend, diagonal_elementwise_binary))
      .def(
        "diagonal_elementwise_unary",
        [](TensorBackend& self,
           DiagonalTensorCPtr a,
           py::function func,
           py::dict func_kwargs,
           bool maps_zero_to_zero) {
            return self.diagonal_elementwise_unary(
              a, block_unary_from_python(func, func_kwargs), maps_zero_to_zero);
        },
        nb::arg("a"),
        nb::arg("func"),
        nb::arg("func_kwargs"),
        nb::arg("maps_zero_to_zero"),
        DOC(lasyten, TensorBackend, diagonal_elementwise_unary))
      .def("diagonal_from_block",
           &TensorBackend::diagonal_from_block,
           nb::arg("a"),
           nb::arg("co_domain"),
           nb::arg("tol"),
           DOC(lasyten, TensorBackend, diagonal_from_block))
      .def(
        "diagonal_from_sector_block_func",
        [](TensorBackend& self, py::function func, TensorProduct::Ptr co_domain) {
            return self.diagonal_from_sector_block_func(sector_block_factory_from_python(func),
                                                        std::move(co_domain));
        },
        nb::arg("func"),
        nb::arg("co_domain"),
        DOC(lasyten, TensorBackend, diagonal_from_sector_block_func))
      .def("diagonal_tensor_from_full_tensor",
           &TensorBackend::diagonal_tensor_from_full_tensor,
           nb::arg("a"),
           nb::arg("tol") = 1e-12,
           DOC(lasyten, TensorBackend, diagonal_tensor_from_full_tensor))
      .def("diagonal_tensor_trace_full", &TensorBackend::diagonal_tensor_trace_full, nb::arg("a"))
      .def("diagonal_tensor_to_block",
           &TensorBackend::diagonal_tensor_to_block,
           nb::arg("a"),
           DOC(lasyten, TensorBackend, diagonal_tensor_to_block))
      .def("diagonal_to_mask",
           &TensorBackend::diagonal_to_mask,
           nb::arg("tens"),
           DOC(lasyten, TensorBackend, diagonal_to_mask))
      .def("diagonal_transpose",
           &TensorBackend::diagonal_transpose,
           nb::arg("tens"),
           DOC(lasyten, TensorBackend, diagonal_transpose))
      .def("eigh",
           &TensorBackend::eigh,
           nb::arg("a"),
           nb::arg("new_leg_dual"),
           nb::arg("sort").none() = nb::none(),
           DOC(lasyten, TensorBackend, eigh))
      .def("eig",
           &TensorBackend::eig,
           nb::arg("a"),
           nb::arg("new_leg_dual"),
           nb::arg("sort").none() = nb::none(),
           DOC(lasyten, TensorBackend, eig))
      .def("eigvalsh",
           &TensorBackend::eigvalsh,
           nb::arg("a"),
           nb::arg("new_leg_dual"),
           nb::arg("sort").none() = nb::none(),
           DOC(lasyten, TensorBackend, eigvalsh))
      .def("eigvals",
           &TensorBackend::eigvals,
           nb::arg("a"),
           nb::arg("new_leg_dual"),
           nb::arg("sort").none() = nb::none(),
           DOC(lasyten, TensorBackend, eigvals))
      .def("eye_data",
           &TensorBackend::eye_data,
           nb::arg("co_domain"),
           nb::arg("dtype"),
           nb::arg("device"),
           DOC(lasyten, TensorBackend, eye_data))
      .def("from_dense_block",
           &TensorBackend::from_dense_block,
           nb::arg("a"),
           nb::arg("codomain"),
           nb::arg("domain"),
           nb::arg("tol"),
           DOC(lasyten, TensorBackend, from_dense_block))
      .def("from_dense_block_trivial_sector",
           &TensorBackend::from_dense_block_trivial_sector,
           nb::arg("block"),
           nb::arg("leg"),
           DOC(lasyten, TensorBackend, from_dense_block_trivial_sector))
      .def(
        "from_grid",
        [](TensorBackend& self,
           py::object grid,
           TensorProduct::Ptr new_codomain,
           TensorProduct::Ptr new_domain,
           std::vector<std::vector<int64>> left_mult_slices,
           std::vector<std::vector<int64>> right_mult_slices,
           Dtype dtype,
           std::string device) {
            return self.from_grid(tensor_grid_from_python(std::move(grid)),
                                  std::move(new_codomain),
                                  std::move(new_domain),
                                  std::move(left_mult_slices),
                                  std::move(right_mult_slices),
                                  dtype,
                                  std::move(device));
        },
        nb::arg("grid"),
        nb::arg("new_codomain"),
        nb::arg("new_domain"),
        nb::arg("left_mult_slices"),
        nb::arg("right_mult_slices"),
        nb::arg("dtype"),
        nb::arg("device"),
        DOC(lasyten, TensorBackend, from_grid))
      .def("from_random_normal",
           &TensorBackend::from_random_normal,
           nb::arg("codomain"),
           nb::arg("domain"),
           nb::arg("sigma"),
           nb::arg("dtype"),
           nb::arg("device"))
      .def(
        "from_sector_block_func",
        [](TensorBackend& self,
           py::function func,
           TensorProduct::Ptr codomain,
           TensorProduct::Ptr domain) {
            return self.from_sector_block_func(
              sector_block_factory_from_python(func), std::move(codomain), std::move(domain));
        },
        nb::arg("func"),
        nb::arg("codomain"),
        nb::arg("domain"),
        DOC(lasyten, TensorBackend, from_sector_block_func))
      .def("from_tree_pairs",
           &TensorBackend::from_tree_pairs,
           nb::arg("trees"),
           nb::arg("codomain"),
           nb::arg("domain"),
           nb::arg("dtype"),
           nb::arg("device"),
           DOC(lasyten, TensorBackend, from_tree_pairs))
      .def("full_data_from_diagonal_tensor",
           &TensorBackend::full_data_from_diagonal_tensor,
           nb::arg("a"))
      .def("full_data_from_mask",
           &TensorBackend::full_data_from_mask,
           nb::arg("a"),
           nb::arg("dtype"),
           DOC(lasyten, TensorBackend, full_data_from_mask))
      .def("get_device_from_data",
           &TensorBackend::get_device_from_data,
           nb::arg("a"),
           DOC(lasyten, TensorBackend, get_device_from_data))
      .def("get_dtype_from_data", &TensorBackend::get_dtype_from_data, nb::arg("a"))
      .def("get_element",
           &TensorBackend::get_element,
           nb::arg("a"),
           nb::arg("idcs"),
           DOC(lasyten, TensorBackend, get_element))
      .def("get_element_diagonal",
           &TensorBackend::get_element_diagonal,
           nb::arg("a"),
           nb::arg("idx"),
           DOC(lasyten, TensorBackend, get_element_diagonal))
      .def("get_element_mask",
           &TensorBackend::get_element_mask,
           nb::arg("a"),
           nb::arg("idcs"),
           DOC(lasyten, TensorBackend, get_element_mask))
      .def("inner",
           &TensorBackend::inner,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("do_dagger"),
           DOC(lasyten, TensorBackend, inner))
      .def("inv_part_from_dense_block_single_sector",
           &TensorBackend::inv_part_from_dense_block_single_sector,
           nb::arg("vector"),
           nb::arg("space"),
           nb::arg("charge_leg"),
           DOC(lasyten, TensorBackend, inv_part_from_dense_block_single_sector))
      .def("inv_part_to_dense_block_single_sector",
           &TensorBackend::inv_part_to_dense_block_single_sector,
           nb::arg("tensor"),
           DOC(lasyten, TensorBackend, inv_part_to_dense_block_single_sector))
      .def("linear_combination",
           &TensorBackend::linear_combination,
           nb::arg("a"),
           nb::arg("v"),
           nb::arg("b"),
           nb::arg("w"),
           DOC(lasyten, TensorBackend, linear_combination))
      .def("lq",
           &TensorBackend::lq,
           nb::arg("tensor"),
           nb::arg("new_co_domain"),
           DOC(lasyten, TensorBackend, lq))
      .def(
        "mask_binary_operand",
        [](TensorBackend& self, MaskCPtr mask1, MaskCPtr mask2, py::function func) {
            return self.mask_binary_operand(mask1, mask2, block_binary_from_python(func));
        },
        nb::arg("mask1"),
        nb::arg("mask2"),
        nb::arg("func"),
        DOC(lasyten, TensorBackend, mask_binary_operand))
      .def("mask_contract_large_leg",
           &TensorBackend::mask_contract_large_leg,
           nb::arg("tensor"),
           nb::arg("mask"),
           nb::arg("leg_idx"),
           DOC(lasyten, TensorBackend, mask_contract_large_leg))
      .def("mask_contract_small_leg",
           &TensorBackend::mask_contract_small_leg,
           nb::arg("tensor"),
           nb::arg("mask"),
           nb::arg("leg_idx"),
           DOC(lasyten, TensorBackend, mask_contract_small_leg))
      .def("mask_dagger", &TensorBackend::mask_dagger, nb::arg("mask"))
      .def("mask_from_block",
           &TensorBackend::mask_from_block,
           nb::arg("a"),
           nb::arg("large_leg"),
           DOC(lasyten, TensorBackend, mask_from_block))
      .def("mask_to_block",
           &TensorBackend::mask_to_block,
           nb::arg("a"),
           DOC(lasyten, TensorBackend, mask_to_block))
      .def("mask_to_diagonal", &TensorBackend::mask_to_diagonal, nb::arg("a"), nb::arg("dtype"))
      .def("mask_transpose",
           &TensorBackend::mask_transpose,
           nb::arg("tens"),
           DOC(lasyten, TensorBackend, mask_transpose))
      .def(
        "mask_unary_operand",
        [](TensorBackend& self, MaskCPtr mask, py::function func) {
            return self.mask_unary_operand(mask, block_unary_from_python(func));
        },
        nb::arg("mask"),
        nb::arg("func"),
        DOC(lasyten, TensorBackend, mask_unary_operand))
      .def("move_to_device",
           &TensorBackend::move_to_device,
           nb::arg("a"),
           nb::arg("device"),
           DOC(lasyten, TensorBackend, move_to_device))
      .def("mul", &TensorBackend::mul, nb::arg("a"), nb::arg("b"))
      .def("norm", &TensorBackend::norm, nb::arg("a"), DOC(lasyten, TensorBackend, norm))
      .def("outer",
           &TensorBackend::outer,
           nb::arg("a"),
           nb::arg("b"),
           DOC(lasyten, TensorBackend, outer))
      .def("partial_compose",
           &TensorBackend::partial_compose,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("a_first_leg"),
           nb::arg("new_codomain"),
           nb::arg("new_domain"),
           DOC(lasyten, TensorBackend, partial_compose))
      .def("partial_trace",
           &TensorBackend::partial_trace,
           nb::arg("tensor"),
           nb::arg("pairs"),
           nb::arg("levels"),
           DOC(lasyten, TensorBackend, partial_trace))
      .def("permute_legs",
           &TensorBackend::permute_legs,
           nb::arg("a"),
           nb::arg("codomain_idcs"),
           nb::arg("domain_idcs"),
           nb::arg("new_codomain"),
           nb::arg("new_domain"),
           nb::arg("mixes_codomain_domain"),
           nb::arg("levels"),
           nb::arg("bend_right"),
           DOC(lasyten, TensorBackend, permute_legs))
      .def("qr",
           &TensorBackend::qr,
           nb::arg("a"),
           nb::arg("new_co_domain"),
           DOC(lasyten, TensorBackend, qr))
      .def(
        "reduce_DiagonalTensor",
        [](TensorBackend& self,
           DiagonalTensorCPtr tensor,
           py::function block_func,
           py::function func) {
            return self.reduce_DiagonalTensor(
              tensor, block_to_scalar_from_python(block_func), scalar_reduce_from_python(func));
        },
        nb::arg("tensor"),
        nb::arg("block_func"),
        nb::arg("func"),
        DOC(lasyten, TensorBackend, reduce_DiagonalTensor))
      .def("scale_axis",
           &TensorBackend::scale_axis,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("leg"),
           DOC(lasyten, TensorBackend, scale_axis))
      .def("split_legs",
           &TensorBackend::split_legs,
           nb::arg("a"),
           nb::arg("leg_idcs"),
           nb::arg("new_codomain"),
           nb::arg("new_domain"),
           DOC(lasyten, TensorBackend, split_legs))
      .def("squeeze_legs",
           &TensorBackend::squeeze_legs,
           nb::arg("a"),
           nb::arg("idcs"),
           DOC(lasyten, TensorBackend, squeeze_legs))
      .def("supports_symmetry", &TensorBackend::supports_symmetry, nb::arg("symmetry"))
      .def("svd",
           &TensorBackend::svd,
           nb::arg("a"),
           nb::arg("new_co_domain"),
           nb::arg("algorithm"),
           DOC(lasyten, TensorBackend, svd))
      .def("state_tensor_product",
           &TensorBackend::state_tensor_product,
           nb::arg("state1"),
           nb::arg("state2"),
           nb::arg("pipe"),
           DOC(lasyten, TensorBackend, state_tensor_product))
      .def("to_block_backend",
           &TensorBackend::to_block_backend,
           nb::arg("data"),
           nb::arg("block_backend"),
           nb::arg("dtype").none() = nb::none(),
           nb::arg("device").none() = nb::none())
      .def("to_dense_block",
           &TensorBackend::to_dense_block,
           nb::arg("a"),
           DOC(lasyten, TensorBackend, to_dense_block))
      .def("to_dense_block_trivial_sector",
           &TensorBackend::to_dense_block_trivial_sector,
           nb::arg("tensor"),
           DOC(lasyten, TensorBackend, to_dense_block_trivial_sector))
      .def("to_dtype",
           &TensorBackend::to_dtype,
           nb::arg("a"),
           nb::arg("dtype"),
           DOC(lasyten, TensorBackend, to_dtype))
      .def("trace_full",
           &TensorBackend::trace_full,
           nb::arg("a"),
           nb::arg("idcs1") = std::vector<int64>{},
           nb::arg("idcs2") = std::vector<int64>{})
      .def("truncate_singular_values",
           &TensorBackend::truncate_singular_values,
           nb::arg("S"),
           nb::arg("chi_max"),
           nb::arg("chi_min"),
           nb::arg("degeneracy_tol"),
           nb::arg("trunc_cut"),
           nb::arg("svd_min"),
           nb::arg("minimize_error") = true,
           DOC(lasyten, TensorBackend, truncate_singular_values))
      .def("_truncate_singular_values_selection",
           static_cast<std::tuple<py::array, float64, float64> (TensorBackend::*)(
             py::array,
             py::object,
             std::optional<int64>,
             int64,
             float64,
             float64,
             std::optional<float64>,
             bool)>(&TensorBackend::_truncate_singular_values_selection),
           nb::arg("S"),
           nb::arg("qdims"),
           nb::arg("chi_max"),
           nb::arg("chi_min"),
           nb::arg("degeneracy_tol"),
           nb::arg("trunc_cut"),
           nb::arg("svd_min"),
           nb::arg("minimize_error") = true,
           DOC(lasyten, TensorBackend, _truncate_singular_values_selection))
      .def("zero_data",
           &TensorBackend::zero_data,
           nb::arg("codomain"),
           nb::arg("domain"),
           nb::arg("dtype"),
           nb::arg("device"),
           nb::arg("all_blocks") = false,
           DOC(lasyten, TensorBackend, zero_data))
      .def("zero_diagonal_data",
           &TensorBackend::zero_diagonal_data,
           nb::arg("co_domain"),
           nb::arg("dtype"),
           nb::arg("device"))
      .def(
        "zero_mask_data", &TensorBackend::zero_mask_data, nb::arg("large_leg"), nb::arg("device"))
      .def("is_real", &TensorBackend::is_real, nb::arg("a"), DOC(lasyten, TensorBackend, is_real))
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5<TensorBackend>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"));

    nb::object classmethod = nb::module_::import_("builtins").attr("classmethod");
    tensor_backend.attr("from_hdf5") = classmethod(
      nb::cpp_function(lasyten::hdf5::wrap_from_hdf5_classmethod<TensorBackend>(),
                       nb::name("from_hdf5"),
                       nb::arg("cls"),
                       nb::arg("hdf5_loader"),
                       nb::arg("h5gr"),
                       nb::arg("subpath"),
                       "Reconstruct a tensor backend from HDF5 by loading its BlockBackend."));

    // Nested Data type under TensorBackend for Python parity with BlockBackend.BlockCls style.
    tensor_backend.attr("Data") = data_cls;

    m.def("conventional_leg_order",
          nb::overload_cast<py::object, py::object>(&conventional_leg_order),
          nb::arg("tensor_or_codomain"),
          nb::arg("domain").none() = nb::none(),
          doc_cpp_ref(R"pydoc(conventional_leg_order)pydoc", "lasyten::conventional_leg_order()"));

    m.def(
      "get_same_backend",
      [](nb::args objs_nb, nb::kwargs kwargs_nb) {
          auto kwargs = pykwargs(kwargs_nb);
          auto objs = pyargs(objs_nb);
          std::string error_msg = "Incompatible backends.";
          if (kwargs.contains("error_msg"))
              error_msg = py::cast<std::string>(kwargs["error_msg"]);
          std::vector<py::object> vec;
          vec.reserve(objs.size());
          for (auto const& o : objs)
              vec.emplace_back(py::reinterpret_borrow<py::object>(o));
          return get_same_backend(vec, std::move(error_msg));
      },
      doc_cpp_ref(R"pydoc(get_same_backend)pydoc", "lasyten::get_same_backend()"));
}

} // namespace lasyten
