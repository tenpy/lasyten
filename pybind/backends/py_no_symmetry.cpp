#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "../tensors/py_callbacks.hpp"
#include "docstrings/backends/no_symmetry.h"

#include <lasyten/backends/no_symmetry.h>
#if LASYTEN_WITH_NUMPY_BACKEND
#include <lasyten/block_backend/numpy.h>
#endif
#include <lasyten/block_backend/torch.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/symmetric_tensor.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

/// Convert a Python BlockBackend (often a non-owning factory singleton) to shared_ptr.
std::shared_ptr<BlockBackend>
as_shared_block_backend(py::object obj)
{
#if LASYTEN_WITH_NUMPY_BACKEND
    if (py::isinstance<NumpyBlockBackend>(obj)) {
        auto* p = py::cast<NumpyBlockBackend*>(obj);
        return NumpyBlockBackend::from_factory_shared(p->default_device);
    }
#endif
    if (py::isinstance<TorchBlockBackend>(obj)) {
        auto* p = py::cast<TorchBlockBackend*>(obj);
        return TorchBlockBackend::from_factory_shared(p->default_device);
    }
    // Other backends (e.g. ArrayApi): keep a non-owning shared_ptr.
    auto* raw = py::cast<BlockBackend*>(obj);
    return std::shared_ptr<BlockBackend>(raw, [](BlockBackend*) {});
}

/// Return Block to Python for NoSymmetry ``Data`` results.
BlockBackend::BlockPtr
py_block(TensorBackend::DataPtr d)
{
    return NoSymmetryBackend::unwrap(std::move(d));
}

TensorBackend::DataPtr
py_data(py::object obj)
{
    if (py::isinstance<TensorBackend::Data>(obj))
        return py::cast<TensorBackend::DataPtr>(obj);
    return NoSymmetryBackend::wrap(py::cast<BlockBackend::BlockPtr>(obj));
}

} // namespace

void
bind_no_symmetry_backend(nb::module_& m)
{
    nb::class_<NoSymmetryBackend::BlockData, TensorBackend::Data>(m, "NoSymmetryBackendBlockData")
      .def(lasyten::ctor_for<NoSymmetryBackend::BlockData, BlockBackend::BlockPtr>(),
           nb::arg("block"))
      .def_rw("block", &NoSymmetryBackend::BlockData::block);

    nb::class_<NoSymmetryBackend, TensorBackend> cls(m, "NoSymmetryBackend");
    cls.doc() = DOC(lasyten, NoSymmetryBackend);

    cls.def(nb::new_([](py::object block_backend) {
                return std::make_shared<NoSymmetryBackend>(as_shared_block_backend(block_backend));
            }),
            nb::arg("block_backend"));

    // Static helpers (useful for tests / debugging; not in original Python API).
    cls.def_static("wrap", &NoSymmetryBackend::wrap, nb::arg("block"));
    cls.def_static("unwrap", &NoSymmetryBackend::unwrap, nb::arg("data"));

    // Overrides that return Data → expose Block to Python (match current Python storage).
    cls.def(
      "act_block_diagonal_square_matrix",
      [](NoSymmetryBackend& self,
         SymmetricTensorCPtr a,
         py::function block_method,
         py::object dtype_map) {
          return py_block(self.act_block_diagonal_square_matrix(
            a, block_unary_from_python(block_method), dtype_map_from_python(dtype_map)));
      },
      nb::arg("a"),
      nb::arg("block_method"),
      nb::arg("dtype_map"),
      DOC(lasyten, NoSymmetryBackend, act_block_diagonal_square_matrix));
    cls.def(
      "add_trivial_leg",
      [](NoSymmetryBackend& self,
         TensorCPtr a,
         int64 legs_pos,
         bool add_to_domain,
         int64 co_domain_pos,
         TensorProduct::Ptr new_codomain,
         TensorProduct::Ptr new_domain) {
          return py_block(self.add_trivial_leg(
            a, legs_pos, add_to_domain, co_domain_pos, new_codomain, new_domain));
      },
      nb::arg("a"),
      nb::arg("legs_pos"),
      nb::arg("add_to_domain"),
      nb::arg("co_domain_pos"),
      nb::arg("new_codomain"),
      nb::arg("new_domain"),
      DOC(lasyten, NoSymmetryBackend, add_trivial_leg));
    cls.def(
      "apply_mask_to_DiagonalTensor",
      [](NoSymmetryBackend& self, DiagonalTensorCPtr tensor, MaskCPtr mask) {
          return py_block(self.apply_mask_to_DiagonalTensor(tensor, mask));
      },
      nb::arg("tensor"),
      nb::arg("mask"));
    cls.def(
      "combine_legs",
      [](NoSymmetryBackend& self,
         TensorCPtr tensor,
         std::vector<std::vector<int64>> leg_idcs_combine,
         std::vector<Leg::Ptr> pipes,
         TensorProduct::Ptr new_codomain,
         TensorProduct::Ptr new_domain) {
          return py_block(self.combine_legs(
            tensor, std::move(leg_idcs_combine), std::move(pipes), new_codomain, new_domain));
      },
      nb::arg("tensor"),
      nb::arg("leg_idcs_combine"),
      nb::arg("pipes"),
      nb::arg("new_codomain"),
      nb::arg("new_domain"),
      DOC(lasyten, NoSymmetryBackend, combine_legs));
    cls.def(
      "compose",
      [](NoSymmetryBackend& self, SymmetricTensorCPtr a, SymmetricTensorCPtr b) {
          return py_block(self.compose(a, b));
      },
      nb::arg("a"),
      nb::arg("b"),
      DOC(lasyten, NoSymmetryBackend, compose));
    cls.def(
      "copy_data",
      [](NoSymmetryBackend& self, TensorCPtr a, std::optional<std::string> device) {
          return py_block(self.copy_data(a, std::move(device)));
      },
      nb::arg("a"),
      nb::arg("device").none() = nb::none(),
      DOC(lasyten, NoSymmetryBackend, copy_data));
    cls.def(
      "dagger",
      [](NoSymmetryBackend& self, TensorCPtr a) { return py_block(self.dagger(a)); },
      nb::arg("a"),
      DOC(lasyten, NoSymmetryBackend, dagger));
    cls.def(
      "item",
      [](NoSymmetryBackend& self, TensorCPtr a) { return self.item(a); },
      nb::arg("a"),
      doc_cpp_ref(R"pydoc(item)pydoc", "lasyten::TensorBackend::item()"));
    cls.def(
      "data_item",
      [](NoSymmetryBackend& self, py::object a) { return self.data_item(py_data(a)); },
      nb::arg("a"),
      DOC(lasyten, NoSymmetryBackend, data_item));
    cls.def(
      "diagonal_elementwise_binary",
      [](NoSymmetryBackend& self,
         DiagonalTensorCPtr a,
         DiagonalTensorCPtr b,
         py::function func,
         py::dict func_kwargs,
         bool partial_zero_is_zero) {
          return py_block(self.diagonal_elementwise_binary(
            a, b, block_binary_from_python(func, func_kwargs), partial_zero_is_zero));
      },
      nb::arg("a"),
      nb::arg("b"),
      nb::arg("func"),
      nb::arg("func_kwargs"),
      nb::arg("partial_zero_is_zero"),
      DOC(lasyten, NoSymmetryBackend, diagonal_elementwise_binary));
    cls.def(
      "diagonal_elementwise_unary",
      [](NoSymmetryBackend& self,
         DiagonalTensorCPtr a,
         py::function func,
         py::dict func_kwargs,
         bool maps_zero_to_zero) {
          return py_block(self.diagonal_elementwise_unary(
            a, block_unary_from_python(func, func_kwargs), maps_zero_to_zero));
      },
      nb::arg("a"),
      nb::arg("func"),
      nb::arg("func_kwargs"),
      nb::arg("maps_zero_to_zero"),
      DOC(lasyten, NoSymmetryBackend, diagonal_elementwise_unary));
    cls.def(
      "diagonal_from_block",
      [](NoSymmetryBackend& self,
         BlockBackend::BlockPtr a,
         TensorProduct::Ptr co_domain,
         float64 tol) { return py_block(self.diagonal_from_block(std::move(a), co_domain, tol)); },
      nb::arg("a"),
      nb::arg("co_domain"),
      nb::arg("tol"),
      DOC(lasyten, NoSymmetryBackend, diagonal_from_block));
    cls.def(
      "diagonal_from_sector_block_func",
      [](NoSymmetryBackend& self, py::function func, TensorProduct::Ptr co_domain) {
          return py_block(self.diagonal_from_sector_block_func(
            sector_block_factory_from_python(func), co_domain));
      },
      nb::arg("func"),
      nb::arg("co_domain"),
      DOC(lasyten, NoSymmetryBackend, diagonal_from_sector_block_func));
    cls.def(
      "diagonal_tensor_from_full_tensor",
      [](NoSymmetryBackend& self, SymmetricTensorCPtr a, std::optional<float64> tol) {
          return py_block(self.diagonal_tensor_from_full_tensor(a, tol));
      },
      nb::arg("a"),
      nb::arg("tol") = 1e-12,
      DOC(lasyten, NoSymmetryBackend, diagonal_tensor_from_full_tensor));
    cls.def(
      "diagonal_to_mask",
      [](NoSymmetryBackend& self, DiagonalTensorCPtr tens) {
          auto [data, leg] = self.diagonal_to_mask(tens);
          return std::make_tuple(py_block(std::move(data)), std::move(leg));
      },
      nb::arg("tens"),
      DOC(lasyten, NoSymmetryBackend, diagonal_to_mask));
    cls.def(
      "diagonal_transpose",
      [](NoSymmetryBackend& self, DiagonalTensorCPtr tens) {
          auto [leg, data] = self.diagonal_transpose(tens);
          return std::make_tuple(std::move(leg), py_block(std::move(data)));
      },
      nb::arg("tens"),
      DOC(lasyten, NoSymmetryBackend, diagonal_transpose));
    cls.def(
      "eigh",
      [](NoSymmetryBackend& self,
         SymmetricTensorCPtr a,
         bool new_leg_dual,
         std::optional<std::string> sort) {
          auto [w, v, leg] = self.eigh(a, new_leg_dual, std::move(sort));
          return std::make_tuple(py_block(std::move(w)), py_block(std::move(v)), std::move(leg));
      },
      nb::arg("a"),
      nb::arg("new_leg_dual"),
      nb::arg("sort").none() = nb::none(),
      DOC(lasyten, NoSymmetryBackend, eigh));
    cls.def(
      "eig",
      [](NoSymmetryBackend& self,
         SymmetricTensorCPtr a,
         bool new_leg_dual,
         std::optional<std::string> sort) {
          auto [w, v, leg] = self.eig(a, new_leg_dual, std::move(sort));
          return std::make_tuple(py_block(std::move(w)), py_block(std::move(v)), std::move(leg));
      },
      nb::arg("a"),
      nb::arg("new_leg_dual"),
      nb::arg("sort").none() = nb::none(),
      DOC(lasyten, NoSymmetryBackend, eig));
    cls.def(
      "eigvalsh",
      [](NoSymmetryBackend& self,
         SymmetricTensorCPtr a,
         bool new_leg_dual,
         std::optional<std::string> sort) {
          auto [w, leg] = self.eigvalsh(a, new_leg_dual, std::move(sort));
          return std::make_tuple(py_block(std::move(w)), std::move(leg));
      },
      nb::arg("a"),
      nb::arg("new_leg_dual"),
      nb::arg("sort").none() = nb::none(),
      DOC(lasyten, NoSymmetryBackend, eigvalsh));
    cls.def(
      "eigvals",
      [](NoSymmetryBackend& self,
         SymmetricTensorCPtr a,
         bool new_leg_dual,
         std::optional<std::string> sort) {
          auto [w, leg] = self.eigvals(a, new_leg_dual, std::move(sort));
          return std::make_tuple(py_block(std::move(w)), std::move(leg));
      },
      nb::arg("a"),
      nb::arg("new_leg_dual"),
      nb::arg("sort").none() = nb::none(),
      DOC(lasyten, NoSymmetryBackend, eigvals));
    cls.def(
      "eye_data",
      [](NoSymmetryBackend& self, TensorProduct::Ptr co_domain, Dtype dtype, std::string device) {
          return py_block(self.eye_data(co_domain, dtype, std::move(device)));
      },
      nb::arg("co_domain"),
      nb::arg("dtype"),
      nb::arg("device"),
      DOC(lasyten, NoSymmetryBackend, eye_data));
    cls.def(
      "from_dense_block",
      [](NoSymmetryBackend& self,
         BlockBackend::BlockPtr a,
         TensorProduct::Ptr codomain,
         TensorProduct::Ptr domain,
         float64 tol) {
          return py_block(self.from_dense_block(std::move(a), codomain, domain, tol));
      },
      nb::arg("a"),
      nb::arg("codomain"),
      nb::arg("domain"),
      nb::arg("tol"),
      DOC(lasyten, NoSymmetryBackend, from_dense_block));
    cls.def(
      "from_dense_block_trivial_sector",
      [](NoSymmetryBackend& self, BlockBackend::BlockPtr block, Space::Ptr leg) {
          return py_block(self.from_dense_block_trivial_sector(std::move(block), leg));
      },
      nb::arg("block"),
      nb::arg("leg"),
      DOC(lasyten, NoSymmetryBackend, from_dense_block_trivial_sector));
    cls.def(
      "from_grid",
      [](NoSymmetryBackend& self,
         py::object grid,
         TensorProduct::Ptr new_codomain,
         TensorProduct::Ptr new_domain,
         std::vector<std::vector<int64>> left_mult_slices,
         std::vector<std::vector<int64>> right_mult_slices,
         Dtype dtype,
         std::string device) {
          return py_block(self.from_grid(tensor_grid_from_python(std::move(grid)),
                                         new_codomain,
                                         new_domain,
                                         std::move(left_mult_slices),
                                         std::move(right_mult_slices),
                                         dtype,
                                         std::move(device)));
      },
      nb::arg("grid"),
      nb::arg("new_codomain"),
      nb::arg("new_domain"),
      nb::arg("left_mult_slices"),
      nb::arg("right_mult_slices"),
      nb::arg("dtype"),
      nb::arg("device"),
      DOC(lasyten, NoSymmetryBackend, from_grid));
    cls.def(
      "from_random_normal",
      [](NoSymmetryBackend& self,
         TensorProduct::Ptr codomain,
         TensorProduct::Ptr domain,
         float64 sigma,
         Dtype dtype,
         std::string device) {
          return py_block(
            self.from_random_normal(codomain, domain, sigma, dtype, std::move(device)));
      },
      nb::arg("codomain"),
      nb::arg("domain"),
      nb::arg("sigma"),
      nb::arg("dtype"),
      nb::arg("device"));
    cls.def(
      "from_sector_block_func",
      [](NoSymmetryBackend& self,
         py::function func,
         TensorProduct::Ptr codomain,
         TensorProduct::Ptr domain) {
          return py_block(
            self.from_sector_block_func(sector_block_factory_from_python(func), codomain, domain));
      },
      nb::arg("func"),
      nb::arg("codomain"),
      nb::arg("domain"),
      DOC(lasyten, NoSymmetryBackend, from_sector_block_func));
    cls.def(
      "from_tree_pairs",
      [](NoSymmetryBackend& self,
         std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
         TensorProduct::Ptr codomain,
         TensorProduct::Ptr domain,
         Dtype dtype,
         std::string device) {
          return py_block(
            self.from_tree_pairs(std::move(trees), codomain, domain, dtype, std::move(device)));
      },
      nb::arg("trees"),
      nb::arg("codomain"),
      nb::arg("domain"),
      nb::arg("dtype"),
      nb::arg("device"),
      DOC(lasyten, NoSymmetryBackend, from_tree_pairs));
    cls.def(
      "full_data_from_diagonal_tensor",
      [](NoSymmetryBackend& self, DiagonalTensorCPtr a) {
          return py_block(self.full_data_from_diagonal_tensor(a));
      },
      nb::arg("a"));
    cls.def(
      "full_data_from_mask",
      [](NoSymmetryBackend& self, MaskCPtr a, Dtype dtype) {
          return py_block(self.full_data_from_mask(a, dtype));
      },
      nb::arg("a"),
      nb::arg("dtype"),
      DOC(lasyten, NoSymmetryBackend, full_data_from_mask));
    cls.def(
      "get_device_from_data",
      [](NoSymmetryBackend& self, py::object a) { return self.get_device_from_data(py_data(a)); },
      nb::arg("a"),
      DOC(lasyten, NoSymmetryBackend, get_device_from_data));
    cls.def(
      "get_dtype_from_data",
      [](NoSymmetryBackend& self, py::object a) { return self.get_dtype_from_data(py_data(a)); },
      nb::arg("a"));
    cls.def(
      "inv_part_from_dense_block_single_sector",
      [](NoSymmetryBackend& self,
         BlockBackend::BlockPtr vector,
         Space::Ptr space,
         ElementarySpace::Ptr charge_leg) {
          return py_block(
            self.inv_part_from_dense_block_single_sector(std::move(vector), space, charge_leg));
      },
      nb::arg("vector"),
      nb::arg("space"),
      nb::arg("charge_leg"),
      DOC(lasyten, NoSymmetryBackend, inv_part_from_dense_block_single_sector));
    cls.def(
      "linear_combination",
      [](NoSymmetryBackend& self,
         BlockBackend::Scalar a,
         TensorCPtr v,
         BlockBackend::Scalar b,
         TensorCPtr w) { return py_block(self.linear_combination(a, v, b, w)); },
      nb::arg("a"),
      nb::arg("v"),
      nb::arg("b"),
      nb::arg("w"),
      DOC(lasyten, NoSymmetryBackend, linear_combination));
    cls.def(
      "lq",
      [](NoSymmetryBackend& self, SymmetricTensorCPtr tensor, TensorProduct::Ptr new_co_domain) {
          auto [l, q] = self.lq(tensor, new_co_domain);
          return std::make_tuple(py_block(std::move(l)), py_block(std::move(q)));
      },
      nb::arg("tensor"),
      nb::arg("new_co_domain"),
      DOC(lasyten, NoSymmetryBackend, lq));
    cls.def(
      "mask_binary_operand",
      [](NoSymmetryBackend& self, MaskCPtr mask1, MaskCPtr mask2, py::function func) {
          auto [data, leg] =
            self.mask_binary_operand(mask1, mask2, block_binary_from_python(func));
          return std::make_tuple(py_block(std::move(data)), std::move(leg));
      },
      nb::arg("mask1"),
      nb::arg("mask2"),
      nb::arg("func"),
      DOC(lasyten, NoSymmetryBackend, mask_binary_operand));
    cls.def(
      "mask_contract_large_leg",
      [](NoSymmetryBackend& self, TensorCPtr tensor, MaskCPtr mask, int64 leg_idx) {
          auto [data, codomain, domain] = self.mask_contract_large_leg(tensor, mask, leg_idx);
          return std::make_tuple(
            py_block(std::move(data)), std::move(codomain), std::move(domain));
      },
      nb::arg("tensor"),
      nb::arg("mask"),
      nb::arg("leg_idx"),
      DOC(lasyten, NoSymmetryBackend, mask_contract_large_leg));
    cls.def(
      "mask_contract_small_leg",
      [](NoSymmetryBackend& self, TensorCPtr tensor, MaskCPtr mask, int64 leg_idx) {
          auto [data, codomain, domain] = self.mask_contract_small_leg(tensor, mask, leg_idx);
          return std::make_tuple(
            py_block(std::move(data)), std::move(codomain), std::move(domain));
      },
      nb::arg("tensor"),
      nb::arg("mask"),
      nb::arg("leg_idx"),
      DOC(lasyten, NoSymmetryBackend, mask_contract_small_leg));
    cls.def(
      "mask_dagger",
      [](NoSymmetryBackend& self, MaskCPtr mask) { return py_block(self.mask_dagger(mask)); },
      nb::arg("mask"));
    cls.def(
      "mask_from_block",
      [](NoSymmetryBackend& self, BlockBackend::BlockPtr a, Space::Ptr large_leg) {
          auto [data, leg] = self.mask_from_block(std::move(a), large_leg);
          return std::make_tuple(py_block(std::move(data)), std::move(leg));
      },
      nb::arg("a"),
      nb::arg("large_leg"),
      DOC(lasyten, NoSymmetryBackend, mask_from_block));
    cls.def(
      "mask_to_diagonal",
      [](NoSymmetryBackend& self, MaskCPtr a, Dtype dtype) {
          return py_block(self.mask_to_diagonal(a, dtype));
      },
      nb::arg("a"),
      nb::arg("dtype"));
    cls.def(
      "mask_transpose",
      [](NoSymmetryBackend& self, MaskCPtr tens) {
          auto [s_in, s_out, data] = self.mask_transpose(tens);
          return std::make_tuple(std::move(s_in), std::move(s_out), py_block(std::move(data)));
      },
      nb::arg("tens"),
      DOC(lasyten, NoSymmetryBackend, mask_transpose));
    cls.def(
      "mask_unary_operand",
      [](NoSymmetryBackend& self, MaskCPtr mask, py::function func) {
          auto [data, leg] = self.mask_unary_operand(mask, block_unary_from_python(func));
          return std::make_tuple(py_block(std::move(data)), std::move(leg));
      },
      nb::arg("mask"),
      nb::arg("func"),
      DOC(lasyten, NoSymmetryBackend, mask_unary_operand));
    cls.def(
      "move_to_device",
      [](NoSymmetryBackend& self, TensorCPtr a, std::string device) {
          return py_block(self.move_to_device(a, std::move(device)));
      },
      nb::arg("a"),
      nb::arg("device"),
      DOC(lasyten, NoSymmetryBackend, move_to_device));
    cls.def(
      "mul",
      [](NoSymmetryBackend& self, BlockBackend::Scalar a, TensorCPtr b) {
          return py_block(self.mul(a, b));
      },
      nb::arg("a"),
      nb::arg("b"));
    cls.def(
      "outer",
      [](NoSymmetryBackend& self, SymmetricTensorCPtr a, SymmetricTensorCPtr b) {
          return py_block(self.outer(a, b));
      },
      nb::arg("a"),
      nb::arg("b"),
      DOC(lasyten, NoSymmetryBackend, outer));
    cls.def(
      "partial_compose",
      [](NoSymmetryBackend& self,
         SymmetricTensorCPtr a,
         SymmetricTensorCPtr b,
         int64 a_first_leg,
         TensorProduct::Ptr new_codomain,
         TensorProduct::Ptr new_domain) {
          return py_block(self.partial_compose(a, b, a_first_leg, new_codomain, new_domain));
      },
      nb::arg("a"),
      nb::arg("b"),
      nb::arg("a_first_leg"),
      nb::arg("new_codomain"),
      nb::arg("new_domain"),
      DOC(lasyten, NoSymmetryBackend, partial_compose));
    cls.def(
      "partial_trace",
      [](NoSymmetryBackend& self,
         SymmetricTensorCPtr tensor,
         std::vector<std::pair<int64, int64>> pairs,
         std::vector<std::optional<int64>> levels) -> py::object {
          auto [data, codomain, domain] =
            self.partial_trace(tensor, std::move(pairs), std::move(levels));
          if (!codomain && !domain) {
              // Match Python: return scalar item when fully traced.
              return py::make_tuple(
                self.block_backend->item(py_block(data)), py::none(), py::none());
          }
          return py::make_tuple(py_block(std::move(data)), std::move(codomain), std::move(domain));
      },
      nb::arg("tensor"),
      nb::arg("pairs"),
      nb::arg("levels").none() = nb::none(),
      DOC(lasyten, NoSymmetryBackend, partial_trace));
    cls.def(
      "permute_legs",
      [](NoSymmetryBackend& self,
         TensorCPtr a,
         std::vector<int64> codomain_idcs,
         std::vector<int64> domain_idcs,
         TensorProduct::Ptr new_codomain,
         TensorProduct::Ptr new_domain,
         bool mixes_codomain_domain,
         std::vector<std::optional<int64>> levels,
         std::vector<std::optional<bool>> bend_right) {
          return py_block(self.permute_legs(a,
                                            std::move(codomain_idcs),
                                            std::move(domain_idcs),
                                            new_codomain,
                                            new_domain,
                                            mixes_codomain_domain,
                                            std::move(levels),
                                            std::move(bend_right)));
      },
      nb::arg("a"),
      nb::arg("codomain_idcs"),
      nb::arg("domain_idcs"),
      nb::arg("new_codomain"),
      nb::arg("new_domain"),
      nb::arg("mixes_codomain_domain"),
      nb::arg("levels"),
      nb::arg("bend_right"),
      DOC(lasyten, NoSymmetryBackend, permute_legs));
    cls.def(
      "qr",
      [](NoSymmetryBackend& self, SymmetricTensorCPtr a, TensorProduct::Ptr new_co_domain) {
          auto [q, r] = self.qr(a, new_co_domain);
          return std::make_tuple(py_block(std::move(q)), py_block(std::move(r)));
      },
      nb::arg("a"),
      nb::arg("new_co_domain"),
      DOC(lasyten, NoSymmetryBackend, qr));
    cls.def(
      "scale_axis",
      [](NoSymmetryBackend& self, TensorCPtr a, DiagonalTensorCPtr b, int64 leg) {
          return py_block(self.scale_axis(a, b, leg));
      },
      nb::arg("a"),
      nb::arg("b"),
      nb::arg("leg"),
      DOC(lasyten, NoSymmetryBackend, scale_axis));
    cls.def(
      "split_legs",
      [](NoSymmetryBackend& self,
         TensorCPtr a,
         std::vector<int64> leg_idcs,
         TensorProduct::Ptr new_codomain,
         TensorProduct::Ptr new_domain) {
          return py_block(self.split_legs(a, std::move(leg_idcs), new_codomain, new_domain));
      },
      nb::arg("a"),
      nb::arg("leg_idcs"),
      nb::arg("new_codomain"),
      nb::arg("new_domain"),
      DOC(lasyten, NoSymmetryBackend, split_legs));
    cls.def(
      "squeeze_legs",
      [](NoSymmetryBackend& self, TensorCPtr a, std::vector<int64> idcs) {
          return py_block(self.squeeze_legs(a, std::move(idcs)));
      },
      nb::arg("a"),
      nb::arg("idcs"),
      DOC(lasyten, NoSymmetryBackend, squeeze_legs));
    cls.def(
      "svd",
      [](NoSymmetryBackend& self,
         SymmetricTensorCPtr a,
         TensorProduct::Ptr new_co_domain,
         std::optional<std::string> algorithm) {
          auto [u, s, vh] = self.svd(a, new_co_domain, std::move(algorithm));
          return std::make_tuple(
            py_block(std::move(u)), py_block(std::move(s)), py_block(std::move(vh)));
      },
      nb::arg("a"),
      nb::arg("new_co_domain"),
      nb::arg("algorithm").none() = nb::none(),
      DOC(lasyten, NoSymmetryBackend, svd));
    cls.def(
      "to_block_backend",
      [](NoSymmetryBackend& self,
         py::object data,
         py::object block_backend,
         std::optional<Dtype> dtype,
         std::optional<std::string> device) {
          return py_block(self.to_block_backend(
            py_data(data), as_shared_block_backend(block_backend), dtype, std::move(device)));
      },
      nb::arg("data"),
      nb::arg("block_backend"),
      nb::arg("dtype").none() = nb::none(),
      nb::arg("device").none() = nb::none());
    cls.def(
      "to_dtype",
      [](NoSymmetryBackend& self, TensorCPtr a, Dtype dtype) {
          return py_block(self.to_dtype(a, dtype));
      },
      nb::arg("a"),
      nb::arg("dtype"),
      DOC(lasyten, NoSymmetryBackend, to_dtype));
    cls.def(
      "trace_full",
      [](NoSymmetryBackend& self,
         SymmetricTensorCPtr a,
         std::vector<int64> idcs1,
         std::vector<int64> idcs2) {
          return self.trace_full(a, std::move(idcs1), std::move(idcs2));
      },
      nb::arg("a"),
      nb::arg("idcs1") = std::vector<int64>{},
      nb::arg("idcs2") = std::vector<int64>{});
    cls.def(
      "truncate_singular_values",
      [](NoSymmetryBackend& self,
         DiagonalTensorCPtr S,
         std::optional<int64> chi_max,
         int64 chi_min,
         float64 degeneracy_tol,
         float64 trunc_cut,
         std::optional<float64> svd_min,
         bool minimize_error) {
          auto [mask, leg, err, new_norm] = self.truncate_singular_values(
            S, chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min, minimize_error);
          return std::make_tuple(py_block(std::move(mask)), std::move(leg), err, new_norm);
      },
      nb::arg("S"),
      nb::arg("chi_max"),
      nb::arg("chi_min"),
      nb::arg("degeneracy_tol"),
      nb::arg("trunc_cut"),
      nb::arg("svd_min"),
      nb::arg("minimize_error") = true,
      DOC(lasyten, NoSymmetryBackend, truncate_singular_values));
    cls.def(
      "zero_data",
      [](NoSymmetryBackend& self,
         TensorProduct::Ptr codomain,
         TensorProduct::Ptr domain,
         Dtype dtype,
         std::string device,
         bool all_blocks) {
          return py_block(self.zero_data(codomain, domain, dtype, std::move(device), all_blocks));
      },
      nb::arg("codomain"),
      nb::arg("domain"),
      nb::arg("dtype"),
      nb::arg("device"),
      nb::arg("all_blocks") = false,
      DOC(lasyten, NoSymmetryBackend, zero_data));
    cls.def(
      "zero_diagonal_data",
      [](NoSymmetryBackend& self, TensorProduct::Ptr co_domain, Dtype dtype, std::string device) {
          return py_block(self.zero_diagonal_data(co_domain, dtype, std::move(device)));
      },
      nb::arg("co_domain"),
      nb::arg("dtype"),
      nb::arg("device"));
    cls.def(
      "zero_mask_data",
      [](NoSymmetryBackend& self, Space::Ptr large_leg, std::string device) {
          return py_block(self.zero_mask_data(large_leg, std::move(device)));
      },
      nb::arg("large_leg"),
      nb::arg("device"));
}

} // namespace lasyten
