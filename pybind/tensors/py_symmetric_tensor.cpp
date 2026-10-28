#include <lasyten/backends/no_symmetry.h>
#include <lasyten/tensors/symmetric_tensor.h>

#include "py_callbacks.hpp"
#include "py_factory_parse.hpp"
#include "py_trampolines.hpp"

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"

#include "docstrings/tensors/symmetric_tensor.h"

#include <pybind11/stl.h>

#include "tools/hdf5_bind.h"
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

namespace {

std::optional<std::vector<std::variant<int64, std::string>>>
optional_leg_order(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    std::vector<std::variant<int64, std::string>> out;
    for (auto item : to_iterable(obj)) {
        if (py::isinstance<py::str>(item)) {
            out.emplace_back(item.cast<std::string>());
        } else {
            out.emplace_back(item.cast<int64>());
        }
    }
    return out;
}

} // namespace

namespace {

std::optional<py::class_<SymmetricTensor, Tensor, PySymmetricTensor, py::smart_holder>>
  g_symmetric_tensor;

} // namespace

void
declare_tensors_symmetric_tensor(py::module_& m)
{
    g_symmetric_tensor = py::class_<SymmetricTensor, Tensor, PySymmetricTensor, py::smart_holder>(
      m, "SymmetricTensor");
    (*g_symmetric_tensor).doc() = DOC(lasyten, SymmetricTensor);
}

void
bind_tensors_symmetric_tensor(py::module_& m)
{
    auto& cls = *g_symmetric_tensor;

    cls.def(
      py::init([](TensorBackend::DataPtr data,
                  py::object codomain,
                  py::object domain,
                  TensorBackend::Ptr backend,
                  py::object labels) {
          auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
          return std::make_shared<SymmetricTensor>(
            std::move(data), init.codomain, init.domain, init.backend, init.symmetry, init.labels);
      }),
      py::arg("data"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none());

    cls.def_property(
      "data",
      [](SymmetricTensor& self) -> py::object {
          // Match Python NoSymmetryBackend: expose the raw Block, not BlockData wrapper.
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              return py::cast(NoSymmetryBackend::unwrap(self.data));
          }
          return py::cast(self.data);
      },
      [](SymmetricTensor& self, py::object obj) {
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              self.data = NoSymmetryBackend::wrap(obj.cast<BlockBackend::BlockPtr>());
          } else {
              self.data = obj.cast<TensorBackend::DataPtr>();
          }
      });

    cls.def(
      "test_sanity", &SymmetricTensor::test_sanity, DOC(lasyten, SymmetricTensor, test_sanity));
    cls.def("verify_dtype", &SymmetricTensor::verify_dtype);

    cls.def_static(
      "from_block_func",
      [](py::function func,
         py::object codomain,
         py::object domain,
         TensorBackend::Ptr backend,
         py::object labels,
         py::object func_kwargs,
         std::optional<std::string> shape_kw,
         std::optional<Dtype> dtype,
         std::optional<std::string> device) {
          auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
          auto dt = SymmetricTensor::_parse_default_dtype(dtype, init.symmetry);
          auto wrapped = block_factory_from_python(
            func, func_kwargs, shape_kw, init.backend->block_backend, dt, device);
          return SymmetricTensor::from_block_func(
            std::move(wrapped), init.codomain, init.domain, init.backend, init.labels, dt, device);
      },
      py::arg("func"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("func_kwargs") = py::none(),
      py::arg("shape_kw") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      DOC(lasyten, SymmetricTensor, from_block_func));

    cls.def_static(
      "from_dense_block",
      [](py::object block,
         py::object codomain,
         py::object domain,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<Dtype> dtype,
         std::optional<std::string> device,
         float64 tol,
         bool understood_braiding) {
          auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
          auto block_ptr = init.backend->block_backend->as_block(block, dtype, device);
          return SymmetricTensor::from_dense_block(block_ptr,
                                                   init.codomain,
                                                   init.domain,
                                                   init.backend,
                                                   init.labels,
                                                   dtype,
                                                   device,
                                                   tol,
                                                   understood_braiding);
      },
      py::arg("block"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      py::arg("tol") = 1e-6,
      py::arg("understood_braiding") = false,
      DOC(lasyten, SymmetricTensor, from_dense_block));

    cls.def_static(
      "from_dense_block_trivial_sector",
      [](py::object vector,
         Leg::Ptr space,
         TensorBackend::Ptr backend,
         std::optional<std::string> device,
         std::string label) {
          if (!backend) {
              backend = get_backend(space->symmetry);
          }
          auto vec = backend->block_backend->as_block(vector, std::nullopt, device);
          return SymmetricTensor::from_dense_block_trivial_sector(
            vec, std::move(space), std::move(backend), device, std::move(label));
      },
      py::arg("vector"),
      py::arg("space"),
      py::arg("backend") = nullptr,
      py::arg("device") = py::none(),
      py::arg("label") = "",
      DOC(lasyten, SymmetricTensor, from_dense_block_trivial_sector));

    cls.def_static(
      "from_eye",
      [](py::object co_domain,
         TensorBackend::Ptr backend,
         py::object labels,
         Dtype dtype,
         std::optional<std::string> device) {
          auto init = parse_tensor_init(co_domain,
                                        co_domain,
                                        std::move(backend),
                                        labels,
                                        /*is_endomorphism=*/true);
          return SymmetricTensor::from_eye(
            init.codomain, init.backend, init.labels, dtype, device);
      },
      py::arg("co_domain"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      DOC(lasyten, SymmetricTensor, from_eye));

    cls.def_static(
      "from_random_normal",
      [](py::object codomain,
         py::object domain,
         py::object mean,
         float64 sigma,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<Dtype> dtype,
         std::optional<std::string> device) {
          auto mean_t = py_optional_tensor(mean);
          TensorProduct::Ptr c;
          TensorProduct::Ptr d;
          std::optional<LegLabelList> labs;
          if (!codomain.is_none()) {
              auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
              c = init.codomain;
              d = init.domain;
              backend = init.backend;
              labs = init.labels;
          } else if (!labels.is_none() && mean_t) {
              labs = parse_tensor_init_labels(labels, mean_t->codomain, mean_t->domain);
          }
          return SymmetricTensor::from_random_normal(std::move(c),
                                                     std::move(d),
                                                     mean_t,
                                                     sigma,
                                                     std::move(backend),
                                                     std::move(labs),
                                                     dtype,
                                                     device);
      },
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("mean") = py::none(),
      py::arg("sigma") = 1.0,
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      DOC(lasyten, SymmetricTensor, from_random_normal));

    cls.def_static(
      "from_random_uniform",
      [](py::object codomain,
         py::object domain,
         TensorBackend::Ptr backend,
         py::object labels,
         Dtype dtype,
         std::optional<std::string> device) {
          auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
          return SymmetricTensor::from_random_uniform(
            init.codomain, init.domain, init.backend, init.labels, dtype, device);
      },
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      DOC(lasyten, SymmetricTensor, from_random_uniform));

    cls.def_static(
      "from_sector_block_func",
      [](py::function func,
         py::object codomain,
         py::object domain,
         TensorBackend::Ptr backend,
         py::object labels,
         py::object func_kwargs,
         std::optional<Dtype> dtype,
         std::optional<std::string> device) {
          auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
          auto dt = SymmetricTensor::_parse_default_dtype(dtype, init.symmetry);
          auto wrapped = sector_block_factory_from_python(
            func, func_kwargs, init.backend->block_backend, dt, device);
          return SymmetricTensor::from_sector_block_func(
            std::move(wrapped), init.codomain, init.domain, init.backend, init.labels, dt, device);
      },
      py::arg("func"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("func_kwargs") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      DOC(lasyten, SymmetricTensor, from_sector_block_func));

    cls.def_static(
      "from_sector_projection",
      [](py::object co_domain,
         Sector sector,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<Dtype> dtype,
         std::optional<std::string> device) {
          auto init = parse_tensor_init(co_domain,
                                        co_domain,
                                        std::move(backend),
                                        labels,
                                        /*is_endomorphism=*/true);
          return SymmetricTensor::from_sector_projection(
            init.codomain, sector, init.backend, init.labels, dtype, device);
      },
      py::arg("co_domain"),
      py::arg("sector"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      DOC(lasyten, SymmetricTensor, from_sector_projection));

    cls.def_static(
      "from_tree_pairs",
      [](py::object trees,
         py::object codomain,
         py::object domain,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<Dtype> dtype,
         std::optional<std::string> device) {
          auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
          return SymmetricTensor::from_tree_pairs(
            trees, init.codomain, init.domain, init.backend, init.labels, dtype, device);
      },
      py::arg("trees"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      DOC(lasyten, SymmetricTensor, from_tree_pairs));

    cls.def_static(
      "from_zero",
      [](py::object codomain,
         py::object domain,
         TensorBackend::Ptr backend,
         py::object labels,
         Dtype dtype,
         std::optional<std::string> device) {
          auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
          return SymmetricTensor::from_zero(
            init.codomain, init.domain, init.backend, init.labels, dtype, device);
      },
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      DOC(lasyten, SymmetricTensor, from_zero));

    cls.def_static("_parse_default_dtype",
                   &SymmetricTensor::_parse_default_dtype,
                   py::arg("dtype"),
                   py::arg("symmetry"));

    cls.def("as_dtype",
            &SymmetricTensor::as_dtype,
            py::arg("dtype"),
            DOC(lasyten, SymmetricTensor, as_dtype));
    cls.def("as_SymmetricTensor",
            &SymmetricTensor::as_SymmetricTensor,
            py::arg("guarantee_copy") = false,
            py::arg("warning") = py::none(),
            DOC(lasyten, SymmetricTensor, as_SymmetricTensor));
    cls.def("copy",
            &SymmetricTensor::copy,
            py::arg("deep") = true,
            py::arg("device") = py::none(),
            py::arg("dtype") = py::none(),
            DOC(lasyten, SymmetricTensor, copy));
    cls.def("diagonal",
            &SymmetricTensor::diagonal,
            py::arg("check_offdiagonal") = false,
            DOC(lasyten, SymmetricTensor, diagonal));
    cls.def("_get_item",
            &SymmetricTensor::_get_item,
            py::arg("idx"),
            DOC(lasyten, SymmetricTensor, _get_item));
    cls.def("move_to_device",
            &SymmetricTensor::move_to_device,
            py::arg("device"),
            DOC(lasyten, SymmetricTensor, move_to_device));
    cls.def("to_backend",
            &SymmetricTensor::to_backend,
            py::arg("backend"),
            py::arg("dtype") = py::none(),
            py::arg("device") = py::none(),
            DOC(lasyten, SymmetricTensor, to_backend));
    cls.def(
      "to_dense_block",
      [](SymmetricTensor& self,
         py::object leg_order,
         std::optional<Dtype> dtype,
         bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false,
      DOC(lasyten, SymmetricTensor, to_dense_block));
    cls.def("to_dense_block_trivial_sector",
            &SymmetricTensor::to_dense_block_trivial_sector,
            DOC(lasyten, SymmetricTensor, to_dense_block_trivial_sector));
    cls.def("save_hdf5",
            lasyten::hdf5::wrap_save_hdf5_const<SymmetricTensor>(),
            py::arg("hdf5_saver"),
            py::arg("h5gr"),
            py::arg("subpath"),
            DOC(lasyten, SymmetricTensor, save_hdf5));
    cls.def_static("from_hdf5",
                   lasyten::hdf5::wrap_from_hdf5<SymmetricTensor>(),
                   py::arg("hdf5_loader"),
                   py::arg("h5gr"),
                   py::arg("subpath"),
                   DOC(lasyten, SymmetricTensor, from_hdf5));
}

} // namespace lasyten
