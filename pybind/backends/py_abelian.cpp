#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "docstrings/backends/abelian.h"

#include "backends/casters.hpp"

#include <lasyten/backends/abelian.h>
#include <lasyten/backends/block_inds_numpy.h>
#if LASYTEN_WITH_NUMPY_BACKEND
#include <lasyten/block_backend/numpy.h>
#endif
#include <lasyten/block_backend/torch.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/symmetric_tensor.h>

#include "tools/hdf5_bind.h"
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

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
    auto* raw = py::cast<BlockBackend*>(obj);
    return std::shared_ptr<BlockBackend>(raw, [](BlockBackend*) {});
}

} // namespace

void
bind_abelian_backend_data(nb::module_& m)
{
    nb::class_<AbelianBackendData, TensorBackend::Data> cls(m, "AbelianBackendData");
    cls.doc() = DOC(lasyten, AbelianBackendData);

    cls.def(nb::new_([](Dtype dtype,
                        std::string device,
                        std::vector<BlockBackend::BlockPtr> blocks,
                        BlockInds block_inds,
                        bool is_sorted) {
                return std::make_shared<AbelianBackendData>(std::move(dtype),
                                                            std::move(device),
                                                            std::move(blocks),
                                                            std::move(block_inds),
                                                            is_sorted);
            }),
            nb::arg("dtype"),
            nb::arg("device"),
            nb::arg("blocks"),
            nb::arg("block_inds"),
            nb::arg("is_sorted") = false);

    cls.def_rw("dtype", &AbelianBackendData::dtype)
      .def_rw("device", &AbelianBackendData::device)
      .def_rw("blocks", &AbelianBackendData::blocks)
      .def_rw("block_inds", &AbelianBackendData::block_inds);

    cls
      .def("get_block_num",
           &AbelianBackendData::get_block_num,
           nb::arg("block_inds"),
           DOC(lasyten, AbelianBackendData, get_block_num))
      .def("get_block",
           &AbelianBackendData::get_block,
           nb::arg("block_inds"),
           DOC(lasyten, AbelianBackendData, get_block))
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<AbelianBackendData>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<AbelianBackendData>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
}

void
bind_abelian_backend(nb::module_& m)
{
    // AbelianBackendData must already be registered (bind_abelian_backend_data).
    nb::class_<AbelianBackend, TensorBackend> cls(m, "AbelianBackend");
    cls.doc() = DOC(lasyten, AbelianBackend);

    cls.def(nb::new_([](py::object block_backend) {
                return std::make_shared<AbelianBackend>(as_shared_block_backend(block_backend));
            }),
            nb::arg("block_backend"));

    cls.def_static("wrap", &AbelianBackend::wrap, nb::arg("data"));
    cls.def_static("unwrap", &AbelianBackend::unwrap, nb::arg("data"));

    cls.def("leg_pipe_map_incoming_block_inds",
            &AbelianBackend::leg_pipe_map_incoming_block_inds,
            nb::arg("pipe"),
            nb::arg("incoming_block_inds"),
            DOC(lasyten, AbelianBackend, leg_pipe_map_incoming_block_inds));

    cls.def(
      "partial_trace",
      [](AbelianBackend& self,
         SymmetricTensorCPtr tensor,
         std::vector<std::pair<int64, int64>> pairs,
         std::vector<std::optional<int64>> levels) -> py::object {
          auto [data, codomain, domain] =
            self.partial_trace(tensor, std::move(pairs), std::move(levels));
          if (!codomain && !domain) {
              // Match Python: return (scalar, None, None) for a full trace.
              auto abd = AbelianBackend::unwrap(data);
              if (abd->blocks.empty()) {
                  return py::make_tuple(
                    self.block_backend->as_scalar(dtype::zero_scalar(abd->dtype), abd->dtype),
                    py::none(),
                    py::none());
              }
              return py::make_tuple(
                self.block_backend->item(abd->blocks[0]), py::none(), py::none());
          }
          return py::make_tuple(std::move(data), std::move(codomain), std::move(domain));
      },
      nb::arg("tensor"),
      nb::arg("pairs"),
      nb::arg("levels").none() = nb::none(),
      DOC(lasyten, AbelianBackend, partial_trace));

    // Expose under both the C++ name and the Python private name.
    m.def("valid_block_inds",
          &valid_block_inds,
          nb::arg("codomain"),
          nb::arg("domain"),
          doc_cpp_ref(R"pydoc(valid_block_inds)pydoc", "lasyten::valid_block_inds()"));
    m.attr("_valid_block_inds") = m.attr("valid_block_inds");
}

} // namespace lasyten
