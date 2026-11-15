#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "docstrings/backends/fusion_tree_backend.h"

#include "backends/casters.hpp"

#include <lasyten/backends/fusion_tree_backend.h>
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
    auto* raw = py::cast<BlockBackend*>(obj);
    return std::shared_ptr<BlockBackend>(raw, [](BlockBackend*) {});
}

} // namespace

void
bind_fusion_tree_data(nb::module_& m)
{
    nb::class_<FusionTreeData, TensorBackend::Data> cls(m, "FusionTreeData");
    cls.doc() = DOC(lasyten, FusionTreeData);

    cls.def(nb::new_([](BlockInds block_inds,
                        std::vector<BlockBackend::BlockPtr> blocks,
                        Dtype dtype,
                        std::string device,
                        bool is_sorted) {
                return std::make_shared<FusionTreeData>(
                  std::move(block_inds), std::move(blocks), dtype, std::move(device), is_sorted);
            }),
            nb::arg("block_inds"),
            nb::arg("blocks"),
            nb::arg("dtype"),
            nb::arg("device"),
            nb::arg("is_sorted") = false);

    cls.def_rw("block_inds", &FusionTreeData::block_inds)
      .def_rw("blocks", &FusionTreeData::blocks)
      .def_rw("dtype", &FusionTreeData::dtype)
      .def_rw("device", &FusionTreeData::device);

    cls
      .def("block_ind_from_coupled",
           &FusionTreeData::block_ind_from_coupled,
           nb::arg("coupled"),
           nb::arg("domain"),
           DOC(lasyten, FusionTreeData, block_ind_from_coupled))
      .def("block_ind_from_domain_sector_ind",
           &FusionTreeData::block_ind_from_domain_sector_ind,
           nb::arg("domain_sector_ind"),
           DOC(lasyten, FusionTreeData, block_ind_from_domain_sector_ind))
      .def(
        "discard_zero_blocks",
        [](FusionTreeData& self, py::object backend, float64 eps) {
            self.discard_zero_blocks(as_shared_block_backend(backend), eps);
        },
        nb::arg("backend"),
        nb::arg("eps"),
        DOC(lasyten, FusionTreeData, discard_zero_blocks))
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<FusionTreeData>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<FusionTreeData>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
}

void
bind_fusion_tree_backend(nb::module_& m)
{
    nb::class_<FusionTreeBackend, TensorBackend> cls(m, "FusionTreeBackend");
    cls.doc() = DOC(lasyten, FusionTreeBackend);

    cls.def(nb::new_([](py::object block_backend) {
                return std::make_shared<FusionTreeBackend>(as_shared_block_backend(block_backend));
            }),
            nb::arg("block_backend"));
    cls.def_static("wrap", &FusionTreeBackend::wrap, nb::arg("data"));
    cls.def_static("unwrap", &FusionTreeBackend::unwrap, nb::arg("data"));

    cls.def(
      "partial_trace",
      [](FusionTreeBackend& self,
         SymmetricTensorCPtr tensor,
         std::vector<std::pair<int64, int64>> pairs,
         std::vector<std::optional<int64>> levels) -> py::object {
          auto [data, codomain, domain] =
            self.partial_trace(tensor, std::move(pairs), std::move(levels));
          if (!codomain && !domain) {
              // Match Python: return (scalar, None, None) for a full trace.
              auto ftd = FusionTreeBackend::unwrap(data);
              if (ftd->blocks.empty()) {
                  return py::make_tuple(
                    self.block_backend->as_scalar(dtype::zero_scalar(ftd->dtype), ftd->dtype),
                    py::none(),
                    py::none());
              }
              return py::make_tuple(
                self.block_backend->item(ftd->blocks[0]), py::none(), py::none());
          }
          return py::make_tuple(std::move(data), std::move(codomain), std::move(domain));
      },
      nb::arg("tensor"),
      nb::arg("pairs"),
      nb::arg("levels").none() = nb::none(),
      DOC(lasyten, FusionTreeBackend, partial_trace));

    cls.def("apply_instructions",
            &FusionTreeBackend::apply_instructions,
            nb::arg("tensor"),
            nb::arg("instructions"),
            nb::arg("codomain_idcs"),
            nb::arg("domain_idcs"),
            nb::arg("new_codomain"),
            nb::arg("new_domain"),
            nb::arg("mixes_codomain_domain"),
            DOC(lasyten, FusionTreeBackend, apply_instructions));
}

} // namespace lasyten
