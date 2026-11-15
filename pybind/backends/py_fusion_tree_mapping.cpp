#include <lasyten/backends/fusion_tree_mapping.h>
#include <lasyten/backends/fusion_tree_permute.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "docstrings/backends/fusion_tree_mapping.h"

#include <format>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lasyten {

void
bind_fusion_tree_mapping(nb::module_& m)
{
    nb::class_<BraidInstruction> braid_cls(m, "BraidInstruction");
    braid_cls.doc() =
      doc_cpp_ref(R"pydoc(BraidInstruction)pydoc", "lasyten::BraidInstruction", "class");

    braid_cls
      .def(lasyten::ctor<decltype(braid_cls), bool, int64, bool>(),
           nb::arg("codomain"),
           nb::arg("idx"),
           nb::arg("overbraid"))
      .def_rw("codomain", &BraidInstruction::codomain)
      .def_rw("idx", &BraidInstruction::idx)
      .def_rw("overbraid", &BraidInstruction::overbraid)
      .def(nb::self == nb::self)
      .def("__repr__", [](BraidInstruction const& i) {
          return std::format(
            "BraidInstruction(codomain={}, idx={}, overbraid={})", i.codomain, i.idx, i.overbraid);
      });

    nb::class_<BendInstruction> bend_cls(m, "BendInstruction");
    bend_cls.doc() =
      doc_cpp_ref(R"pydoc(BendInstruction)pydoc", "lasyten::BendInstruction", "class");

    bend_cls.def(lasyten::ctor<decltype(bend_cls), bool>(), nb::arg("bend_down"))
      .def_rw("bend_down", &BendInstruction::bend_down)
      .def(nb::self == nb::self)
      .def("__repr__", [](BendInstruction const& i) {
          return std::format("BendInstruction(bend_down={})", i.bend_down);
      });

    nb::class_<TwistInstruction> twist_cls(m, "TwistInstruction");
    twist_cls.doc() =
      doc_cpp_ref(R"pydoc(TwistInstruction)pydoc", "lasyten::TwistInstruction", "class");

    twist_cls
      .def(lasyten::ctor<decltype(twist_cls), bool, std::vector<int64>, bool>(),
           nb::arg("codomain"),
           nb::arg("idcs"),
           nb::arg("overtwist"))
      .def_rw("codomain", &TwistInstruction::codomain)
      .def_rw("idcs", &TwistInstruction::idcs)
      .def_rw("overtwist", &TwistInstruction::overtwist)
      .def(nb::self == nb::self)
      .def("__repr__", [](TwistInstruction const& i) {
          std::string idcs_str = "[";
          for (std::size_t k = 0; k < i.idcs.size(); ++k) {
              if (k > 0) {
                  idcs_str += ", ";
              }
              idcs_str += std::to_string(i.idcs[k]);
          }
          idcs_str += "]";
          return std::format("TwistInstruction(codomain={}, idcs={}, overtwist={})",
                             i.codomain,
                             idcs_str,
                             i.overtwist);
      });

    nb::class_<TreePairMapping> tree_pair(m, "TreePairMapping");
    tree_pair.doc() = DOC(lasyten, TreePairMapping);

    tree_pair
      .def(lasyten::ctor<decltype(tree_pair), SparseMappingFusionTreePair, bool>(),
           nb::arg("mapping"),
           nb::arg("is_real"))
      .def_rw("is_real", &TreePairMapping::is_real)
      .def_rw("mapping", &TreePairMapping::mapping)
      .def_static("from_identity",
                  &TreePairMapping::from_identity,
                  nb::arg("codomain"),
                  nb::arg("domain"),
                  nb::arg("block_inds").none() = nb::none(),
                  DOC(lasyten, TreePairMapping, from_identity))
      .def_static("from_instructions",
                  &TreePairMapping::from_instructions,
                  nb::arg("instructions"),
                  nb::arg("codomain"),
                  nb::arg("domain"),
                  nb::arg("block_inds").none() = nb::none())
      .def("prune",
           &TreePairMapping::prune,
           nb::arg("tol") = 1e-15,
           doc_cpp_ref(R"pydoc(prune)pydoc", "lasyten::TreePairMapping::prune()"))
      .def(
        "transform_tensor",
        [](TreePairMapping const& self,
           FusionTreeData const& data,
           TensorProduct::Ptr codomain,
           TensorProduct::Ptr domain,
           TensorProduct::Ptr new_codomain,
           TensorProduct::Ptr new_domain,
           std::vector<int64> codomain_idcs,
           std::vector<int64> domain_idcs,
           py::object block_backend) {
            auto backend = block_backend.cast<std::shared_ptr<BlockBackend>>();
            return self.transform_tensor(data,
                                         codomain,
                                         domain,
                                         new_codomain,
                                         new_domain,
                                         codomain_idcs,
                                         domain_idcs,
                                         backend);
        },
        nb::arg("data"),
        nb::arg("codomain"),
        nb::arg("domain"),
        nb::arg("new_codomain"),
        nb::arg("new_domain"),
        nb::arg("codomain_idcs"),
        nb::arg("domain_idcs"),
        nb::arg("block_backend"),
        doc_cpp_ref(R"pydoc(transform_tensor)pydoc",
                    "lasyten::TreePairMapping::transform_tensor()"));

    nb::class_<FactorizedTreeMapping> fact(m, "FactorizedTreeMapping");
    fact.doc() = DOC(lasyten, FactorizedTreeMapping);

    fact
      .def(
        lasyten::ctor<decltype(fact), FusionTreeMappingVariant, FusionTreeMappingVariant, bool>(),
        nb::arg("splitting_tree_mapping"),
        nb::arg("fusion_tree_mapping"),
        nb::arg("is_real"))
      .def_rw("is_real", &FactorizedTreeMapping::is_real)
      .def_rw("splitting_tree_mapping", &FactorizedTreeMapping::splitting_tree_mapping)
      .def_rw("fusion_tree_mapping", &FactorizedTreeMapping::fusion_tree_mapping)
      .def_static("from_identity",
                  &FactorizedTreeMapping::from_identity,
                  nb::arg("codomain"),
                  nb::arg("domain"),
                  nb::arg("block_inds").none() = nb::none(),
                  doc_cpp_ref(R"pydoc(from_identity)pydoc",
                              "lasyten::FactorizedTreeMapping::from_identity()"))
      .def_static("from_instructions",
                  &FactorizedTreeMapping::from_instructions,
                  nb::arg("instructions"),
                  nb::arg("codomain"),
                  nb::arg("domain"),
                  nb::arg("block_inds").none() = nb::none())
      .def("prune",
           &FactorizedTreeMapping::prune,
           nb::arg("tol") = 1e-15,
           doc_cpp_ref(R"pydoc(prune)pydoc", "lasyten::FactorizedTreeMapping::prune()"))
      .def(
        "transform_tensor",
        [](FactorizedTreeMapping const& self,
           FusionTreeData const& data,
           TensorProduct::Ptr codomain,
           TensorProduct::Ptr domain,
           TensorProduct::Ptr new_codomain,
           TensorProduct::Ptr new_domain,
           std::vector<int64> codomain_idcs,
           std::vector<int64> domain_idcs,
           py::object block_backend) {
            auto backend = block_backend.cast<std::shared_ptr<BlockBackend>>();
            return self.transform_tensor(data,
                                         codomain,
                                         domain,
                                         new_codomain,
                                         new_domain,
                                         codomain_idcs,
                                         domain_idcs,
                                         backend);
        },
        nb::arg("data"),
        nb::arg("codomain"),
        nb::arg("domain"),
        nb::arg("new_codomain"),
        nb::arg("new_domain"),
        nb::arg("codomain_idcs"),
        nb::arg("domain_idcs"),
        nb::arg("block_backend"),
        doc_cpp_ref(R"pydoc(transform_tensor)pydoc",
                    "lasyten::FactorizedTreeMapping::transform_tensor()"));

    nb::class_<PermuteLegsInstructionEngine> perm_cls(m, "PermuteLegsInstructionEngine");
    perm_cls.doc() = doc_cpp_ref(R"pydoc(PermuteLegsInstructionEngine)pydoc",
                                 "lasyten::PermuteLegsInstructionEngine",
                                 "class");

    perm_cls
      .def(lasyten::ctor<decltype(perm_cls),
                         int64,
                         int64,
                         std::vector<int64>,
                         std::vector<int64>,
                         std::vector<std::optional<int64>>,
                         std::vector<std::optional<bool>>,
                         bool>(),
           nb::arg("num_codomain_legs"),
           nb::arg("num_domain_legs"),
           nb::arg("codomain_idcs"),
           nb::arg("domain_idcs"),
           nb::arg("levels"),
           nb::arg("bend_right"),
           nb::arg("has_symmetric_braid"))
      .def_rw("num_legs", &PermuteLegsInstructionEngine::num_legs)
      .def_rw("has_symmetric_braid", &PermuteLegsInstructionEngine::has_symmetric_braid)
      .def_rw("num_codomain_legs", &PermuteLegsInstructionEngine::num_codomain_legs)
      .def_rw("num_domain_legs", &PermuteLegsInstructionEngine::num_domain_legs)
      .def_rw("target_positions", &PermuteLegsInstructionEngine::target_positions)
      .def_rw("levels", &PermuteLegsInstructionEngine::levels)
      .def("evaluate_instructions",
           [](PermuteLegsInstructionEngine& self) {
               auto inst = self.evaluate_instructions();
               py::list out;
               for (Instruction const& i : inst) {
                   std::visit([&](auto const& x) { out.append(py::cast(x)); }, i);
               }
               return out;
           })
      .def("verify",
           &PermuteLegsInstructionEngine::verify,
           nb::arg("num_codomain_legs"),
           nb::arg("num_domain_legs"),
           nb::arg("codomain_idcs"),
           nb::arg("domain_idcs"),
           doc_cpp_ref(R"pydoc(verify)pydoc", "lasyten::PermuteLegsInstructionEngine::verify()"));
}

} // namespace lasyten
