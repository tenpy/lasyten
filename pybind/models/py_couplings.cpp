#include <lasyten/models/couplings.h>

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"
#include "docstrings/models/couplings.h"

#include <optional>
#include <string>
#include <vector>

namespace lasyten {

namespace {

py::object
not_implemented()
{
    return py::module_::import("builtins").attr("NotImplemented");
}

} // namespace

void
bind_models_couplings(py::module_& m)
{
    m.def("freeze", &freeze, py::arg("obj"), DOC(lasyten, freeze));

    m.def("space_to_dict", &space_to_dict, py::arg("space"));

    py::class_<Coupling, py::smart_holder> coupling(m, "Coupling");
    coupling.doc() = DOC(lasyten, Coupling);

    coupling
      .def(py::init<std::vector<Site::Ptr>,
                    std::vector<SymmetricTensorPtr>,
                    std::optional<std::string>>(),
           py::arg("sites"),
           py::arg("factorization"),
           py::arg("name") = py::none())
      .def_readwrite("sites", &Coupling::sites)
      .def_readwrite("factorization", &Coupling::factorization)
      .def_readwrite("name", &Coupling::name)
      .def_readwrite("_levels", &Coupling::_levels)
      .def_property_readonly("_permuted",
                             [](Coupling const& self) {
                                 py::list out;
                                 for (auto const& [perm, obj] : self._permuted_py) {
                                     out.append(py::make_tuple(py::cast(perm), obj));
                                 }
                                 return out;
                             })
      .def_static("from_dense_block",
                  &Coupling::from_dense_block,
                  py::arg("operator"),
                  py::arg("sites"),
                  py::arg("name") = py::none(),
                  py::arg("dtype") = py::none(),
                  py::arg("understood_braiding") = false,
                  py::arg("cutoff_singular_values") = py::none(),
                  DOC(lasyten, Coupling, from_dense_block))
      .def_static("from_tensor",
                  &Coupling::from_tensor,
                  py::arg("operator"),
                  py::arg("sites"),
                  py::arg("name") = py::none(),
                  py::arg("cutoff") = py::none(),
                  DOC(lasyten, Coupling, from_tensor))
      .def("to_tensor", &Coupling::to_tensor, DOC(lasyten, Coupling, to_tensor))
      .def("to_numpy",
           &Coupling::to_numpy,
           py::arg("leg_order") = py::none(),
           py::arg("dtype") = py::none(),
           py::arg("understood_braiding") = false,
           DOC(lasyten, Coupling, to_numpy))
      .def("stretch_with_identities",
           &Coupling::stretch_with_identities,
           py::arg("all_sites"),
           py::arg("coupling_positions"),
           DOC(lasyten, Coupling, stretch_with_identities))
      .def("elementary_decomposition",
           &Coupling::elementary_decomposition,
           DOC(lasyten, Coupling, elementary_decomposition))
      .def(
        "permute",
        [](
          Coupling& self, std::vector<int64> const& permutation, py::object levels) -> py::object {
            for (auto const& [key, obj] : self._permuted_py) {
                if (key == permutation) {
                    return obj;
                }
            }
            std::optional<LevelsSpec> levels_spec;
            if (!levels.is_none()) {
                levels_spec = levels.cast<LevelsSpec>();
            }
            Coupling result = self.permute(permutation, levels_spec);
            py::object result_obj = py::cast(std::move(result));
            self._permuted_py.emplace_back(permutation, result_obj);
            return result_obj;
        },
        py::arg("permutation"),
        py::arg("levels") = py::none(),
        DOC(lasyten, Coupling, permute))
      .def(
        "_key",
        [](Coupling const& self) { return std::get<0>(self.key()); },
        doc_cpp_ref(R"pydoc(_key)pydoc", "lasyten::Coupling::_key()"))
      .def(
        "__eq__",
        [](Coupling const& self, py::handle other) -> py::object {
            if (!py::isinstance<Coupling>(other)) {
                return not_implemented();
            }
            return py::cast(self == other.cast<Coupling>());
        },
        py::arg("other"))
      .def_property_readonly("num_sites", &Coupling::num_sites)
      .def("__hash__", &Coupling::hash)
      .def("__repr__", &Coupling::repr)
      .def("test_sanity", &Coupling::test_sanity, DOC(lasyten, Coupling, test_sanity));

    m.def("spin_spin_coupling",
          &spin_spin_coupling,
          py::arg("sites"),
          py::arg("Jx") = 0,
          py::arg("Jy") = 0,
          py::arg("Jz") = 0,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, spin_spin_coupling));

    m.def("spin_field_coupling",
          &spin_field_coupling,
          py::arg("sites"),
          py::arg("hx") = 0,
          py::arg("hy") = 0,
          py::arg("hz") = 0,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, spin_field_coupling));

    m.def("aklt_coupling",
          &aklt_coupling,
          py::arg("sites"),
          py::arg("J") = 1,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, aklt_coupling));

    m.def("heisenberg_coupling",
          &heisenberg_coupling,
          py::arg("sites"),
          py::arg("J") = 1,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, heisenberg_coupling));

    m.def("chiral_3spin_coupling",
          &chiral_3spin_coupling,
          py::arg("sites"),
          py::arg("chi") = 1,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, chiral_3spin_coupling));

    m.def("chemical_potential",
          &chemical_potential,
          py::arg("sites"),
          py::arg("mu"),
          py::arg_v("species", all_species_sentinel(), "ALL_SPECIES"),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, chemical_potential));

    m.def("onsite_interaction",
          &onsite_interaction,
          py::arg("sites"),
          py::arg("U") = 1,
          py::arg_v("species", all_species_sentinel(), "ALL_SPECIES"),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, onsite_interaction));

    m.def("density_density_interaction",
          &density_density_interaction,
          py::arg("sites"),
          py::arg("V") = 1,
          py::arg_v("species_i", all_species_sentinel(), "ALL_SPECIES"),
          py::arg_v("species_j", all_species_sentinel(), "ALL_SPECIES"),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, density_density_interaction));

    m.def("hopping",
          &hopping,
          py::arg("sites"),
          py::arg("t") = 1,
          py::arg("species") = py::none(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, hopping));

    m.def("pairing",
          &pairing,
          py::arg("sites"),
          py::arg("Delta") = 1,
          py::arg("species") = py::none(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, pairing));

    m.def("onsite_pairing",
          &onsite_pairing,
          py::arg("sites"),
          py::arg("Delta") = 1,
          py::arg("species") = py::none(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, onsite_pairing));

    m.def("clock_clock_coupling",
          &clock_clock_coupling,
          py::arg("sites"),
          py::arg("Jx") = 0,
          py::arg("Jz") = 0,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, clock_clock_coupling));

    m.def("clock_field_coupling",
          &clock_field_coupling,
          py::arg("sites"),
          py::arg("hx") = py::none(),
          py::arg("hz") = py::none(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, clock_field_coupling));

    m.def("sector_projection_coupling",
          &sector_projection_coupling,
          py::arg("sites"),
          py::arg("J"),
          py::arg("sector"),
          py::arg("name"),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          DOC(lasyten, sector_projection_coupling));

    m.def("gold_coupling",
          &gold_coupling,
          py::arg("sites"),
          py::arg("J") = 1,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          DOC(lasyten, gold_coupling));
}

} // namespace lasyten
