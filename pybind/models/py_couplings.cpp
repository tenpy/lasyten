#include <lasyten/models/couplings.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
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
bind_models_couplings(nb::module_& m)
{
    m.def("freeze", &freeze, nb::arg("obj"), DOC(lasyten, freeze));

    m.def("space_to_dict", &space_to_dict, nb::arg("space"));

    nb::class_<Coupling> coupling(m, "Coupling");
    coupling.doc() = DOC(lasyten, Coupling);

    coupling
      .def(lasyten::ctor<decltype(coupling),
                         std::vector<Site::Ptr>,
                         std::vector<SymmetricTensorPtr>,
                         std::optional<std::string>>(),
           nb::arg("sites"),
           nb::arg("factorization"),
           nb::arg("name").none() = nb::none())
      .def_rw("sites", &Coupling::sites)
      .def_rw("factorization", &Coupling::factorization)
      .def_rw("name", &Coupling::name)
      .def_rw("_levels", &Coupling::_levels)
      .def_prop_ro("_permuted",
                   [](Coupling const& self) {
                       py::list out;
                       for (auto const& [perm, obj] : self._permuted_py) {
                           out.append(py::make_tuple(py::cast(perm), obj));
                       }
                       return out;
                   })
      .def_static("from_dense_block",
                  &Coupling::from_dense_block,
                  nb::arg("operator"),
                  nb::arg("sites"),
                  nb::arg("name").none() = nb::none(),
                  nb::arg("dtype").none() = nb::none(),
                  nb::arg("understood_braiding") = false,
                  nb::arg("cutoff_singular_values").none() = nb::none(),
                  DOC(lasyten, Coupling, from_dense_block))
      .def_static("from_tensor",
                  &Coupling::from_tensor,
                  nb::arg("operator"),
                  nb::arg("sites"),
                  nb::arg("name").none() = nb::none(),
                  nb::arg("cutoff").none() = nb::none(),
                  DOC(lasyten, Coupling, from_tensor))
      .def("to_tensor", &Coupling::to_tensor, DOC(lasyten, Coupling, to_tensor))
      .def("dagger",
           &Coupling::dagger,
           nb::arg("new_name").none() = nb::none(),
           "The hermitian conjugate of this coupling. Keeps the name unless `new_name` is given.")
      .def_prop_ro(
        "hc", [](Coupling const& self) { return self.dagger(); }, "Synonym for ``dagger()``.")
      .def("to_numpy",
           &Coupling::to_numpy,
           nb::arg("leg_order").none() = nb::none(),
           nb::arg("dtype").none() = nb::none(),
           nb::arg("understood_braiding") = false,
           DOC(lasyten, Coupling, to_numpy))
      .def("stretch_with_identities",
           &Coupling::stretch_with_identities,
           nb::arg("all_sites"),
           nb::arg("coupling_positions"),
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
                levels_spec = py::cast<LevelsSpec>(levels);
            }
            Coupling result = self.permute(permutation, levels_spec);
            py::object result_obj = py::cast(std::move(result));
            self._permuted_py.emplace_back(permutation, result_obj);
            return result_obj;
        },
        nb::arg("permutation"),
        nb::arg("levels").none() = nb::none(),
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
            return py::cast(self == py::cast<Coupling>(other));
        },
        nb::arg("other"))
      .def_prop_ro("num_sites", &Coupling::num_sites)
      .def("__hash__", &Coupling::hash)
      .def("__repr__", &Coupling::repr)
      .def("test_sanity", &Coupling::test_sanity, DOC(lasyten, Coupling, test_sanity));

    m.def("spin_spin_coupling",
          &spin_spin_coupling,
          nb::arg("sites"),
          nb::arg("Jx") = 0,
          nb::arg("Jy") = 0,
          nb::arg("Jz") = 0,
          nb::arg("name").none() = nb::none(),
          nb::arg("Jpp") = complex128{},
          nb::arg("Jmp") = complex128{},
          nb::arg("Jpm") = complex128{},
          nb::arg("Jmm") = complex128{},
          nb::arg("dof") = std::vector<std::string>{},
          DOC(lasyten, spin_spin_coupling));

    m.def("spin_field_coupling",
          &spin_field_coupling,
          nb::arg("sites"),
          nb::arg("hx") = 0,
          nb::arg("hy") = 0,
          nb::arg("hz") = 0,
          nb::arg("name").none() = nb::none(),
          nb::arg("dof") = DOF_SPIN,
          DOC(lasyten, spin_field_coupling));

    m.def("aklt_coupling",
          &aklt_coupling,
          nb::arg("sites"),
          nb::arg("J") = 1,
          nb::arg("name").none() = nb::none(),
          nb::arg("dof") = std::vector<std::string>{},
          DOC(lasyten, aklt_coupling));

    m.def("heisenberg_coupling",
          &heisenberg_coupling,
          nb::arg("sites"),
          nb::arg("J") = 1,
          nb::arg("name").none() = nb::none(),
          nb::arg("dof") = std::vector<std::string>{},
          DOC(lasyten, heisenberg_coupling));

    m.def("chiral_3spin_coupling",
          &chiral_3spin_coupling,
          nb::arg("sites"),
          nb::arg("chi") = 1,
          nb::arg("name").none() = nb::none(),
          nb::arg("dof") = std::vector<std::string>{},
          DOC(lasyten, chiral_3spin_coupling));

    m.def("chemical_potential",
          &chemical_potential,
          nb::arg("sites"),
          nb::arg("mu"),
          nb::arg("species") = all_species_sentinel(),
          nb::arg("name").none() = nb::none(),
          DOC(lasyten, chemical_potential));

    m.def("onsite_interaction",
          &onsite_interaction,
          nb::arg("sites"),
          nb::arg("U") = 1,
          nb::arg("species") = all_species_sentinel(),
          nb::arg("name").none() = nb::none(),
          DOC(lasyten, onsite_interaction));

    m.def("density_density_interaction",
          &density_density_interaction,
          nb::arg("sites"),
          nb::arg("V") = 1,
          nb::arg("species_i") = all_species_sentinel(),
          nb::arg("species_j") = all_species_sentinel(),
          nb::arg("name").none() = nb::none(),
          DOC(lasyten, density_density_interaction));

    m.def("hopping",
          &hopping,
          nb::arg("sites"),
          nb::arg("t_left") = 1,
          nb::arg("t_right") = 1,
          nb::arg("species").none() = nb::none(),
          nb::arg("name").none() = nb::none(),
          DOC(lasyten, hopping));

    m.def("pairing",
          &pairing,
          nb::arg("sites"),
          nb::arg("Delta") = 1,
          nb::arg("species").none() = nb::none(),
          nb::arg("name").none() = nb::none(),
          DOC(lasyten, pairing));

    m.def("onsite_pairing",
          &onsite_pairing,
          nb::arg("sites"),
          nb::arg("Delta") = 1,
          nb::arg("species").none() = nb::none(),
          nb::arg("name").none() = nb::none(),
          DOC(lasyten, onsite_pairing));

    m.def("clock_clock_coupling",
          &clock_clock_coupling,
          nb::arg("sites"),
          nb::arg("Jx") = 0,
          nb::arg("Jz") = 0,
          nb::arg("plus_hc") = true,
          nb::arg("name").none() = nb::none(),
          DOC(lasyten, clock_clock_coupling));

    m.def("clock_field_coupling",
          &clock_field_coupling,
          nb::arg("sites"),
          nb::arg("hx").none() = nb::none(),
          nb::arg("hz").none() = nb::none(),
          nb::arg("name").none() = nb::none(),
          DOC(lasyten, clock_field_coupling));

    m.def("sector_projection_coupling",
          &sector_projection_coupling,
          nb::arg("sites"),
          nb::arg("J"),
          nb::arg("sector"),
          nb::arg("name"),
          DOC(lasyten, sector_projection_coupling));

    m.def("gold_coupling",
          &gold_coupling,
          nb::arg("sites"),
          nb::arg("J") = 1,
          nb::arg("name").none() = nb::none(),
          DOC(lasyten, gold_coupling));
}

} // namespace lasyten
