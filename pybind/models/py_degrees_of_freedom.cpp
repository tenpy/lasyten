#include <lasyten/models/degrees_of_freedom.h>

#include "../block_backend/eigen_ndarray.h"
#include "../block_backend/ndarray_dispatch.h"
#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "docstrings/models/degrees_of_freedom.h"

#include "tools/hdf5_bind.h"
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

using AnyCpuArray = eigen_ndarray::AnyCpuArray;

/// Dual path for DOF ``py::array`` fields (issue #326 / leftover #10):
/// import copies into owning ``py::array``; property getters return zero-copy views;
/// ``*_numpy`` / factory returns use owning copies.
[[nodiscard]] py::array
array_from_ndarray(AnyCpuArray const& arr)
{
    eigen_ndarray::require_cpu_device(arr.device_type(), "models DOF");
    return py::array::ensure(nb::cast(arr).ptr());
}

[[nodiscard]] nb::object
dof_array_as_ndarray(py::array const& arr, bool writable = false)
{
    return ndarray_dispatch::numpy_array_as_ndarray(arr, writable);
}

[[nodiscard]] nb::object
dof_array_to_numpy(py::array const& arr)
{
    nb::object np = nb::module_::import_("numpy");
    return np.attr("array")(nb::borrow(arr.ptr()), nb::arg("copy") = true);
}

[[nodiscard]] py::object
dim_to_python(float64 dim)
{
    if (std::isfinite(dim) && std::floor(dim) == dim) {
        return py::int_(static_cast<long long>(dim));
    }
    return py::float_(dim);
}

py::dict
dofs_as_dict(Site const& self)
{
    py::dict out;
    for (auto const& [name, dof] : self.dofs) {
        out[py::str(name)] = py::cast(dof);
    }
    return out;
}

void
set_dofs_from_dict(Site& self, py::dict const& mapping)
{
    self.dofs.clear();
    for (auto item : mapping) {
        auto name = py::cast<std::string>(item.first);
        self.set_dof(name, py::cast<DegreeOfFreedom::Ptr>(item.second));
    }
}

template<class T>
py::object
dof_or_none(Site const& self, std::string const& name)
{
    auto dof = self.get_dof<T>(name);
    if (!dof) {
        return py::none();
    }
    return py::cast(dof);
}

} // namespace

void
bind_models_degrees_of_freedom(nb::module_& m)
{
    m.attr("ALL_SPECIES") = all_species_sentinel();

    nb::class_<DegreeOfFreedom> degree_of_freedom(m, "DegreeOfFreedom");
    degree_of_freedom.doc() =
      "Base class for degrees of freedom composed into a Site (not site mixins).";
    degree_of_freedom.def("test_sanity", &DegreeOfFreedom::test_sanity);

    nb::class_<Site> site(m, "Site");
    site.doc() = DOC(lasyten, Site);

    site
      .def(lasyten::ctor<decltype(site),
                         ElementarySpace::Ptr,
                         std::map<std::string, int64>,
                         std::map<std::string, SymmetricTensorPtr>,
                         TensorBackend::Ptr,
                         std::optional<std::string>,
                         std::map<std::string, DegreeOfFreedom::Ptr>>(),
           nb::arg("leg"),
           nb::arg("state_labels") = std::map<std::string, int64>{},
           nb::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           nb::arg("backend").none() = nullptr,
           nb::arg("default_device").none() = nb::none(),
           nb::arg("dofs") = std::map<std::string, DegreeOfFreedom::Ptr>{})
      .def_rw("leg", &Site::leg)
      .def_rw("state_labels", &Site::state_labels)
      .def_rw("backend", &Site::backend)
      .def_rw("default_device", &Site::default_device)
      .def_rw("onsite_operators", &Site::onsite_operators)
      .def_rw("hc_ops", &Site::hc_ops)
      .def_prop_rw(
        "dofs",
        [](Site const& self) { return dofs_as_dict(self); },
        [](Site& self, py::dict const& mapping) { set_dofs_from_dict(self, mapping); })
      .def_prop_ro("spin", [](Site const& self) { return dof_or_none<SpinDOF>(self, DOF_SPIN); })
      .def_prop_ro(
        "occupation",
        [](Site const& self) { return dof_or_none<OccupationDOF>(self, DOF_OCCUPATION); })
      .def_prop_ro("clock",
                   [](Site const& self) { return dof_or_none<ClockDOF>(self, DOF_CLOCK); })
      .def_prop_ro("anyon",
                   [](Site const& self) { return dof_or_none<AnyonDOF>(self, DOF_ANYON); })
      .def("set_dof",
           &Site::set_dof,
           nb::arg("name"),
           nb::arg("dof"),
           "Register (or replace) a named degree of freedom.")
      .def("test_sanity", &Site::test_sanity, DOC(lasyten, Site, test_sanity))
      .def_prop_ro("symmetry", &Site::symmetry)
      .def_prop_ro("dim", [](Site const& self) { return dim_to_python(self.dim()); })
      .def(
        "change_leg",
        [](Site& self, ElementarySpace::Ptr new_leg) -> Site& {
            self.change_leg(std::move(new_leg));
            return self;
        },
        nb::arg("new_leg") = ElementarySpace::Ptr{},
        nb::rv_policy::reference_internal,
        DOC(lasyten, Site, change_leg))
      .def("add_onsite_operator",
           &Site::add_onsite_operator,
           nb::arg("name"),
           nb::arg("op"),
           nb::arg("is_diagonal").none() = nb::none(),
           nb::arg("understood_braiding") = false,
           nb::arg("hc").none() = nb::none(),
           DOC(lasyten, Site, add_onsite_operator))
      .def("valid_opname", &Site::valid_opname, nb::arg("name"), DOC(lasyten, Site, valid_opname))
      .def("test_op",
           &Site::test_op,
           nb::arg("opname"),
           "Check that an onsite operator name is known and available under this site's symmetry.")
      .def("rename_op",
           &Site::rename_op,
           nb::arg("old_name"),
           nb::arg("new_name"),
           "Rename an onsite operator while preserving its hermitian-conjugate mapping.")
      .def("get_op", &Site::get_op, nb::arg("name"), DOC(lasyten, Site, get_op))
      .def("get_hc_op_name",
           &Site::get_hc_op_name,
           nb::arg("name"),
           "Return the name of the hermitian conjugate of an operator or operator product.")
      .def("multiply_op_names",
           &Site::multiply_op_names,
           nb::arg("names"),
           DOC(lasyten, Site, multiply_op_names))
      .def("multiply_operators",
           &Site::multiply_operators,
           nb::arg("operators"),
           DOC(lasyten, Site, multiply_operators))
      .def("identity_tensor",
           &Site::identity_tensor,
           nb::arg("w"),
           nb::arg("overbraid") = true,
           DOC(lasyten, Site, identity_tensor))
      .def("get_onsite_W_block",
           &Site::get_onsite_W_block,
           nb::arg("opname"),
           nb::arg("w"),
           nb::arg("overbraid") = true,
           nb::arg("wL_label") = "wL",
           nb::arg("wR_label") = "wR",
           DOC(lasyten, Site, get_onsite_W_block))
      .def("state_index", &Site::state_index, nb::arg("label"), DOC(lasyten, Site, state_index))
      .def("state_indices",
           &Site::state_indices,
           nb::arg("labels"),
           DOC(lasyten, Site, state_indices))
      .def("__repr__", &Site::repr)
      .def(
        "save_hdf5",
        [](Site const& self, py::object py_saver, py::object py_h5gr, std::string subpath) {
            lasyten::hdf5::PyBridge bridge(py_saver, py::none(), py_h5gr);
            lasyten::hdf5::py_save(subpath + "init_kwargs", self.hdf5_init_kwargs());
        },
        nb::arg("hdf5_saver"),
        nb::arg("h5gr"),
        nb::arg("subpath"),
        DOC(lasyten, Site, save_hdf5));

    nb::object classmethod = nb::module_::import_("builtins").attr("classmethod");
    site.attr("from_hdf5") = classmethod(nb::cpp_function(
      [](py::object cls, py::object py_loader, py::object py_h5gr, std::string subpath) {
          lasyten::hdf5::PyBridge bridge(py::none(), py_loader, py_h5gr);
          py::object kwargs = lasyten::hdf5::py_load(subpath + "init_kwargs");
          py::object obj = cls(**kwargs);
          py_loader.attr("memorize_load")(py_h5gr, obj);
          return obj;
      },
      nb::name("from_hdf5"),
      nb::arg("cls"),
      nb::arg("hdf5_loader"),
      nb::arg("h5gr"),
      nb::arg("subpath"),
      "Reconstruct a Site (or subclass) from HDF5."));

    nb::class_<SpinDOF, DegreeOfFreedom> spin_dof(m, "SpinDOF");
    spin_dof.doc() = DOC(lasyten, SpinDOF);

    spin_dof
      .def(nb::new_([](ElementarySpace::Ptr leg, AnyCpuArray const& spin_vector) {
               return std::make_shared<SpinDOF>(std::move(leg), array_from_ndarray(spin_vector));
           }),
           nb::arg("leg"),
           nb::arg("spin_vector"))
      .def_prop_rw(
        "spin_vector",
        [](SpinDOF& self) { return dof_array_as_ndarray(self.spin_vector); },
        [](SpinDOF& self, AnyCpuArray const& arr) {
            self.spin_vector = as_immutable_array(array_from_ndarray(arr));
        },
        "Zero-copy ``nb::ndarray`` view of the stored spin operators (read-only). "
        "Assignment copies from any CPU array-like. Use ``np.array(..., copy=True)`` for an "
        "owning NumPy copy.")
      .def(
        "to_numpy",
        [](SpinDOF const& self) { return dof_array_to_numpy(self.spin_vector); },
        "Owning NumPy copy of ``spin_vector`` (dual path vs the property view).")
      .def("test_sanity",
           &SpinDOF::test_sanity,
           doc_cpp_ref(R"pydoc(test_sanity)pydoc", "lasyten::SpinDOF::test_sanity()"))
      .def_static(
        "spin_vector_from_Sp",
        [](AnyCpuArray const& Sz, AnyCpuArray const& Sp) {
            return dof_array_to_numpy(
              SpinDOF::spin_vector_from_Sp(array_from_ndarray(Sz), array_from_ndarray(Sp)));
        },
        nb::arg("Sz"),
        nb::arg("Sp"),
        DOC(lasyten, SpinDOF, spin_vector_from_Sp))
      .def_static("conservation_law_to_symmetry",
                  &SpinDOF::conservation_law_to_symmetry,
                  nb::arg("conserve"),
                  nb::arg("use_test_su2") = false,
                  DOC(lasyten, SpinDOF, conservation_law_to_symmetry));

    nb::class_<ClockDOF, DegreeOfFreedom> clock_dof(m, "ClockDOF");
    clock_dof.doc() = DOC(lasyten, ClockDOF);

    clock_dof
      .def(nb::new_([](ElementarySpace::Ptr leg, AnyCpuArray const& clock_operators) {
               return std::make_shared<ClockDOF>(std::move(leg),
                                                 array_from_ndarray(clock_operators));
           }),
           nb::arg("leg"),
           nb::arg("clock_operators"))
      .def_prop_rw(
        "clock_operators",
        [](ClockDOF& self) { return dof_array_as_ndarray(self.clock_operators); },
        [](ClockDOF& self, AnyCpuArray const& arr) {
            self.clock_operators = as_immutable_array(array_from_ndarray(arr));
        },
        "Zero-copy ``nb::ndarray`` view of the stored clock operators (read-only).")
      .def(
        "to_numpy",
        [](ClockDOF const& self) { return dof_array_to_numpy(self.clock_operators); },
        "Owning NumPy copy of ``clock_operators``.")
      .def("test_sanity",
           &ClockDOF::test_sanity,
           doc_cpp_ref(R"pydoc(test_sanity)pydoc", "lasyten::ClockDOF::test_sanity()"))
      .def("register_default_ops",
           &ClockDOF::register_default_ops,
           nb::arg("site"),
           "Register default Z / Zhc / Zphc onsite operators on the given site.")
      .def_static("conservation_law_to_symmetry",
                  &ClockDOF::conservation_law_to_symmetry,
                  nb::arg("conserve"),
                  nb::arg("use_test_su2") = false,
                  DOC(lasyten, ClockDOF, conservation_law_to_symmetry));

    nb::class_<AnyonDOF, DegreeOfFreedom> anyon_dof(m, "AnyonDOF");
    anyon_dof.doc() = DOC(lasyten, AnyonDOF);

    anyon_dof
      .def(lasyten::ctor<decltype(anyon_dof), ElementarySpace::Ptr, std::vector<std::string>>(),
           nb::arg("leg"),
           nb::arg("sector_names") = std::vector<std::string>{})
      .def_rw("sector_names", &AnyonDOF::sector_names)
      .def("test_sanity",
           &AnyonDOF::test_sanity,
           doc_cpp_ref(R"pydoc(test_sanity)pydoc", "lasyten::AnyonDOF::test_sanity()"))
      .def("register_default_ops",
           &AnyonDOF::register_default_ops,
           nb::arg("site"),
           nb::arg("backend").none() = nullptr,
           nb::arg("default_device").none() = nb::none(),
           "Register sector-projection onsite operators on the given site.");

    nb::class_<OccupationDOF, DegreeOfFreedom> occupation_dof(m, "OccupationDOF");
    occupation_dof.doc() = DOC(lasyten, OccupationDOF);

    occupation_dof.def_rw("num_species", &OccupationDOF::num_species)
      .def_prop_rw(
        "creators",
        [](OccupationDOF& self) { return dof_array_as_ndarray(self.creators); },
        [](OccupationDOF& self, AnyCpuArray const& arr) {
            self.creators = as_immutable_array(array_from_ndarray(arr));
        })
      .def_prop_rw(
        "annihilators",
        [](OccupationDOF& self) { return dof_array_as_ndarray(self.annihilators); },
        [](OccupationDOF& self, AnyCpuArray const& arr) {
            self.annihilators = as_immutable_array(array_from_ndarray(arr));
        })
      .def_rw("anti_commute_sign", &OccupationDOF::anti_commute_sign)
      .def_rw("species_names", &OccupationDOF::species_names)
      .def_prop_rw(
        "number_operators",
        [](OccupationDOF& self) { return dof_array_as_ndarray(self.number_operators); },
        [](OccupationDOF& self, AnyCpuArray const& arr) {
            self.number_operators = as_immutable_array(array_from_ndarray(arr));
        })
      .def_prop_rw(
        "n_tot",
        [](OccupationDOF& self) { return dof_array_as_ndarray(self.n_tot); },
        [](OccupationDOF& self, AnyCpuArray const& arr) {
            self.n_tot = as_immutable_array(array_from_ndarray(arr));
        })
      .def("test_sanity",
           &OccupationDOF::test_sanity,
           doc_cpp_ref(R"pydoc(test_sanity)pydoc", "lasyten::OccupationDOF::test_sanity()"))
      .def("add_individual_occupation_ops",
           &OccupationDOF::add_individual_occupation_ops,
           nb::arg("site"),
           DOC(lasyten, OccupationDOF, add_individual_occupation_ops))
      .def("add_total_occupation_ops",
           &OccupationDOF::add_total_occupation_ops,
           nb::arg("site"),
           doc_cpp_ref(R"pydoc(add_total_occupation_ops)pydoc",
                       "lasyten::OccupationDOF::add_total_occupation_ops()"))
      .def(
        "get_annihilator_numpy",
        [](OccupationDOF& self, py::object species, bool include_JW) {
            return dof_array_to_numpy(self.get_annihilator_numpy(std::move(species), include_JW));
        },
        nb::arg("species"),
        nb::arg("include_JW") = false,
        DOC(lasyten, OccupationDOF, get_annihilator_numpy))
      .def(
        "get_creator_numpy",
        [](OccupationDOF& self, py::object species, bool include_JW) {
            return dof_array_to_numpy(self.get_creator_numpy(std::move(species), include_JW));
        },
        nb::arg("species"),
        nb::arg("include_JW") = false,
        DOC(lasyten, OccupationDOF, get_creator_numpy))
      .def(
        "get_occupation_numpy",
        [](OccupationDOF& self, py::object species) {
            return dof_array_to_numpy(self.get_occupation_numpy(std::move(species)));
        },
        nb::arg("species") = all_species_sentinel(),
        DOC(lasyten, OccupationDOF, get_occupation_numpy))
      .def("get_species_idx", &OccupationDOF::get_species_idx, nb::arg("species"));

    nb::class_<BosonicDOF, OccupationDOF> bosonic_dof(m, "BosonicDOF");
    bosonic_dof.doc() = DOC(lasyten, BosonicDOF);

    bosonic_dof
      .def(nb::new_([](ElementarySpace::Ptr leg,
                       AnyCpuArray const& Nmax,
                       AnyCpuArray const& creators,
                       AnyCpuArray const& annihilators,
                       std::vector<std::optional<std::string>> species_names) {
               return std::make_shared<BosonicDOF>(std::move(leg),
                                                   array_from_ndarray(Nmax),
                                                   array_from_ndarray(creators),
                                                   array_from_ndarray(annihilators),
                                                   std::move(species_names));
           }),
           nb::arg("leg"),
           nb::arg("Nmax"),
           nb::arg("creators"),
           nb::arg("annihilators"),
           nb::arg("species_names") = std::vector<std::optional<std::string>>{})
      .def_prop_rw(
        "Nmax",
        [](BosonicDOF& self) { return dof_array_as_ndarray(self.Nmax); },
        [](BosonicDOF& self, AnyCpuArray const& arr) {
            self.Nmax = as_immutable_array(array_from_ndarray(arr));
        })
      .def_prop_rw(
        "JW",
        [](BosonicDOF& self) { return dof_array_as_ndarray(self.JW); },
        [](BosonicDOF& self, AnyCpuArray const& arr) {
            self.JW = as_immutable_array(array_from_ndarray(arr));
        })
      .def_prop_rw(
        "_JW",
        [](BosonicDOF& self) { return dof_array_as_ndarray(self.JW); },
        [](BosonicDOF& self, AnyCpuArray const& arr) {
            self.JW = as_immutable_array(array_from_ndarray(arr));
        })
      .def("test_sanity",
           &BosonicDOF::test_sanity,
           doc_cpp_ref(R"pydoc(test_sanity)pydoc", "lasyten::BosonicDOF::test_sanity()"))
      .def("add_individual_occupation_ops",
           &BosonicDOF::add_individual_occupation_ops,
           nb::arg("site"),
           doc_cpp_ref(R"pydoc(add_individual_occupation_ops)pydoc",
                       "lasyten::BosonicDOF::add_individual_occupation_ops()"))
      .def(
        "get_annihilator_numpy",
        [](BosonicDOF& self, py::object species, bool include_JW) {
            return dof_array_to_numpy(self.get_annihilator_numpy(std::move(species), include_JW));
        },
        nb::arg("species"),
        nb::arg("include_JW") = false)
      .def(
        "get_creator_numpy",
        [](BosonicDOF& self, py::object species, bool include_JW) {
            return dof_array_to_numpy(self.get_creator_numpy(std::move(species), include_JW));
        },
        nb::arg("species"),
        nb::arg("include_JW") = false)
      .def_static("conservation_law_to_symmetry",
                  &BosonicDOF::conservation_law_to_symmetry,
                  nb::arg("conserve"),
                  doc_cpp_ref(R"pydoc(conservation_law_to_symmetry)pydoc",
                              "lasyten::BosonicDOF::conservation_law_to_symmetry()"))
      .def_static(
        "creation_annihilation_op_from_single_Nmax",
        [](int64 Nmax, int64 dim) {
            auto [c, a] = BosonicDOF::creation_annihilation_op_from_single_Nmax(Nmax, dim);
            return py::make_tuple(dof_array_to_numpy(c), dof_array_to_numpy(a));
        },
        nb::arg("Nmax"),
        nb::arg("dim"))
      .def_static(
        "creation_annihilation_ops_from_Nmax",
        [](AnyCpuArray const& Nmax, int64 dim) {
            auto [c, a] =
              BosonicDOF::creation_annihilation_ops_from_Nmax(array_from_ndarray(Nmax), dim);
            return py::make_tuple(dof_array_to_numpy(c), dof_array_to_numpy(a));
        },
        nb::arg("Nmax"),
        nb::arg("dim"))
      .def_static(
        "creation_annihilation_ops",
        [](int64 num_species, AnyCpuArray const& Nmax, int64 dim) {
            auto [c, a] =
              BosonicDOF::creation_annihilation_ops(num_species, array_from_ndarray(Nmax), dim);
            return py::make_tuple(dof_array_to_numpy(c), dof_array_to_numpy(a));
        },
        nb::arg("num_species"),
        nb::arg("Nmax"),
        nb::arg("dim"));

    nb::class_<FermionicDOF, OccupationDOF> fermionic_dof(m, "FermionicDOF");
    fermionic_dof.doc() = DOC(lasyten, FermionicDOF);

    fermionic_dof
      .def(nb::new_([](ElementarySpace::Ptr leg,
                       AnyCpuArray const& creators,
                       AnyCpuArray const& annihilators,
                       std::vector<std::optional<std::string>> species_names) {
               return std::make_shared<FermionicDOF>(std::move(leg),
                                                     array_from_ndarray(creators),
                                                     array_from_ndarray(annihilators),
                                                     std::move(species_names));
           }),
           nb::arg("leg"),
           nb::arg("creators"),
           nb::arg("annihilators"),
           nb::arg("species_names") = std::vector<std::optional<std::string>>{})
      .def_prop_rw(
        "partial_JWs",
        [](FermionicDOF& self) { return dof_array_as_ndarray(self.partial_JWs); },
        [](FermionicDOF& self, AnyCpuArray const& arr) {
            self.partial_JWs = as_immutable_array(array_from_ndarray(arr));
        })
      .def_prop_rw(
        "JW",
        [](FermionicDOF& self) { return dof_array_as_ndarray(self.JW); },
        [](FermionicDOF& self, AnyCpuArray const& arr) {
            self.JW = as_immutable_array(array_from_ndarray(arr));
        })
      .def_prop_rw(
        "_JW",
        [](FermionicDOF& self) { return dof_array_as_ndarray(self.JW); },
        [](FermionicDOF& self, AnyCpuArray const& arr) {
            self.JW = as_immutable_array(array_from_ndarray(arr));
        })
      .def("test_sanity",
           &FermionicDOF::test_sanity,
           doc_cpp_ref(R"pydoc(test_sanity)pydoc", "lasyten::FermionicDOF::test_sanity()"))
      .def(
        "get_annihilator_numpy",
        [](FermionicDOF& self, py::object species, bool include_JW) {
            return dof_array_to_numpy(self.get_annihilator_numpy(std::move(species), include_JW));
        },
        nb::arg("species"),
        nb::arg("include_JW") = false)
      .def(
        "get_creator_numpy",
        [](FermionicDOF& self, py::object species, bool include_JW) {
            return dof_array_to_numpy(self.get_creator_numpy(std::move(species), include_JW));
        },
        nb::arg("species"),
        nb::arg("include_JW") = false)
      .def_static("conservation_law_to_symmetry",
                  &FermionicDOF::conservation_law_to_symmetry,
                  nb::arg("conserve"),
                  doc_cpp_ref(R"pydoc(conservation_law_to_symmetry)pydoc",
                              "lasyten::FermionicDOF::conservation_law_to_symmetry()"))
      .def_static(
        "creation_annihilation_ops",
        [](int64 num_species) {
            auto [c, a] = FermionicDOF::creation_annihilation_ops(num_species);
            return py::make_tuple(dof_array_to_numpy(c), dof_array_to_numpy(a));
        },
        nb::arg("num_species"));

    m.def(
      "set_common_symmetry",
      [](py::sequence sites_obj, py::object symmetry_combine, py::object new_symmetry_obj) {
          std::vector<Site::Ptr> sites;
          sites.reserve(static_cast<std::size_t>(py::len(sites_obj)));
          for (py::handle item : sites_obj) {
              sites.push_back(py::cast<Site::Ptr>(item));
          }
          Symmetry::Ptr new_symmetry;
          if (!new_symmetry_obj.is_none()) {
              new_symmetry = py::cast<Symmetry::Ptr>(new_symmetry_obj);
          }

          if (py::isinstance<py::str>(symmetry_combine)) {
              set_common_symmetry(sites, py::cast<std::string>(symmetry_combine), new_symmetry);
              return;
          }
          if (py::isinstance<py::list>(symmetry_combine) ||
              py::isinstance<py::tuple>(symmetry_combine)) {
              CommonSymmetryComboSpec spec;
              for (py::handle col_obj : symmetry_combine) {
                  CommonSymmetryComboColumn col;
                  for (py::handle term_obj : col_obj) {
                      py::tuple term = py::to_tuple(term_obj);
                      if (term.size() != 3) {
                          throw py::value_error(
                            "symmetry_combine terms must be (prefactor, site_idx, old_col_idx)");
                      }
                      col.emplace_back(py::cast<float64>(term[0]),
                                       py::cast<int64>(term[1]),
                                       py::cast<int64>(term[2]));
                  }
                  spec.push_back(std::move(col));
              }
              set_common_symmetry(sites, spec, new_symmetry);
              return;
          }
          if (py::isinstance<py::function>(symmetry_combine) ||
              py::hasattr(symmetry_combine, "__call__")) {
              py::object fn = symmetry_combine;
              SiteSectorMapFn map = [fn](int64 site_idx, SectorArray const& old_sectors) {
                  return py::cast<SectorArray>(fn(site_idx, old_sectors));
              };
              set_common_symmetry(sites, std::move(map), new_symmetry);
              return;
          }
          throw py::type_error("symmetry_combine must be str, callable, or list; got " +
                               py::cast<std::string>(py::str(py::type::of(symmetry_combine))));
      },
      nb::arg("sites"),
      nb::arg("symmetry_combine") = "by_name",
      nb::arg("new_symmetry").none() = nb::none(),
      doc_cpp_ref(
        R"pydoc(Adjust the symmetries of the given sites *in place* so they can be used together.

Parameters
----------
sites
    Sites to combine. Modified **in place**; the same object must not appear twice.
symmetry_combine : ``'by_name'`` | ``'same'`` | ``'drop'`` | ``'independent'`` | callable | list
    How the common symmetry is built from the individual ones.
new_symmetry
    Required for callable / list modes; ignored for the string presets.
)pydoc",
        "lasyten::set_common_symmetry()"));
}

} // namespace lasyten
