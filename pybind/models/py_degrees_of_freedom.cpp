#include <lasyten/models/degrees_of_freedom.h>

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"
#include "docstrings/models/degrees_of_freedom.h"

#include "tools/hdf5_bind.h"
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace lasyten {

namespace {

class PySite
  : public Site
  , public py::trampoline_self_life_support
{
  public:
    using Site::Site;

    void test_sanity() override { PYBIND11_OVERRIDE(void, Site, test_sanity); }
};

class PyOccupationDOF
  : public OccupationDOF
  , public py::trampoline_self_life_support
{
  public:
    using OccupationDOF::OccupationDOF;

    py::array get_annihilator_numpy(py::object species, bool include_JW = false) override
    {
        PYBIND11_OVERRIDE_PURE(
          py::array, OccupationDOF, get_annihilator_numpy, species, include_JW);
    }

    py::array get_creator_numpy(py::object species, bool include_JW = false) override
    {
        PYBIND11_OVERRIDE_PURE(py::array, OccupationDOF, get_creator_numpy, species, include_JW);
    }
};

py::object
dim_to_python(float64 dim)
{
    if (std::isfinite(dim) && std::floor(dim) == dim) {
        return py::int_(static_cast<long long>(dim));
    }
    return py::float_(dim);
}

} // namespace

void
bind_models_degrees_of_freedom(py::module_& m)
{
    m.attr("ALL_SPECIES") = all_species_sentinel();

    py::class_<Site, PySite, py::smart_holder> site(m, "Site");
    site.doc() = DOC(lasyten, Site);

    site
      .def(py::init<ElementarySpace::Ptr,
                    std::map<std::string, int64>,
                    std::map<std::string, SymmetricTensorPtr>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("leg"),
           py::arg("state_labels") = std::map<std::string, int64>{},
           py::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("leg", &Site::leg)
      .def_readwrite("state_labels", &Site::state_labels)
      .def_readwrite("backend", &Site::backend)
      .def_readwrite("default_device", &Site::default_device)
      .def_readwrite("onsite_operators", &Site::onsite_operators)
      .def_readwrite("hc_ops", &Site::hc_ops)
      .def("test_sanity", &Site::test_sanity, DOC(lasyten, Site, test_sanity))
      .def_property_readonly("symmetry", &Site::symmetry)
      .def_property_readonly("dim", [](Site const& self) { return dim_to_python(self.dim()); })
      .def("add_onsite_operator",
           &Site::add_onsite_operator,
           py::arg("name"),
           py::arg("op"),
           py::arg("is_diagonal") = py::none(),
           py::arg("understood_braiding") = false,
           py::arg("hc") = py::none(),
           DOC(lasyten, Site, add_onsite_operator))
      .def("valid_opname", &Site::valid_opname, py::arg("name"), DOC(lasyten, Site, valid_opname))
      .def("test_op",
           &Site::test_op,
           py::arg("opname"),
           "Check that an onsite operator name is known and available under this site's symmetry.")
      .def("rename_op",
           &Site::rename_op,
           py::arg("old_name"),
           py::arg("new_name"),
           "Rename an onsite operator while preserving its hermitian-conjugate mapping.")
      .def("get_op", &Site::get_op, py::arg("name"), DOC(lasyten, Site, get_op))
      .def("get_hc_op_name",
           &Site::get_hc_op_name,
           py::arg("name"),
           "Return the name of the hermitian conjugate of an operator or operator product.")
      .def("multiply_op_names",
           &Site::multiply_op_names,
           py::arg("names"),
           DOC(lasyten, Site, multiply_op_names))
      .def("multiply_operators",
           &Site::multiply_operators,
           py::arg("operators"),
           DOC(lasyten, Site, multiply_operators))
      .def("identity_tensor",
           &Site::identity_tensor,
           py::arg("w"),
           py::arg("overbraid") = true,
           DOC(lasyten, Site, identity_tensor))
      .def("get_onsite_W_block",
           &Site::get_onsite_W_block,
           py::arg("opname"),
           py::arg("w"),
           py::arg("overbraid") = true,
           py::arg("wL_label") = "wL",
           py::arg("wR_label") = "wR",
           DOC(lasyten, Site, get_onsite_W_block))
      .def("state_index", &Site::state_index, py::arg("label"), DOC(lasyten, Site, state_index))
      .def("state_indices",
           &Site::state_indices,
           py::arg("labels"),
           DOC(lasyten, Site, state_indices))
      .def("__repr__", &Site::repr)
      .def(
        "save_hdf5",
        [](Site const& self, py::object py_saver, py::object py_h5gr, std::string subpath) {
            // Use subclass hdf5_init_kwargs (S/conserve for SpinSite, etc.) rather than the
            // base Site leg/state_labels format, which many site constructors do not accept.
            lasyten::hdf5::PyBridge bridge(py_saver, py::none(), py_h5gr);
            lasyten::hdf5::py_save(subpath + "init_kwargs", self.hdf5_init_kwargs());
        },
        py::arg("hdf5_saver"),
        py::arg("h5gr"),
        py::arg("subpath"),
        DOC(lasyten, Site, save_hdf5));

    py::object classmethod = py::module_::import("builtins").attr("classmethod");
    site.attr("from_hdf5") = classmethod(py::cpp_function(
      [](py::object cls, py::object py_loader, py::object py_h5gr, std::string subpath) {
          lasyten::hdf5::PyBridge bridge(py::none(), py_loader, py_h5gr);
          py::object kwargs = lasyten::hdf5::py_load(subpath + "init_kwargs");
          py::object obj = cls(**kwargs);
          py_loader.attr("memorize_load")(py_h5gr, obj);
          return obj;
      },
      py::name("from_hdf5"),
      py::arg("cls"),
      py::arg("hdf5_loader"),
      py::arg("h5gr"),
      py::arg("subpath"),
      "Reconstruct a Site (or subclass) from HDF5."));

    py::class_<SpinDOF, Site, py::smart_holder> spin_dof(m, "SpinDOF");
    spin_dof.doc() = DOC(lasyten, SpinDOF);

    spin_dof
      .def(py::init<ElementarySpace::Ptr,
                    py::array,
                    std::map<std::string, int64>,
                    std::map<std::string, SymmetricTensorPtr>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("leg"),
           py::arg("spin_vector"),
           py::arg("state_labels") = std::map<std::string, int64>{},
           py::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("spin_vector", &SpinDOF::spin_vector)
      .def("test_sanity", &SpinDOF::test_sanity, DOC(lasyten, SpinDOF, test_sanity))
      .def_static("spin_vector_from_Sp",
                  &SpinDOF::spin_vector_from_Sp,
                  py::arg("Sz"),
                  py::arg("Sp"),
                  DOC(lasyten, SpinDOF, spin_vector_from_Sp))
      .def_static("conservation_law_to_symmetry",
                  &SpinDOF::conservation_law_to_symmetry,
                  py::arg("conserve"),
                  py::arg("use_test_su2") = false,
                  DOC(lasyten, SpinDOF, conservation_law_to_symmetry));

    py::class_<ClockDOF, Site, py::smart_holder> clock_dof(m, "ClockDOF");
    clock_dof.doc() = DOC(lasyten, ClockDOF);

    clock_dof
      .def(py::init<ElementarySpace::Ptr,
                    py::array,
                    std::map<std::string, int64>,
                    std::map<std::string, SymmetricTensorPtr>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("leg"),
           py::arg("clock_operators"),
           py::arg("state_labels") = std::map<std::string, int64>{},
           py::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("clock_operators", &ClockDOF::clock_operators)
      .def("test_sanity", &ClockDOF::test_sanity, DOC(lasyten, ClockDOF, test_sanity))
      .def_static("conservation_law_to_symmetry",
                  &ClockDOF::conservation_law_to_symmetry,
                  py::arg("conserve"),
                  py::arg("use_test_su2") = false,
                  DOC(lasyten, ClockDOF, conservation_law_to_symmetry));

    py::class_<AnyonDOF, Site, py::smart_holder> anyon_dof(m, "AnyonDOF");
    anyon_dof.doc() = DOC(lasyten, AnyonDOF);

    anyon_dof
      .def(py::init<ElementarySpace::Ptr,
                    std::vector<std::string>,
                    std::map<std::string, int64>,
                    std::map<std::string, SymmetricTensorPtr>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("leg"),
           py::arg("sector_names") = std::vector<std::string>{},
           py::arg("state_labels") = std::map<std::string, int64>{},
           py::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("sector_names", &AnyonDOF::sector_names)
      .def("test_sanity", &AnyonDOF::test_sanity, DOC(lasyten, AnyonDOF, test_sanity));

    py::class_<OccupationDOF, Site, PyOccupationDOF, py::smart_holder> occupation_dof(
      m, "OccupationDOF");
    occupation_dof.doc() = DOC(lasyten, OccupationDOF);

    occupation_dof.def_readwrite("num_species", &OccupationDOF::num_species)
      .def_readwrite("creators", &OccupationDOF::creators)
      .def_readwrite("annihilators", &OccupationDOF::annihilators)
      .def_readwrite("anti_commute_sign", &OccupationDOF::anti_commute_sign)
      .def_readwrite("species_names", &OccupationDOF::species_names)
      .def_readwrite("number_operators", &OccupationDOF::number_operators)
      .def_readwrite("n_tot", &OccupationDOF::n_tot)
      .def("test_sanity",
           &OccupationDOF::test_sanity,
           doc_cpp_ref(R"pydoc(test_sanity)pydoc", "lasyten::OccupationDOF::test_sanity()"))
      .def("add_individual_occupation_ops",
           &OccupationDOF::add_individual_occupation_ops,
           DOC(lasyten, OccupationDOF, add_individual_occupation_ops))
      .def("add_total_occupation_ops",
           &OccupationDOF::add_total_occupation_ops,
           doc_cpp_ref(R"pydoc(add_total_occupation_ops)pydoc",
                       "lasyten::OccupationDOF::add_total_occupation_ops()"))
      .def("get_annihilator_numpy",
           &OccupationDOF::get_annihilator_numpy,
           py::arg("species"),
           py::arg("include_JW") = false,
           DOC(lasyten, OccupationDOF, get_annihilator_numpy))
      .def("get_creator_numpy",
           &OccupationDOF::get_creator_numpy,
           py::arg("species"),
           py::arg("include_JW") = false,
           DOC(lasyten, OccupationDOF, get_creator_numpy))
      .def("get_occupation_numpy",
           &OccupationDOF::get_occupation_numpy,
           py::arg_v("species", all_species_sentinel(), "ALL_SPECIES"),
           DOC(lasyten, OccupationDOF, get_occupation_numpy))
      .def("get_species_idx", &OccupationDOF::get_species_idx, py::arg("species"));

    py::class_<BosonicDOF, OccupationDOF, py::smart_holder> bosonic_dof(m, "BosonicDOF");
    bosonic_dof.doc() = DOC(lasyten, BosonicDOF);

    bosonic_dof
      .def(py::init<ElementarySpace::Ptr,
                    py::array,
                    py::array,
                    py::array,
                    std::vector<std::optional<std::string>>,
                    std::map<std::string, int64>,
                    std::map<std::string, SymmetricTensorPtr>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("leg"),
           py::arg("Nmax"),
           py::arg("creators"),
           py::arg("annihilators"),
           py::arg("species_names") = std::vector<std::optional<std::string>>{},
           py::arg("state_labels") = std::map<std::string, int64>{},
           py::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("Nmax", &BosonicDOF::Nmax)
      .def_readwrite("JW", &BosonicDOF::JW)
      .def_readwrite("_JW", &BosonicDOF::JW)
      .def("test_sanity",
           &BosonicDOF::test_sanity,
           doc_cpp_ref(R"pydoc(test_sanity)pydoc", "lasyten::BosonicDOF::test_sanity()"))
      .def("add_individual_occupation_ops",
           &BosonicDOF::add_individual_occupation_ops,
           doc_cpp_ref(R"pydoc(add_individual_occupation_ops)pydoc",
                       "lasyten::BosonicDOF::add_individual_occupation_ops()"))
      .def("get_annihilator_numpy",
           &BosonicDOF::get_annihilator_numpy,
           py::arg("species"),
           py::arg("include_JW") = false)
      .def("get_creator_numpy",
           &BosonicDOF::get_creator_numpy,
           py::arg("species"),
           py::arg("include_JW") = false)
      .def_static("conservation_law_to_symmetry",
                  &BosonicDOF::conservation_law_to_symmetry,
                  py::arg("conserve"),
                  doc_cpp_ref(R"pydoc(conservation_law_to_symmetry)pydoc",
                              "lasyten::BosonicDOF::conservation_law_to_symmetry()"))
      .def_static("creation_annihilation_op_from_single_Nmax",
                  &BosonicDOF::creation_annihilation_op_from_single_Nmax,
                  py::arg("Nmax"),
                  py::arg("dim"))
      .def_static("creation_annihilation_ops_from_Nmax",
                  &BosonicDOF::creation_annihilation_ops_from_Nmax,
                  py::arg("Nmax"),
                  py::arg("dim"))
      .def_static("creation_annihilation_ops",
                  &BosonicDOF::creation_annihilation_ops,
                  py::arg("num_species"),
                  py::arg("Nmax"),
                  py::arg("dim"));

    py::class_<FermionicDOF, OccupationDOF, py::smart_holder> fermionic_dof(m, "FermionicDOF");
    fermionic_dof.doc() = DOC(lasyten, FermionicDOF);

    fermionic_dof
      .def(py::init<ElementarySpace::Ptr,
                    py::array,
                    py::array,
                    std::vector<std::optional<std::string>>,
                    std::map<std::string, int64>,
                    std::map<std::string, SymmetricTensorPtr>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("leg"),
           py::arg("creators"),
           py::arg("annihilators"),
           py::arg("species_names") = std::vector<std::optional<std::string>>{},
           py::arg("state_labels") = std::map<std::string, int64>{},
           py::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("partial_JWs", &FermionicDOF::partial_JWs)
      .def_readwrite("JW", &FermionicDOF::JW)
      .def_readwrite("_JW", &FermionicDOF::JW)
      .def("test_sanity",
           &FermionicDOF::test_sanity,
           doc_cpp_ref(R"pydoc(test_sanity)pydoc", "lasyten::FermionicDOF::test_sanity()"))
      .def("get_annihilator_numpy",
           &FermionicDOF::get_annihilator_numpy,
           py::arg("species"),
           py::arg("include_JW") = false)
      .def("get_creator_numpy",
           &FermionicDOF::get_creator_numpy,
           py::arg("species"),
           py::arg("include_JW") = false)
      .def_static("conservation_law_to_symmetry",
                  &FermionicDOF::conservation_law_to_symmetry,
                  py::arg("conserve"),
                  doc_cpp_ref(R"pydoc(conservation_law_to_symmetry)pydoc",
                              "lasyten::FermionicDOF::conservation_law_to_symmetry()"))
      .def_static("creation_annihilation_ops",
                  &FermionicDOF::creation_annihilation_ops,
                  py::arg("num_species"));
}

} // namespace lasyten
