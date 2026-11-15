#include <lasyten/models/sites.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "docstrings/models/sites.h"

#include <optional>
#include <string>
#include <vector>

namespace lasyten {

void
bind_models_sites(nb::module_& m)
{
    nb::class_<SpinSite, Site> spin_site(m, "SpinSite");
    spin_site.doc() = DOC(lasyten, SpinSite);

    spin_site
      .def(lasyten::ctor<decltype(spin_site),
                         float64,
                         std::optional<std::string>,
                         TensorBackend::Ptr,
                         std::optional<std::string>,
                         bool>(),
           nb::arg("S") = 0.5,
           nb::arg("conserve").none() = nb::none(),
           nb::arg("backend").none() = nullptr,
           nb::arg("default_device").none() = nb::none(),
           nb::arg("use_test_su2") = false)
      .def_rw("S", &SpinSite::S)
      .def_rw("double_total_spin", &SpinSite::double_total_spin)
      .def_rw("conserve", &SpinSite::conserve)
      .def_rw("use_test_su2", &SpinSite::use_test_su2)
      .def("test_sanity", &SpinSite::test_sanity, DOC(lasyten, SpinSite, test_sanity))
      .def("__repr__", &SpinSite::repr);

    nb::class_<SpinlessBosonSite, Site> spinless_boson_site(m, "SpinlessBosonSite");
    spinless_boson_site.doc() = DOC(lasyten, SpinlessBosonSite);

    spinless_boson_site
      .def(lasyten::ctor<decltype(spinless_boson_site),
                         py::object,
                         py::object,
                         std::optional<float64>,
                         TensorBackend::Ptr,
                         std::optional<std::string>>(),
           nb::arg("Nmax") = 1,
           nb::arg("conserve").none() = nb::none(),
           nb::arg("filling").none() = nb::none(),
           nb::arg("backend").none() = nullptr,
           nb::arg("default_device").none() = nb::none())
      .def_rw("conserve", &SpinlessBosonSite::conserve)
      .def_rw("filling", &SpinlessBosonSite::filling)
      .def("__repr__", &SpinlessBosonSite::repr);

    nb::class_<SpinlessFermionSite, Site> spinless_fermion_site(m, "SpinlessFermionSite");
    spinless_fermion_site.doc() = DOC(lasyten, SpinlessFermionSite);

    spinless_fermion_site
      .def(lasyten::ctor<decltype(spinless_fermion_site),
                         int64,
                         py::object,
                         std::optional<float64>,
                         TensorBackend::Ptr,
                         std::optional<std::string>>(),
           nb::arg("num_species") = 1,
           nb::arg("conserve") = "parity",
           nb::arg("filling").none() = nb::none(),
           nb::arg("backend").none() = nullptr,
           nb::arg("default_device").none() = nb::none())
      .def_rw("num_species", &SpinlessFermionSite::num_species)
      .def_rw("conserve", &SpinlessFermionSite::conserve)
      .def_rw("filling", &SpinlessFermionSite::filling)
      .def("__repr__", &SpinlessFermionSite::repr);

    nb::class_<SpinHalfFermionSite, Site> spin_half_fermion_site(m, "SpinHalfFermionSite");
    spin_half_fermion_site.doc() = DOC(lasyten, SpinHalfFermionSite);

    spin_half_fermion_site
      .def(lasyten::ctor<decltype(spin_half_fermion_site),
                         std::string,
                         std::optional<std::string>,
                         std::optional<float64>,
                         TensorBackend::Ptr,
                         std::optional<std::string>,
                         bool>(),
           nb::arg("conserve_N") = "parity",
           nb::arg("conserve_S").none() = nb::none(),
           nb::arg("filling").none() = nb::none(),
           nb::arg("backend").none() = nullptr,
           nb::arg("default_device").none() = nb::none(),
           nb::arg("use_test_su2") = false)
      .def_rw("conserve_N", &SpinHalfFermionSite::conserve_N)
      .def_rw("conserve_S", &SpinHalfFermionSite::conserve_S)
      .def_rw("filling", &SpinHalfFermionSite::filling)
      .def_rw("use_test_su2", &SpinHalfFermionSite::use_test_su2)
      .def("test_sanity",
           &SpinHalfFermionSite::test_sanity,
           doc_cpp_ref(R"pydoc(test_sanity)pydoc", "lasyten::SpinHalfFermionSite::test_sanity()"))
      .def("__repr__", &SpinHalfFermionSite::repr);

    nb::class_<ClockSite, Site> clock_site(m, "ClockSite");
    clock_site.doc() = DOC(lasyten, ClockSite);

    clock_site
      .def(lasyten::ctor<decltype(clock_site),
                         int64,
                         std::optional<std::string>,
                         TensorBackend::Ptr,
                         std::optional<std::string>>(),
           nb::arg("q"),
           nb::arg("conserve").none() = nb::none(),
           nb::arg("backend").none() = nullptr,
           nb::arg("default_device").none() = nb::none())
      .def_rw("q", &ClockSite::q)
      .def_rw("conserve", &ClockSite::conserve)
      .def("__repr__", &ClockSite::repr);

    nb::class_<AnyonSite, Site> anyon_site(m, "AnyonSite");
    anyon_site.doc() = DOC(lasyten, AnyonSite);

    anyon_site
      .def(lasyten::ctor<decltype(anyon_site),
                         Symmetry::Ptr,
                         TensorBackend::Ptr,
                         std::optional<std::string>>(),
           nb::arg("symmetry"),
           nb::arg("backend").none() = nullptr,
           nb::arg("default_device").none() = nb::none())
      .def("__repr__", &AnyonSite::repr);

    nb::class_<FibonacciAnyonSite, AnyonSite> fibonacci_anyon_site(m, "FibonacciAnyonSite");
    fibonacci_anyon_site.doc() = DOC(lasyten, FibonacciAnyonSite);

    fibonacci_anyon_site
      .def(lasyten::ctor<decltype(fibonacci_anyon_site),
                         TensorBackend::Ptr,
                         std::optional<std::string>>(),
           nb::arg("backend").none() = nullptr,
           nb::arg("default_device").none() = nb::none())
      .def("__repr__", &FibonacciAnyonSite::repr);

    nb::class_<IsingAnyonSite, AnyonSite> ising_anyon_site(m, "IsingAnyonSite");
    ising_anyon_site.doc() = DOC(lasyten, IsingAnyonSite);

    ising_anyon_site
      .def(
        lasyten::
          ctor<decltype(ising_anyon_site), int, TensorBackend::Ptr, std::optional<std::string>>(),
        nb::arg("nu") = 1,
        nb::arg("backend").none() = nullptr,
        nb::arg("default_device").none() = nb::none())
      .def("__repr__", &IsingAnyonSite::repr);

    nb::class_<GoldenSite, Site> golden_site(m, "GoldenSite");
    golden_site.doc() = DOC(lasyten, GoldenSite);

    golden_site
      .def(lasyten::ctor<decltype(golden_site),
                         std::string,
                         TensorBackend::Ptr,
                         std::optional<std::string>>(),
           nb::arg("handedness") = "left",
           nb::arg("backend").none() = nullptr,
           nb::arg("default_device").none() = nb::none())
      .def("__repr__", &GoldenSite::repr);

    nb::class_<SU2kSpin1Site, Site> su2k_spin1_site(m, "SU2kSpin1Site");
    su2k_spin1_site.doc() = DOC(lasyten, SU2kSpin1Site);

    su2k_spin1_site
      .def(
        lasyten::
          ctor<decltype(su2k_spin1_site), int64, TensorBackend::Ptr, std::optional<std::string>>(),
        nb::arg("k"),
        nb::arg("backend").none() = nullptr,
        nb::arg("default_device").none() = nb::none())
      .def_rw("k", &SU2kSpin1Site::k)
      .def("__repr__", &SU2kSpin1Site::repr);

    nb::class_<GroupedSite, Site> grouped_site(m, "GroupedSite");
    grouped_site.doc() = DOC(lasyten, GroupedSite);

    grouped_site
      .def(lasyten::ctor<decltype(grouped_site),
                         std::vector<Site::Ptr>,
                         std::optional<std::vector<std::string>>,
                         std::string>(),
           nb::arg("sites"),
           nb::arg("labels").none() = nb::none(),
           nb::arg("symmetry_combine") = "same")
      .def_rw("n_sites", &GroupedSite::n_sites)
      .def_rw("sites", &GroupedSite::sites)
      .def_rw("labels", &GroupedSite::labels)
      .def_rw("symmetry_combine", &GroupedSite::symmetry_combine)
      .def("kroneckerproduct",
           &GroupedSite::kroneckerproduct,
           nb::arg("ops"),
           DOC(lasyten, GroupedSite, kroneckerproduct))
      .def("__repr__", &GroupedSite::repr);

    m.def("group_sites",
          &group_sites,
          nb::arg("sites"),
          nb::arg("n") = 2,
          nb::arg("labels").none() = nb::none(),
          nb::arg("symmetry_combine") = "same",
          DOC(lasyten, group_sites));
}

} // namespace lasyten
