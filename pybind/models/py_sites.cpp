#include <lasyten/models/sites.h>

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"
#include "docstrings/models/sites.h"

#include <optional>
#include <string>

namespace lasyten {

void
bind_models_sites(py::module_& m)
{
    py::class_<SpinSite, Site, py::smart_holder> spin_site(m, "SpinSite");
    spin_site.doc() = DOC(lasyten, SpinSite);

    spin_site
      .def(py::init<float64,
                    std::optional<std::string>,
                    TensorBackend::Ptr,
                    std::optional<std::string>,
                    bool>(),
           py::arg("S") = 0.5,
           py::arg("conserve") = py::none(),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none(),
           py::arg("use_test_su2") = false)
      .def_readwrite("S", &SpinSite::S)
      .def_readwrite("double_total_spin", &SpinSite::double_total_spin)
      .def_readwrite("conserve", &SpinSite::conserve)
      .def_readwrite("use_test_su2", &SpinSite::use_test_su2)
      .def("test_sanity", &SpinSite::test_sanity, DOC(lasyten, SpinSite, test_sanity))
      .def("__repr__", &SpinSite::repr);

    py::class_<SpinlessBosonSite, Site, py::smart_holder> spinless_boson_site(m,
                                                                              "SpinlessBosonSite");
    spinless_boson_site.doc() = DOC(lasyten, SpinlessBosonSite);

    spinless_boson_site
      .def(py::init<py::object,
                    py::object,
                    std::optional<float64>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("Nmax") = 1,
           py::arg("conserve") = py::none(),
           py::arg("filling") = py::none(),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("conserve", &SpinlessBosonSite::conserve)
      .def_readwrite("filling", &SpinlessBosonSite::filling)
      .def("__repr__", &SpinlessBosonSite::repr);

    py::class_<SpinlessFermionSite, Site, py::smart_holder> spinless_fermion_site(
      m, "SpinlessFermionSite");
    spinless_fermion_site.doc() = DOC(lasyten, SpinlessFermionSite);

    spinless_fermion_site
      .def(py::init<int64,
                    py::object,
                    std::optional<float64>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("num_species") = 1,
           py::arg("conserve") = "parity",
           py::arg("filling") = py::none(),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("num_species", &SpinlessFermionSite::num_species)
      .def_readwrite("conserve", &SpinlessFermionSite::conserve)
      .def_readwrite("filling", &SpinlessFermionSite::filling)
      .def("__repr__", &SpinlessFermionSite::repr);

    py::class_<SpinHalfFermionSite, Site, py::smart_holder> spin_half_fermion_site(
      m, "SpinHalfFermionSite");
    spin_half_fermion_site.doc() = DOC(lasyten, SpinHalfFermionSite);

    spin_half_fermion_site
      .def(py::init<std::string,
                    std::optional<std::string>,
                    std::optional<float64>,
                    TensorBackend::Ptr,
                    std::optional<std::string>,
                    bool>(),
           py::arg("conserve_N") = "parity",
           py::arg("conserve_S") = py::none(),
           py::arg("filling") = py::none(),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none(),
           py::arg("use_test_su2") = false)
      .def_readwrite("conserve_N", &SpinHalfFermionSite::conserve_N)
      .def_readwrite("conserve_S", &SpinHalfFermionSite::conserve_S)
      .def_readwrite("filling", &SpinHalfFermionSite::filling)
      .def_readwrite("use_test_su2", &SpinHalfFermionSite::use_test_su2)
      .def("test_sanity",
           &SpinHalfFermionSite::test_sanity,
           doc_cpp_ref(R"pydoc(test_sanity)pydoc", "lasyten::SpinHalfFermionSite::test_sanity()"))
      .def("__repr__", &SpinHalfFermionSite::repr);

    py::class_<ClockSite, Site, py::smart_holder> clock_site(m, "ClockSite");
    clock_site.doc() = DOC(lasyten, ClockSite);

    clock_site
      .def(py::init<int64,
                    std::optional<std::string>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("q"),
           py::arg("conserve") = py::none(),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("q", &ClockSite::q)
      .def_readwrite("conserve", &ClockSite::conserve)
      .def("__repr__", &ClockSite::repr);

    py::class_<AnyonSite, Site, py::smart_holder> anyon_site(m, "AnyonSite");
    anyon_site.doc() = DOC(lasyten, AnyonSite);

    anyon_site
      .def(py::init<Symmetry::Ptr, TensorBackend::Ptr, std::optional<std::string>>(),
           py::arg("symmetry"),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def("__repr__", &AnyonSite::repr);

    py::class_<FibonacciAnyonSite, AnyonSite, py::smart_holder> fibonacci_anyon_site(
      m, "FibonacciAnyonSite");
    fibonacci_anyon_site.doc() = DOC(lasyten, FibonacciAnyonSite);

    fibonacci_anyon_site
      .def(py::init<TensorBackend::Ptr, std::optional<std::string>>(),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def("__repr__", &FibonacciAnyonSite::repr);

    py::class_<IsingAnyonSite, AnyonSite, py::smart_holder> ising_anyon_site(m, "IsingAnyonSite");
    ising_anyon_site.doc() = DOC(lasyten, IsingAnyonSite);

    ising_anyon_site
      .def(py::init<int, TensorBackend::Ptr, std::optional<std::string>>(),
           py::arg("nu") = 1,
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def("__repr__", &IsingAnyonSite::repr);

    py::class_<GoldenSite, Site, py::smart_holder> golden_site(m, "GoldenSite");
    golden_site.doc() = DOC(lasyten, GoldenSite);

    golden_site
      .def(py::init<std::string, TensorBackend::Ptr, std::optional<std::string>>(),
           py::arg("handedness") = "left",
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def("__repr__", &GoldenSite::repr);

    py::class_<SU2kSpin1Site, Site, py::smart_holder> su2k_spin1_site(m, "SU2kSpin1Site");
    su2k_spin1_site.doc() = DOC(lasyten, SU2kSpin1Site);

    su2k_spin1_site
      .def(py::init<int64, TensorBackend::Ptr, std::optional<std::string>>(),
           py::arg("k"),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("k", &SU2kSpin1Site::k)
      .def("__repr__", &SU2kSpin1Site::repr);
}

} // namespace lasyten
