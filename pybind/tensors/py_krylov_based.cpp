#include <lasyten/tensors/krylov_based.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"

#include "docstrings/tensors/krylov_based.h"

#include <optional>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

py::object
optional_float_to_py(std::optional<float64> const& v)
{
    if (!v.has_value()) {
        return py::none();
    }
    return py::float_(*v);
}

py::object
optional_complex_to_py(std::optional<complex128> const& v)
{
    if (!v.has_value()) {
        return py::none();
    }
    return py::cast(*v);
}

py::array
eigenvalues_to_numpy(std::vector<complex128> const& E0s)
{
    py::array_t<complex128> arr(static_cast<py::ssize_t>(E0s.size()));
    auto r = arr.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < arr.shape(0); ++i) {
        r(i) = E0s[static_cast<std::size_t>(i)];
    }
    return arr;
}

} // namespace

void
bind_tensors_krylov_based(nb::module_& m)
{
    nb::class_<KrylovBased> krylov_based(m, "KrylovBased");
    krylov_based.doc() = DOC(lasyten, KrylovBased);

    krylov_based
      .def(
        lasyten::ctor<decltype(krylov_based), LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
        nb::arg("H"),
        nb::arg("psi0"),
        nb::arg("options").none() = nb::none())
      .def_rw("H", &KrylovBased::H)
      .def_rw("psi0", &KrylovBased::psi0)
      .def_prop_rw(
        "options",
        [](KrylovBased const& self) { return krylov_options_to_py(self.options); },
        [](KrylovBased& self, py::object v) { self.options = krylov_options_from_py(v); })
      .def_rw("N_min", &KrylovBased::N_min)
      .def_rw("N_max", &KrylovBased::N_max)
      .def_rw("N_cache", &KrylovBased::N_cache)
      .def_rw("P_tol", &KrylovBased::P_tol)
      .def_rw("min_gap", &KrylovBased::min_gap)
      .def_rw("reortho", &KrylovBased::reortho)
      .def_prop_rw(
        "E_shift",
        [](KrylovBased const& self) { return optional_float_to_py(self.E_shift); },
        [](KrylovBased& self, py::object v) {
            if (v.is_none()) {
                self.E_shift = std::nullopt;
            } else {
                self.E_shift = py::cast<float64>(v);
            }
        })
      .def_rw("_cutoff", &KrylovBased::_cutoff)
      .def_prop_ro("Es", &KrylovBased::Es_numpy)
      .def_prop_ro("_h_krylov", &KrylovBased::h_krylov_numpy)
      .def_prop_ro("_result_krylov", &KrylovBased::result_krylov_numpy)
      .def("_reset_krylov_state",
           &KrylovBased::_reset_krylov_state,
           "Clear cached Krylov vectors and the projected Hessenberg matrix.");

    nb::class_<GMRES> gmres(m, "GMRES");
    gmres.doc() = DOC(lasyten, GMRES);

    gmres
      .def(lasyten::ctor<decltype(gmres),
                         LinearOperator::Ptr,
                         VectorLike::Ptr,
                         VectorLike::Ptr,
                         py::object>(),
           nb::arg("A"),
           nb::arg("x"),
           nb::arg("b"),
           nb::arg("options").none() = nb::none())
      .def_rw("A", &GMRES::A)
      .def_rw("x", &GMRES::x)
      .def_rw("b", &GMRES::b)
      .def_prop_rw(
        "options",
        [](GMRES const& self) { return krylov_options_to_py(self.options); },
        [](GMRES& self, py::object v) { self.options = krylov_options_from_py(v); })
      .def_rw("N_min", &GMRES::N_min)
      .def_rw("N_max", &GMRES::N_max)
      .def_rw("restart", &GMRES::restart)
      .def_rw("res", &GMRES::res)
      .def("run", &GMRES::run)
      .def("arnoldi", &GMRES::arnoldi, nb::arg("k"))
      .def("apply_givens_rotation", &GMRES::apply_givens_rotation, nb::arg("k"))
      .def("givens_rotation", &GMRES::givens_rotation, nb::arg("k"))
      .def("backsolve", &GMRES::backsolve, nb::arg("k"))
      .def("reset", &GMRES::reset);

    nb::class_<Arnoldi, KrylovBased> arnoldi(m, "Arnoldi");
    arnoldi.doc() = DOC(lasyten, Arnoldi);

    arnoldi
      .def(lasyten::ctor<decltype(arnoldi), LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           nb::arg("H"),
           nb::arg("psi0"),
           nb::arg("options").none() = nb::none())
      .def_rw("E_tol", &Arnoldi::E_tol)
      .def_rw("which", &Arnoldi::which)
      .def_rw("num_ev", &Arnoldi::num_ev)
      .def(
        "run",
        [](Arnoldi& self) {
            auto [E0s, psis, N] = self.run();
            return py::make_tuple(eigenvalues_to_numpy(E0s), std::move(psis), N);
        },
        DOC(lasyten, Arnoldi, run));

    nb::class_<ArnoldiEvolution, Arnoldi> arnoldi_evolution(m, "ArnoldiEvolution");
    arnoldi_evolution.doc() = DOC(lasyten, ArnoldiEvolution);

    arnoldi_evolution
      .def(lasyten::
             ctor<decltype(arnoldi_evolution), LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           nb::arg("H"),
           nb::arg("psi0"),
           nb::arg("options").none() = nb::none())
      .def_rw("_result_norm", &ArnoldiEvolution::_result_norm)
      .def_prop_rw(
        "delta",
        [](ArnoldiEvolution const& self) { return optional_complex_to_py(self.delta); },
        [](ArnoldiEvolution& self, py::object v) {
            if (v.is_none()) {
                self.delta = std::nullopt;
            } else {
                self.delta = py::cast<complex128>(v);
            }
        })
      .def("run",
           &ArnoldiEvolution::run,
           nb::arg("delta"),
           nb::arg("normalize").none() = nb::none(),
           DOC(lasyten, ArnoldiEvolution, run));

    nb::class_<LanczosGroundState, KrylovBased> lanczos_ground_state(m, "LanczosGroundState");
    lanczos_ground_state.doc() = DOC(lasyten, LanczosGroundState);

    lanczos_ground_state
      .def(
        lasyten::
          ctor<decltype(lanczos_ground_state), LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
        nb::arg("H"),
        nb::arg("psi0"),
        nb::arg("options").none() = nb::none())
      .def_rw("E_tol", &LanczosGroundState::E_tol)
      .def("run", &LanczosGroundState::run, DOC(lasyten, LanczosGroundState, run));

    nb::class_<LanczosEvolution, LanczosGroundState> lanczos_evolution(m, "LanczosEvolution");
    lanczos_evolution.doc() = DOC(lasyten, LanczosEvolution);

    lanczos_evolution
      .def(lasyten::
             ctor<decltype(lanczos_evolution), LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           nb::arg("H"),
           nb::arg("psi0"),
           nb::arg("options").none() = nb::none())
      .def_rw("_result_norm", &LanczosEvolution::_result_norm)
      .def_prop_rw(
        "delta",
        [](LanczosEvolution const& self) { return optional_complex_to_py(self.delta); },
        [](LanczosEvolution& self, py::object v) {
            if (v.is_none()) {
                self.delta = std::nullopt;
            } else {
                self.delta = py::cast<complex128>(v);
            }
        })
      .def("run",
           &LanczosEvolution::run,
           nb::arg("delta"),
           nb::arg("normalize").none() = nb::none(),
           DOC(lasyten, LanczosEvolution, run));

    m.def("lanczos",
          static_cast<std::tuple<float64, VectorLike::Ptr, int64> (*)(
            LinearOperator::Ptr, VectorLike::Ptr, py::object)>(&lanczos),
          nb::arg("H"),
          nb::arg("psi"),
          nb::arg("options").none() = nb::none(),
          DOC(lasyten, lanczos));
}

} // namespace lasyten
