#include "../doc_plus.h"
#include "docstrings/symmetries/base_symmetry.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <lasyten/symmetries/base_symmetry.h>

#include <cmath>
#include <vector>

namespace lasyten {

void declare_symmetry(nb::module_& m);

void
bind_base_symmetry(nb::module_& m)
{
    nb::class_<BaseSymmetry> cls(m, "BaseSymmetry", DOC(lasyten, BaseSymmetry));

    // Register Symmetry before methods that return it (signatures / stubgen).
    declare_symmetry(m);

    cls.def(
      lasyten::ctor<decltype(cls), FusionStyle, BraidingStyle, Sector, float64, bool, bool>(),
      nb::arg("fusion_style"),
      nb::arg("braiding_style"),
      nb::arg("trivial_sector"),
      nb::arg("num_sectors"),
      nb::arg("has_complex_topological_data"),
      nb::arg("trivial_shift"));

    cls.def_ro("fusion_style", &BaseSymmetry::fusion_style)
      .def_ro("braiding_style", &BaseSymmetry::braiding_style)
      .def_prop_ro("trivial_sector", [](BaseSymmetry const& self) { return self.trivial_sector; })
      .def_prop_ro("num_sectors",
                   [](BaseSymmetry const& self) -> py::object {
                       // Match Python ``int | float``: finite counts as int, else
                       // float('inf').
                       if (std::isfinite(self.num_sectors)) {
                           return py::int_(static_cast<long long>(self.num_sectors));
                       }
                       return py::float_(self.num_sectors);
                   })
      .def_ro("sector_ind_len", &BaseSymmetry::sector_ind_len)
      .def_prop_ro("empty_sector_array",
                   [](BaseSymmetry const& self) { return self.empty_sector_array; })
      .def_ro("has_complex_topological_data", &BaseSymmetry::has_complex_topological_data)
      .def_ro("trivial_shift", &BaseSymmetry::trivial_shift);

    cls
      .def_prop_ro("can_be_dropped",
                   &BaseSymmetry::can_be_dropped,
                   DOC(lasyten, BaseSymmetry, can_be_dropped))
      .def_prop_ro("has_symmetric_braid", &BaseSymmetry::has_symmetric_braid)
      .def_prop_ro("has_trivial_braid", &BaseSymmetry::has_trivial_braid)
      .def_prop_ro("is_abelian", &BaseSymmetry::is_abelian, DOC(lasyten, BaseSymmetry, is_abelian))
      .def_prop_ro("has_unique_fusion",
                   &BaseSymmetry::has_unique_fusion,
                   DOC(lasyten, BaseSymmetry, has_unique_fusion));

    cls
      .def("dual_sector",
           &BaseSymmetry::dual_sector,
           nb::arg("a"),
           DOC(lasyten, BaseSymmetry, dual_sector))
      .def("_n_symbol",
           &BaseSymmetry::_n_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           DOC(lasyten, BaseSymmetry, _n_symbol))
      .def("_f_symbol",
           &BaseSymmetry::_f_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           nb::arg("d"),
           nb::arg("e"),
           nb::arg("f"),
           DOC(lasyten, BaseSymmetry, _f_symbol))
      .def("_r_symbol",
           &BaseSymmetry::_r_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           DOC(lasyten, BaseSymmetry, _r_symbol))
      .def("as_Symmetry", &BaseSymmetry::as_Symmetry, DOC(lasyten, BaseSymmetry, as_Symmetry))
      .def(
        "is_valid_sector",
        [](BaseSymmetry const& self, py::object a) {
            // Accept Sector or ndarray; lists/scalars are not valid sectors.
            if (lasyten::py_isinstance<Sector>(a)) {
                return self.is_valid_sector(a.cast<Sector>());
            }
            if (!py::isinstance<py::array>(a)) {
                return false;
            }
            try {
                return self.is_valid_sector(sector_from_numpy(a));
            } catch (...) {
                return false;
            }
        },
        nb::arg("a"),
        DOC(lasyten, BaseSymmetry, is_valid_sector))
      .def("fusion_outcomes",
           &BaseSymmetry::fusion_outcomes,
           nb::arg("a"),
           nb::arg("b"),
           DOC(lasyten, BaseSymmetry, fusion_outcomes))
      .def("_fusion_tensor",
           &BaseSymmetry::_fusion_tensor,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           nb::arg("Z_a"),
           nb::arg("Z_b"),
           DOC(lasyten, BaseSymmetry, _fusion_tensor))
      .def("swap_gate",
           &BaseSymmetry::swap_gate,
           nb::arg("a"),
           nb::arg("b"),
           DOC(lasyten, BaseSymmetry, swap_gate))
      .def("Z_iso", &BaseSymmetry::Z_iso, nb::arg("a"), DOC(lasyten, BaseSymmetry, Z_iso))
      .def("all_sectors", &BaseSymmetry::all_sectors, DOC(lasyten, BaseSymmetry, all_sectors))
      .def("n_symbol",
           &BaseSymmetry::n_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           DOC(lasyten, BaseSymmetry, n_symbol))
      .def("f_symbol",
           &BaseSymmetry::f_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           nb::arg("d"),
           nb::arg("e"),
           nb::arg("f"),
           DOC(lasyten, BaseSymmetry, f_symbol))
      .def("b_symbol",
           &BaseSymmetry::b_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           DOC(lasyten, BaseSymmetry, b_symbol))
      .def("r_symbol",
           &BaseSymmetry::r_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           DOC(lasyten, BaseSymmetry, r_symbol))
      .def("c_symbol",
           &BaseSymmetry::c_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           nb::arg("d"),
           nb::arg("e"),
           nb::arg("f"),
           DOC(lasyten, BaseSymmetry, c_symbol))
      .def("fusion_tensor",
           &BaseSymmetry::fusion_tensor,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           nb::arg("Z_a") = false,
           nb::arg("Z_b") = false,
           DOC(lasyten, BaseSymmetry, fusion_tensor))
      .def(
        "are_valid_sectors",
        [](BaseSymmetry const& self, py::object sectors) {
            if (lasyten::py_isinstance<SectorArray>(sectors)) {
                return self.are_valid_sectors(sectors.cast<SectorArray>());
            }
            if (!py::isinstance<py::array>(sectors)) {
                return false;
            }
            try {
                return self.are_valid_sectors(sector_array_from_numpy(sectors));
            } catch (...) {
                return false;
            }
        },
        nb::arg("sectors"))
      .def("fusion_outcomes_broadcast",
           &BaseSymmetry::fusion_outcomes_broadcast,
           nb::arg("a"),
           nb::arg("b"),
           DOC(lasyten, BaseSymmetry, fusion_outcomes_broadcast))
      .def("multiple_fusion",
           [](BaseSymmetry const& self, nb::args args_nb) {
               auto args = pyargs(args_nb);
               std::vector<Sector> sectors;
               sectors.reserve(args.size());
               for (auto h : args) {
                   sectors.push_back(h.cast<Sector>());
               }
               return self.multiple_fusion(sectors);
           })
      .def(
        "multiple_fusion_broadcast",
        [](BaseSymmetry const& self, nb::args args_nb) {
            auto args = pyargs(args_nb);
            std::vector<SectorArray> sectors;
            sectors.reserve(args.size());
            for (auto h : args) {
                sectors.push_back(h.cast<SectorArray>());
            }
            return self.multiple_fusion_broadcast(sectors);
        },
        DOC(lasyten, BaseSymmetry, multiple_fusion_broadcast))
      .def(
        "_multiple_fusion_broadcast",
        [](BaseSymmetry const& self, nb::args args_nb) {
            auto args = pyargs(args_nb);
            std::vector<SectorArray> sectors;
            sectors.reserve(args.size());
            for (auto h : args) {
                sectors.push_back(h.cast<SectorArray>());
            }
            return self._multiple_fusion_broadcast(sectors);
        },
        DOC(lasyten, BaseSymmetry, _multiple_fusion_broadcast))
      .def("can_fuse_to",
           &BaseSymmetry::can_fuse_to,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           DOC(lasyten, BaseSymmetry, can_fuse_to))
      .def("sector_dim",
           &BaseSymmetry::sector_dim,
           nb::arg("a"),
           DOC(lasyten, BaseSymmetry, sector_dim))
      .def(
        "batch_sector_dim",
        [](BaseSymmetry const& self, SectorArray const& a) {
            return vector_i64_to_numpy(self.batch_sector_dim(a));
        },
        nb::arg("a"),
        DOC(lasyten, BaseSymmetry, batch_sector_dim))
      .def(
        "batch_qdim",
        [](BaseSymmetry const& self, SectorArray const& a) {
            return vector_f64_to_numpy(self.batch_qdim(a));
        },
        nb::arg("a"),
        DOC(lasyten, BaseSymmetry, batch_qdim))
      .def("sector_str",
           &BaseSymmetry::sector_str,
           nb::arg("a"),
           DOC(lasyten, BaseSymmetry, sector_str))
      .def("get_sector",
           &BaseSymmetry::get_sector,
           nb::arg("s"),
           DOC(lasyten, BaseSymmetry, get_sector))
      .def("dual_sectors",
           &BaseSymmetry::dual_sectors,
           nb::arg("sectors"),
           DOC(lasyten, BaseSymmetry, dual_sectors))
      .def("frobenius_schur",
           &BaseSymmetry::frobenius_schur,
           nb::arg("a"),
           DOC(lasyten, BaseSymmetry, frobenius_schur))
      .def("qdim", &BaseSymmetry::qdim, nb::arg("a"), DOC(lasyten, BaseSymmetry, qdim))
      .def(
        "sqrt_qdim", &BaseSymmetry::sqrt_qdim, nb::arg("a"), DOC(lasyten, BaseSymmetry, sqrt_qdim))
      .def("inv_sqrt_qdim",
           &BaseSymmetry::inv_sqrt_qdim,
           nb::arg("a"),
           DOC(lasyten, BaseSymmetry, inv_sqrt_qdim))
      .def("total_qdim", &BaseSymmetry::total_qdim, DOC(lasyten, BaseSymmetry, total_qdim))
      .def("_b_symbol",
           &BaseSymmetry::_b_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           DOC(lasyten, BaseSymmetry, _b_symbol))
      .def("_c_symbol",
           &BaseSymmetry::_c_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           nb::arg("d"),
           nb::arg("e"),
           nb::arg("f"),
           DOC(lasyten, BaseSymmetry, _c_symbol))
      .def(
        "topological_twist",
        [](BaseSymmetry const& self, Sector const& a) -> py::object {
            complex128 const z = self.topological_twist(a);
            // Python historically returned int/float for real twists (±1). Returning
            // complex(±1+0j) makes is_real fusion-tree mappings multiply float blocks
            // by complex128 and trip ComplexWarning-as-error under pytest.
            if (z.imag() == 0.0) {
                return py::cast(z.real());
            }
            return py::cast(z);
        },
        nb::arg("a"),
        DOC(lasyten, BaseSymmetry, topological_twist))
      .def("s_matrix_element",
           &BaseSymmetry::s_matrix_element,
           nb::arg("a"),
           nb::arg("b"),
           DOC(lasyten, BaseSymmetry, s_matrix_element))
      .def("s_matrix", &BaseSymmetry::s_matrix, DOC(lasyten, BaseSymmetry, s_matrix));
}

} // namespace lasyten
