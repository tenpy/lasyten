#include "../doc_plus.h"
#include "docstrings/symmetries/base_symmetry.h"
#include "py_lasyten_pybind11.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <lasyten/symmetries/base_symmetry.h>

#include <cmath>
#include <vector>

namespace lasyten {

void declare_symmetry(py::module_& m);

void
bind_base_symmetry(py::module_& m)
{
    py::class_<BaseSymmetry, PyBaseSymmetry, py::smart_holder> cls(
      m, "BaseSymmetry", DOC(lasyten, BaseSymmetry));

    // Register Symmetry before methods that return it (signatures / stubgen).
    declare_symmetry(m);

    cls.def(py::init<FusionStyle, BraidingStyle, Sector, float64, bool, bool>(),
            py::arg("fusion_style"),
            py::arg("braiding_style"),
            py::arg("trivial_sector"),
            py::arg("num_sectors"),
            py::arg("has_complex_topological_data"),
            py::arg("trivial_shift"));

    cls.def_readonly("fusion_style", &BaseSymmetry::fusion_style)
      .def_readonly("braiding_style", &BaseSymmetry::braiding_style)
      .def_property_readonly("trivial_sector",
                             [](BaseSymmetry const& self) { return self.trivial_sector; })
      .def_property_readonly("num_sectors",
                             [](BaseSymmetry const& self) -> py::object {
                                 // Match Python ``int | float``: finite counts as int, else
                                 // float('inf').
                                 if (std::isfinite(self.num_sectors)) {
                                     return py::int_(static_cast<long long>(self.num_sectors));
                                 }
                                 return py::float_(self.num_sectors);
                             })
      .def_readonly("sector_ind_len", &BaseSymmetry::sector_ind_len)
      .def_property_readonly("empty_sector_array",
                             [](BaseSymmetry const& self) { return self.empty_sector_array; })
      .def_readonly("has_complex_topological_data", &BaseSymmetry::has_complex_topological_data)
      .def_readonly("trivial_shift", &BaseSymmetry::trivial_shift);

    cls
      .def_property_readonly("can_be_dropped",
                             &BaseSymmetry::can_be_dropped,
                             DOC(lasyten, BaseSymmetry, can_be_dropped))
      .def_property_readonly("has_symmetric_braid", &BaseSymmetry::has_symmetric_braid)
      .def_property_readonly("has_trivial_braid", &BaseSymmetry::has_trivial_braid)
      .def_property_readonly(
        "is_abelian", &BaseSymmetry::is_abelian, DOC(lasyten, BaseSymmetry, is_abelian))
      .def_property_readonly("has_unique_fusion",
                             &BaseSymmetry::has_unique_fusion,
                             DOC(lasyten, BaseSymmetry, has_unique_fusion));

    cls
      .def("dual_sector",
           &BaseSymmetry::dual_sector,
           py::arg("a"),
           DOC(lasyten, BaseSymmetry, dual_sector))
      .def("_n_symbol",
           &BaseSymmetry::_n_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           DOC(lasyten, BaseSymmetry, _n_symbol))
      .def("_f_symbol",
           &BaseSymmetry::_f_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           DOC(lasyten, BaseSymmetry, _f_symbol))
      .def("_r_symbol",
           &BaseSymmetry::_r_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           DOC(lasyten, BaseSymmetry, _r_symbol))
      .def("as_Symmetry", &BaseSymmetry::as_Symmetry, DOC(lasyten, BaseSymmetry, as_Symmetry))
      .def(
        "is_valid_sector",
        [](BaseSymmetry const& self, py::object a) {
            // Accept Sector or ndarray; lists/scalars are not valid sectors.
            if (py::isinstance<Sector>(a)) {
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
        py::arg("a"),
        DOC(lasyten, BaseSymmetry, is_valid_sector))
      .def("fusion_outcomes",
           &BaseSymmetry::fusion_outcomes,
           py::arg("a"),
           py::arg("b"),
           DOC(lasyten, BaseSymmetry, fusion_outcomes))
      .def("_fusion_tensor",
           &BaseSymmetry::_fusion_tensor,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("Z_a"),
           py::arg("Z_b"),
           DOC(lasyten, BaseSymmetry, _fusion_tensor))
      .def("swap_gate",
           &BaseSymmetry::swap_gate,
           py::arg("a"),
           py::arg("b"),
           DOC(lasyten, BaseSymmetry, swap_gate))
      .def("Z_iso", &BaseSymmetry::Z_iso, py::arg("a"), DOC(lasyten, BaseSymmetry, Z_iso))
      .def("all_sectors", &BaseSymmetry::all_sectors, DOC(lasyten, BaseSymmetry, all_sectors))
      .def("n_symbol",
           &BaseSymmetry::n_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           DOC(lasyten, BaseSymmetry, n_symbol))
      .def("f_symbol",
           &BaseSymmetry::f_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           DOC(lasyten, BaseSymmetry, f_symbol))
      .def("b_symbol",
           &BaseSymmetry::b_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           DOC(lasyten, BaseSymmetry, b_symbol))
      .def("r_symbol",
           &BaseSymmetry::r_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           DOC(lasyten, BaseSymmetry, r_symbol))
      .def("c_symbol",
           &BaseSymmetry::c_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           DOC(lasyten, BaseSymmetry, c_symbol))
      .def("fusion_tensor",
           &BaseSymmetry::fusion_tensor,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("Z_a") = false,
           py::arg("Z_b") = false,
           DOC(lasyten, BaseSymmetry, fusion_tensor))
      .def(
        "are_valid_sectors",
        [](BaseSymmetry const& self, py::object sectors) {
            if (py::isinstance<SectorArray>(sectors)) {
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
        py::arg("sectors"))
      .def("fusion_outcomes_broadcast",
           &BaseSymmetry::fusion_outcomes_broadcast,
           py::arg("a"),
           py::arg("b"),
           DOC(lasyten, BaseSymmetry, fusion_outcomes_broadcast))
      .def("multiple_fusion",
           [](BaseSymmetry const& self, py::args args) {
               std::vector<Sector> sectors;
               sectors.reserve(args.size());
               for (auto h : args) {
                   sectors.push_back(h.cast<Sector>());
               }
               return self.multiple_fusion(sectors);
           })
      .def(
        "multiple_fusion_broadcast",
        [](BaseSymmetry const& self, py::args args) {
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
        [](BaseSymmetry const& self, py::args args) {
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
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           DOC(lasyten, BaseSymmetry, can_fuse_to))
      .def("sector_dim",
           &BaseSymmetry::sector_dim,
           py::arg("a"),
           DOC(lasyten, BaseSymmetry, sector_dim))
      .def(
        "batch_sector_dim",
        [](BaseSymmetry const& self, SectorArray const& a) {
            return vector_i64_to_numpy(self.batch_sector_dim(a));
        },
        py::arg("a"),
        DOC(lasyten, BaseSymmetry, batch_sector_dim))
      .def(
        "batch_qdim",
        [](BaseSymmetry const& self, SectorArray const& a) {
            return vector_f64_to_numpy(self.batch_qdim(a));
        },
        py::arg("a"),
        DOC(lasyten, BaseSymmetry, batch_qdim))
      .def("sector_str",
           &BaseSymmetry::sector_str,
           py::arg("a"),
           DOC(lasyten, BaseSymmetry, sector_str))
      .def("get_sector",
           &BaseSymmetry::get_sector,
           py::arg("s"),
           DOC(lasyten, BaseSymmetry, get_sector))
      .def("dual_sectors",
           &BaseSymmetry::dual_sectors,
           py::arg("sectors"),
           DOC(lasyten, BaseSymmetry, dual_sectors))
      .def("frobenius_schur",
           &BaseSymmetry::frobenius_schur,
           py::arg("a"),
           DOC(lasyten, BaseSymmetry, frobenius_schur))
      .def("qdim", &BaseSymmetry::qdim, py::arg("a"), DOC(lasyten, BaseSymmetry, qdim))
      .def(
        "sqrt_qdim", &BaseSymmetry::sqrt_qdim, py::arg("a"), DOC(lasyten, BaseSymmetry, sqrt_qdim))
      .def("inv_sqrt_qdim",
           &BaseSymmetry::inv_sqrt_qdim,
           py::arg("a"),
           DOC(lasyten, BaseSymmetry, inv_sqrt_qdim))
      .def("total_qdim", &BaseSymmetry::total_qdim, DOC(lasyten, BaseSymmetry, total_qdim))
      .def("_b_symbol",
           &BaseSymmetry::_b_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           DOC(lasyten, BaseSymmetry, _b_symbol))
      .def("_c_symbol",
           &BaseSymmetry::_c_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
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
        py::arg("a"),
        DOC(lasyten, BaseSymmetry, topological_twist))
      .def("s_matrix_element",
           &BaseSymmetry::s_matrix_element,
           py::arg("a"),
           py::arg("b"),
           DOC(lasyten, BaseSymmetry, s_matrix_element))
      .def("s_matrix", &BaseSymmetry::s_matrix, DOC(lasyten, BaseSymmetry, s_matrix));
}

} // namespace lasyten
