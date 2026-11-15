#include "../doc_plus.h"
#include "docstrings/symmetries/symmetry.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <lasyten/symmetries/symmetry.h>
#include <lasyten/symmetries/symmetry_factor.h>

#include "tools/hdf5_bind.h"
#include <memory>
#include <optional>
#include <vector>

namespace lasyten {

namespace {

/// Flatten a Python sequence of SymmetryFactor / Symmetry into SymmetryFactor::Ptr list.
std::vector<SymmetryFactor::Ptr>
flatten_factors_from_python(py::sequence seq)
{
    std::vector<SymmetryFactor::Ptr> out;
    for (auto item : seq) {
        if (py::isinstance<Symmetry>(item)) {
            auto const& sym = py::cast<Symmetry const&>(item);
            out.insert(out.end(), sym.factors.begin(), sym.factors.end());
        } else {
            out.push_back(py::cast<SymmetryFactor::Ptr>(item));
        }
    }
    return out;
}

std::optional<nb::class_<Symmetry, BaseSymmetry>> g_symmetry;

} // namespace

void
declare_symmetry(nb::module_& m)
{
    g_symmetry = nb::class_<Symmetry, BaseSymmetry>(m, "Symmetry", DOC(lasyten, Symmetry));
}

void
bind_symmetry(nb::module_& m)
{
    (void)m;
    auto& cls = *g_symmetry;

    cls.def(nb::new_([](py::sequence factors) {
                return std::make_shared<Symmetry>(flatten_factors_from_python(factors));
            }),
            nb::arg("factors"));

    cls
      .def_prop_ro("factors",
                   [](Symmetry const& self) {
                       py::list out;
                       for (auto const& f : self.factors) {
                           out.append(py::cast(f));
                       }
                       return out;
                   })
      .def_prop_ro("sector_slices",
                   [](Symmetry const& self) {
                       auto np = py::module_::import("numpy");
                       return np.attr("array")(self.sector_slices,
                                               nb::arg("dtype") = np.attr("int64"));
                   })
      .def_rw("fusion_tensor_dtype", &Symmetry::fusion_tensor_dtype)
      .def_prop_ro("num_factors", &Symmetry::num_factors);

    cls
      .def(
        "as_Symmetry",
        [](py::object self) { return py::cast<Symmetry::Ptr>(self); },
        DOC(lasyten, Symmetry, as_Symmetry))
      .def(
        "is_valid_sector",
        [](Symmetry const& self, py::object a) {
            if (py::isinstance<Sector>(a)) {
                return self.is_valid_sector(py::cast<Sector>(a));
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
        doc_cpp_ref(R"pydoc(is_valid_sector)pydoc", "lasyten::BaseSymmetry::is_valid_sector()"))
      .def(
        "are_valid_sectors",
        [](Symmetry const& self, py::object sectors) {
            if (py::isinstance<SectorArray>(sectors)) {
                return self.are_valid_sectors(py::cast<SectorArray>(sectors));
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
      .def("fusion_outcomes",
           &Symmetry::fusion_outcomes,
           nb::arg("a"),
           nb::arg("b"),
           DOC(lasyten, Symmetry, fusion_outcomes))
      .def("fusion_outcomes_broadcast",
           &Symmetry::fusion_outcomes_broadcast,
           nb::arg("a"),
           nb::arg("b"),
           DOC(lasyten, Symmetry, fusion_outcomes_broadcast))
      .def(
        "has_factor",
        [](Symmetry const& self, py::object other) {
            if (py::isinstance<SymmetryFactor>(other)) {
                return self.has_factor(py::cast<SymmetryFactor const&>(other));
            }
            // type check: any factor isinstance of the given type
            if (PyType_Check(other.ptr())) {
                for (auto const& f : self.factors) {
                    if (py::isinstance(py::cast(f), other)) {
                        return true;
                    }
                }
                return false;
            }
            throw py::type_error("Expected instance or subclass of SymmetryFactor.");
        },
        nb::arg("other"))
      .def(
        "dual_sector", &Symmetry::dual_sector, nb::arg("a"), DOC(lasyten, Symmetry, dual_sector))
      .def("dual_sectors",
           &Symmetry::dual_sectors,
           nb::arg("sectors"),
           DOC(lasyten, Symmetry, dual_sectors))
      .def("_n_symbol",
           &Symmetry::_n_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           DOC(lasyten, Symmetry, _n_symbol))
      .def("all_sectors", &Symmetry::all_sectors, DOC(lasyten, Symmetry, all_sectors))
      .def("factor_where",
           &Symmetry::factor_where,
           nb::arg("descriptive_name"),
           DOC(lasyten, Symmetry, factor_where))
      .def("qdim", &Symmetry::qdim, nb::arg("a"), DOC(lasyten, Symmetry, qdim))
      .def("sector_dim", &Symmetry::sector_dim, nb::arg("a"), DOC(lasyten, Symmetry, sector_dim))
      .def(
        "batch_sector_dim",
        [](Symmetry const& self, SectorArray const& a) {
            return vector_i64_to_numpy(self.batch_sector_dim(a));
        },
        nb::arg("a"),
        DOC(lasyten, Symmetry, batch_sector_dim))
      .def(
        "batch_qdim",
        [](Symmetry const& self, SectorArray const& a) {
            return vector_f64_to_numpy(self.batch_qdim(a));
        },
        nb::arg("a"),
        DOC(lasyten, Symmetry, batch_qdim))
      .def("sector_str", &Symmetry::sector_str, nb::arg("a"), DOC(lasyten, Symmetry, sector_str))
      .def("get_sector", &Symmetry::get_sector, nb::arg("s"), DOC(lasyten, Symmetry, get_sector))
      .def("_f_symbol",
           &Symmetry::_f_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           nb::arg("d"),
           nb::arg("e"),
           nb::arg("f"),
           DOC(lasyten, Symmetry, _f_symbol))
      .def("_r_symbol",
           &Symmetry::_r_symbol,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           DOC(lasyten, Symmetry, _r_symbol))
      .def("_fusion_tensor",
           &Symmetry::_fusion_tensor,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           nb::arg("Z_a") = false,
           nb::arg("Z_b") = false,
           DOC(lasyten, Symmetry, _fusion_tensor))
      .def("swap_gate",
           &Symmetry::swap_gate,
           nb::arg("a"),
           nb::arg("b"),
           DOC(lasyten, Symmetry, swap_gate))
      .def("Z_iso", &Symmetry::Z_iso, nb::arg("a"), DOC(lasyten, Symmetry, Z_iso))
      .def(
        "is_equivalent_to",
        [](Symmetry const& self, py::object other, bool strict_ordering) {
            Symmetry::Ptr other_sym;
            if (py::isinstance<Symmetry>(other)) {
                other_sym = py::cast<Symmetry::Ptr>(other);
            } else if (py::isinstance<SymmetryFactor>(other)) {
                other_sym = std::make_shared<Symmetry>(
                  std::vector<SymmetryFactor::Ptr>{ py::cast<SymmetryFactor::Ptr>(other) });
            } else {
                throw py::type_error("Expected Symmetry or SymmetryFactor");
            }
            return self.is_equivalent_to(*other_sym, strict_ordering);
        },
        nb::arg("other"),
        nb::arg("strict_ordering") = false,
        DOC(lasyten, Symmetry, is_equivalent_to))
      .def("__repr__", &Symmetry::repr)
      .def("__str__", &Symmetry::str)
      .def("__eq__",
           [](Symmetry const& self, py::object other) {
               if (!py::isinstance<Symmetry>(other)) {
                   return false;
               }
               return self.equals(py::cast<Symmetry const&>(other));
           })
      .def("__mul__",
           [](Symmetry const& self, py::object other) -> py::object {
               if (py::isinstance<Symmetry>(other)) {
                   return py::cast(self.mul(py::cast<Symmetry const&>(other)));
               }
               if (py::isinstance<SymmetryFactor>(other)) {
                   return py::cast(self.mul(py::cast<SymmetryFactor::Ptr>(other)));
               }
               return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
           })
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<Symmetry>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<Symmetry>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));
}

} // namespace lasyten
