#include "../doc_plus.h"
#include "docstrings/symmetries/symmetry.h"
#include "py_lasyten_pybind11.h"

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
            auto const& sym = item.cast<Symmetry const&>();
            out.insert(out.end(), sym.factors.begin(), sym.factors.end());
        } else {
            out.push_back(item.cast<SymmetryFactor::Ptr>());
        }
    }
    return out;
}

std::optional<py::class_<Symmetry, BaseSymmetry, py::smart_holder>> g_symmetry;

} // namespace

void
declare_symmetry(py::module_& m)
{
    g_symmetry =
      py::class_<Symmetry, BaseSymmetry, py::smart_holder>(m, "Symmetry", DOC(lasyten, Symmetry));
}

void
bind_symmetry(py::module_& m)
{
    (void)m;
    auto& cls = *g_symmetry;

    cls.def(py::init([](py::sequence factors) {
                return std::make_shared<Symmetry>(flatten_factors_from_python(factors));
            }),
            py::arg("factors"));

    cls
      .def_property_readonly("factors",
                             [](Symmetry const& self) {
                                 py::list out;
                                 for (auto const& f : self.factors) {
                                     out.append(py::cast(f));
                                 }
                                 return out;
                             })
      .def_property_readonly("sector_slices",
                             [](Symmetry const& self) {
                                 auto np = py::module_::import("numpy");
                                 return np.attr("array")(self.sector_slices,
                                                         py::arg("dtype") = np.attr("int64"));
                             })
      .def_readwrite("fusion_tensor_dtype", &Symmetry::fusion_tensor_dtype)
      .def_property_readonly("num_factors", &Symmetry::num_factors);

    cls
      .def(
        "as_Symmetry",
        [](py::object self) { return self.cast<Symmetry::Ptr>(); },
        DOC(lasyten, Symmetry, as_Symmetry))
      .def(
        "is_valid_sector",
        [](Symmetry const& self, py::object a) {
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
        doc_cpp_ref(R"pydoc(is_valid_sector)pydoc", "lasyten::BaseSymmetry::is_valid_sector()"))
      .def(
        "are_valid_sectors",
        [](Symmetry const& self, py::object sectors) {
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
      .def("fusion_outcomes",
           &Symmetry::fusion_outcomes,
           py::arg("a"),
           py::arg("b"),
           DOC(lasyten, Symmetry, fusion_outcomes))
      .def("fusion_outcomes_broadcast",
           &Symmetry::fusion_outcomes_broadcast,
           py::arg("a"),
           py::arg("b"),
           DOC(lasyten, Symmetry, fusion_outcomes_broadcast))
      .def(
        "has_factor",
        [](Symmetry const& self, py::object other) {
            if (py::isinstance<SymmetryFactor>(other)) {
                return self.has_factor(other.cast<SymmetryFactor const&>());
            }
            // type check: any factor isinstance of the given type
            if (py::isinstance<py::type>(other)) {
                for (auto const& f : self.factors) {
                    if (py::isinstance(py::cast(f), other)) {
                        return true;
                    }
                }
                return false;
            }
            throw py::type_error("Expected instance or subclass of SymmetryFactor.");
        },
        py::arg("other"))
      .def(
        "dual_sector", &Symmetry::dual_sector, py::arg("a"), DOC(lasyten, Symmetry, dual_sector))
      .def("dual_sectors",
           &Symmetry::dual_sectors,
           py::arg("sectors"),
           DOC(lasyten, Symmetry, dual_sectors))
      .def("_n_symbol",
           &Symmetry::_n_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           DOC(lasyten, Symmetry, _n_symbol))
      .def("all_sectors", &Symmetry::all_sectors, DOC(lasyten, Symmetry, all_sectors))
      .def("factor_where",
           &Symmetry::factor_where,
           py::arg("descriptive_name"),
           DOC(lasyten, Symmetry, factor_where))
      .def("qdim", &Symmetry::qdim, py::arg("a"), DOC(lasyten, Symmetry, qdim))
      .def("sector_dim", &Symmetry::sector_dim, py::arg("a"), DOC(lasyten, Symmetry, sector_dim))
      .def(
        "batch_sector_dim",
        [](Symmetry const& self, SectorArray const& a) {
            return vector_i64_to_numpy(self.batch_sector_dim(a));
        },
        py::arg("a"),
        DOC(lasyten, Symmetry, batch_sector_dim))
      .def(
        "batch_qdim",
        [](Symmetry const& self, SectorArray const& a) {
            return vector_f64_to_numpy(self.batch_qdim(a));
        },
        py::arg("a"),
        DOC(lasyten, Symmetry, batch_qdim))
      .def("sector_str", &Symmetry::sector_str, py::arg("a"), DOC(lasyten, Symmetry, sector_str))
      .def("get_sector", &Symmetry::get_sector, py::arg("s"), DOC(lasyten, Symmetry, get_sector))
      .def("_f_symbol",
           &Symmetry::_f_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           DOC(lasyten, Symmetry, _f_symbol))
      .def("_r_symbol",
           &Symmetry::_r_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           DOC(lasyten, Symmetry, _r_symbol))
      .def("_fusion_tensor",
           &Symmetry::_fusion_tensor,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("Z_a") = false,
           py::arg("Z_b") = false,
           DOC(lasyten, Symmetry, _fusion_tensor))
      .def("swap_gate",
           &Symmetry::swap_gate,
           py::arg("a"),
           py::arg("b"),
           DOC(lasyten, Symmetry, swap_gate))
      .def("Z_iso", &Symmetry::Z_iso, py::arg("a"), DOC(lasyten, Symmetry, Z_iso))
      .def(
        "is_equivalent_to",
        [](Symmetry const& self, py::object other, bool strict_ordering) {
            Symmetry::Ptr other_sym;
            if (py::isinstance<Symmetry>(other)) {
                other_sym = other.cast<Symmetry::Ptr>();
            } else if (py::isinstance<SymmetryFactor>(other)) {
                other_sym = std::make_shared<Symmetry>(
                  std::vector<SymmetryFactor::Ptr>{ other.cast<SymmetryFactor::Ptr>() });
            } else {
                throw py::type_error("Expected Symmetry or SymmetryFactor");
            }
            return self.is_equivalent_to(*other_sym, strict_ordering);
        },
        py::arg("other"),
        py::arg("strict_ordering") = false,
        DOC(lasyten, Symmetry, is_equivalent_to))
      .def("__repr__", &Symmetry::repr)
      .def("__str__", &Symmetry::str)
      .def("__eq__",
           [](Symmetry const& self, py::object other) {
               if (!py::isinstance<Symmetry>(other)) {
                   return false;
               }
               return self.equals(other.cast<Symmetry const&>());
           })
      .def("__mul__",
           [](Symmetry const& self, py::object other) -> py::object {
               if (py::isinstance<Symmetry>(other)) {
                   return py::cast(self.mul(other.cast<Symmetry const&>()));
               }
               if (py::isinstance<SymmetryFactor>(other)) {
                   return py::cast(self.mul(other.cast<SymmetryFactor::Ptr>()));
               }
               return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
           })
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<Symmetry>(),
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<Symmetry>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace lasyten
