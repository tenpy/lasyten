#include "../doc_plus.h"
#include "docstrings/symmetries/trees.h"
#include "py_lasyten_nanobind.h"

#include "symmetries/casters.hpp"

#include <lasyten/block_backend/block_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/symmetries/sector.h>
#include <lasyten/symmetries/sector_numpy.h>
#include <lasyten/symmetries/trees.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace lasyten {

namespace {

std::vector<std::uint8_t>
are_dual_from_python(py::handle obj, std::size_t expected_len)
{
    if (obj.is_none()) {
        return std::vector<std::uint8_t>(expected_len, 0);
    }
    py::array arr = py::array::ensure(obj);
    if (!arr || arr.ndim() != 1) {
        throw py::type_error("are_dual must be a 1D sequence of bool");
    }
    auto casted =
      py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>::ensure(arr);
    auto r = casted.unchecked<1>();
    if (static_cast<std::size_t>(r.shape(0)) != expected_len) {
        throw py::value_error("are_dual length mismatch");
    }
    std::vector<std::uint8_t> out(expected_len);
    for (std::size_t i = 0; i < expected_len; ++i) {
        out[i] = r(static_cast<py::ssize_t>(i)) ? 1 : 0;
    }
    return out;
}

std::optional<std::vector<int64>>
multiplicities_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    py::array arr = py::array::ensure(obj);
    if (!arr || arr.ndim() != 1) {
        throw py::type_error("multiplicities must be a 1D integer sequence");
    }
    auto casted = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(arr);
    auto r = casted.unchecked<1>();
    std::vector<int64> out(static_cast<std::size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = r(i);
    }
    return out;
}

SectorArray
sector_array_from_python(py::handle obj, Symmetry const& symmetry)
{
    if (obj.is_none()) {
        return symmetry.empty_sector_array;
    }
    if (lasyten::py_isinstance<SectorArray>(obj)) {
        return obj.cast<SectorArray>();
    }
    if (lasyten::py_isinstance<Sector>(obj)) {
        return SectorArray::from_sector(obj.cast<Sector>());
    }
    // empty list / sequence → empty SectorArray with correct sector_ind_len
    if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj)) {
        auto seq = obj.cast<py::sequence>();
        if (seq.size() == 0) {
            return symmetry.empty_sector_array;
        }
    }
    return sector_array_from_numpy(obj);
}

Symmetry::Ptr
symmetry_from_python(py::object symmetry_obj)
{
    if (lasyten::py_isinstance<Symmetry>(symmetry_obj)) {
        return symmetry_obj.cast<Symmetry::Ptr>();
    }
    if (lasyten::py_isinstance<SymmetryFactor>(symmetry_obj)) {
        auto ptr = symmetry_obj.cast<SymmetryFactor::Ptr>();
        return std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ ptr });
    }
    return symmetry_obj.attr("as_Symmetry")().cast<Symmetry::Ptr>();
}

BlockBackend*
block_backend_from_python(py::object backend)
{
    if (backend.is_none()) {
        return nullptr;
    }
    if (lasyten::py_isinstance<BlockBackend>(backend)) {
        return backend.cast<BlockBackend*>();
    }
    return backend.attr("block_backend").cast<BlockBackend*>();
}

/// Match historical Python: real topological coeffs as float, not complex(x+0j).
/// Complex coeffs make is_real fusion-tree mappings multiply float blocks by
/// complex128 and trip ComplexWarning-as-error under pytest.
py::object
scalar_to_python(complex128 z)
{
    if (z.imag() == 0.0) {
        return py::cast(z.real());
    }
    return py::cast(z);
}

py::dict
linear_combination_to_python(FusionTreeLinearCombination const& lc)
{
    py::dict out;
    for (auto const& [tree, coeff] : lc) {
        out[py::cast(tree)] = scalar_to_python(coeff);
    }
    return out;
}

py::dict
pair_linear_combination_to_python(FusionTreePairLinearCombination const& lc)
{
    py::dict out;
    for (auto const& [pair, coeff] : lc) {
        out[py::cast(pair)] = scalar_to_python(coeff);
    }
    return out;
}

} // namespace

void
bind_trees(nb::module_& m)
{
    nb::class_<FusionTree> cls(m, "FusionTree", DOC(lasyten, FusionTree));

    cls.def(nb::new_([](py::object symmetry_obj,
                        py::object uncoupled,
                        Sector coupled,
                        py::object are_dual,
                        py::object inner_sectors,
                        py::object multiplicities) {
                auto symmetry = symmetry_from_python(symmetry_obj);
                SectorArray unc = sector_array_from_python(uncoupled, *symmetry);
                SectorArray inner = sector_array_from_python(inner_sectors, *symmetry);
                auto dual = are_dual_from_python(are_dual, unc.size());
                return FusionTree(std::move(symmetry),
                                  std::move(unc),
                                  coupled,
                                  std::move(dual),
                                  std::move(inner),
                                  multiplicities_from_python(multiplicities));
            }),
            nb::arg("symmetry"),
            nb::arg("uncoupled"),
            nb::arg("coupled"),
            nb::arg("are_dual"),
            nb::arg("inner_sectors"),
            nb::arg("multiplicities").none() = nb::none());

    cls.def_rw("symmetry", &FusionTree::symmetry)
      .def_rw("uncoupled", &FusionTree::uncoupled)
      .def_rw("coupled", &FusionTree::coupled)
      .def_rw("inner_sectors", &FusionTree::inner_sectors)
      .def_ro("num_uncoupled", &FusionTree::num_uncoupled)
      .def_ro("num_vertices", &FusionTree::num_vertices)
      .def_ro("num_inner_edges", &FusionTree::num_inner_edges)
      .def_ro("fusion_style", &FusionTree::fusion_style)
      .def_ro("is_abelian", &FusionTree::is_abelian)
      .def_ro("braiding_style", &FusionTree::braiding_style);

    // Writable NumPy views into C++ storage so in-place updates (e.g. fusion_trees)
    // remain visible on the FusionTree instance.
    cls
      .def_prop_rw(
        "are_dual",
        [](py::object self_obj) {
            auto& self = self_obj.cast<FusionTree&>();
            return py::array(py::dtype::of<bool>(),
                             { self.are_dual.size() },
                             { sizeof(std::uint8_t) },
                             self.are_dual.data(),
                             self_obj);
        },
        [](FusionTree& self, py::object value) {
            self.are_dual = are_dual_from_python(value, self.num_uncoupled);
        })
      .def_prop_rw(
        "multiplicities",
        [](py::object self_obj) {
            auto& self = self_obj.cast<FusionTree&>();
            return py::array(py::dtype::of<int64>(),
                             { self.multiplicities.size() },
                             { sizeof(int64) },
                             self.multiplicities.data(),
                             self_obj);
        },
        [](FusionTree& self, py::object value) {
            auto opt = multiplicities_from_python(value);
            if (!opt) {
                self.multiplicities.assign(self.num_vertices, 0);
            } else {
                if (opt->size() != self.num_vertices) {
                    throw py::value_error("multiplicities length mismatch");
                }
                self.multiplicities = std::move(*opt);
            }
        });

    cls.def_prop_ro(
      "pre_Z_uncoupled", &FusionTree::pre_Z_uncoupled, DOC(lasyten, FusionTree, pre_Z_uncoupled));

    cls.def("test_sanity", &FusionTree::test_sanity, DOC(lasyten, FusionTree, test_sanity))
      .def_static(
        "from_abelian_symmetry",
        [](py::object symmetry_obj, py::object uncoupled, py::object are_dual) {
            auto symmetry = symmetry_from_python(symmetry_obj);
            SectorArray unc = sector_array_from_python(uncoupled, *symmetry);
            auto dual = are_dual_from_python(are_dual, unc.size());
            return FusionTree::from_abelian_symmetry(std::move(symmetry), unc, dual);
        },
        nb::arg("symmetry"),
        nb::arg("uncoupled"),
        nb::arg("are_dual"),
        DOC(lasyten, FusionTree, from_abelian_symmetry))
      .def_static(
        "from_empty",
        [](py::object symmetry_obj) {
            return FusionTree::from_empty(symmetry_from_python(symmetry_obj));
        },
        nb::arg("symmetry"),
        DOC(lasyten, FusionTree, from_empty))
      .def_static(
        "from_sector",
        [](py::object symmetry_obj, Sector sector, bool is_dual) {
            return FusionTree::from_sector(symmetry_from_python(symmetry_obj), sector, is_dual);
        },
        nb::arg("symmetry"),
        nb::arg("sector"),
        nb::arg("is_dual"),
        DOC(lasyten, FusionTree, from_sector))
      .def("__hash__", &FusionTree::hash)
      .def("__eq__", &FusionTree::operator==, nb::arg("other"))
      .def("ascii_diagram",
           &FusionTree::ascii_diagram,
           nb::arg("dagger") = false,
           DOC(lasyten, FusionTree, ascii_diagram))
      .def_static(
        "_str_uncoupled_coupled",
        [](Symmetry const& symmetry,
           SectorArray const& uncoupled,
           Sector coupled,
           py::object are_dual) {
            auto dual = are_dual_from_python(are_dual, uncoupled.size());
            return FusionTree::str_uncoupled_coupled(symmetry, uncoupled, coupled, dual);
        },
        nb::arg("symmetry"),
        nb::arg("uncoupled"),
        nb::arg("coupled"),
        nb::arg("are_dual"),
        doc_cpp_ref(DOC(lasyten, FusionTree), "lasyten::FusionTree::_str_uncoupled_coupled()"))
      .def_static(
        "bend_leg",
        [](FusionTree const& X, FusionTree const& Y, bool bend_downward, bool do_conj) {
            return pair_linear_combination_to_python(
              FusionTree::bend_leg(X, Y, bend_downward, do_conj));
        },
        nb::arg("X"),
        nb::arg("Y"),
        nb::arg("bend_downward"),
        nb::arg("do_conj") = false,
        DOC(lasyten, FusionTree, bend_leg))
      .def(
        "braid",
        [](FusionTree const& self, int64 j, bool overbraid, float64 cutoff, bool do_conj) {
            return linear_combination_to_python(self.braid(j, overbraid, cutoff, do_conj));
        },
        nb::arg("j"),
        nb::arg("overbraid"),
        nb::arg("cutoff") = 1e-16,
        nb::arg("do_conj") = false,
        DOC(lasyten, FusionTree, braid))
      .def("vertex_labels",
           &FusionTree::vertex_labels,
           nb::arg("n"),
           DOC(lasyten, FusionTree, vertex_labels))
      .def("modify_vertex_labels",
           &FusionTree::modify_vertex_labels,
           nb::arg("n"),
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("mu"),
           nb::arg("c"),
           nb::arg("copy") = true,
           DOC(lasyten, FusionTree, modify_vertex_labels))
      .def("__str__", &FusionTree::str)
      .def("__repr__", &FusionTree::repr)
      .def(
        "to_dense_block",
        [](
          FusionTree const& self, py::object backend, py::object dtype, bool understood_braiding) {
            std::optional<Dtype> dt;
            if (!dtype.is_none()) {
                dt = dtype.cast<Dtype>();
            }
            return self.to_dense_block(
              block_backend_from_python(backend), dt, understood_braiding);
        },
        nb::arg("backend").none() = nb::none(),
        nb::arg("dtype").none() = nb::none(),
        nb::arg("understood_braiding") = false,
        DOC(lasyten, FusionTree, to_dense_block))
      .def("copy", &FusionTree::copy, nb::arg("deep") = true, DOC(lasyten, FusionTree, copy))
      .def("extended",
           &FusionTree::extended,
           nb::arg("new_uncoupled"),
           nb::arg("mu"),
           nb::arg("new_coupled"),
           nb::arg("is_dual"),
           DOC(lasyten, FusionTree, extended))
      .def("insert", &FusionTree::insert, nb::arg("t2"), DOC(lasyten, FusionTree, insert))
      .def(
        "insert_at",
        [](FusionTree const& self, int64 n, FusionTree const& t2, float64 eps) {
            return linear_combination_to_python(self.insert_at(n, t2, eps));
        },
        nb::arg("n"),
        nb::arg("t2"),
        nb::arg("eps") = 1.0e-14,
        DOC(lasyten, FusionTree, insert_at))
      .def(
        "outer",
        [](FusionTree const& self, FusionTree const& right_tree, float64 eps) {
            return linear_combination_to_python(self.outer(right_tree, eps));
        },
        nb::arg("right_tree"),
        nb::arg("eps") = 1.0e-14,
        DOC(lasyten, FusionTree, outer))
      .def("split", &FusionTree::split, nb::arg("n"), DOC(lasyten, FusionTree, split))
      .def("split_bottom_vertex",
           &FusionTree::split_bottom_vertex,
           DOC(lasyten, FusionTree, split_bottom_vertex))
      .def(
        "twist",
        [](FusionTree const& self, std::vector<int64> const& idcs, bool overtwist) {
            return linear_combination_to_python(self.twist(idcs, overtwist));
        },
        nb::arg("idcs"),
        nb::arg("overtwist"),
        DOC(lasyten, FusionTree, twist));

    nb::class_<fusion_trees> ft(m, "fusion_trees", DOC(lasyten, fusion_trees));

    ft.def(
      nb::new_(
        [](py::object symmetry_obj, py::object uncoupled, Sector coupled, py::object are_dual) {
            auto symmetry = symmetry_from_python(symmetry_obj);
            SectorArray unc = sector_array_from_python(uncoupled, *symmetry);
            std::optional<std::vector<std::uint8_t>> dual;
            if (!are_dual.is_none()) {
                dual = are_dual_from_python(are_dual, unc.size());
            }
            return fusion_trees(std::move(symmetry), std::move(unc), coupled, std::move(dual));
        }),
      nb::arg("symmetry"),
      nb::arg("uncoupled"),
      nb::arg("coupled"),
      nb::arg("are_dual").none() = nb::none());

    ft.def_rw("symmetry", &fusion_trees::symmetry)
      .def_rw("uncoupled", &fusion_trees::uncoupled)
      .def_rw("coupled", &fusion_trees::coupled)
      .def_ro("num_uncoupled", &fusion_trees::num_uncoupled)
      .def_prop_rw(
        "are_dual",
        [](py::object self_obj) {
            auto& self = self_obj.cast<fusion_trees&>();
            return py::array(py::dtype::of<bool>(),
                             { self.are_dual.size() },
                             { sizeof(std::uint8_t) },
                             self.are_dual.data(),
                             self_obj);
        },
        [](fusion_trees& self, py::object value) {
            self.are_dual = are_dual_from_python(value, self.num_uncoupled);
        });

    ft.def("__iter__",
           [](fusion_trees const& self) {
               py::list out;
               for (auto const& t : self.all_trees()) {
                   out.append(t);
               }
               return py::iter(out);
           })
      .def("__len__", &fusion_trees::size)
      .def("__str__", &fusion_trees::str)
      .def("__repr__", &fusion_trees::repr)
      .def("index", &fusion_trees::index, nb::arg("tree"), DOC(lasyten, fusion_trees, index));
}

} // namespace lasyten
