#include "../doc_plus.h"
#include "../ndarray_utils.h"
#include "docstrings/symmetries/spaces.h"
#include "py_lasyten_nanobind.h"

#include "backends/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <lasyten/backends/tensor_backend.h>
#include <lasyten/symmetries/sector_numpy.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/symmetry.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/symmetric_tensor.h>

#include "tools/hdf5_bind.h"
#include "tools/pickle_bind.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

/// ``blockmask`` argument of ``take_slice``: any 1D array-like of bool.
std::vector<bool>
blockmask_from_python(py::object blockmask)
{
    py::array const arr = py::array::ensure(blockmask);
    if (!arr) {
        throw py::type_error("blockmask must be a 1D array of bool");
    }
    auto casted = py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(arr);
    if (!casted || casted.ndim() != 1) {
        throw py::type_error("blockmask must be a 1D array of bool");
    }
    auto const view = casted.unchecked<1>();
    std::vector<bool> out(static_cast<std::size_t>(view.shape(0)));
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = view(i);
    }
    return out;
}

Symmetry::Ptr
symmetry_from_python(py::object symmetry_obj)
{
    if (py::isinstance<Symmetry>(symmetry_obj)) {
        return py::cast<Symmetry::Ptr>(symmetry_obj);
    }
    if (py::isinstance<SymmetryFactor>(symmetry_obj)) {
        auto ptr = py::cast<SymmetryFactor::Ptr>(symmetry_obj);
        return std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ ptr });
    }
    return py::cast<Symmetry::Ptr>(symmetry_obj.attr("as_Symmetry")());
}

std::optional<std::vector<int64>>
perm_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    py::array arr = py::array::ensure(obj);
    if (!arr || arr.ndim() != 1) {
        throw py::type_error("basis permutation must be a 1D integer sequence or None");
    }
    auto casted = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(arr);
    auto r = casted.unchecked<1>();
    std::vector<int64> out(static_cast<std::size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = r(i);
    }
    return out;
}

nb::object
perm_to_numpy(std::vector<int64> perm)
{
    return nb::cast(ndarray_utils::vector_to_numpy_1d(std::move(perm)));
}

py::object
dim_to_python(float64 dim)
{
    if (std::isfinite(dim) && std::floor(dim) == dim) {
        return py::int_(static_cast<long long>(dim));
    }
    return py::float_(dim);
}

std::optional<std::vector<int64>>
multiplicities_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return perm_from_python(obj);
}

std::optional<std::string>
sector_order_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return py::cast<std::string>(obj);
}

nb::object
f64_vector_to_numpy(std::vector<float64> v)
{
    return nb::cast(ndarray_utils::vector_to_numpy_1d(std::move(v)));
}

nb::object
slices_to_numpy(std::optional<std::vector<std::array<int64, 2>>> const& slices)
{
    if (!slices) {
        return nb::none();
    }
    std::size_t const rows = slices->size();
    std::vector<int64> flat(rows * 2);
    for (std::size_t i = 0; i < rows; ++i) {
        flat[i * 2] = (*slices)[i][0];
        flat[i * 2 + 1] = (*slices)[i][1];
    }
    return nb::cast(ndarray_utils::vector_to_numpy_2d(std::move(flat), rows, 2));
}

nb::object
sector_array_to_ndarray(SectorArray const& src)
{
    std::size_t const rows = src.size();
    std::size_t const cols = src.sector_ind_len();
    std::vector<std::int64_t> v(rows * cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::uint8_t j = 0; j < cols; ++j) {
            v[i * cols + j] = src[i][j];
        }
    }
    // Return as ``nb::object`` so ``def_prop_ro`` does not apply ``reference_internal`` to an
    // ndarray that already has a capsule owner.
    return nb::cast(ndarray_utils::vector_to_numpy_2d(std::move(v), rows, cols));
}

std::optional<std::vector<int64>>
drop_which_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    if (py::isinstance<py::str>(obj)) {
        auto s = py::cast<std::string>(obj);
        if (s == "all") {
            return std::nullopt;
        }
        throw py::value_error("which must be 'all', an int, or a list of ints");
    }
    if (py::isinstance<py::int_>(obj)) {
        return std::vector<int64>{ py::cast<int64>(obj) };
    }
    return py::cast<std::vector<int64>>(obj);
}

SectorArray
sector_array_from_python(py::handle obj, Symmetry const& symmetry)
{
    if (py::isinstance<SectorArray>(obj)) {
        return py::cast<SectorArray>(obj);
    }
    if (py::isinstance<Sector>(obj)) {
        return SectorArray::from_sector(py::cast<Sector>(obj));
    }
    // empty list / sequence → empty SectorArray with correct sector_ind_len
    // (np.asarray([]) is 1D and rejected by sector_array_from_numpy)
    if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj) &&
        !py::isinstance<py::array>(obj)) {
        auto seq = py::reinterpret_borrow<py::sequence>(obj);
        if (seq.size() == 0) {
            return SectorArray::empty(symmetry.sector_ind_len);
        }
    }
    auto arr = sector_array_from_numpy(obj);
    if (arr.sector_ind_len() == 0 && arr.size() == 0) {
        return SectorArray::empty(symmetry.sector_ind_len);
    }
    return arr;
}

Symmetry::Ptr
optional_symmetry_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return nullptr;
    }
    return symmetry_from_python(py::reinterpret_borrow<py::object>(obj));
}

SectorMapFn
sector_map_from_python(py::function sector_map)
{
    return [sector_map](SectorArray const& sectors) {
        return py::cast<SectorArray>(sector_map(sectors));
    };
}

Sector
sector_from_python(py::handle obj)
{
    if (py::isinstance<Sector>(obj)) {
        return py::cast<Sector>(obj);
    }
    return sector_from_numpy(obj);
}

py::slice
index_slice_to_python(IndexSlice const& slc)
{
    return py::slice(py::int_(slc.start), py::int_(slc.stop), py::none());
}

std::vector<py::object>
objects_from_python(py::handle obj)
{
    std::vector<py::object> out;
    for (py::handle item : obj) {
        out.push_back(py::reinterpret_borrow<py::object>(item));
    }
    return out;
}

std::vector<Leg::Ptr>
legs_from_python(py::handle obj)
{
    std::vector<Leg::Ptr> out;
    for (py::handle item : obj) {
        out.push_back(py::cast_non_null<Leg::Ptr>(item, "elements of factors"));
    }
    return out;
}

py::list
objects_to_python(std::vector<py::object> const& objects)
{
    py::list out;
    for (auto const& obj : objects) {
        out.append(obj);
    }
    return out;
}

std::optional<nb::class_<Space>> g_space;
std::optional<nb::class_<ElementarySpace, Space>> g_elementary_space;

/// Bind axis attributes shared by `ElementarySpace` and `LegPipe` (formerly on polymorphic `Leg`).
template<typename Class, typename T>
void
bind_axis_common(Class& cls)
{
    cls
      .def_prop_rw(
        "basis_perm",
        [](T const& self) { return perm_to_numpy(self.basis_perm()); },
        [](T& self, py::object basis_perm) { self.set_basis_perm(perm_from_python(basis_perm)); },
        doc_cpp_ref(DOC(lasyten, Leg), "lasyten::Leg::basis_perm()"))
      .def_prop_rw(
        "_basis_perm",
        [](T const& self) -> py::object {
            if (!self.has_custom_basis_perm()) {
                return py::none();
            }
            return perm_to_numpy(self.basis_perm());
        },
        [](T& self, py::object basis_perm) { self.set_basis_perm(perm_from_python(basis_perm)); })
      .def_prop_rw(
        "inverse_basis_perm",
        [](T const& self) { return perm_to_numpy(self.inverse_basis_perm()); },
        [](T& self, py::object inverse_basis_perm) {
            self.set_inverse_basis_perm(perm_from_python(inverse_basis_perm));
        },
        doc_cpp_ref(DOC(lasyten, Leg), "lasyten::Leg::inverse_basis_perm()"))
      .def_prop_rw(
        "_inverse_basis_perm",
        [](T const& self) -> py::object {
            if (!self.has_custom_basis_perm()) {
                return py::none();
            }
            return perm_to_numpy(self.inverse_basis_perm());
        },
        [](T& self, py::object inverse_basis_perm) {
            self.set_inverse_basis_perm(perm_from_python(inverse_basis_perm));
        })
      .def_prop_ro("flat_legs", &T::flat_legs, DOC(lasyten, LegPipe, flat_legs))
      .def_prop_ro("flat_spaces", &T::flat_spaces, DOC(lasyten, LegPipe, flat_spaces))
      .def_prop_ro("num_flat_legs", &T::num_flat_legs, DOC(lasyten, LegPipe, num_flat_legs))
      .def_prop_ro("ascii_arrow", &T::ascii_arrow, DOC(lasyten, LegPipe, ascii_arrow))
      .def("_flat_leg_permutation",
           &T::_flat_leg_permutation,
           nb::arg("offset") = 0,
           DOC(lasyten, LegPipe, _flat_leg_permutation))
      .def("apply_basis_perm",
           &T::apply_basis_perm,
           nb::arg("arr"),
           nb::arg("axis") = 0,
           nb::arg("inverse") = false,
           nb::arg("pre_compose") = false,
           DOC(lasyten, LegPipe, apply_basis_perm))
      .def_prop_ro("is_pipe", &T::is_pipe, DOC(lasyten, LegPipe, is_pipe))
      .def_prop_ro("is_abelian_leg_pipe",
                   &T::is_abelian_leg_pipe,
                   DOC(lasyten, LegPipe, is_abelian_leg_pipe));
}

void bind_elementary_space(nb::module_& m);

void bind_direct_sum_space(nb::module_& m);

void bind_tensor_product(nb::module_& m);

void bind_abelian_leg_pipe(nb::module_& m);

} // namespace

void
bind_spaces(nb::module_& m)
{
    // `Leg` is a C++ value-type handle; Python exposes it as an ABC in spaces.py.
    // Declare Space / ElementarySpace early for return types (stubgen / cross-refs).
    // No trampolines: pybind11 smart_holder + enable_shared_from_this + trampoline
    // is unsupported ("trampoline-self-life-support for external shared_ptr").
    g_space = nb::class_<Space>(m, "Space", DOC(lasyten, Space));
    g_elementary_space =
      nb::class_<ElementarySpace, Space>(m, "ElementarySpace", DOC(lasyten, ElementarySpace));

    auto& space = *g_space;

    space
      .def_rw("symmetry", &Space::symmetry)
      // Return SectorArray by value: def_readwrite member refs need a shared_ptr keep-alive
      // that fails for smart_holder + enable_shared_from_this on some construction paths.
      .def_prop_rw(
        "sector_decomposition",
        [](Space const& self) { return self.sector_decomposition; },
        [](Space& self, SectorArray const& v) { self.sector_decomposition = v; })
      .def_rw("sector_order", &Space::sector_order)
      .def_prop_rw(
        "multiplicities",
        [](Space const& self) { return perm_to_numpy(self.multiplicities); },
        [](Space& self, py::object obj) {
            auto m = multiplicities_from_python(obj);
            if (!m) {
                self.multiplicities.assign(static_cast<std::size_t>(self.num_sectors), 1);
            } else {
                self.multiplicities = std::move(*m);
            }
        })
      .def_ro("num_sectors",
              &Space::num_sectors,
              R"pydoc(The number of unique sectors appearing in this space, regardless of
their multiplicity. See :attr:`~lasyten.symmetries.spaces.Space.total_multiplicity` for the sum of
the sector multiplicities.)pydoc")
      .def_prop_ro(
        "total_multiplicity", &Space::total_multiplicity, DOC(lasyten, Space, total_multiplicity))
      .def_prop_ro("sector_dims",
                   [](Space const& self) -> py::object {
                       if (!self.sector_dims) {
                           return py::none();
                       }
                       return perm_to_numpy(*self.sector_dims);
                   })
      .def_prop_ro("sector_qdims",
                   [](Space const& self) { return f64_vector_to_numpy(self.sector_qdims); })
      .def_prop_ro("slices", [](Space const& self) { return slices_to_numpy(self.slices); })
      .def_prop_ro("dim", [](Space const& self) { return dim_to_python(self.dim); });

    space.def_prop_ro("dual", &Space::dual, DOC(lasyten, Space, dual))
      .def_prop_ro("is_trivial",
                   &Space::is_trivial,
                   doc_cpp_ref(DOC(lasyten, Space), "lasyten::Space::is_trivial()"));

    space
      .def("test_sanity",
           &Space::test_sanity,
           doc_cpp_ref(DOC(lasyten, Space), "lasyten::Space::test_sanity()"))
      .def("__eq__", &Space::operator==, nb::arg("other"))
      .def("is_isomorphic_to",
           &Space::is_isomorphic_to,
           nb::arg("other"),
           doc_cpp_ref(DOC(lasyten, Space), "lasyten::Space::is_isomorphic_to()"))
      .def("is_subspace_of",
           &Space::is_subspace_of,
           nb::arg("other"),
           doc_cpp_ref(DOC(lasyten, Space), "lasyten::Space::is_subspace_of()"))
      .def("as_ElementarySpace",
           &Space::as_ElementarySpace,
           nb::arg("is_dual") = false,
           DOC(lasyten, ElementarySpace, as_ElementarySpace))
      .def(
        "change_symmetry",
        [](Space& self, py::object symmetry_obj, py::function sector_map, bool injective) {
            auto symmetry = symmetry_from_python(symmetry_obj);
            SectorMapFn map = [sector_map](SectorArray const& sectors) {
                return py::cast<SectorArray>(sector_map(sectors));
            };
            return self.change_symmetry(std::move(symmetry), std::move(map), injective);
        },
        nb::arg("symmetry"),
        nb::arg("sector_map"),
        nb::arg("injective") = false,
        DOC(lasyten, Space, change_symmetry))
      .def(
        "drop_symmetry",
        [](Space& self, py::object which) {
            return self.drop_symmetry(drop_which_from_python(which));
        },
        nb::arg("which") = "all",
        DOC(lasyten, Space, drop_symmetry))
      .def("as_Space", &Space::as_Space, DOC(lasyten, Space, as_Space))
      .def(
        "sector_decomposition_where",
        [](Space const& self, Sector sector) -> py::object {
            auto idx = self.sector_decomposition_where(sector);
            if (!idx) {
                return py::none();
            }
            return py::int_(*idx);
        },
        nb::arg("sector"),
        DOC(lasyten, Space, sector_decomposition_where))
      .def("sector_multiplicity",
           &Space::sector_multiplicity,
           nb::arg("sector"),
           DOC(lasyten, Space, sector_multiplicity));

    nb::class_<LegPipe> pipe(m, "LegPipe", DOC(lasyten, LegPipe));

    pipe.def(nb::new_([](py::sequence legs_obj, bool is_dual, bool combine_cstyle) {
                 std::vector<Leg::Ptr> legs;
                 legs.reserve(static_cast<std::size_t>(legs_obj.size()));
                 for (py::handle item : legs_obj) {
                     legs.push_back(py::cast_non_null<Leg::Ptr>(item, "elements of legs"));
                 }
                 return std::make_shared<LegPipe>(std::move(legs), is_dual, combine_cstyle);
             }),
             nb::arg("legs"),
             nb::arg("is_dual") = false,
             nb::arg("combine_cstyle") = true);

    pipe.def_rw("symmetry", &LegPipe::symmetry)
      .def_prop_rw(
        "dim",
        [](LegPipe const& self) { return dim_to_python(self.dim); },
        [](LegPipe& self, py::object dim_obj) {
            self.dim = py::cast<float64>(py::float_(dim_obj));
        })
      .def_rw("is_dual", &LegPipe::is_dual)
      .def_rw("legs", &LegPipe::legs)
      .def_ro("num_legs", &LegPipe::num_legs)
      .def_rw("combine_cstyle", &LegPipe::combine_cstyle);

    // Pickle state matches HDF5: ``LegPipe(legs, is_dual, combine_cstyle)``.
    lasyten::pickle::def_reduce_ctor(pipe, [](LegPipe const& self) {
        return nb::make_tuple(nb::cast(self.legs), self.is_dual, self.combine_cstyle);
    });

    bind_axis_common<decltype(pipe), LegPipe>(pipe);

    pipe
      .def_prop_ro(
        "dual", &LegPipe::dual, doc_cpp_ref(DOC(lasyten, LegPipe), "lasyten::LegPipe::dual()"))
      .def_prop_ro("is_trivial", &LegPipe::is_trivial)
      .def("test_sanity",
           &LegPipe::test_sanity,
           doc_cpp_ref(DOC(lasyten, LegPipe), "lasyten::LegPipe::test_sanity()"))
      .def("as_Space", &LegPipe::as_space, DOC(lasyten, as_space))
      .def(
        "as_ElementarySpace",
        [](LegPipe const& self, bool is_dual) {
            return self.as_space()->as_ElementarySpace(is_dual);
        },
        nb::arg("is_dual") = false,
        DOC(lasyten, ElementarySpace, as_ElementarySpace))
      .def("__eq__",
           [](LegPipe const& self, py::object other) -> py::object {
               if (!py::isinstance<LegPipe>(other)) {
                   return py::reinterpret_borrow<py::object>(Py_NotImplemented);
               }
               return py::cast(self == py::cast<LegPipe const&>(other));
           })
      .def("__getitem__", &LegPipe::operator[], nb::arg("idx"))
      .def("__len__", [](LegPipe const& self) { return self.num_legs; })
      .def(
        "__iter__",
        [](LegPipe& self) {
            return nb::make_iterator(
              nb::type<LegPipe>(), "_LegPipeIterator", self.legs.begin(), self.legs.end());
        },
        nb::keep_alive<0, 1>())
      .def("__repr__", [](LegPipe const& self) { return self.repr(); })
      .def("repr", &LegPipe::repr, nb::arg("show_symmetry") = true, nb::arg("one_line") = false)
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<LegPipe>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<LegPipe>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    bind_elementary_space(m);
    bind_direct_sum_space(m);
    bind_tensor_product(m);
    bind_abelian_leg_pipe(m);
}

namespace {

void
bind_elementary_space(nb::module_& m)
{
    (void)m;
    auto& cls = *g_elementary_space;

    cls.def(nb::new_([](py::object symmetry_obj,
                        py::object defining_sectors,
                        py::object multiplicities,
                        bool is_dual,
                        py::object basis_perm) {
                auto symmetry = symmetry_from_python(symmetry_obj);
                return std::make_shared<ElementarySpace>(
                  symmetry,
                  sector_array_from_python(defining_sectors, *symmetry),
                  multiplicities_from_python(multiplicities),
                  is_dual,
                  perm_from_python(basis_perm));
            }),
            nb::arg("symmetry"),
            nb::arg("defining_sectors"),
            nb::arg("multiplicities").none() = nb::none(),
            nb::arg("is_dual") = false,
            nb::arg("basis_perm").none() = nb::none());

    cls.def_rw("is_dual", &ElementarySpace::is_dual)
      .def_rw("defining_sectors", &ElementarySpace::defining_sectors)
      .def_prop_ro(
        "sectors_of_basis",
        [](ElementarySpace const& self) {
            return sector_array_to_ndarray(self.sectors_of_basis());
        },
        DOC(lasyten, ElementarySpace, sectors_of_basis))
      .def_prop_ro("dual", &ElementarySpace::dual_es, DOC(lasyten, Space, dual));

    bind_axis_common<decltype(cls), ElementarySpace>(cls);

    cls.def_static(
      "from_basis",
      [](py::object symmetry_obj, py::object sectors_of_basis) {
          auto symmetry = symmetry_from_python(symmetry_obj);
          return ElementarySpace::from_basis(
            symmetry, sector_array_from_python(sectors_of_basis, *symmetry));
      },
      nb::arg("symmetry"),
      nb::arg("sectors_of_basis"),
      doc_cpp_ref(DOC(lasyten, ElementarySpace), "lasyten::ElementarySpace::from_basis()"));

    cls.def_static(
      "from_independent_symmetries",
      [](py::sequence independent_descriptions) {
          std::vector<ElementarySpace::Ptr> descriptions;
          descriptions.reserve(static_cast<std::size_t>(independent_descriptions.size()));
          for (py::handle item : independent_descriptions) {
              descriptions.push_back(py::cast_non_null<ElementarySpace::Ptr>(
                item, "elements of independent_descriptions"));
          }
          return ElementarySpace::from_independent_symmetries(descriptions);
      },
      nb::arg("independent_descriptions"),
      doc_cpp_ref(DOC(lasyten, ElementarySpace),
                  "lasyten::ElementarySpace::from_independent_symmetries()"));

    cls.def_static(
      "from_largest_common_subspace",
      [](nb::args spaces_obj_nb, bool is_dual) {
          auto spaces_obj = pyargs(spaces_obj_nb);
          std::vector<Space::Ptr> spaces;
          spaces.reserve(static_cast<std::size_t>(spaces_obj.size()));
          for (py::handle item : spaces_obj) {
              spaces.push_back(py::cast_non_null<Space::Ptr>(item, "spaces"));
          }
          return ElementarySpace::from_largest_common_subspace(spaces, is_dual);
      },
      nb::arg("spaces"),
      nb::arg("is_dual") = false,
      doc_cpp_ref(DOC(lasyten, ElementarySpace),
                  "lasyten::ElementarySpace::from_largest_common_subspace()"));

    cls.def_static(
      "from_null_space",
      [](py::object symmetry_obj, bool is_dual) {
          return ElementarySpace::from_null_space(symmetry_from_python(symmetry_obj), is_dual);
      },
      nb::arg("symmetry"),
      nb::arg("is_dual") = false,
      doc_cpp_ref(DOC(lasyten, ElementarySpace), "lasyten::ElementarySpace::from_null_space()"));

    cls.def_static(
      "from_defining_sectors",
      [](py::object symmetry_obj,
         py::object defining_sectors,
         py::object multiplicities,
         bool is_dual,
         py::object basis_perm,
         bool unique_sectors,
         bool return_sorting_perm) -> py::object {
          auto symmetry = symmetry_from_python(symmetry_obj);
          std::vector<std::size_t> sort;
          auto res = ElementarySpace::from_defining_sectors(
            symmetry,
            sector_array_from_python(defining_sectors, *symmetry),
            multiplicities_from_python(multiplicities),
            is_dual,
            perm_from_python(basis_perm),
            unique_sectors,
            return_sorting_perm ? &sort : nullptr);
          if (!return_sorting_perm) {
              return py::cast(res);
          }
          std::vector<int64> const sort_perm(sort.begin(), sort.end());
          return py::make_tuple(py::cast(res), perm_to_numpy(sort_perm));
      },
      nb::arg("symmetry"),
      nb::arg("defining_sectors"),
      nb::arg("multiplicities").none() = nb::none(),
      nb::arg("is_dual") = false,
      nb::arg("basis_perm").none() = nb::none(),
      nb::arg("unique_sectors") = false,
      nb::arg("return_sorting_perm") = false,
      doc_cpp_ref(DOC(lasyten, ElementarySpace),
                  "lasyten::ElementarySpace::from_defining_sectors()"));

    cls.def_static(
      "from_sector_decomposition",
      [](py::object symmetry_obj,
         py::object sector_decomposition,
         py::object multiplicities,
         bool is_dual,
         py::object basis_perm,
         bool unique_sectors) {
          auto symmetry = symmetry_from_python(symmetry_obj);
          return ElementarySpace::from_sector_decomposition(
            symmetry,
            sector_array_from_python(sector_decomposition, *symmetry),
            multiplicities_from_python(multiplicities),
            is_dual,
            perm_from_python(basis_perm),
            unique_sectors);
      },
      nb::arg("symmetry"),
      nb::arg("sector_decomposition"),
      nb::arg("multiplicities").none() = nb::none(),
      nb::arg("is_dual") = false,
      nb::arg("basis_perm").none() = nb::none(),
      nb::arg("unique_sectors") = false,
      doc_cpp_ref(DOC(lasyten, ElementarySpace),
                  "lasyten::ElementarySpace::from_sector_decomposition()"));

    cls.def_static(
      "from_trivial_sector",
      [](int64 dim, py::object symmetry_obj, bool is_dual, py::object basis_perm) {
          return ElementarySpace::from_trivial_sector(dim,
                                                      optional_symmetry_from_python(symmetry_obj),
                                                      is_dual,
                                                      perm_from_python(basis_perm));
      },
      nb::arg("dim") = 1,
      nb::arg("symmetry").none() = nb::none(),
      nb::arg("is_dual") = false,
      nb::arg("basis_perm").none() = nb::none(),
      doc_cpp_ref(DOC(lasyten, ElementarySpace),
                  "lasyten::ElementarySpace::from_trivial_sector()"));

    cls
      .def("test_sanity",
           &ElementarySpace::test_sanity,
           doc_cpp_ref(DOC(lasyten, ElementarySpace), "lasyten::ElementarySpace::test_sanity()"))
      .def("__repr__", [](ElementarySpace const& self) { return self.repr(); })
      .def("repr",
           &ElementarySpace::repr,
           nb::arg("show_symmetry") = true,
           nb::arg("one_line") = false)
      .def("__eq__",
           [](ElementarySpace const& self, py::object other) -> py::object {
               if (!py::isinstance<ElementarySpace>(other)) {
                   return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
               }
               return py::cast(self.equals_es(py::cast<ElementarySpace const&>(other)));
           })
      .def("as_Space", &ElementarySpace::as_Space, DOC(lasyten, Space, as_Space))
      .def("as_ElementarySpace",
           &ElementarySpace::as_ElementarySpace,
           nb::arg("is_dual") = false,
           DOC(lasyten, ElementarySpace, as_ElementarySpace))
      .def("as_ket_space",
           &ElementarySpace::as_ket_space,
           DOC(lasyten, ElementarySpace, as_ket_space))
      .def("as_bra_space",
           &ElementarySpace::as_bra_space,
           DOC(lasyten, ElementarySpace, as_bra_space))
      .def(
        "change_symmetry",
        [](ElementarySpace& self,
           py::object symmetry_obj,
           py::function sector_map,
           bool injective) {
            return self.change_symmetry(
              symmetry_from_python(symmetry_obj), sector_map_from_python(sector_map), injective);
        },
        nb::arg("symmetry"),
        nb::arg("sector_map"),
        nb::arg("injective") = false,
        DOC(lasyten, ElementarySpace, change_symmetry))
      .def(
        "direct_sum",
        [](ElementarySpace const& self,
           nb::args others_obj_nb,
           std::optional<LegLabelList> summand_labels) {
            auto others_obj = pyargs(others_obj_nb);
            std::vector<ElementarySpace::Ptr> others;
            others.reserve(static_cast<std::size_t>(others_obj.size()));
            for (py::handle item : others_obj) {
                others.push_back(py::cast_non_null<ElementarySpace::Ptr>(item, "others"));
            }
            return self.direct_sum(others, std::move(summand_labels));
        },
        nb::arg("others"),
        nb::kw_only(),
        nb::arg("summand_labels").none() = nb::none(),
        doc_cpp_ref(DOC(lasyten, ElementarySpace), "lasyten::ElementarySpace::direct_sum()"))
      .def(
        "drop_symmetry",
        [](ElementarySpace& self, py::object which) {
            return self.drop_symmetry(drop_which_from_python(which));
        },
        nb::arg("which") = "all",
        DOC(lasyten, Space, drop_symmetry))
      .def(
        "parse_index",
        [](ElementarySpace const& self, int64 idx) {
            auto const [sector_idx, multiplicity_idx] = self.parse_index(idx);
            return py::make_tuple(sector_idx, multiplicity_idx);
        },
        nb::arg("idx"),
        doc_cpp_ref(DOC(lasyten, ElementarySpace), "lasyten::ElementarySpace::parse_index()"))
      .def("idx_to_sector", &ElementarySpace::idx_to_sector, nb::arg("idx"))
      .def(
        "take_slice",
        [](ElementarySpace& self, py::object blockmask) {
            return self.take_slice(blockmask_from_python(blockmask));
        },
        nb::arg("blockmask"),
        DOC(lasyten, ElementarySpace, take_slice))
      .def("with_opposite_duality",
           &ElementarySpace::with_opposite_duality,
           DOC(lasyten, ElementarySpace, with_opposite_duality))
      .def("with_is_dual",
           &ElementarySpace::with_is_dual,
           nb::arg("is_dual"),
           DOC(lasyten, ElementarySpace, with_is_dual))
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<ElementarySpace>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<ElementarySpace>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"))
      .def_prop_ro("is_direct_sum_space",
                   &ElementarySpace::is_direct_sum_space,
                   DOC(lasyten, ElementarySpace, is_direct_sum_space));

    // Pickle state: the fields ``ElementarySpace.from_hdf5`` consumes (HDF5 sub-group names);
    // everything else (sector_decomposition, slices, ...) is derived by the constructor.
    lasyten::pickle::def_state_pickle(
      cls,
      [](ElementarySpace const& self) {
          nb::dict d;
          d["symmetry"] = self.symmetry;
          d["defining_sectors"] = self.defining_sectors;
          d["multiplicities"] = perm_to_numpy(self.multiplicities);
          d["_basis_perm"] =
            self.has_custom_basis_perm() ? perm_to_numpy(self.basis_perm()) : nb::none();
          d["is_dual"] = self.is_dual;
          return d;
      },
      [](nb::dict state) {
          return std::make_shared<ElementarySpace>(
            lasyten::pickle::get_item<Symmetry::Ptr>(state, "symmetry"),
            lasyten::pickle::get_item<SectorArray>(state, "defining_sectors"),
            multiplicities_from_python(state["multiplicities"]),
            lasyten::pickle::get_item<bool>(state, "is_dual"),
            perm_from_python(state["_basis_perm"]));
      });
}

void
bind_direct_sum_space(nb::module_& m)
{
    nb::class_<DirectSumSpace, ElementarySpace> cls(
      m, "DirectSumSpace", DOC(lasyten, DirectSumSpace));

    cls.def(
      nb::new_(
        [](py::sequence spaces_obj, bool is_dual, std::optional<LegLabelList> summand_labels) {
            std::vector<ElementarySpace::Ptr> spaces;
            spaces.reserve(static_cast<std::size_t>(spaces_obj.size()));
            for (py::handle item : spaces_obj) {
                spaces.push_back(
                  py::cast_non_null<ElementarySpace::Ptr>(item, "elements of spaces"));
            }
            return std::make_shared<DirectSumSpace>(
              std::move(spaces), is_dual, std::move(summand_labels));
        }),
      nb::arg("spaces"),
      nb::arg("is_dual") = false,
      nb::arg("summand_labels").none() = nb::none());

    cls.def_prop_ro("spaces", [](DirectSumSpace const& self) { return self.spaces; });
    cls.def_prop_ro("summand_labels",
                    [](DirectSumSpace const& self) { return self.summand_labels; });

    cls
      .def_static(
        "from_spaces",
        [](py::sequence spaces_obj, bool is_dual, std::optional<LegLabelList> summand_labels) {
            std::vector<ElementarySpace::Ptr> spaces;
            spaces.reserve(static_cast<std::size_t>(spaces_obj.size()));
            for (py::handle item : spaces_obj) {
                spaces.push_back(
                  py::cast_non_null<ElementarySpace::Ptr>(item, "elements of spaces"));
            }
            return DirectSumSpace::from_spaces(
              std::move(spaces), is_dual, std::move(summand_labels));
        },
        nb::arg("spaces"),
        nb::arg("is_dual") = false,
        nb::arg("summand_labels").none() = nb::none(),
        DOC(lasyten, DirectSumSpace, from_spaces))
      .def(
        "get_summand_idx",
        [](DirectSumSpace const& self, py::object which) {
            if (py::isinstance<py::str>(which)) {
                return self.get_summand_idx(py::cast<std::string>(which));
            }
            return self.get_summand_idx(py::cast<int64>(which));
        },
        nb::arg("which"),
        DOC(lasyten, DirectSumSpace, get_summand_idx))
      .def("has_summand_label",
           &DirectSumSpace::has_summand_label,
           nb::arg("label"),
           DOC(lasyten, DirectSumSpace, has_summand_label))
      .def("mult_slices", &DirectSumSpace::mult_slices, DOC(lasyten, DirectSumSpace, mult_slices))
      .def("as_plain_ElementarySpace",
           &DirectSumSpace::as_plain_ElementarySpace,
           DOC(lasyten, DirectSumSpace, as_plain_ElementarySpace))
      .def("test_sanity", &DirectSumSpace::test_sanity)
      .def("as_Space", &DirectSumSpace::as_Space, DOC(lasyten, Space, as_Space))
      .def("as_ElementarySpace",
           &DirectSumSpace::as_ElementarySpace,
           nb::arg("is_dual") = false,
           DOC(lasyten, ElementarySpace, as_ElementarySpace))
      .def_prop_ro("dual", &DirectSumSpace::dual_dss, DOC(lasyten, Space, dual))
      .def(
        "change_symmetry",
        [](
          DirectSumSpace& self, py::object symmetry_obj, py::function sector_map, bool injective) {
            return self.change_symmetry(
              symmetry_from_python(symmetry_obj), sector_map_from_python(sector_map), injective);
        },
        nb::arg("symmetry"),
        nb::arg("sector_map"),
        nb::arg("injective") = false,
        DOC(lasyten, ElementarySpace, change_symmetry))
      .def(
        "drop_symmetry",
        [](DirectSumSpace& self, py::object which) {
            return self.drop_symmetry(drop_which_from_python(which));
        },
        nb::arg("which") = "all",
        DOC(lasyten, Space, drop_symmetry))
      .def(
        "take_slice",
        [](DirectSumSpace& self, py::object blockmask) {
            return self.take_slice(blockmask_from_python(blockmask));
        },
        nb::arg("blockmask"),
        DOC(lasyten, ElementarySpace, take_slice))
      .def("with_opposite_duality",
           &DirectSumSpace::with_opposite_duality,
           DOC(lasyten, ElementarySpace, with_opposite_duality))
      .def("projection_onto_summands",
           &DirectSumSpace::projection_onto_summands,
           nb::arg("keep_indices").none() = nb::none(),
           nb::arg("discard_indices").none() = nb::none(),
           nb::arg("backend").none() = nullptr,
           nb::arg("labels").none() = nb::none(),
           nb::arg("device").none() = nb::none(),
           DOC(lasyten, DirectSumSpace, projection_onto_summands))
      .def("inclusion_of_summands",
           &DirectSumSpace::inclusion_of_summands,
           nb::arg("keep_indices").none() = nb::none(),
           nb::arg("discard_indices").none() = nb::none(),
           nb::arg("backend").none() = nullptr,
           nb::arg("labels").none() = nb::none(),
           nb::arg("device").none() = nb::none(),
           DOC(lasyten, DirectSumSpace, inclusion_of_summands))
      .def(
        "projection_onto_summand",
        [](DirectSumSpace const& self,
           py::object which,
           std::shared_ptr<TensorBackend> backend,
           std::optional<LegLabelList> labels,
           std::optional<std::string> device) {
            DirectSumSpace::SummandRef ref =
              py::isinstance<py::str>(which)
                ? DirectSumSpace::SummandRef{ py::cast<std::string>(which) }
                : DirectSumSpace::SummandRef{ py::cast<int64>(which) };
            return self.projection_onto_summand(
              std::move(ref), std::move(backend), std::move(labels), std::move(device));
        },
        nb::arg("which"),
        nb::arg("backend").none() = nullptr,
        nb::arg("labels").none() = nb::none(),
        nb::arg("device").none() = nb::none(),
        DOC(lasyten, DirectSumSpace, projection_onto_summand))
      .def(
        "inclusion_of_summand",
        [](DirectSumSpace const& self,
           py::object which,
           std::shared_ptr<TensorBackend> backend,
           std::optional<LegLabelList> labels,
           std::optional<std::string> device) {
            DirectSumSpace::SummandRef ref =
              py::isinstance<py::str>(which)
                ? DirectSumSpace::SummandRef{ py::cast<std::string>(which) }
                : DirectSumSpace::SummandRef{ py::cast<int64>(which) };
            return self.inclusion_of_summand(
              std::move(ref), std::move(backend), std::move(labels), std::move(device));
        },
        nb::arg("which"),
        nb::arg("backend").none() = nullptr,
        nb::arg("labels").none() = nb::none(),
        nb::arg("device").none() = nb::none(),
        DOC(lasyten, DirectSumSpace, inclusion_of_summand))
      .def("unit_vector_of_summand",
           &DirectSumSpace::unit_vector_of_summand,
           nb::arg("i"),
           nb::arg("backend").none() = nullptr,
           nb::arg("labels").none() = nb::none(),
           nb::arg("dtype").none() = nb::none(),
           nb::arg("device").none() = nb::none(),
           DOC(lasyten, DirectSumSpace, unit_vector_of_summand))
      .def(
        "set_basis_perm",
        [](DirectSumSpace& self, nb::args, nb::kwargs) { self.set_basis_perm(std::nullopt); },
        "Can not set basis_perm for DirectSumSpace.")
      .def(
        "set_inverse_basis_perm",
        [](DirectSumSpace& self, nb::args, nb::kwargs) {
            self.set_inverse_basis_perm(std::nullopt);
        },
        "Can not set basis_perm for DirectSumSpace.")
      .def("__repr__", [](DirectSumSpace const& self) { return self.repr(); })
      .def("repr",
           &DirectSumSpace::repr,
           nb::arg("show_symmetry") = true,
           nb::arg("one_line") = false)
      .def("__eq__",
           [](DirectSumSpace const& self, py::object other) -> py::object {
               if (!py::isinstance<DirectSumSpace>(other)) {
                   return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
               }
               return py::cast(self.equals_dss(py::cast<DirectSumSpace const&>(other)));
           })
      // Shadow unsupported ElementarySpace factories
      .def_static("from_basis",
                  [](nb::args, nb::kwargs) -> py::object {
                      throw py::type_error("from_basis is not supported for DirectSumSpace");
                  })
      .def_static("from_null_space",
                  [](nb::args, nb::kwargs) -> py::object {
                      throw py::type_error("from_null_space is not supported for DirectSumSpace");
                  })
      .def_static("from_defining_sectors",
                  [](nb::args, nb::kwargs) -> py::object {
                      throw py::type_error(
                        "from_defining_sectors is not supported for DirectSumSpace");
                  })
      .def_static("from_trivial_sector",
                  [](nb::args, nb::kwargs) -> py::object {
                      throw py::type_error(
                        "from_trivial_sector is not supported for DirectSumSpace");
                  })
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<DirectSumSpace>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<DirectSumSpace>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    // Pickle state: ``spaces``, ``is_dual``, ``summand_labels`` (as consumed by ``from_hdf5``).
    lasyten::pickle::def_state_pickle(
      cls,
      [](DirectSumSpace const& self) {
          nb::dict d;
          d["spaces"] = nb::cast(self.spaces);
          d["is_dual"] = self.is_dual;
          d["summand_labels"] = nb::cast(self.summand_labels);
          return d;
      },
      [](nb::dict state) {
          return DirectSumSpace::from_spaces(
            lasyten::pickle::get_item<std::vector<ElementarySpace::Ptr>>(state, "spaces"),
            lasyten::pickle::get_item<bool>(state, "is_dual"),
            lasyten::pickle::get_item<std::optional<LegLabelList>>(state, "summand_labels"));
      });
}

void
bind_tensor_product(nb::module_& m)
{
    nb::class_<TensorProduct, Space> cls(m, "TensorProduct", DOC(lasyten, TensorProduct));

    cls.def(nb::new_([](py::iterable factors_obj,
                        py::object symmetry_obj,
                        py::object sector_decomposition,
                        py::object multiplicities) {
                auto factors = legs_from_python(factors_obj);
                auto symmetry = optional_symmetry_from_python(symmetry_obj);
                std::optional<SectorArray> sectors;
                if (!sector_decomposition.is_none()) {
                    auto sym = symmetry;
                    if (!sym && !factors.empty()) {
                        sym = factors.front().symmetry();
                    }
                    if (!sym) {
                        throw py::value_error("If spaces is empty, the symmetry arg is required.");
                    }
                    sectors = sector_array_from_python(sector_decomposition, *sym);
                }
                return std::make_shared<TensorProduct>(std::move(factors),
                                                       std::move(symmetry),
                                                       std::move(sectors),
                                                       multiplicities_from_python(multiplicities));
            }),
            nb::arg("factors"),
            nb::arg("symmetry").none() = nb::none(),
            nb::arg("_sector_decomposition").none() = nb::none(),
            nb::arg("_multiplicities").none() = nb::none());

    cls
      .def_prop_rw(
        "factors",
        [](TensorProduct const& self) { return self.factors; },
        [](TensorProduct& self, py::iterable factors_obj) {
            self.factors = legs_from_python(factors_obj);
            self.num_factors = static_cast<int64>(self.factors.size());
        })
      .def_ro("num_factors", &TensorProduct::num_factors)
      .def_prop_ro("dual", &TensorProduct::dual_space, DOC(lasyten, Space, dual))
      .def_prop_ro("has_pipes",
                   &TensorProduct::has_pipes,
                   doc_cpp_ref(DOC(lasyten, TensorProduct), "lasyten::TensorProduct::has_pipes()"))
      .def_prop_ro("flat_legs", &TensorProduct::flat_legs, DOC(lasyten, LegPipe, flat_legs))
      .def_prop_ro("flat_spaces", &TensorProduct::flat_spaces, DOC(lasyten, LegPipe, flat_spaces))
      .def_prop_ro(
        "num_flat_legs", &TensorProduct::num_flat_legs, DOC(lasyten, LegPipe, num_flat_legs));

    cls.def_static(
      "from_partial_products",
      [](nb::args factors_obj_nb) {
          auto factors_obj = pyargs(factors_obj_nb);
          std::vector<TensorProduct::Ptr> factors;
          factors.reserve(static_cast<std::size_t>(factors_obj.size()));
          for (py::handle item : factors_obj) {
              factors.push_back(py::cast_non_null<TensorProduct::Ptr>(item, "factors"));
          }
          return TensorProduct::from_partial_products(factors);
      },
      doc_cpp_ref(DOC(lasyten, TensorProduct), "lasyten::TensorProduct::from_partial_products()"));

    cls
      .def("test_sanity",
           &TensorProduct::test_sanity,
           doc_cpp_ref(DOC(lasyten, TensorProduct), "lasyten::TensorProduct::test_sanity()"))
      .def(
        "block_size",
        [](TensorProduct const& self, py::object coupled) {
            if (py::isinstance<py::int_>(coupled)) {
                return self.block_size(py::cast<int64>(coupled));
            }
            return self.block_size(sector_from_python(coupled));
        },
        nb::arg("coupled"),
        DOC(lasyten, TensorProduct, block_size))
      .def(
        "change_symmetry",
        [](TensorProduct& self, py::object symmetry_obj, py::function sector_map, bool injective) {
            return self.change_symmetry(
              symmetry_from_python(symmetry_obj), sector_map_from_python(sector_map), injective);
        },
        nb::arg("symmetry"),
        nb::arg("sector_map"),
        nb::arg("injective") = false,
        DOC(lasyten, TensorProduct, change_symmetry))
      .def(
        "drop_symmetry",
        [](TensorProduct& self, py::object which) {
            return self.drop_symmetry(drop_which_from_python(which));
        },
        nb::arg("which") = "all",
        DOC(lasyten, Space, drop_symmetry))
      .def("flat_legs_nesting",
           &TensorProduct::flat_legs_nesting,
           doc_cpp_ref(DOC(lasyten, TensorProduct), "lasyten::TensorProduct::flat_legs_nesting()"))
      .def("flat_leg_idcs",
           &TensorProduct::flat_leg_idcs,
           nb::arg("i"),
           doc_cpp_ref(DOC(lasyten, TensorProduct), "lasyten::TensorProduct::flat_leg_idcs()"))
      .def(
        "forest_block_size",
        [](TensorProduct const& self, py::object uncoupled, py::object coupled) {
            return self.forest_block_size(sector_array_from_python(uncoupled, *self.symmetry),
                                          sector_from_python(coupled));
        },
        nb::arg("uncoupled"),
        nb::arg("coupled"),
        DOC(lasyten, TensorProduct, forest_block_size))
      .def(
        "forest_block_slice",
        [](TensorProduct const& self, py::object uncoupled, py::object coupled) {
            return index_slice_to_python(self.forest_block_slice(
              sector_array_from_python(uncoupled, *self.symmetry), sector_from_python(coupled)));
        },
        nb::arg("uncoupled"),
        nb::arg("coupled"),
        DOC(lasyten, TensorProduct, forest_block_slice))
      .def("insert_multiply",
           &TensorProduct::insert_multiply,
           nb::arg("other"),
           nb::arg("pos"),
           DOC(lasyten, TensorProduct, insert_multiply))
      .def(
        "iter_tree_blocks",
        [](TensorProduct const& self, py::object coupled) {
            py::list out;
            for (auto const& item :
                 self.iter_tree_blocks(sector_array_from_python(coupled, *self.symmetry))) {
                out.append(py::make_tuple(py::cast(item.tree),
                                          index_slice_to_python(item.slice),
                                          perm_to_numpy(item.multiplicities),
                                          item.coupled_idx));
            }
            return py::iter(out);
        },
        nb::arg("coupled"),
        DOC(lasyten, TensorProduct, iter_tree_blocks))
      .def(
        "iter_forest_blocks",
        [](TensorProduct const& self, py::object coupled) {
            py::list out;
            for (auto const& item :
                 self.iter_forest_blocks(sector_array_from_python(coupled, *self.symmetry))) {
                out.append(py::make_tuple(
                  py::cast(item.uncoupled), index_slice_to_python(item.slice), item.coupled_idx));
            }
            return py::iter(out);
        },
        nb::arg("coupled"),
        DOC(lasyten, TensorProduct, iter_forest_blocks))
      .def(
        "iter_uncoupled",
        [](TensorProduct const& self, bool yield_slices) {
            py::list out;
            for (auto const& item : self.iter_uncoupled(yield_slices)) {
                auto uncoupled = py::cast(item.uncoupled);
                auto mults = perm_to_numpy(item.multiplicities);
                if (!yield_slices) {
                    out.append(py::make_tuple(std::move(uncoupled), std::move(mults)));
                    continue;
                }
                py::list slices;
                for (auto const& slc : *item.slices) {
                    slices.append(index_slice_to_python(slc));
                }
                out.append(
                  py::make_tuple(std::move(uncoupled), std::move(mults), std::move(slices)));
            }
            return py::iter(out);
        },
        nb::arg("yield_slices") = false,
        DOC(lasyten, TensorProduct, iter_uncoupled))
      .def("left_multiply",
           &TensorProduct::left_multiply,
           nb::arg("other"),
           DOC(lasyten, TensorProduct, left_multiply))
      .def("permuted",
           &TensorProduct::permuted,
           nb::arg("perm"),
           DOC(lasyten, TensorProduct, permuted))
      .def("right_multiply",
           &TensorProduct::right_multiply,
           nb::arg("other"),
           DOC(lasyten, TensorProduct, right_multiply))
      .def(
        "tree_block_size",
        [](TensorProduct const& self, py::object uncoupled) {
            return self.tree_block_size(sector_array_from_python(uncoupled, *self.symmetry));
        },
        nb::arg("uncoupled"),
        DOC(lasyten, TensorProduct, tree_block_size))
      .def(
        "tree_block_slice",
        [](TensorProduct const& self, FusionTree const& tree) {
            return index_slice_to_python(self.tree_block_slice(tree));
        },
        nb::arg("tree"),
        DOC(lasyten, TensorProduct, tree_block_slice));

    cls
      .def("__eq__",
           [](TensorProduct const& self, py::object other) -> py::object {
               if (!py::isinstance<TensorProduct>(other)) {
                   return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
               }
               return py::cast(self.operator==(
                 static_cast<Space const&>(py::cast<TensorProduct const&>(other))));
           })
      .def("__getitem__",
           [](TensorProduct const& self, py::object idx) -> py::object {
               if (py::isinstance<py::slice>(idx)) {
                   py::list all;
                   for (auto const& f : self.factors) {
                       all.append(py::cast(f));
                   }
                   return all.attr("__getitem__")(idx);
               }
               return py::cast(self[py::cast<int64>(idx)]);
           })
      .def("__len__", [](TensorProduct const& self) { return self.num_factors; })
      .def(
        "__iter__",
        [](TensorProduct const& self) {
            return nb::make_iterator(nb::type<TensorProduct>(),
                                     "_TensorProductIterator",
                                     self.factors.begin(),
                                     self.factors.end());
        },
        nb::keep_alive<0, 1>())
      .def("__repr__",
           &TensorProduct::repr,
           nb::arg("show_symmetry") = true,
           nb::arg("one_line") = false)
      .def(
        "repr", &TensorProduct::repr, nb::arg("show_symmetry") = true, nb::arg("one_line") = false)
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<TensorProduct>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<TensorProduct>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    // Pickle state: ``factors``, ``symmetry``, ``sector_decomposition``, ``multiplicities``
    // (as consumed by ``from_hdf5``).
    lasyten::pickle::def_state_pickle(
      cls,
      [](TensorProduct const& self) {
          nb::dict d;
          d["factors"] = nb::cast(self.factors);
          d["symmetry"] = self.symmetry;
          d["sector_decomposition"] = self.sector_decomposition;
          d["multiplicities"] = perm_to_numpy(self.multiplicities);
          return d;
      },
      [](nb::dict state) {
          return std::make_shared<TensorProduct>(
            lasyten::pickle::get_item<std::vector<Leg::Ptr>>(state, "factors"),
            lasyten::pickle::get_item<Symmetry::Ptr>(state, "symmetry"),
            lasyten::pickle::get_item<SectorArray>(state, "sector_decomposition"),
            multiplicities_from_python(state["multiplicities"]));
      });
}

void
bind_abelian_leg_pipe(nb::module_& m)
{
    // Single base: AbelianLegPipe inherits only ElementarySpace (pipe fields composed).
    nb::class_<AbelianLegPipe, ElementarySpace> cls(
      m, "AbelianLegPipe", DOC(lasyten, AbelianLegPipe));

    cls.def(nb::new_([](py::sequence legs_obj, bool is_dual, bool combine_cstyle) {
                std::vector<ElementarySpace::Ptr> legs;
                legs.reserve(static_cast<std::size_t>(legs_obj.size()));
                for (py::handle item : legs_obj) {
                    legs.push_back(
                      py::cast_non_null<ElementarySpace::Ptr>(item, "elements of legs"));
                }
                return std::make_shared<AbelianLegPipe>(std::move(legs), is_dual, combine_cstyle);
            }),
            nb::arg("legs"),
            nb::arg("is_dual") = false,
            nb::arg("combine_cstyle") = true);

    cls.def_rw("legs", &AbelianLegPipe::legs)
      .def_ro("num_legs", &AbelianLegPipe::num_legs)
      .def_rw("combine_cstyle", &AbelianLegPipe::combine_cstyle);

    cls
      .def_prop_ro("sector_strides",
                   [](AbelianLegPipe const& self) { return perm_to_numpy(self.sector_strides); })
      .def_prop_ro(
        "fusion_outcomes_sort",
        [](AbelianLegPipe const& self) { return perm_to_numpy(self.fusion_outcomes_sort); })
      .def_prop_ro(
        "block_ind_map_slices",
        [](AbelianLegPipe const& self) { return perm_to_numpy(self.block_ind_map_slices); })
      .def_prop_ro("block_ind_map", [](AbelianLegPipe const& self) { return self.block_ind_map; });

    cls.def_prop_ro("dual", &AbelianLegPipe::dual_pipe, DOC(lasyten, Space, dual))
      .def_prop_ro(
        "is_trivial",
        &AbelianLegPipe::is_trivial,
        doc_cpp_ref(DOC(lasyten, AbelianLegPipe), "lasyten::AbelianLegPipe::is_trivial()"))
      .def_prop_ro("flat_spaces", &AbelianLegPipe::flat_spaces, DOC(lasyten, LegPipe, flat_spaces))
      .def_prop_ro("flat_legs", &AbelianLegPipe::flat_legs, DOC(lasyten, LegPipe, flat_legs))
      .def_prop_ro("is_pipe", &AbelianLegPipe::is_pipe, DOC(lasyten, LegPipe, is_pipe))
      .def_prop_ro(
        "ascii_arrow", &AbelianLegPipe::ascii_arrow, DOC(lasyten, AbelianLegPipe, ascii_arrow));

    cls.def_static(
      "from_independent_symmetries",
      [](py::sequence independent_descriptions) {
          std::vector<AbelianLegPipe::Ptr> descriptions;
          descriptions.reserve(static_cast<std::size_t>(independent_descriptions.size()));
          for (py::handle item : independent_descriptions) {
              descriptions.push_back(py::cast_non_null<AbelianLegPipe::Ptr>(
                item, "elements of independent_descriptions"));
          }
          return AbelianLegPipe::from_independent_symmetries(descriptions);
      },
      nb::arg("independent_descriptions"),
      DOC(lasyten, AbelianLegPipe, from_independent_symmetries));

    // The unsupported ElementarySpace factories. They are bound (and raise) such that they
    // shadow the inherited versions, which would silently return a plain ElementarySpace.
    cls
      .def_static(
        "from_basis",
        [](nb::args, nb::kwargs) -> py::object {
            throw py::type_error("from_basis is not supported for AbelianLegPipe");
        },
        DOC(lasyten, AbelianLegPipe, from_basis))
      .def_static(
        "from_null_space",
        [](nb::args, nb::kwargs) -> py::object {
            throw py::type_error("from_null_space is not supported for AbelianLegPipe");
        },
        DOC(lasyten, AbelianLegPipe, from_null_space))
      .def_static(
        "from_defining_sectors",
        [](nb::args, nb::kwargs) -> py::object {
            throw py::type_error("from_defining_sectors is not supported for AbelianLegPipe");
        },
        doc_cpp_ref(DOC(lasyten, AbelianLegPipe),
                    "lasyten::AbelianLegPipe::from_defining_sectors()"))
      .def_static(
        "from_trivial_sector",
        [](nb::args, nb::kwargs) -> py::object {
            throw py::type_error("from_trivial_sector is not supported for AbelianLegPipe");
        },
        DOC(lasyten, AbelianLegPipe, from_trivial_sector));

    cls.def("test_sanity", &AbelianLegPipe::test_sanity, DOC(lasyten, AbelianLegPipe, test_sanity))
      .def("as_Space", &AbelianLegPipe::as_Space, DOC(lasyten, Space, as_Space))
      .def("as_ElementarySpace",
           &AbelianLegPipe::as_ElementarySpace,
           nb::arg("is_dual") = false,
           DOC(lasyten, ElementarySpace, as_ElementarySpace))
      .def(
        "change_symmetry",
        [](
          AbelianLegPipe& self, py::object symmetry_obj, py::function sector_map, bool injective) {
            return self.change_symmetry(
              symmetry_from_python(symmetry_obj), sector_map_from_python(sector_map), injective);
        },
        nb::arg("symmetry"),
        nb::arg("sector_map"),
        nb::arg("injective") = false,
        DOC(lasyten, Space, change_symmetry))
      .def(
        "drop_symmetry",
        [](AbelianLegPipe& self, py::object which) {
            return self.drop_symmetry(drop_which_from_python(which));
        },
        nb::arg("which") = "all",
        DOC(lasyten, Space, drop_symmetry))
      .def(
        "set_basis_perm",
        [](AbelianLegPipe& self, nb::args, nb::kwargs) { self.set_basis_perm(std::nullopt); },
        doc_cpp_ref(DOC(lasyten, AbelianLegPipe), "lasyten::AbelianLegPipe::set_basis_perm()"))
      .def(
        "take_slice",
        [](AbelianLegPipe& self, py::object blockmask) {
            return self.take_slice(blockmask_from_python(blockmask));
        },
        nb::arg("blockmask"),
        doc_cpp_ref(DOC(lasyten, AbelianLegPipe), "lasyten::AbelianLegPipe::take_slice()"))
      .def("with_opposite_duality",
           &AbelianLegPipe::with_opposite_duality,
           doc_cpp_ref(DOC(lasyten, AbelianLegPipe),
                       "lasyten::AbelianLegPipe::with_opposite_duality()"))
      .def(
        "_get_fusion_outcomes_perm",
        [](AbelianLegPipe const& self, py::object multiplicities) {
            auto mults = multiplicities_from_python(multiplicities);
            return perm_to_numpy(
              self.get_fusion_outcomes_perm(mults.value_or(self.multiplicities)));
        },
        nb::arg("multiplicities"),
        doc_cpp_ref(DOC(lasyten, AbelianLegPipe),
                    "lasyten::AbelianLegPipe::_get_fusion_outcomes_perm()"))
      .def("__getitem__", &AbelianLegPipe::operator[], nb::arg("idx"))
      .def("__len__", [](AbelianLegPipe const& self) { return self.num_legs; })
      .def(
        "__iter__",
        [](AbelianLegPipe& self) {
            return nb::make_iterator(nb::type<AbelianLegPipe>(),
                                     "_AbelianLegPipeIterator",
                                     self.legs.begin(),
                                     self.legs.end());
        },
        nb::keep_alive<0, 1>())
      .def("__eq__",
           [](AbelianLegPipe const& self, py::object other) -> py::object {
               if (!py::isinstance<AbelianLegPipe>(other)) {
                   return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
               }
               return py::cast(self.operator==(
                 static_cast<Space const&>(py::cast<AbelianLegPipe const&>(other))));
           })
      .def("__repr__", [](AbelianLegPipe const& self) { return self.repr(); })
      .def("repr",
           &AbelianLegPipe::repr,
           nb::arg("show_symmetry") = true,
           nb::arg("one_line") = false)
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<AbelianLegPipe>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<AbelianLegPipe>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    // Pickle state: ``legs``, ``is_dual``, ``combine_cstyle`` (as consumed by ``from_hdf5``).
    lasyten::pickle::def_state_pickle(
      cls,
      [](AbelianLegPipe const& self) {
          nb::dict d;
          d["legs"] = nb::cast(self.legs);
          d["is_dual"] = self.is_dual;
          d["combine_cstyle"] = self.combine_cstyle;
          return d;
      },
      [](nb::dict state) {
          return std::make_shared<AbelianLegPipe>(
            lasyten::pickle::get_item<std::vector<ElementarySpace::Ptr>>(state, "legs"),
            lasyten::pickle::get_item<bool>(state, "is_dual"),
            lasyten::pickle::get_item<bool>(state, "combine_cstyle"));
      });

    m.def("swap_gate",
          &swap_gate,
          nb::arg("V"),
          nb::arg("W"),
          doc_cpp_ref(DOC(lasyten, AbelianLegPipe), "lasyten::swap_gate()"));

    m.def("twist_gate",
          &twist_gate,
          nb::arg("V"),
          doc_cpp_ref(DOC(lasyten, AbelianLegPipe), "lasyten::twist_gate()"));

    m.def("_twist_gate_diag", &twist_gate_diag, nb::arg("V"));

    m.def(
      "_flat_leg_permutation",
      [](py::sequence legs_obj) {
          std::vector<Leg::Ptr> legs;
          legs.reserve(static_cast<std::size_t>(legs_obj.size()));
          for (py::handle item : legs_obj) {
              legs.push_back(py::cast_non_null<Leg::Ptr>(item, "elements of legs"));
          }
          return flat_leg_permutation(legs);
      },
      nb::arg("legs"),
      doc_cpp_ref(DOC(lasyten, AbelianLegPipe),
                  "lasyten::AbelianLegPipe::_flat_leg_permutation()"));

    m.def(
      "_unique_sorted_sectors",
      [](SectorArray const& sectors, py::object multiplicities) {
          auto mults = multiplicities_from_python(multiplicities).value_or(std::vector<int64>{});
          auto [s, m, perm] = unique_sorted_sectors(sectors, mults);
          return py::make_tuple(
            s, perm_to_numpy(m), perm_to_numpy(std::vector<int64>(perm.begin(), perm.end())));
      },
      nb::arg("unsorted_sectors"),
      nb::arg("unsorted_multiplicities"),
      doc_cpp_ref(DOC(lasyten, AbelianLegPipe),
                  "lasyten::AbelianLegPipe::_unique_sorted_sectors()"));

    m.def(
      "_sort_sectors",
      [](SectorArray const& sectors, py::object multiplicities) {
          auto mults = multiplicities_from_python(multiplicities).value_or(std::vector<int64>{});
          auto [s, m, perm] = sort_sectors_public(sectors, mults);
          return py::make_tuple(
            s, perm_to_numpy(m), perm_to_numpy(std::vector<int64>(perm.begin(), perm.end())));
      },
      nb::arg("sectors"),
      nb::arg("multiplicities"));

    m.def(
      "_parse_inputs_drop_symmetry",
      [](py::object which, py::object symmetry_obj) -> py::tuple {
          auto symmetry = symmetry_from_python(symmetry_obj);
          auto [factors, remaining] =
            parse_inputs_drop_symmetry_public(drop_which_from_python(which), std::move(symmetry));
          if (!factors) {
              return py::make_tuple(py::str("all"), remaining);
          }
          return py::make_tuple(perm_to_numpy(*factors), remaining);
      },
      nb::arg("which"),
      nb::arg("symmetry"),
      doc_cpp_ref(DOC(lasyten, AbelianLegPipe),
                  "lasyten::AbelianLegPipe::_parse_inputs_drop_symmetry()"));
}

} // namespace

} // namespace lasyten
