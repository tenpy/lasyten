#include "../doc_plus.h"
#include "docstrings/symmetries/spaces.h"
#include "py_lasyten_pybind11.h"

#include "backends/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <lasyten/backends/tensor_backend.h>
#include <lasyten/symmetries/sector_numpy.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/symmetry.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/symmetric_tensor.h>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "tools/hdf5_bind.h"
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace lasyten {

namespace {

Symmetry::Ptr
symmetry_from_python(py::object symmetry_obj)
{
    if (py::isinstance<Symmetry>(symmetry_obj)) {
        return symmetry_obj.cast<Symmetry::Ptr>();
    }
    if (py::isinstance<SymmetryFactor>(symmetry_obj)) {
        auto ptr = symmetry_obj.cast<SymmetryFactor::Ptr>();
        return std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ ptr });
    }
    return symmetry_obj.attr("as_Symmetry")().cast<Symmetry::Ptr>();
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

py::array_t<int64>
perm_to_numpy(std::vector<int64> const& perm)
{
    py::array_t<int64> arr(static_cast<py::ssize_t>(perm.size()));
    auto buf = arr.mutable_unchecked<1>();
    for (std::size_t i = 0; i < perm.size(); ++i) {
        buf(static_cast<py::ssize_t>(i)) = perm[i];
    }
    return arr;
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
    return obj.cast<std::string>();
}

py::array_t<float64>
f64_vector_to_numpy(std::vector<float64> const& v)
{
    py::array_t<float64> arr(static_cast<py::ssize_t>(v.size()));
    auto buf = arr.mutable_unchecked<1>();
    for (std::size_t i = 0; i < v.size(); ++i) {
        buf(static_cast<py::ssize_t>(i)) = v[i];
    }
    return arr;
}

py::object
slices_to_numpy(std::optional<std::vector<std::array<int64, 2>>> const& slices)
{
    if (!slices) {
        return py::none();
    }
    py::array_t<int64> arr({ static_cast<py::ssize_t>(slices->size()), py::ssize_t{ 2 } });
    auto buf = arr.mutable_unchecked<2>();
    for (std::size_t i = 0; i < slices->size(); ++i) {
        buf(static_cast<py::ssize_t>(i), 0) = (*slices)[i][0];
        buf(static_cast<py::ssize_t>(i), 1) = (*slices)[i][1];
    }
    return arr;
}

std::optional<std::vector<int64>>
drop_which_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    if (py::isinstance<py::str>(obj)) {
        auto s = obj.cast<std::string>();
        if (s == "all") {
            return std::nullopt;
        }
        throw py::value_error("which must be 'all', an int, or a list of ints");
    }
    if (py::isinstance<py::int_>(obj)) {
        return std::vector<int64>{ obj.cast<int64>() };
    }
    return obj.cast<std::vector<int64>>();
}

SectorArray
sector_array_from_python(py::handle obj, Symmetry const& symmetry)
{
    if (py::isinstance<SectorArray>(obj)) {
        return obj.cast<SectorArray>();
    }
    if (py::isinstance<Sector>(obj)) {
        return SectorArray::from_sector(obj.cast<Sector>());
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
    return
      [sector_map](SectorArray const& sectors) { return sector_map(sectors).cast<SectorArray>(); };
}

Sector
sector_from_python(py::handle obj)
{
    if (py::isinstance<Sector>(obj)) {
        return obj.cast<Sector>();
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
        out.push_back(item.cast<Leg::Ptr>());
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

std::optional<py::class_<Space, PySpace, py::smart_holder>> g_space;
std::optional<py::class_<ElementarySpace, Space, Leg, PyElementarySpace, py::smart_holder>>
  g_elementary_space;

void bind_elementary_space(py::module_& m);

void bind_direct_sum_space(py::module_& m);

void bind_tensor_product(py::module_& m);

void bind_abelian_leg_pipe(py::module_& m);

} // namespace

void
bind_spaces(py::module_& m)
{
    py::class_<Leg, PyLeg, py::smart_holder> cls(m, "Leg", DOC(lasyten, Leg));

    // Declare Space / ElementarySpace before Leg methods that return them (stubgen).
    g_space = py::class_<Space, PySpace, py::smart_holder>(m, "Space", DOC(lasyten, Space));
    g_elementary_space =
      py::class_<ElementarySpace, Space, Leg, PyElementarySpace, py::smart_holder>(
        m, "ElementarySpace", DOC(lasyten, ElementarySpace));

    cls.def(
      py::init(
        [](py::object symmetry_obj, py::object dim_obj, bool is_dual, py::object basis_perm) {
            auto symmetry = symmetry_from_python(symmetry_obj);
            float64 dim = py::float_(dim_obj).cast<float64>();
            return std::make_shared<PyLeg>(
              std::move(symmetry), dim, is_dual, perm_from_python(basis_perm));
        }),
      py::arg("symmetry"),
      py::arg("dim"),
      py::arg("is_dual"),
      py::arg("basis_perm") = py::none());

    cls.def_readwrite("symmetry", &Leg::symmetry)
      .def_property(
        "dim",
        [](Leg const& self) { return dim_to_python(self.dim); },
        [](Leg& self, py::object dim_obj) { self.dim = py::float_(dim_obj).cast<float64>(); })
      .def_readwrite("is_dual", &Leg::is_dual);

    cls.def_property_readonly("dual", &Leg::dual, DOC(lasyten, Leg, dual))
      .def_property_readonly("is_trivial", &Leg::is_trivial)
      .def_property(
        "basis_perm",
        [](Leg const& self) { return perm_to_numpy(self.basis_perm()); },
        [](Leg& self, py::object basis_perm) {
            self.set_basis_perm(perm_from_python(basis_perm));
        },
        doc_cpp_ref(DOC(lasyten, Leg), "lasyten::Leg::basis_perm()"))
      .def_property(
        "_basis_perm",
        [](Leg const& self) -> py::object {
            if (!self.has_custom_basis_perm()) {
                return py::none();
            }
            return perm_to_numpy(self.basis_perm());
        },
        [](Leg& self, py::object basis_perm) {
            self.set_basis_perm(perm_from_python(basis_perm));
        })
      .def_property(
        "inverse_basis_perm",
        [](Leg const& self) { return perm_to_numpy(self.inverse_basis_perm()); },
        [](Leg& self, py::object inverse_basis_perm) {
            self.set_inverse_basis_perm(perm_from_python(inverse_basis_perm));
        },
        DOC(lasyten, Leg, inverse_basis_perm))
      .def_property(
        "_inverse_basis_perm",
        [](Leg const& self) -> py::object {
            if (!self.has_custom_basis_perm()) {
                return py::none();
            }
            return perm_to_numpy(self.inverse_basis_perm());
        },
        [](Leg& self, py::object inverse_basis_perm) {
            self.set_inverse_basis_perm(perm_from_python(inverse_basis_perm));
        })
      .def_property_readonly("flat_legs", &Leg::flat_legs, DOC(lasyten, Leg, flat_legs))
      .def_property_readonly("flat_spaces", &Leg::flat_spaces, DOC(lasyten, Leg, flat_spaces))
      .def_property_readonly(
        "num_flat_legs", &Leg::num_flat_legs, DOC(lasyten, Leg, num_flat_legs))
      .def_property_readonly("ascii_arrow", &Leg::ascii_arrow, DOC(lasyten, Leg, ascii_arrow));

    cls.def("test_sanity", &Leg::test_sanity, DOC(lasyten, Leg, test_sanity))
      .def("as_Space", &Leg::as_space_obj, DOC(lasyten, Leg, as_space))
      .def("as_ElementarySpace",
           &Leg::as_ElementarySpace,
           py::arg("is_dual") = false,
           DOC(lasyten, Leg, as_ElementarySpace))
      .def("_flat_leg_permutation",
           &Leg::_flat_leg_permutation,
           py::arg("offset") = 0,
           doc_cpp_ref(DOC(lasyten, Leg), "lasyten::Leg::_flat_leg_permutation()"))
      .def("__eq__", &Leg::operator==, py::arg("other"))
      .def("apply_basis_perm",
           &Leg::apply_basis_perm,
           py::arg("arr"),
           py::arg("axis") = 0,
           py::arg("inverse") = false,
           py::arg("pre_compose") = false,
           DOC(lasyten, Leg, apply_basis_perm));

    auto& space = *g_space;

    space.def(py::init([](py::object symmetry_obj,
                          py::object sector_decomposition,
                          py::object multiplicities,
                          py::object sector_order) {
                  auto symmetry = symmetry_from_python(symmetry_obj);
                  return std::make_shared<PySpace>(
                    symmetry,
                    sector_array_from_python(sector_decomposition, *symmetry),
                    multiplicities_from_python(multiplicities),
                    sector_order_from_python(sector_order));
              }),
              py::arg("symmetry"),
              py::arg("sector_decomposition"),
              py::arg("multiplicities") = py::none(),
              py::arg("sector_order") = py::none());

    space.def_readwrite("symmetry", &Space::symmetry)
      .def_readwrite("sector_decomposition", &Space::sector_decomposition)
      .def_readwrite("sector_order", &Space::sector_order)
      .def_property(
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
      .def_readonly("num_sectors",
                    &Space::num_sectors,
                    R"pydoc(The number of unique sectors appearing in this space, regardless of
their multiplicity. See :attr:`~lasyten.symmetries.spaces.Space.total_multiplicity` for the sum of
the sector multiplicities.)pydoc")
      .def_property_readonly(
        "total_multiplicity", &Space::total_multiplicity, DOC(lasyten, Space, total_multiplicity))
      .def_property_readonly("sector_dims",
                             [](Space const& self) -> py::object {
                                 if (!self.sector_dims) {
                                     return py::none();
                                 }
                                 return perm_to_numpy(*self.sector_dims);
                             })
      .def_property_readonly(
        "sector_qdims", [](Space const& self) { return f64_vector_to_numpy(self.sector_qdims); })
      .def_property_readonly("slices",
                             [](Space const& self) { return slices_to_numpy(self.slices); })
      .def_property_readonly("dim", [](Space const& self) { return dim_to_python(self.dim); });

    space.def_property_readonly("dual", &Space::dual, DOC(lasyten, Space, dual))
      .def_property_readonly("is_trivial",
                             &Space::is_trivial,
                             doc_cpp_ref(DOC(lasyten, Space), "lasyten::Space::is_trivial()"));

    space.def("test_sanity", &Space::test_sanity, DOC(lasyten, Space, test_sanity))
      .def("__eq__", &Space::operator==, py::arg("other"))
      .def("is_isomorphic_to",
           &Space::is_isomorphic_to,
           py::arg("other"),
           doc_cpp_ref(DOC(lasyten, Space), "lasyten::Space::is_isomorphic_to()"))
      .def("is_subspace_of",
           &Space::is_subspace_of,
           py::arg("other"),
           doc_cpp_ref(DOC(lasyten, Space), "lasyten::Space::is_subspace_of()"))
      .def("as_ElementarySpace",
           &Space::as_ElementarySpace,
           py::arg("is_dual") = false,
           DOC(lasyten, Leg, as_ElementarySpace))
      .def(
        "change_symmetry",
        [](Space& self, py::object symmetry_obj, py::function sector_map, bool injective) {
            auto symmetry = symmetry_from_python(symmetry_obj);
            SectorMapFn map = [sector_map](SectorArray const& sectors) {
                return sector_map(sectors).cast<SectorArray>();
            };
            return self.change_symmetry(std::move(symmetry), std::move(map), injective);
        },
        py::arg("symmetry"),
        py::arg("sector_map"),
        py::arg("injective") = false,
        DOC(lasyten, Space, change_symmetry))
      .def(
        "drop_symmetry",
        [](Space& self, py::object which) {
            return self.drop_symmetry(drop_which_from_python(which));
        },
        py::arg("which") = "all",
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
        py::arg("sector"),
        DOC(lasyten, Space, sector_decomposition_where))
      .def("sector_multiplicity",
           &Space::sector_multiplicity,
           py::arg("sector"),
           DOC(lasyten, Space, sector_multiplicity));

    py::class_<LegPipe, Leg, PyLegPipe, py::smart_holder> pipe(
      m, "LegPipe", DOC(lasyten, LegPipe));

    pipe.def(py::init([](py::sequence legs_obj, bool is_dual, bool combine_cstyle) {
                 std::vector<Leg::Ptr> legs;
                 legs.reserve(static_cast<std::size_t>(legs_obj.size()));
                 for (py::handle item : legs_obj) {
                     legs.push_back(item.cast<Leg::Ptr>());
                 }
                 return std::make_shared<PyLegPipe>(std::move(legs), is_dual, combine_cstyle);
             }),
             py::arg("legs"),
             py::arg("is_dual") = false,
             py::arg("combine_cstyle") = true);

    pipe.def_readwrite("legs", &LegPipe::legs)
      .def_readonly("num_legs", &LegPipe::num_legs)
      .def_readwrite("combine_cstyle", &LegPipe::combine_cstyle);

    pipe.def("test_sanity", &LegPipe::test_sanity, DOC(lasyten, LegPipe, test_sanity))
      .def("__eq__",
           [](LegPipe const& self, py::object other) -> py::object {
               if (!py::isinstance<LegPipe>(other)) {
                   return py::reinterpret_borrow<py::object>(Py_NotImplemented);
               }
               return py::cast(
                 self.operator==(static_cast<Leg const&>(other.cast<LegPipe const&>())));
           })
      .def("__getitem__", &LegPipe::operator[], py::arg("idx"))
      .def("__len__", [](LegPipe const& self) { return self.num_legs; })
      .def(
        "__iter__",
        [](LegPipe& self) { return py::make_iterator(self.legs.begin(), self.legs.end()); },
        py::keep_alive<0, 1>())
      .def("__repr__", [](LegPipe const& self) { return self.repr(); })
      .def("repr", &LegPipe::repr, py::arg("show_symmetry") = true, py::arg("one_line") = false);

    bind_elementary_space(m);
    bind_direct_sum_space(m);
    bind_tensor_product(m);
    bind_abelian_leg_pipe(m);
}

namespace {

void
bind_elementary_space(py::module_& m)
{
    (void)m;
    auto& cls = *g_elementary_space;

    cls.def(py::init([](py::object symmetry_obj,
                        py::object defining_sectors,
                        py::object multiplicities,
                        bool is_dual,
                        py::object basis_perm) {
                auto symmetry = symmetry_from_python(symmetry_obj);
                return std::make_shared<PyElementarySpace>(
                  symmetry,
                  sector_array_from_python(defining_sectors, *symmetry),
                  multiplicities_from_python(multiplicities),
                  is_dual,
                  perm_from_python(basis_perm));
            }),
            py::arg("symmetry"),
            py::arg("defining_sectors"),
            py::arg("multiplicities") = py::none(),
            py::arg("is_dual") = false,
            py::arg("basis_perm") = py::none());

    // symmetry and dim exist on both the Space and the Leg base; keep them in sync.
    cls
      .def_property(
        "symmetry",
        [](ElementarySpace const& self) { return self.Space::symmetry; },
        [](ElementarySpace& self, py::object symmetry_obj) {
            auto symmetry = symmetry_from_python(symmetry_obj);
            self.Space::symmetry = symmetry;
            self.Leg::symmetry = std::move(symmetry);
        })
      .def_property(
        "dim",
        [](ElementarySpace const& self) { return dim_to_python(self.Space::dim); },
        [](ElementarySpace& self, py::object dim_obj) {
            auto const dim = py::float_(dim_obj).cast<float64>();
            self.Space::dim = dim;
            self.Leg::dim = dim;
        })
      .def_readwrite("defining_sectors", &ElementarySpace::defining_sectors)
      .def_property_readonly(
        "sectors_of_basis",
        [](ElementarySpace const& self) { return sector_array_to_numpy(self.sectors_of_basis()); },
        DOC(lasyten, ElementarySpace, sectors_of_basis))
      .def_property_readonly("dual", &ElementarySpace::dual_es, DOC(lasyten, Space, dual));

    cls.def_static(
      "from_basis",
      [](py::object symmetry_obj, py::object sectors_of_basis) {
          auto symmetry = symmetry_from_python(symmetry_obj);
          return ElementarySpace::from_basis(
            symmetry, sector_array_from_python(sectors_of_basis, *symmetry));
      },
      py::arg("symmetry"),
      py::arg("sectors_of_basis"),
      doc_cpp_ref(DOC(lasyten, ElementarySpace), "lasyten::ElementarySpace::from_basis()"));

    cls.def_static(
      "from_independent_symmetries",
      [](py::sequence independent_descriptions) {
          std::vector<ElementarySpace::Ptr> descriptions;
          descriptions.reserve(static_cast<std::size_t>(independent_descriptions.size()));
          for (py::handle item : independent_descriptions) {
              descriptions.push_back(item.cast<ElementarySpace::Ptr>());
          }
          return ElementarySpace::from_independent_symmetries(descriptions);
      },
      py::arg("independent_descriptions"),
      doc_cpp_ref(DOC(lasyten, ElementarySpace),
                  "lasyten::ElementarySpace::from_independent_symmetries()"));

    cls.def_static(
      "from_largest_common_subspace",
      [](py::args spaces_obj, bool is_dual) {
          std::vector<Space::Ptr> spaces;
          spaces.reserve(static_cast<std::size_t>(spaces_obj.size()));
          for (py::handle item : spaces_obj) {
              spaces.push_back(item.cast<Space::Ptr>());
          }
          return ElementarySpace::from_largest_common_subspace(spaces, is_dual);
      },
      py::arg("is_dual") = false,
      doc_cpp_ref(DOC(lasyten, ElementarySpace),
                  "lasyten::ElementarySpace::from_largest_common_subspace()"));

    cls.def_static(
      "from_null_space",
      [](py::object symmetry_obj, bool is_dual) {
          return ElementarySpace::from_null_space(symmetry_from_python(symmetry_obj), is_dual);
      },
      py::arg("symmetry"),
      py::arg("is_dual") = false,
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
      py::arg("symmetry"),
      py::arg("defining_sectors"),
      py::arg("multiplicities") = py::none(),
      py::arg("is_dual") = false,
      py::arg("basis_perm") = py::none(),
      py::arg("unique_sectors") = false,
      py::arg("return_sorting_perm") = false,
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
      py::arg("symmetry"),
      py::arg("sector_decomposition"),
      py::arg("multiplicities") = py::none(),
      py::arg("is_dual") = false,
      py::arg("basis_perm") = py::none(),
      py::arg("unique_sectors") = false,
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
      py::arg("dim") = 1,
      py::arg("symmetry") = py::none(),
      py::arg("is_dual") = false,
      py::arg("basis_perm") = py::none(),
      doc_cpp_ref(DOC(lasyten, ElementarySpace),
                  "lasyten::ElementarySpace::from_trivial_sector()"));

    cls.def("test_sanity", &ElementarySpace::test_sanity, DOC(lasyten, Space, test_sanity))
      .def("__repr__", [](ElementarySpace const& self) { return self.repr(); })
      .def("repr",
           &ElementarySpace::repr,
           py::arg("show_symmetry") = true,
           py::arg("one_line") = false)
      .def("__eq__",
           [](ElementarySpace const& self, py::object other) -> py::object {
               if (!py::isinstance<ElementarySpace>(other)) {
                   return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
               }
               return py::cast(self.equals_es(other.cast<ElementarySpace const&>()));
           })
      .def("as_Space", &ElementarySpace::as_space_obj, DOC(lasyten, ElementarySpace, as_space_obj))
      .def("as_ElementarySpace",
           &ElementarySpace::as_ElementarySpace,
           py::arg("is_dual") = false,
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
        py::arg("symmetry"),
        py::arg("sector_map"),
        py::arg("injective") = false,
        DOC(lasyten, ElementarySpace, change_symmetry))
      .def(
        "direct_sum",
        [](ElementarySpace const& self,
           py::args others_obj,
           std::optional<OptionalLabels> summand_labels) {
            std::vector<ElementarySpace::Ptr> others;
            others.reserve(static_cast<std::size_t>(others_obj.size()));
            for (py::handle item : others_obj) {
                others.push_back(item.cast<ElementarySpace::Ptr>());
            }
            return self.direct_sum(others, std::move(summand_labels));
        },
        py::kw_only(),
        py::arg("summand_labels") = py::none(),
        doc_cpp_ref(DOC(lasyten, ElementarySpace), "lasyten::ElementarySpace::direct_sum()"))
      .def(
        "drop_symmetry",
        [](ElementarySpace& self, py::object which) {
            return self.drop_symmetry(drop_which_from_python(which));
        },
        py::arg("which") = "all",
        DOC(lasyten, Space, drop_symmetry))
      .def(
        "parse_index",
        [](ElementarySpace const& self, int64 idx) {
            auto const [sector_idx, multiplicity_idx] = self.parse_index(idx);
            return py::make_tuple(sector_idx, multiplicity_idx);
        },
        py::arg("idx"),
        doc_cpp_ref(DOC(lasyten, ElementarySpace), "lasyten::ElementarySpace::parse_index()"))
      .def("idx_to_sector", &ElementarySpace::idx_to_sector, py::arg("idx"))
      .def(
        "take_slice",
        [](ElementarySpace& self, py::object blockmask) {
            return self.take_slice(py::array::ensure(blockmask));
        },
        py::arg("blockmask"),
        DOC(lasyten, ElementarySpace, take_slice))
      .def("with_opposite_duality",
           &ElementarySpace::with_opposite_duality,
           DOC(lasyten, ElementarySpace, with_opposite_duality))
      .def("with_is_dual",
           &ElementarySpace::with_is_dual,
           py::arg("is_dual"),
           DOC(lasyten, ElementarySpace, with_is_dual))
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<ElementarySpace>(),
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<ElementarySpace>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"))
      .def_property_readonly("is_direct_sum_space",
                             &ElementarySpace::is_direct_sum_space,
                             DOC(lasyten, ElementarySpace, is_direct_sum_space));
}

void
bind_direct_sum_space(py::module_& m)
{
    py::class_<DirectSumSpace, ElementarySpace, PyDirectSumSpace, py::smart_holder> cls(
      m, "DirectSumSpace", DOC(lasyten, DirectSumSpace));

    cls.def(
      py::init(
        [](py::sequence spaces_obj, bool is_dual, std::optional<OptionalLabels> summand_labels) {
            std::vector<ElementarySpace::Ptr> spaces;
            spaces.reserve(static_cast<std::size_t>(spaces_obj.size()));
            for (py::handle item : spaces_obj) {
                spaces.push_back(item.cast<ElementarySpace::Ptr>());
            }
            return std::make_shared<PyDirectSumSpace>(
              std::move(spaces), is_dual, std::move(summand_labels));
        }),
      py::arg("spaces"),
      py::arg("is_dual") = false,
      py::arg("summand_labels") = py::none());

    cls.def_property_readonly("spaces", [](DirectSumSpace const& self) { return self.spaces; });
    cls.def_property_readonly("summand_labels",
                              [](DirectSumSpace const& self) { return self.summand_labels; });

    cls
      .def_static(
        "from_spaces",
        [](py::sequence spaces_obj, bool is_dual, std::optional<OptionalLabels> summand_labels) {
            std::vector<ElementarySpace::Ptr> spaces;
            spaces.reserve(static_cast<std::size_t>(spaces_obj.size()));
            for (py::handle item : spaces_obj) {
                spaces.push_back(item.cast<ElementarySpace::Ptr>());
            }
            return DirectSumSpace::from_spaces(
              std::move(spaces), is_dual, std::move(summand_labels));
        },
        py::arg("spaces"),
        py::arg("is_dual") = false,
        py::arg("summand_labels") = py::none(),
        DOC(lasyten, DirectSumSpace, from_spaces))
      .def(
        "get_summand_idx",
        [](DirectSumSpace const& self, py::object which) {
            if (py::isinstance<py::str>(which)) {
                return self.get_summand_idx(which.cast<std::string>());
            }
            return self.get_summand_idx(which.cast<int64>());
        },
        py::arg("which"),
        DOC(lasyten, DirectSumSpace, get_summand_idx))
      .def("has_summand_label",
           &DirectSumSpace::has_summand_label,
           py::arg("label"),
           DOC(lasyten, DirectSumSpace, has_summand_label))
      .def("mult_slices", &DirectSumSpace::mult_slices, DOC(lasyten, DirectSumSpace, mult_slices))
      .def("as_plain_ElementarySpace",
           &DirectSumSpace::as_plain_ElementarySpace,
           DOC(lasyten, DirectSumSpace, as_plain_ElementarySpace))
      .def("test_sanity", &DirectSumSpace::test_sanity)
      .def("as_Space", &DirectSumSpace::as_space_obj, DOC(lasyten, DirectSumSpace, as_space_obj))
      .def("as_ElementarySpace",
           &DirectSumSpace::as_ElementarySpace,
           py::arg("is_dual") = false,
           DOC(lasyten, Leg, as_ElementarySpace))
      .def_property_readonly("dual", &DirectSumSpace::dual_dss, DOC(lasyten, Leg, dual))
      .def(
        "change_symmetry",
        [](
          DirectSumSpace& self, py::object symmetry_obj, py::function sector_map, bool injective) {
            return self.change_symmetry(
              symmetry_from_python(symmetry_obj), sector_map_from_python(sector_map), injective);
        },
        py::arg("symmetry"),
        py::arg("sector_map"),
        py::arg("injective") = false,
        DOC(lasyten, ElementarySpace, change_symmetry))
      .def(
        "drop_symmetry",
        [](DirectSumSpace& self, py::object which) {
            return self.drop_symmetry(drop_which_from_python(which));
        },
        py::arg("which") = "all",
        DOC(lasyten, Space, drop_symmetry))
      .def(
        "take_slice",
        [](DirectSumSpace& self, py::object blockmask) {
            return self.take_slice(py::array::ensure(blockmask));
        },
        py::arg("blockmask"),
        DOC(lasyten, ElementarySpace, take_slice))
      .def("with_opposite_duality",
           &DirectSumSpace::with_opposite_duality,
           DOC(lasyten, ElementarySpace, with_opposite_duality))
      .def("projection_onto_summands",
           &DirectSumSpace::projection_onto_summands,
           py::arg("keep_indices") = py::none(),
           py::arg("discard_indices") = py::none(),
           py::arg("backend") = nullptr,
           py::arg("labels") = py::none(),
           py::arg("device") = py::none(),
           DOC(lasyten, DirectSumSpace, projection_onto_summands))
      .def("inclusion_of_summands",
           &DirectSumSpace::inclusion_of_summands,
           py::arg("keep_indices") = py::none(),
           py::arg("discard_indices") = py::none(),
           py::arg("backend") = nullptr,
           py::arg("labels") = py::none(),
           py::arg("device") = py::none(),
           DOC(lasyten, DirectSumSpace, inclusion_of_summands))
      .def(
        "projection_onto_summand",
        [](DirectSumSpace const& self,
           py::object which,
           std::shared_ptr<TensorBackend> backend,
           std::optional<OptionalLabels> labels,
           std::optional<std::string> device) {
            DirectSumSpace::SummandRef ref =
              py::isinstance<py::str>(which)
                ? DirectSumSpace::SummandRef{ which.cast<std::string>() }
                : DirectSumSpace::SummandRef{ which.cast<int64>() };
            return self.projection_onto_summand(
              std::move(ref), std::move(backend), std::move(labels), std::move(device));
        },
        py::arg("which"),
        py::arg("backend") = nullptr,
        py::arg("labels") = py::none(),
        py::arg("device") = py::none(),
        DOC(lasyten, DirectSumSpace, projection_onto_summand))
      .def(
        "inclusion_of_summand",
        [](DirectSumSpace const& self,
           py::object which,
           std::shared_ptr<TensorBackend> backend,
           std::optional<OptionalLabels> labels,
           std::optional<std::string> device) {
            DirectSumSpace::SummandRef ref =
              py::isinstance<py::str>(which)
                ? DirectSumSpace::SummandRef{ which.cast<std::string>() }
                : DirectSumSpace::SummandRef{ which.cast<int64>() };
            return self.inclusion_of_summand(
              std::move(ref), std::move(backend), std::move(labels), std::move(device));
        },
        py::arg("which"),
        py::arg("backend") = nullptr,
        py::arg("labels") = py::none(),
        py::arg("device") = py::none(),
        DOC(lasyten, DirectSumSpace, inclusion_of_summand))
      .def("unit_vector_of_summand",
           &DirectSumSpace::unit_vector_of_summand,
           py::arg("i"),
           py::arg("backend") = nullptr,
           py::arg("labels") = py::none(),
           py::arg("dtype") = py::none(),
           py::arg("device") = py::none(),
           DOC(lasyten, DirectSumSpace, unit_vector_of_summand))
      .def(
        "set_basis_perm",
        [](DirectSumSpace& self, py::args, py::kwargs) { self.set_basis_perm(std::nullopt); },
        "Can not set basis_perm for DirectSumSpace.")
      .def(
        "set_inverse_basis_perm",
        [](DirectSumSpace& self, py::args, py::kwargs) {
            self.set_inverse_basis_perm(std::nullopt);
        },
        "Can not set basis_perm for DirectSumSpace.")
      .def("__repr__", [](DirectSumSpace const& self) { return self.repr(); })
      .def("repr",
           &DirectSumSpace::repr,
           py::arg("show_symmetry") = true,
           py::arg("one_line") = false)
      .def("__eq__",
           [](DirectSumSpace const& self, py::object other) -> py::object {
               if (!py::isinstance<DirectSumSpace>(other)) {
                   return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
               }
               return py::cast(self.equals_dss(other.cast<DirectSumSpace const&>()));
           })
      // Shadow unsupported ElementarySpace factories
      .def_static("from_basis",
                  [](py::args, py::kwargs) -> py::object {
                      throw py::type_error("from_basis is not supported for DirectSumSpace");
                  })
      .def_static("from_null_space",
                  [](py::args, py::kwargs) -> py::object {
                      throw py::type_error("from_null_space is not supported for DirectSumSpace");
                  })
      .def_static("from_defining_sectors",
                  [](py::args, py::kwargs) -> py::object {
                      throw py::type_error(
                        "from_defining_sectors is not supported for DirectSumSpace");
                  })
      .def_static("from_trivial_sector",
                  [](py::args, py::kwargs) -> py::object {
                      throw py::type_error(
                        "from_trivial_sector is not supported for DirectSumSpace");
                  })
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<DirectSumSpace>(),
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<DirectSumSpace>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

void
bind_tensor_product(py::module_& m)
{
    py::class_<TensorProduct, Space, PyTensorProduct, py::smart_holder> cls(
      m, "TensorProduct", DOC(lasyten, TensorProduct));

    cls.def(py::init([](py::iterable factors_obj,
                        py::object symmetry_obj,
                        py::object sector_decomposition,
                        py::object multiplicities) {
                auto factors = legs_from_python(factors_obj);
                auto symmetry = optional_symmetry_from_python(symmetry_obj);
                std::optional<SectorArray> sectors;
                if (!sector_decomposition.is_none()) {
                    auto sym = symmetry;
                    if (!sym && !factors.empty()) {
                        sym = factors.front()->symmetry;
                    }
                    if (!sym) {
                        throw py::value_error("If spaces is empty, the symmetry arg is required.");
                    }
                    sectors = sector_array_from_python(sector_decomposition, *sym);
                }
                return std::make_shared<PyTensorProduct>(
                  std::move(factors),
                  std::move(symmetry),
                  std::move(sectors),
                  multiplicities_from_python(multiplicities));
            }),
            py::arg("factors"),
            py::arg("symmetry") = py::none(),
            py::arg("_sector_decomposition") = py::none(),
            py::arg("_multiplicities") = py::none());

    cls
      .def_property(
        "factors",
        [](TensorProduct const& self) { return self.factors; },
        [](TensorProduct& self, py::iterable factors_obj) {
            self.factors = legs_from_python(factors_obj);
            self.num_factors = static_cast<int64>(self.factors.size());
        })
      .def_readonly("num_factors", &TensorProduct::num_factors)
      .def_property_readonly("dual", &TensorProduct::dual_space, DOC(lasyten, Space, dual))
      .def_property_readonly(
        "has_pipes",
        &TensorProduct::has_pipes,
        doc_cpp_ref(DOC(lasyten, TensorProduct), "lasyten::TensorProduct::has_pipes()"))
      .def_property_readonly("flat_legs", &TensorProduct::flat_legs, DOC(lasyten, Leg, flat_legs))
      .def_property_readonly(
        "flat_spaces", &TensorProduct::flat_spaces, DOC(lasyten, Leg, flat_spaces))
      .def_property_readonly(
        "num_flat_legs", &TensorProduct::num_flat_legs, DOC(lasyten, Leg, num_flat_legs));

    cls.def_static(
      "from_partial_products",
      [](py::args factors_obj) {
          std::vector<TensorProduct::Ptr> factors;
          factors.reserve(static_cast<std::size_t>(factors_obj.size()));
          for (py::handle item : factors_obj) {
              factors.push_back(item.cast<TensorProduct::Ptr>());
          }
          return TensorProduct::from_partial_products(factors);
      },
      doc_cpp_ref(DOC(lasyten, TensorProduct), "lasyten::TensorProduct::from_partial_products()"));

    cls.def("test_sanity", &TensorProduct::test_sanity, DOC(lasyten, Space, test_sanity))
      .def(
        "block_size",
        [](TensorProduct const& self, py::object coupled) {
            if (py::isinstance<py::int_>(coupled)) {
                return self.block_size(coupled.cast<int64>());
            }
            return self.block_size(sector_from_python(coupled));
        },
        py::arg("coupled"),
        DOC(lasyten, TensorProduct, block_size))
      .def(
        "change_symmetry",
        [](TensorProduct& self, py::object symmetry_obj, py::function sector_map, bool injective) {
            return self.change_symmetry(
              symmetry_from_python(symmetry_obj), sector_map_from_python(sector_map), injective);
        },
        py::arg("symmetry"),
        py::arg("sector_map"),
        py::arg("injective") = false,
        DOC(lasyten, TensorProduct, change_symmetry))
      .def(
        "drop_symmetry",
        [](TensorProduct& self, py::object which) {
            return self.drop_symmetry(drop_which_from_python(which));
        },
        py::arg("which") = "all",
        DOC(lasyten, Space, drop_symmetry))
      .def("flat_legs_nesting",
           &TensorProduct::flat_legs_nesting,
           doc_cpp_ref(DOC(lasyten, TensorProduct), "lasyten::TensorProduct::flat_legs_nesting()"))
      .def("flat_leg_idcs",
           &TensorProduct::flat_leg_idcs,
           py::arg("i"),
           doc_cpp_ref(DOC(lasyten, TensorProduct), "lasyten::TensorProduct::flat_leg_idcs()"))
      .def(
        "forest_block_size",
        [](TensorProduct const& self, py::object uncoupled, py::object coupled) {
            return self.forest_block_size(sector_array_from_python(uncoupled, *self.symmetry),
                                          sector_from_python(coupled));
        },
        py::arg("uncoupled"),
        py::arg("coupled"),
        DOC(lasyten, TensorProduct, forest_block_size))
      .def(
        "forest_block_slice",
        [](TensorProduct const& self, py::object uncoupled, py::object coupled) {
            return index_slice_to_python(self.forest_block_slice(
              sector_array_from_python(uncoupled, *self.symmetry), sector_from_python(coupled)));
        },
        py::arg("uncoupled"),
        py::arg("coupled"),
        DOC(lasyten, TensorProduct, forest_block_slice))
      .def("insert_multiply",
           &TensorProduct::insert_multiply,
           py::arg("other"),
           py::arg("pos"),
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
        py::arg("coupled"),
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
        py::arg("coupled"),
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
        py::arg("yield_slices") = false,
        DOC(lasyten, TensorProduct, iter_uncoupled))
      .def("left_multiply",
           &TensorProduct::left_multiply,
           py::arg("other"),
           DOC(lasyten, TensorProduct, left_multiply))
      .def("permuted",
           &TensorProduct::permuted,
           py::arg("perm"),
           DOC(lasyten, TensorProduct, permuted))
      .def("right_multiply",
           &TensorProduct::right_multiply,
           py::arg("other"),
           DOC(lasyten, TensorProduct, right_multiply))
      .def(
        "tree_block_size",
        [](TensorProduct const& self, py::object uncoupled) {
            return self.tree_block_size(sector_array_from_python(uncoupled, *self.symmetry));
        },
        py::arg("uncoupled"),
        DOC(lasyten, TensorProduct, tree_block_size))
      .def(
        "tree_block_slice",
        [](TensorProduct const& self, FusionTree const& tree) {
            return index_slice_to_python(self.tree_block_slice(tree));
        },
        py::arg("tree"),
        DOC(lasyten, TensorProduct, tree_block_slice));

    cls
      .def("__eq__",
           [](TensorProduct const& self, py::object other) -> py::object {
               if (!py::isinstance<TensorProduct>(other)) {
                   return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
               }
               return py::cast(
                 self.operator==(static_cast<Space const&>(other.cast<TensorProduct const&>())));
           })
      .def("__getitem__",
           [](TensorProduct const& self, py::object idx) -> py::object {
               if (py::isinstance<py::slice>(idx)) {
                   py::list all;
                   for (auto const& f : self.factors) {
                       all.append(py::cast(f));
                   }
                   return all[idx];
               }
               return py::cast(self[idx.cast<int64>()]);
           })
      .def("__len__", [](TensorProduct const& self) { return self.num_factors; })
      .def(
        "__iter__",
        [](TensorProduct const& self) { return py::make_iterator(self.factors); },
        py::keep_alive<0, 1>())
      .def("__repr__",
           &TensorProduct::repr,
           py::arg("show_symmetry") = true,
           py::arg("one_line") = false)
      .def(
        "repr", &TensorProduct::repr, py::arg("show_symmetry") = true, py::arg("one_line") = false)
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<TensorProduct>(),
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<TensorProduct>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

void
bind_abelian_leg_pipe(py::module_& m)
{
    // Diamond MI: the Python MRO is AbelianLegPipe, LegPipe, ElementarySpace, Space, Leg.
    py::class_<AbelianLegPipe, LegPipe, ElementarySpace, PyAbelianLegPipe, py::smart_holder> cls(
      m, "AbelianLegPipe", DOC(lasyten, AbelianLegPipe));

    cls.def(py::init([](py::sequence legs_obj, bool is_dual, bool combine_cstyle) {
                std::vector<ElementarySpace::Ptr> legs;
                legs.reserve(static_cast<std::size_t>(legs_obj.size()));
                for (py::handle item : legs_obj) {
                    legs.push_back(item.cast<ElementarySpace::Ptr>());
                }
                return std::make_shared<PyAbelianLegPipe>(
                  std::move(legs), is_dual, combine_cstyle);
            }),
            py::arg("legs"),
            py::arg("is_dual") = false,
            py::arg("combine_cstyle") = true);

    cls
      .def_property_readonly(
        "sector_strides",
        [](AbelianLegPipe const& self) { return perm_to_numpy(self.sector_strides); })
      .def_property_readonly(
        "fusion_outcomes_sort",
        [](AbelianLegPipe const& self) { return perm_to_numpy(self.fusion_outcomes_sort); })
      .def_property_readonly(
        "block_ind_map_slices",
        [](AbelianLegPipe const& self) { return perm_to_numpy(self.block_ind_map_slices); })
      .def_property_readonly("block_ind_map",
                             [](AbelianLegPipe const& self) { return self.block_ind_map; });

    cls.def_property_readonly("dual", &AbelianLegPipe::dual_pipe, DOC(lasyten, Leg, dual))
      .def_property_readonly(
        "is_trivial",
        &AbelianLegPipe::is_trivial,
        doc_cpp_ref(DOC(lasyten, AbelianLegPipe), "lasyten::AbelianLegPipe::is_trivial()"))
      .def_property_readonly(
        "flat_spaces", &AbelianLegPipe::flat_spaces, DOC(lasyten, LegPipe, flat_spaces))
      .def_property_readonly(
        "ascii_arrow", &AbelianLegPipe::ascii_arrow, DOC(lasyten, AbelianLegPipe, ascii_arrow));

    cls.def_static(
      "from_independent_symmetries",
      [](py::sequence independent_descriptions) {
          std::vector<AbelianLegPipe::Ptr> descriptions;
          descriptions.reserve(static_cast<std::size_t>(independent_descriptions.size()));
          for (py::handle item : independent_descriptions) {
              descriptions.push_back(item.cast<AbelianLegPipe::Ptr>());
          }
          return AbelianLegPipe::from_independent_symmetries(descriptions);
      },
      py::arg("independent_descriptions"),
      DOC(lasyten, AbelianLegPipe, from_independent_symmetries));

    // The unsupported ElementarySpace factories. They are bound (and raise) such that they
    // shadow the inherited versions, which would silently return a plain ElementarySpace.
    cls
      .def_static(
        "from_basis",
        [](py::args, py::kwargs) -> py::object {
            throw py::type_error("from_basis is not supported for AbelianLegPipe");
        },
        DOC(lasyten, AbelianLegPipe, from_basis))
      .def_static(
        "from_null_space",
        [](py::args, py::kwargs) -> py::object {
            throw py::type_error("from_null_space is not supported for AbelianLegPipe");
        },
        DOC(lasyten, AbelianLegPipe, from_null_space))
      .def_static(
        "from_defining_sectors",
        [](py::args, py::kwargs) -> py::object {
            throw py::type_error("from_defining_sectors is not supported for AbelianLegPipe");
        },
        doc_cpp_ref(DOC(lasyten, AbelianLegPipe),
                    "lasyten::AbelianLegPipe::from_defining_sectors()"))
      .def_static(
        "from_trivial_sector",
        [](py::args, py::kwargs) -> py::object {
            throw py::type_error("from_trivial_sector is not supported for AbelianLegPipe");
        },
        DOC(lasyten, AbelianLegPipe, from_trivial_sector));

    cls.def("test_sanity", &AbelianLegPipe::test_sanity, DOC(lasyten, AbelianLegPipe, test_sanity))
      .def("as_Space", &AbelianLegPipe::as_space_obj, DOC(lasyten, Leg, as_space))
      .def("as_ElementarySpace",
           &AbelianLegPipe::as_ElementarySpace,
           py::arg("is_dual") = false,
           DOC(lasyten, Leg, as_ElementarySpace))
      .def(
        "change_symmetry",
        [](
          AbelianLegPipe& self, py::object symmetry_obj, py::function sector_map, bool injective) {
            return self.change_symmetry(
              symmetry_from_python(symmetry_obj), sector_map_from_python(sector_map), injective);
        },
        py::arg("symmetry"),
        py::arg("sector_map"),
        py::arg("injective") = false,
        DOC(lasyten, Space, change_symmetry))
      .def(
        "drop_symmetry",
        [](AbelianLegPipe& self, py::object which) {
            return self.drop_symmetry(drop_which_from_python(which));
        },
        py::arg("which") = "all",
        DOC(lasyten, Space, drop_symmetry))
      .def(
        "set_basis_perm",
        [](AbelianLegPipe& self, py::args, py::kwargs) { self.set_basis_perm(std::nullopt); },
        doc_cpp_ref(DOC(lasyten, AbelianLegPipe), "lasyten::AbelianLegPipe::set_basis_perm()"))
      .def(
        "take_slice",
        [](AbelianLegPipe& self, py::object blockmask) {
            return self.take_slice(py::array::ensure(blockmask));
        },
        py::arg("blockmask"),
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
        py::arg("multiplicities"),
        doc_cpp_ref(DOC(lasyten, AbelianLegPipe),
                    "lasyten::AbelianLegPipe::_get_fusion_outcomes_perm()"))
      .def("__eq__",
           [](AbelianLegPipe const& self, py::object other) -> py::object {
               if (!py::isinstance<LegPipe>(other)) {
                   return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
               }
               return py::cast(
                 self.operator==(static_cast<Leg const&>(other.cast<LegPipe const&>())));
           })
      .def("__repr__", [](AbelianLegPipe const& self) { return self.repr(); })
      .def("repr",
           &AbelianLegPipe::repr,
           py::arg("show_symmetry") = true,
           py::arg("one_line") = false)
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<AbelianLegPipe>(),
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<AbelianLegPipe>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));

    m.def("swap_gate",
          &swap_gate,
          py::arg("V"),
          py::arg("W"),
          doc_cpp_ref(DOC(lasyten, AbelianLegPipe), "lasyten::swap_gate()"));

    m.def("twist_gate",
          &twist_gate,
          py::arg("V"),
          doc_cpp_ref(DOC(lasyten, AbelianLegPipe), "lasyten::twist_gate()"));

    m.def("_twist_gate_diag", &twist_gate_diag, py::arg("V"));

    m.def(
      "_flat_leg_permutation",
      [](py::sequence legs_obj) {
          std::vector<Leg::Ptr> legs;
          legs.reserve(static_cast<std::size_t>(legs_obj.size()));
          for (py::handle item : legs_obj) {
              legs.push_back(item.cast<Leg::Ptr>());
          }
          return flat_leg_permutation(legs);
      },
      py::arg("legs"),
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
      py::arg("unsorted_sectors"),
      py::arg("unsorted_multiplicities"),
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
      py::arg("sectors"),
      py::arg("multiplicities"));

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
      py::arg("which"),
      py::arg("symmetry"),
      doc_cpp_ref(DOC(lasyten, AbelianLegPipe),
                  "lasyten::AbelianLegPipe::_parse_inputs_drop_symmetry()"));
}

} // namespace

} // namespace lasyten
