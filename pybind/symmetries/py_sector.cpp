#include "../doc_plus.h"
#include "../ndarray_utils.h"
#include "docstrings/symmetries/sector.h"
#include "py_lasyten_nanobind.h"
#include "tools/hdf5_bind.h"

#include <lasyten/symmetries/sector.h>
#include <lasyten/symmetries/sector_numpy.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace lasyten {

namespace {

nb::ndarray<nb::numpy, std::size_t>
indices_to_numpy(std::vector<std::size_t> v)
{
    return ndarray_utils::vector_to_numpy_1d(std::move(v));
}

nb::ndarray<nb::numpy, std::int64_t>
i64_to_numpy(std::vector<std::int64_t> v)
{
    return ndarray_utils::vector_to_numpy_1d(std::move(v));
}

nb::ndarray<nb::numpy, std::int64_t>
sector_to_ndarray(Sector const& src)
{
    std::vector<std::int64_t> v(src.len());
    for (std::uint8_t i = 0; i < src.len(); ++i) {
        v[i] = src.q[i];
    }
    return ndarray_utils::vector_to_numpy_1d(std::move(v));
}

nb::ndarray<nb::numpy, std::int64_t>
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
    return ndarray_utils::vector_to_numpy_2d(std::move(v), rows, cols);
}

std::vector<std::size_t>
indices_from_py(py::handle key, std::size_t n_rows)
{
    // Python int or NumPy integer scalar
    try {
        if (!py::isinstance<py::slice>(key) && !py::isinstance<py::tuple>(key) &&
            !py::isinstance<py::list>(key) && !py::isinstance<py::array>(key)) {
            auto i = py::cast<std::ptrdiff_t>(key);
            if (i < 0) {
                i += static_cast<std::ptrdiff_t>(n_rows);
            }
            if (i < 0 || static_cast<std::size_t>(i) >= n_rows) {
                throw py::index_error("SectorArray index out of range");
            }
            return { static_cast<std::size_t>(i) };
        }
    } catch (py::cast_error const&) {
        // not a scalar int
    }
    if (py::isinstance<py::slice>(key)) {
        auto sl = py::cast<py::slice>(key);
        size_t start = 0, stop = 0, step = 0, slicelength = 0;
        if (!sl.compute(n_rows, &start, &stop, &step, &slicelength)) {
            throw py::error_already_set();
        }
        std::vector<std::size_t> out(slicelength);
        for (size_t i = 0; i < slicelength; ++i) {
            out[i] = start + i * step;
        }
        return out;
    }
    // bool mask or fancy int index via NumPy
    py::array arr = py::array::ensure(key);
    if (!arr) {
        throw py::type_error("SectorArray indices must be int, slice, or array");
    }
    if (arr.ndim() == 0) {
        // 0-d numpy integer
        auto i = py::cast<std::ptrdiff_t>(arr);
        if (i < 0) {
            i += static_cast<std::ptrdiff_t>(n_rows);
        }
        if (i < 0 || static_cast<std::size_t>(i) >= n_rows) {
            throw py::index_error("SectorArray index out of range");
        }
        return { static_cast<std::size_t>(i) };
    }
    if (arr.ndim() != 1) {
        throw py::type_error("SectorArray fancy index must be 1D");
    }
    auto info = arr.request();
    if (py::isinstance<py::array_t<bool>>(arr) || info.format == "?") {
        py::array_t<bool> mask = py::array_t<bool>::ensure(arr);
        auto r = mask.unchecked<1>();
        if (static_cast<std::size_t>(r.shape(0)) != n_rows) {
            throw py::value_error("boolean index length mismatch");
        }
        std::vector<std::size_t> out;
        for (py::ssize_t i = 0; i < r.shape(0); ++i) {
            if (r(i)) {
                out.push_back(static_cast<std::size_t>(i));
            }
        }
        return out;
    }
    // Also treat uint8/bool-like masks from np.any(...)
    if (info.itemsize == 1 && (info.format == "B" || info.format == "b" || info.format == "?")) {
        // Could be bool mask stored as uint8
        auto* ptr = static_cast<unsigned char const*>(info.ptr);
        if (static_cast<std::size_t>(info.shape[0]) == n_rows) {
            bool looks_bool = true;
            for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
                if (ptr[i] > 1) {
                    looks_bool = false;
                    break;
                }
            }
            if (looks_bool) {
                std::vector<std::size_t> out;
                for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
                    if (ptr[i]) {
                        out.push_back(static_cast<std::size_t>(i));
                    }
                }
                return out;
            }
        }
    }
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> idx =
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(arr);
    auto r = idx.unchecked<1>();
    std::vector<std::size_t> out(static_cast<std::size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        auto v = static_cast<std::ptrdiff_t>(r(i));
        if (v < 0) {
            v += static_cast<std::ptrdiff_t>(n_rows);
        }
        if (v < 0 || static_cast<std::size_t>(v) >= n_rows) {
            throw py::index_error("SectorArray index out of range");
        }
        out[static_cast<std::size_t>(i)] = static_cast<std::size_t>(v);
    }
    return out;
}

std::string
sector_repr(Sector const& s)
{
    std::ostringstream oss;
    oss << "Sector([";
    for (std::uint8_t i = 0; i < s.len(); ++i) {
        if (i) {
            oss << ", ";
        }
        oss << s.q[i];
    }
    oss << "])";
    return oss.str();
}

std::string
sector_array_repr(SectorArray const& a)
{
    std::ostringstream oss;
    oss << "SectorArray(shape=(" << a.size() << ", " << static_cast<unsigned>(a.sector_ind_len())
        << "))";
    return oss.str();
}

} // namespace

void
bind_sector(nb::module_& m)
{
    nb::class_<Sector>(m, "Sector", DOC(lasyten, Sector))
      .def(nb::init<>())
      .def(nb::new_([](py::object values) { return sector_from_numpy(values); }),
           nb::arg("values"),
           "Construct from a 1D integer sequence or ndarray.")
      .def_prop_ro("shape", [](Sector const& self) { return py::make_tuple(self.len()); })
      .def_prop_ro("sector_ind_len",
                   [](Sector const& self) { return static_cast<int>(self.len()); })
      .def("__len__", [](Sector const& self) { return static_cast<std::size_t>(self.len()); })
      .def("__getitem__",
           [](Sector const& self, py::object key) -> py::object {
               if (py::isinstance<py::int_>(key)) {
                   auto i = py::cast<std::ptrdiff_t>(key);
                   auto const n = static_cast<std::ptrdiff_t>(self.len());
                   if (i < 0) {
                       i += n;
                   }
                   if (i < 0 || i >= n) {
                       throw py::index_error("Sector index out of range");
                   }
                   return py::int_(self[static_cast<std::size_t>(i)]);
               }
               // slice → list of ints
               if (py::isinstance<py::slice>(key)) {
                   auto sl = py::cast<py::slice>(key);
                   size_t start = 0, stop = 0, step = 0, slicelength = 0;
                   if (!sl.compute(self.len(), &start, &stop, &step, &slicelength)) {
                       throw py::error_already_set();
                   }
                   py::list out;
                   for (size_t i = 0; i < slicelength; ++i) {
                       out.append(self[start + i * step]);
                   }
                   return out;
               }
               throw py::type_error("Sector indices must be int or slice");
           })
      .def(
        "__iter__",
        [](Sector const& self) {
            return nb::make_iterator(
              nb::type<Sector>(), "_SectorIterator", self.span().begin(), self.span().end());
        },
        nb::keep_alive<0, 1>())
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def("__hash__", [](Sector const& self) { return std::hash<Sector>{}(self); })
      .def("__repr__", &sector_repr)
      .def("__str__", &sector_repr)
      .def("copy", [](Sector const& self) { return Sector(self); })
      .def(
        "to_numpy",
        [](Sector const& self) { return sector_to_ndarray(self); },
        "Return a copy as a 1D ``int64`` NumPy array (``nb::ndarray``).")
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<Sector>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<Sector>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    nb::implicitly_convertible<py::array, Sector>();

    nb::class_<SectorArray>(m, "SectorArray", DOC(lasyten, SectorArray))
      .def(nb::init<>())
      .def(nb::new_([](py::object values) { return sector_array_from_numpy(values); }),
           nb::arg("values"),
           "Construct from a 2D integer sequence or ndarray.")
      .def_static(
        "empty",
        [](int sector_ind_len) {
            return SectorArray::empty(static_cast<std::uint8_t>(sector_ind_len));
        },
        nb::arg("sector_ind_len"))
      .def_static("from_sector", &SectorArray::from_sector, nb::arg("sector"))
      .def_static("repeat", &SectorArray::repeat, nb::arg("sector"), nb::arg("n"))
      .def_prop_ro(
        "shape",
        [](SectorArray const& self) { return py::make_tuple(self.size(), self.sector_ind_len()); })
      .def_prop_ro("num_sectors", [](SectorArray const& self) { return self.size(); })
      .def_prop_ro("sector_ind_len",
                   [](SectorArray const& self) { return static_cast<int>(self.sector_ind_len()); })
      .def("__len__", [](SectorArray const& self) { return self.size(); })
      .def("__getitem__",
           [](SectorArray const& self, py::object key) -> py::object {
               // ``arr[i, :]`` / ``arr[i, ...]`` → treat as row index only
               if (py::isinstance<py::tuple>(key)) {
                   py::tuple t = py::to_tuple(key);
                   if (t.size() == 2 &&
                       (py::isinstance<py::slice>(t[1]) || py::isinstance<py::ellipsis>(t[1]))) {
                       key = t[0];
                   } else if (t.size() == 1) {
                       key = t[0];
                   } else {
                       throw py::type_error(
                         "SectorArray only supports row indexing (int/slice/fancy/bool), "
                         "optionally as arr[rows, :]");
                   }
               }
               auto idx = indices_from_py(key, self.size());
               if (idx.size() == 1 && !py::isinstance<py::slice>(key) &&
                   !py::isinstance<py::array>(key) && !py::isinstance<py::list>(key)) {
                   // scalar index → Sector (including NumPy integer scalars)
                   try {
                       (void)py::cast<std::ptrdiff_t>(key);
                       return py::cast(self[idx[0]]);
                   } catch (py::cast_error const&) {
                       // fall through — e.g. length-1 fancy index
                   }
               }
               return py::cast(self.take(idx));
           })
      .def("__setitem__",
           [](SectorArray& self, py::object key, py::object value) {
               if (py::isinstance<py::tuple>(key)) {
                   py::tuple t = py::to_tuple(key);
                   if (t.size() == 2 &&
                       (py::isinstance<py::slice>(t[1]) || py::isinstance<py::ellipsis>(t[1]))) {
                       key = t[0];
                   } else if (t.size() == 1) {
                       key = t[0];
                   } else {
                       throw py::type_error("SectorArray item assignment: use arr[i] or arr[i,:]");
                   }
               }
               if (py::isinstance<py::int_>(key)) {
                   auto i = py::cast<std::ptrdiff_t>(key);
                   auto const n = static_cast<std::ptrdiff_t>(self.size());
                   if (i < 0) {
                       i += n;
                   }
                   if (i < 0 || i >= n) {
                       throw py::index_error("SectorArray index out of range");
                   }
                   Sector s;
                   if (py::isinstance<Sector>(value)) {
                       s = py::cast<Sector>(value);
                   } else if (py::isinstance<SectorArray>(value)) {
                       auto sa = py::cast<SectorArray>(value);
                       if (sa.size() != 1) {
                           throw py::value_error("assigning multiple rows to a single index");
                       }
                       s = sa[0];
                   } else {
                       s = sector_from_numpy(value);
                   }
                   if (s.len() != self.sector_ind_len()) {
                       throw py::value_error("Sector length mismatch");
                   }
                   self[static_cast<std::size_t>(i)] = s;
                   return;
               }
               // fancy int index assignment of rows
               auto idx = indices_from_py(key, self.size());
               SectorArray src;
               if (py::isinstance<SectorArray>(value)) {
                   src = py::cast<SectorArray>(value);
               } else {
                   src = sector_array_from_numpy(value);
               }
               if (src.size() != idx.size()) {
                   throw py::value_error("SectorArray assignment length mismatch");
               }
               for (std::size_t i = 0; i < idx.size(); ++i) {
                   self[idx[i]] = src[i];
               }
           })
      .def(
        "__iter__",
        [](SectorArray const& self) {
            return nb::make_iterator(
              nb::type<SectorArray>(), "_SectorArrayIterator", self.begin(), self.end());
        },
        nb::keep_alive<0, 1>())
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def("__repr__", &sector_array_repr)
      .def("__str__", &sector_array_repr)
      .def("copy", [](SectorArray const& self) { return SectorArray(self); })
      .def(
        "to_numpy",
        [](SectorArray const& self) { return sector_array_to_ndarray(self); },
        "Return a copy as a 2D ``int64`` NumPy array (``nb::ndarray``).")
      .def("lexsort_indices",
           [](SectorArray const& self) { return indices_to_numpy(self.lexsort_indices()); })
      .def("sorted",
           [](SectorArray const& self) {
               auto [sorted, perm] = self.sorted();
               return py::make_tuple(sorted, indices_to_numpy(perm));
           })
      .def(
        "find_row_differences",
        [](SectorArray const& self, bool include_len) {
            return indices_to_numpy(self.find_row_differences(include_len));
        },
        nb::arg("include_len") = false)
      .def(
        "unique_sorted",
        [](SectorArray const& self, py::object multiplicities) {
            std::vector<std::int64_t> mults;
            if (multiplicities.is_none()) {
                mults.assign(self.size(), 1);
            } else {
                py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> arr =
                  py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(
                    multiplicities);
                auto r = arr.unchecked<1>();
                mults.resize(static_cast<std::size_t>(r.shape(0)));
                for (py::ssize_t i = 0; i < r.shape(0); ++i) {
                    mults[static_cast<std::size_t>(i)] = r(i);
                }
            }
            auto [uniq, um, perm] = self.unique_sorted(mults);
            return py::make_tuple(uniq, i64_to_numpy(um), indices_to_numpy(perm));
        },
        nb::arg("multiplicities").none() = nb::none())
      .def(
        "row_where",
        [](SectorArray const& self, Sector const& sector) -> py::object {
            auto idx = self.row_where(sector);
            if (!idx) {
                return py::none();
            }
            return py::int_(*idx);
        },
        nb::arg("sector"))
      .def("concat", &SectorArray::concat, nb::arg("other"))
      .def(
        "take",
        [](SectorArray const& self, py::object indices) {
            if (py::isinstance<py::array_t<bool>>(indices) ||
                (py::isinstance<py::array>(indices) &&
                 py::array::ensure(indices).request().format == "?")) {
                py::array_t<bool> mask = py::array_t<bool>::ensure(indices);
                auto r = mask.unchecked<1>();
                std::vector<bool> m(static_cast<std::size_t>(r.shape(0)));
                for (py::ssize_t i = 0; i < r.shape(0); ++i) {
                    m[static_cast<std::size_t>(i)] = r(i);
                }
                return self.take_mask(m);
            }
            auto idx = indices_from_py(indices, self.size());
            return self.take(idx);
        },
        nb::arg("indices"))
      .def("slice", &SectorArray::slice, nb::arg("start"), nb::arg("stop"))
      .def_static(
        "iter_common_sorted",
        [](SectorArray const& a, SectorArray const& b, bool a_strict, bool b_strict) {
            py::list out;
            SectorArray::iter_common_sorted(
              a, b, a_strict, b_strict, [&](std::ptrdiff_t i, std::ptrdiff_t j) {
                  out.append(py::make_tuple(i, j));
              });
            return out;
        },
        nb::arg("a"),
        nb::arg("b"),
        nb::arg("a_strict") = true,
        nb::arg("b_strict") = true)
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<SectorArray>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<SectorArray>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    nb::implicitly_convertible<py::array, SectorArray>();
}

} // namespace lasyten
