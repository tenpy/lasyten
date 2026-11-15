#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "docstrings/backends/block_inds.h"

#include "backends/casters.hpp"

#include <lasyten/backends/block_inds.h>
#include <lasyten/backends/block_inds_numpy.h>

#include "tools/hdf5_bind.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// g_block_inds is defined below (declare_block_inds / bind_block_inds).

namespace lasyten {

namespace {

py::array_t<std::size_t>
indices_to_numpy(std::vector<std::size_t> const& v)
{
    py::array_t<std::size_t> arr(static_cast<py::ssize_t>(v.size()));
    if (!v.empty()) {
        std::copy(v.begin(), v.end(), arr.mutable_data());
    }
    return arr;
}

std::vector<std::size_t>
indices_from_py(py::handle key, std::size_t n_rows)
{
    try {
        if (!py::isinstance<py::slice>(key) && !py::isinstance<py::tuple>(key) &&
            !py::isinstance<py::list>(key) && !py::isinstance<py::array>(key)) {
            auto i = key.cast<std::ptrdiff_t>();
            if (i < 0) {
                i += static_cast<std::ptrdiff_t>(n_rows);
            }
            if (i < 0 || static_cast<std::size_t>(i) >= n_rows) {
                throw py::index_error("BlockInds index out of range");
            }
            return { static_cast<std::size_t>(i) };
        }
    } catch (py::cast_error const&) {
    }
    if (py::isinstance<py::slice>(key)) {
        auto sl = key.cast<py::slice>();
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
    py::array arr = py::array::ensure(key);
    if (!arr) {
        throw py::type_error("BlockInds indices must be int, slice, or array");
    }
    if (arr.ndim() == 0) {
        auto i = arr.cast<std::ptrdiff_t>();
        if (i < 0) {
            i += static_cast<std::ptrdiff_t>(n_rows);
        }
        if (i < 0 || static_cast<std::size_t>(i) >= n_rows) {
            throw py::index_error("BlockInds index out of range");
        }
        return { static_cast<std::size_t>(i) };
    }
    if (arr.ndim() != 1) {
        throw py::type_error("BlockInds fancy index must be 1D");
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
            throw py::index_error("BlockInds index out of range");
        }
        out[static_cast<std::size_t>(i)] = static_cast<std::size_t>(v);
    }
    return out;
}

std::vector<std::size_t>
col_indices_from_slice(py::slice sl, std::size_t ncols)
{
    size_t start = 0, stop = 0, step = 0, slicelength = 0;
    if (!sl.compute(ncols, &start, &stop, &step, &slicelength)) {
        throw py::error_already_set();
    }
    std::vector<std::size_t> out(slicelength);
    for (size_t i = 0; i < slicelength; ++i) {
        out[i] = start + i * step;
    }
    return out;
}

std::string
block_inds_repr(BlockInds const& a)
{
    std::ostringstream oss;
    oss << "BlockInds(shape=(" << a.nrows() << ", " << a.ncols() << "))";
    return oss.str();
}

std::optional<nb::class_<BlockInds>> g_block_inds;

} // namespace

void
declare_block_inds(nb::module_& m)
{
    g_block_inds = nb::class_<BlockInds>(m, "BlockInds", DOC(lasyten, BlockInds));
}

void
bind_block_inds(nb::module_& m)
{
    (void)m;
    auto& cls = *g_block_inds;
    cls.def(lasyten::ctor<decltype(cls)>())
      .def(nb::new_([](py::object values) { return block_inds_from_numpy(values); }),
           nb::arg("values"),
           "Construct from a 2D integer sequence or ndarray.")
      .def_static(
        "empty", [](std::size_t ncols) { return BlockInds::empty(ncols); }, nb::arg("ncols"))
      .def_static(
        "zeros",
        [](std::size_t nrows, std::size_t ncols) { return BlockInds::zeros(nrows, ncols); },
        nb::arg("nrows"),
        nb::arg("ncols"))
      .def_prop_ro(
        "shape", [](BlockInds const& self) { return py::make_tuple(self.nrows(), self.ncols()); })
      .def_prop_ro("nrows", &BlockInds::nrows)
      .def_prop_ro("ncols", &BlockInds::ncols)
      .def_prop_ro("ndim", [](BlockInds const&) { return 2; })
      .def("__len__", [](BlockInds const& self) { return self.nrows(); })
      .def("__repr__", &block_inds_repr)
      .def("__str__", &block_inds_repr)
      // Element-wise like ndarray; do not return a scalar bool (breaks broadcasting).
      .def("__eq__",
           [](BlockInds const& self, py::object other) -> py::object {
               py::object self_np = block_inds_to_numpy(self);
               if (lasyten::py_isinstance<BlockInds>(other)) {
                   other = block_inds_to_numpy(other.cast<BlockInds>());
               }
               return self_np.attr("__eq__")(other);
           })
      .def("__ne__",
           [](BlockInds const& self, py::object other) -> py::object {
               py::object self_np = block_inds_to_numpy(self);
               if (lasyten::py_isinstance<BlockInds>(other)) {
                   other = block_inds_to_numpy(other.cast<BlockInds>());
               }
               return self_np.attr("__ne__")(other);
           })
      .def(
        "__array__",
        [](BlockInds const& self, py::object /*dtype*/, py::object /*copy*/) {
            return block_inds_to_numpy(self);
        },
        nb::arg("dtype").none() = nb::none(),
        nb::arg("copy").none() = nb::none())
      .def("to_numpy", &block_inds_to_numpy)
      .def("lexsort_indices",
           [](BlockInds const& self) { return indices_to_numpy(self.lexsort_indices()); })
      .def("sorted",
           [](BlockInds const& self) {
               auto [sorted, perm] = self.sorted();
               return py::make_tuple(std::move(sorted), indices_to_numpy(perm));
           })
      .def(
        "find_row_differences",
        [](BlockInds const& self, bool include_len) {
            return indices_to_numpy(self.find_row_differences(include_len));
        },
        nb::arg("include_len") = false)
      .def(
        "take",
        [](BlockInds const& self, py::object indices) {
            return self.take(indices_from_py(indices, self.nrows()));
        },
        nb::arg("indices"))
      .def("concat", &BlockInds::concat, nb::arg("other"))
      .def("reverse_columns", &BlockInds::reverse_columns)
      .def("repeat_columns", &BlockInds::repeat_columns, nb::arg("times"))
      .def("insert_column", &BlockInds::insert_column, nb::arg("col"), nb::arg("fill") = 0)
      .def("__getitem__",
           [](BlockInds const& self, py::object key) -> py::object {
               // ``arr[:, ::-1]`` / ``arr[rows, cols]``
               if (py::isinstance<py::tuple>(key)) {
                   py::tuple t = key.cast<py::tuple>();
                   if (t.size() != 2) {
                       throw py::type_error("BlockInds 2D indexing expects two indices");
                   }
                   // column-only slice with full rows: arr[:, cols]
                   bool rows_all = py::isinstance<py::slice>(t[0]);
                   if (rows_all) {
                       auto sl = t[0].cast<py::slice>();
                       size_t start = 0, stop = 0, step = 0, slicelength = 0;
                       if (!sl.compute(self.nrows(), &start, &stop, &step, &slicelength)) {
                           throw py::error_already_set();
                       }
                       if (!(start == 0 && stop == self.nrows() && step == 1 &&
                             slicelength == self.nrows())) {
                           // general row slice + column selector
                           auto rows = indices_from_py(t[0], self.nrows());
                           BlockInds taken = self.take(rows);
                           if (py::isinstance<py::slice>(t[1]) ||
                               py::isinstance<py::ellipsis>(t[1])) {
                               if (py::isinstance<py::ellipsis>(t[1])) {
                                   return py::cast(taken);
                               }
                               return py::cast(taken.take_columns(
                                 col_indices_from_slice(t[1].cast<py::slice>(), taken.ncols())));
                           }
                           if (py::isinstance<py::int_>(t[1])) {
                               auto c = t[1].cast<std::ptrdiff_t>();
                               if (c < 0) {
                                   c += static_cast<std::ptrdiff_t>(taken.ncols());
                               }
                               auto col = taken.column(static_cast<std::size_t>(c));
                               py::array_t<int64> arr(static_cast<py::ssize_t>(col.size()));
                               if (!col.empty()) {
                                   std::copy(col.begin(), col.end(), arr.mutable_data());
                               }
                               return arr;
                           }
                           throw py::type_error("unsupported BlockInds column index");
                       }
                   }
                   if (py::isinstance<py::slice>(t[0]) || py::isinstance<py::ellipsis>(t[0])) {
                       // full row range: column select only
                       if (py::isinstance<py::ellipsis>(t[1]) ||
                           (py::isinstance<py::slice>(t[1]) && [&] {
                               auto sl = t[1].cast<py::slice>();
                               size_t start = 0, stop = 0, step = 0, slicelength = 0;
                               sl.compute(self.ncols(), &start, &stop, &step, &slicelength);
                               return start == 0 && stop == self.ncols() && step == 1;
                           }())) {
                           return py::cast(self);
                       }
                       if (py::isinstance<py::slice>(t[1])) {
                           return py::cast(self.take_columns(
                             col_indices_from_slice(t[1].cast<py::slice>(), self.ncols())));
                       }
                       if (py::isinstance<py::int_>(t[1])) {
                           auto c = t[1].cast<std::ptrdiff_t>();
                           if (c < 0) {
                               c += static_cast<std::ptrdiff_t>(self.ncols());
                           }
                           auto col = self.column(static_cast<std::size_t>(c));
                           py::array_t<int64> arr(static_cast<py::ssize_t>(col.size()));
                           if (!col.empty()) {
                               std::copy(col.begin(), col.end(), arr.mutable_data());
                           }
                           return arr;
                       }
                   }
                   // scalar element: arr[i, j]
                   if (py::isinstance<py::int_>(t[0]) && py::isinstance<py::int_>(t[1])) {
                       auto r = t[0].cast<std::ptrdiff_t>();
                       auto c = t[1].cast<std::ptrdiff_t>();
                       if (r < 0) {
                           r += static_cast<std::ptrdiff_t>(self.nrows());
                       }
                       if (c < 0) {
                           c += static_cast<std::ptrdiff_t>(self.ncols());
                       }
                       if (r < 0 || static_cast<std::size_t>(r) >= self.nrows() || c < 0 ||
                           static_cast<std::size_t>(c) >= self.ncols()) {
                           throw py::index_error("BlockInds index out of range");
                       }
                       return py::int_(
                         self(static_cast<std::size_t>(r), static_cast<std::size_t>(c)));
                   }
                   // row fancy + optional full columns / column slice
                   if (py::isinstance<py::slice>(t[1]) || py::isinstance<py::ellipsis>(t[1])) {
                       auto rows = indices_from_py(t[0], self.nrows());
                       BlockInds taken = self.take(rows);
                       if (py::isinstance<py::ellipsis>(t[1])) {
                           // single row → 1D numpy, matching ndarray[i, ...] / ndarray[i, :]
                           if (rows.size() == 1 && py::isinstance<py::int_>(t[0])) {
                               auto row = taken.row(0);
                               py::array_t<int64> arr(static_cast<py::ssize_t>(row.size()));
                               if (!row.empty()) {
                                   std::copy(row.begin(), row.end(), arr.mutable_data());
                               }
                               return arr;
                           }
                           return py::cast(taken);
                       }
                       auto cols = col_indices_from_slice(t[1].cast<py::slice>(), taken.ncols());
                       // full column range on a single integer row → 1D numpy
                       if (rows.size() == 1 && py::isinstance<py::int_>(t[0]) &&
                           cols.size() == taken.ncols()) {
                           auto row = taken.row(0);
                           py::array_t<int64> arr(static_cast<py::ssize_t>(row.size()));
                           if (!row.empty()) {
                               std::copy(row.begin(), row.end(), arr.mutable_data());
                           }
                           return arr;
                       }
                       return py::cast(taken.take_columns(cols));
                   }
                   // single row + single column already handled; fancy/int row + columns
                   if (py::isinstance<py::int_>(t[1])) {
                       auto rows = indices_from_py(t[0], self.nrows());
                       BlockInds taken = self.take(rows);
                       auto c = t[1].cast<std::ptrdiff_t>();
                       if (c < 0) {
                           c += static_cast<std::ptrdiff_t>(taken.ncols());
                       }
                       if (c < 0 || static_cast<std::size_t>(c) >= taken.ncols()) {
                           throw py::index_error("BlockInds index out of range");
                       }
                       auto col = taken.column(static_cast<std::size_t>(c));
                       if (rows.size() == 1 && py::isinstance<py::int_>(t[0])) {
                           return py::int_(col[0]);
                       }
                       py::array_t<int64> arr(static_cast<py::ssize_t>(col.size()));
                       if (!col.empty()) {
                           std::copy(col.begin(), col.end(), arr.mutable_data());
                       }
                       return arr;
                   }
                   if (py::isinstance<py::int_>(t[0])) {
                       auto rows = indices_from_py(t[0], self.nrows());
                       BlockInds taken = self.take(rows);
                       auto cols = indices_from_py(t[1], taken.ncols());
                       if (cols.size() == 1 && py::isinstance<py::int_>(t[1])) {
                           return py::int_(taken(0, cols[0]));
                       }
                       return py::cast(taken.take_columns(cols));
                   }
                   // fancy rows + fancy columns
                   {
                       auto rows = indices_from_py(t[0], self.nrows());
                       BlockInds taken = self.take(rows);
                       auto cols = indices_from_py(t[1], taken.ncols());
                       return py::cast(taken.take_columns(cols));
                   }
               }
               auto idx = indices_from_py(key, self.nrows());
               // scalar → 1D numpy row (matches ndarray[i] for 2D array)
               if (idx.size() == 1 && !py::isinstance<py::slice>(key) &&
                   !py::isinstance<py::array>(key) && !py::isinstance<py::list>(key)) {
                   try {
                       (void)key.cast<std::ptrdiff_t>();
                       auto row = self.row(idx[0]);
                       py::array_t<int64> arr(static_cast<py::ssize_t>(row.size()));
                       if (!row.empty()) {
                           std::copy(row.begin(), row.end(), arr.mutable_data());
                       }
                       return arr;
                   } catch (py::cast_error const&) {
                   }
               }
               return py::cast(self.take(idx));
           })
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<BlockInds>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<BlockInds>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    nb::implicitly_convertible<py::array, BlockInds>();
}

} // namespace lasyten
