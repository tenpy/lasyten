#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/sun.h"
#include "py_lasyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/sun.h>

#include "tools/hdf5_bind.h"
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace lasyten {

namespace {

std::string
path_from_file_or_str(py::handle obj)
{
    if (py::isinstance<py::str>(obj)) {
        return obj.cast<std::string>();
    }
    // Open h5py.File (or anything with a .filename attribute).
    return py::str(obj.attr("filename")).cast<std::string>();
}

py::dict
sector_int_map_to_py(std::map<Sector, int64> const& m)
{
    py::dict out;
    for (auto const& [sec, val] : m) {
        py::tuple k(sec.len());
        for (std::uint8_t j = 0; j < sec.len(); ++j) {
            k[j] = sec.q[j];
        }
        out[k] = val;
    }
    return out;
}

} // namespace

void
bind_sun(py::module_& m)
{
    py::class_<SUN, Group, py::smart_holder> cls(m, "SUN", DOC(lasyten, SUN));

    cls.def(py::init([](int N,
                        std::optional<int64> hweight,
                        std::optional<int64> cg_hweight,
                        std::optional<int64> f_hweight,
                        std::optional<int64> r_hweight,
                        std::optional<std::string> path,
                        std::optional<std::string> filename_base,
                        std::optional<std::string> descriptive_name) {
                return SUN::from_config(N,
                                        hweight,
                                        cg_hweight,
                                        f_hweight,
                                        r_hweight,
                                        std::move(path),
                                        std::move(filename_base),
                                        std::move(descriptive_name));
            }),
            py::arg("N"),
            py::arg("hweight") = py::none(),
            py::kw_only(),
            py::arg("cg_hweight") = py::none(),
            py::arg("f_hweight") = py::none(),
            py::arg("r_hweight") = py::none(),
            py::arg("path") = py::none(),
            py::arg("filename_base") = py::none(),
            py::arg("descriptive_name") = py::none());

    cls.def(py::init([](int N,
                        py::object CGfile,
                        py::object Ffile,
                        py::object Rfile,
                        std::optional<std::string> descriptive_name) {
                return std::make_shared<SUN>(N,
                                             path_from_file_or_str(CGfile),
                                             path_from_file_or_str(Ffile),
                                             path_from_file_or_str(Rfile),
                                             std::move(descriptive_name));
            }),
            py::arg("N"),
            py::arg("CGfile"),
            py::arg("Ffile"),
            py::arg("Rfile"),
            py::arg("descriptive_name") = py::none());

    m.def("su_n_data_filename",
          &su_n_data_filename,
          py::arg("N"),
          py::arg("kind"),
          py::arg("hweight"),
          py::arg("filename_base") = py::none(),
          DOC(lasyten, su_n_data_filename));

    m.def("su_n_data_file_path",
          &su_n_data_file_path,
          py::arg("N"),
          py::arg("kind"),
          py::arg("hweight"),
          py::arg("path") = py::none(),
          py::arg("filename_base") = py::none(),
          DOC(lasyten, su_n_data_file_path));

    cls.def_readonly("N", &SUN::N)
      .def_readonly("CGpath", &SUN::CGpath)
      .def_readonly("Fpath", &SUN::Fpath)
      .def_readonly("Rpath", &SUN::Rpath)
      // Back-compat aliases for the former h5py handle attributes (expose paths).
      .def_property_readonly("CGfile", [](SUN const& self) { return py::str(self.CGpath); })
      .def_property_readonly("Ffile", [](SUN const& self) { return py::str(self.Fpath); })
      .def_property_readonly("Rfile", [](SUN const& self) { return py::str(self.Rpath); })
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<SUN>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));

    cls.def("hweight_from_CG_hdf5", &SUN::hweight_from_CG_hdf5)
      .def("hweight_from_F_hdf5", &SUN::hweight_from_F_hdf5)
      .def("hweight_from_R_hdf5", &SUN::hweight_from_R_hdf5)
      .def("S_index_irrep_weight",
           &SUN::S_index_irrep_weight,
           py::arg("a"),
           DOC(lasyten, SUN, S_index_irrep_weight))
      .def("highest_irrep_in_decomp",
           &SUN::highest_irrep_in_decomp,
           py::arg("a"),
           py::arg("b"),
           DOC(lasyten, SUN, highest_irrep_in_decomp))
      .def(
        "dims_of_irreps",
        [](SUN const& self, Sector a, Sector b) {
            return sector_int_map_to_py(self.dims_of_irreps(a, b));
        },
        py::arg("a"),
        py::arg("b"),
        DOC(lasyten, SUN, dims_of_irreps))
      .def(
        "outer_multiplicity_from_CG",
        [](SUN const& self, Sector a, Sector b) {
            return sector_int_map_to_py(self.outer_multiplicity_from_CG(a, b));
        },
        py::arg("a"),
        py::arg("b"),
        DOC(lasyten, SUN, outer_multiplicity_from_CG))
      .def("clebschgordan",
           &SUN::clebschgordan,
           py::arg("a"),
           py::arg("q_a"),
           py::arg("b"),
           py::arg("q_b"),
           py::arg("c"),
           py::arg("q_c"),
           py::arg("mu"),
           DOC(lasyten, SUN, clebschgordan))
      .def("_f_symbol_from_CG",
           &SUN::_f_symbol_from_CG,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           DOC(lasyten, SUN, _f_symbol_from_CG))
      .def("_r_symbol_from_CG",
           &SUN::_r_symbol_from_CG,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           DOC(lasyten, SUN, _r_symbol_from_CG))
      .def(
        "has_data_in_group",
        [](SUN const& self, py::object group) {
            // Accept HighFive-wrapped or h5py objects via .id.id / .id
            hid_t hid = -1;
            if (py::hasattr(group, "id")) {
                py::object idobj = group.attr("id");
                if (py::hasattr(idobj, "id")) {
                    hid = idobj.attr("id").cast<hid_t>();
                } else {
                    hid = idobj.cast<hid_t>();
                }
            } else {
                throw py::type_error("has_data_in_group expects an h5py Group/Dataset");
            }
            return self.has_data_in_group(hid);
        },
        py::arg("group"))
      .def(
        "sanity_check_hdf5",
        [](SUN const& self, py::object file) {
            auto path = path_from_file_or_str(file);
            HighFive::File hf(path, HighFive::File::ReadOnly);
            self.sanity_check_hdf5(hf);
        },
        py::arg("file"),
        DOC(lasyten, SUN, sanity_check_hdf5));
}

} // namespace lasyten
