#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/sun.h"
#include "py_lasyten_nanobind.h"

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
        return py::cast<std::string>(obj);
    }
    // Open h5py.File (or anything with a .filename attribute).
    return py::cast<std::string>(py::str(obj.attr("filename")));
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
bind_sun(nb::module_& m)
{
    nb::class_<SUN, Group> cls(m, "SUN", DOC(lasyten, SUN));

    cls.def(nb::new_([](int N,
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
            nb::arg("N"),
            nb::arg("hweight").none() = nb::none(),
            nb::kw_only(),
            nb::arg("cg_hweight").none() = nb::none(),
            nb::arg("f_hweight").none() = nb::none(),
            nb::arg("r_hweight").none() = nb::none(),
            nb::arg("path").none() = nb::none(),
            nb::arg("filename_base").none() = nb::none(),
            nb::arg("descriptive_name").none() = nb::none());

    cls.def(nb::new_([](int N,
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
            nb::arg("N"),
            nb::arg("CGfile"),
            nb::arg("Ffile"),
            nb::arg("Rfile"),
            nb::arg("descriptive_name").none() = nb::none());

    m.def("su_n_data_filename",
          &su_n_data_filename,
          nb::arg("N"),
          nb::arg("kind"),
          nb::arg("hweight"),
          nb::arg("filename_base").none() = nb::none(),
          DOC(lasyten, su_n_data_filename));

    m.def("su_n_data_file_path",
          &su_n_data_file_path,
          nb::arg("N"),
          nb::arg("kind"),
          nb::arg("hweight"),
          nb::arg("path").none() = nb::none(),
          nb::arg("filename_base").none() = nb::none(),
          DOC(lasyten, su_n_data_file_path));

    cls.def_ro("N", &SUN::N)
      .def_ro("CGpath", &SUN::CGpath)
      .def_ro("Fpath", &SUN::Fpath)
      .def_ro("Rpath", &SUN::Rpath)
      // Back-compat aliases for the former h5py handle attributes (expose paths).
      .def_prop_ro("CGfile", [](SUN const& self) { return py::str(self.CGpath); })
      .def_prop_ro("Ffile", [](SUN const& self) { return py::str(self.Fpath); })
      .def_prop_ro("Rfile", [](SUN const& self) { return py::str(self.Rpath); })
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<SUN>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    cls.def("hweight_from_CG_hdf5", &SUN::hweight_from_CG_hdf5)
      .def("hweight_from_F_hdf5", &SUN::hweight_from_F_hdf5)
      .def("hweight_from_R_hdf5", &SUN::hweight_from_R_hdf5)
      .def("S_index_irrep_weight",
           &SUN::S_index_irrep_weight,
           nb::arg("a"),
           DOC(lasyten, SUN, S_index_irrep_weight))
      .def("highest_irrep_in_decomp",
           &SUN::highest_irrep_in_decomp,
           nb::arg("a"),
           nb::arg("b"),
           DOC(lasyten, SUN, highest_irrep_in_decomp))
      .def(
        "dims_of_irreps",
        [](SUN const& self, Sector a, Sector b) {
            return sector_int_map_to_py(self.dims_of_irreps(a, b));
        },
        nb::arg("a"),
        nb::arg("b"),
        DOC(lasyten, SUN, dims_of_irreps))
      .def(
        "outer_multiplicity_from_CG",
        [](SUN const& self, Sector a, Sector b) {
            return sector_int_map_to_py(self.outer_multiplicity_from_CG(a, b));
        },
        nb::arg("a"),
        nb::arg("b"),
        DOC(lasyten, SUN, outer_multiplicity_from_CG))
      .def("clebschgordan",
           &SUN::clebschgordan,
           nb::arg("a"),
           nb::arg("q_a"),
           nb::arg("b"),
           nb::arg("q_b"),
           nb::arg("c"),
           nb::arg("q_c"),
           nb::arg("mu"),
           DOC(lasyten, SUN, clebschgordan))
      .def("_f_symbol_from_CG",
           &SUN::_f_symbol_from_CG,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           nb::arg("d"),
           nb::arg("e"),
           nb::arg("f"),
           DOC(lasyten, SUN, _f_symbol_from_CG))
      .def("_r_symbol_from_CG",
           &SUN::_r_symbol_from_CG,
           nb::arg("a"),
           nb::arg("b"),
           nb::arg("c"),
           DOC(lasyten, SUN, _r_symbol_from_CG))
      .def(
        "has_data_in_group",
        [](SUN const& self, py::object group) {
            // Accept HighFive-wrapped or h5py objects via .id.id / .id
            hid_t hid = -1;
            if (py::hasattr(group, "id")) {
                py::object idobj = group.attr("id");
                if (py::hasattr(idobj, "id")) {
                    hid = py::cast<hid_t>(idobj.attr("id"));
                } else {
                    hid = py::cast<hid_t>(idobj);
                }
            } else {
                throw py::type_error("has_data_in_group expects an h5py Group/Dataset");
            }
            return self.has_data_in_group(hid);
        },
        nb::arg("group"))
      .def(
        "sanity_check_hdf5",
        [](SUN const& self, py::object file) {
            auto path = path_from_file_or_str(file);
            HighFive::File hf(path, HighFive::File::ReadOnly);
            self.sanity_check_hdf5(hf);
        },
        nb::arg("file"),
        DOC(lasyten, SUN, sanity_check_hdf5));
}

} // namespace lasyten
