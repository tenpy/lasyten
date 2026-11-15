#include <lasyten/config.h>

#include "py_lasyten_nanobind.h"
#include "tools/hdf5_bind.h"

namespace py = pybind11;
namespace lasyten {

namespace {

/// Context manager: saves a copy of _global_config on enter, restores on exit.
class TemporaryOptions
{
  public:
    explicit TemporaryOptions(py::dict options)
      : options_(std::move(options))
    {
    }

    TemporaryOptions& enter()
    {
        if (!entered_) {
            saved_ = get_config();
        }
        _global_config.update(options_);
        entered_ = true;
        return *this;
    }

    void exit(py::object /*exc_type*/, py::object /*exc_value*/, py::object /*traceback*/)
    {
        if (entered_) {
            _global_config = saved_;
            entered_ = false;
        }
    }

  private:
    LasytenConfig saved_;
    py::dict options_;
    bool entered_ = false;
};

} // namespace

void
bind_config(nb::module_& m)
{
    nb::class_<LasytenConfig>(m, "LasytenConfig")
      .def(nb::init<>())
      .def_ro("print_linewidth", &LasytenConfig::print_linewidth)
      .def_ro("print_indent", &LasytenConfig::print_indent)
      .def_ro("maxlines_spaces", &LasytenConfig::maxlines_spaces)
      .def_ro("maxlines_tensors", &LasytenConfig::maxlines_tensors)
      .def_ro("check_fusion", &LasytenConfig::check_fusion)
      .def_ro("implicit_scalar_conversion", &LasytenConfig::implicit_scalar_conversion)
      .def_ro("default_tensor_backend", &LasytenConfig::default_tensor_backend)
      .def_ro("default_block_backend", &LasytenConfig::default_block_backend)
      .def_ro("fusion_tree_eps", &LasytenConfig::fusion_tree_eps)
      .def_ro("su_n_data_path", &LasytenConfig::su_n_data_path)
      .def_ro("su_n_data_filename_base", &LasytenConfig::su_n_data_filename_base)
      .def_ro("su_n_symbol_cache_mb", &LasytenConfig::su_n_symbol_cache_mb)
      .def_ro("tree_plan_cache_mb", &LasytenConfig::tree_plan_cache_mb)
      .def_ro("tree_size_memo_mb", &LasytenConfig::tree_size_memo_mb)
      .def_ro("su_n_packed_ram_mb", &LasytenConfig::su_n_packed_ram_mb)
      .def_ro("su2_cg_hweight", &LasytenConfig::su2_cg_hweight)
      .def_ro("su2_f_hweight", &LasytenConfig::su2_f_hweight)
      .def_ro("su2_r_hweight", &LasytenConfig::su2_r_hweight)
      .def_ro("su_n_hweights", &LasytenConfig::su_n_hweights)
      .def_ro("coupling_cutoff", &LasytenConfig::coupling_cutoff)
      .def_static("all_option_keys",
                  &LasytenConfig::all_option_keys,
                  "Names of all recognized config options")
      .def_static("env_var_name",
                  &LasytenConfig::env_var_name,
                  nb::arg("key"),
                  "Environment variable name for a config option")
      .def("set_option",
           nb::overload_cast<const std::string&, bool>(&LasytenConfig::set_option),
           nb::arg("key"),
           nb::arg("value"))
      .def("set_option",
           nb::overload_cast<const std::string&, int64>(&LasytenConfig::set_option),
           nb::arg("key"),
           nb::arg("value"))
      .def("set_option",
           nb::overload_cast<const std::string&, float64>(&LasytenConfig::set_option),
           nb::arg("key"),
           nb::arg("value"))
      .def("set_option",
           nb::overload_cast<const std::string&, const std::string&>(&LasytenConfig::set_option),
           nb::arg("key"),
           nb::arg("value"))
      .def("update", nb::overload_cast<py::dict>(&LasytenConfig::update), nb::arg("options"))
      .def("update",
           nb::overload_cast<const LasytenConfig&>(&LasytenConfig::update),
           nb::arg("other"))
      .def("update_from_env", &LasytenConfig::update_from_env)
      .def(
        "update_from_yaml",
        [](LasytenConfig& self, std::string const& yaml_text) {
            // YAML parsing stays in the Python binding layer (PyYAML); C++ accepts mappings.
            py::module_ yaml = py::module_::import("yaml");
            py::object data = yaml.attr("safe_load")(yaml_text);
            if (data.is_none()) {
                return;
            }
            if (!py::isinstance<py::dict>(data)) {
                throw py::type_error("Config must contain a mapping");
            }
            self.update_from_mapping(py::reinterpret_borrow<py::dict>(data));
        },
        nb::arg("yaml_text"))
      .def("update_from_file", &LasytenConfig::update_from_file, nb::arg("filename"))
      .def("get_option", &LasytenConfig::get_option, nb::arg("key"))
      .def("str", &LasytenConfig::str)
      .def("__str__", &LasytenConfig::str)
      .def("__repr__", &LasytenConfig::str)
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<LasytenConfig>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"),
           "Export config to hdf5 such that it can be re-imported with from_hdf5")
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<LasytenConfig>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"),
                  "Load config from hdf5");

    nb::class_<TemporaryOptions>(m, "TemporaryOptions")
      .def(nb::init<py::dict>(), nb::arg("options"))
      .def("__enter__", &TemporaryOptions::enter)
      .def("__exit__",
           &TemporaryOptions::exit,
           nb::arg("exc_type").none(),
           nb::arg("exc_value").none(),
           nb::arg("traceback").none());

    m.def(
      "get_config", &get_config, nb::rv_policy::reference, "Get the global configuration object");
    m.def("get_option", &get_option, nb::arg("key"), "Get a config option by name");
    m.def(
      "set_option",
      [](const std::string& key, py::handle value) { set_option(key, value); },
      nb::arg("key"),
      nb::arg("value"),
      "Set a single config option on the global config");
    m.def(
      "set_options",
      [](nb::kwargs kwargs_nb) {
          auto kwargs = pykwargs(kwargs_nb);
          set_options(py::dict(kwargs));
      },
      "Set config options on the global config");
    m.def(
      "temporary_options",
      [](nb::kwargs kwargs_nb) {
          auto kwargs = pykwargs(kwargs_nb);
          return TemporaryOptions(py::dict(kwargs));
      },
      "Context manager for temporary config overrides");
    m.def("restore_defaults",
          &restore_defaults,
          nb::arg("use_user_file") = true,
          nb::arg("use_local_file") = true,
          nb::arg("use_env_vars") = true,
          "Reset config to defaults and optionally reload files/env");
}

} // namespace lasyten
