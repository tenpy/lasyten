#include <lasyten/config.h>

#include "py_lasyten_pybind11.h"
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
bind_config(py::module_& m)
{
    py::class_<LasytenConfig>(m, "LasytenConfig")
      .def(py::init<>())
      .def_readonly("print_linewidth", &LasytenConfig::print_linewidth)
      .def_readonly("print_indent", &LasytenConfig::print_indent)
      .def_readonly("maxlines_spaces", &LasytenConfig::maxlines_spaces)
      .def_readonly("maxlines_tensors", &LasytenConfig::maxlines_tensors)
      .def_readonly("check_fusion", &LasytenConfig::check_fusion)
      .def_readonly("implicit_scalar_conversion", &LasytenConfig::implicit_scalar_conversion)
      .def_readonly("default_tensor_backend", &LasytenConfig::default_tensor_backend)
      .def_readonly("default_block_backend", &LasytenConfig::default_block_backend)
      .def_readonly("fusion_tree_eps", &LasytenConfig::fusion_tree_eps)
      .def_readonly("su_n_data_path", &LasytenConfig::su_n_data_path)
      .def_readonly("su_n_data_filename_base", &LasytenConfig::su_n_data_filename_base)
      .def_readonly("su_n_symbol_cache_mb", &LasytenConfig::su_n_symbol_cache_mb)
      .def_readonly("tree_plan_cache_mb", &LasytenConfig::tree_plan_cache_mb)
      .def_readonly("tree_size_memo_mb", &LasytenConfig::tree_size_memo_mb)
      .def_readonly("su_n_packed_ram_mb", &LasytenConfig::su_n_packed_ram_mb)
      .def_readonly("su2_cg_hweight", &LasytenConfig::su2_cg_hweight)
      .def_readonly("su2_f_hweight", &LasytenConfig::su2_f_hweight)
      .def_readonly("su2_r_hweight", &LasytenConfig::su2_r_hweight)
      .def_readonly("su_n_hweights", &LasytenConfig::su_n_hweights)
      .def_readonly("coupling_cutoff", &LasytenConfig::coupling_cutoff)
      .def_static("all_option_keys",
                  &LasytenConfig::all_option_keys,
                  "Names of all recognized config options")
      .def_static("env_var_name",
                  &LasytenConfig::env_var_name,
                  py::arg("key"),
                  "Environment variable name for a config option")
      .def("set_option",
           py::overload_cast<const std::string&, bool>(&LasytenConfig::set_option),
           py::arg("key"),
           py::arg("value"))
      .def("set_option",
           py::overload_cast<const std::string&, int64>(&LasytenConfig::set_option),
           py::arg("key"),
           py::arg("value"))
      .def("set_option",
           py::overload_cast<const std::string&, float64>(&LasytenConfig::set_option),
           py::arg("key"),
           py::arg("value"))
      .def("set_option",
           py::overload_cast<const std::string&, const std::string&>(&LasytenConfig::set_option),
           py::arg("key"),
           py::arg("value"))
      .def("update", py::overload_cast<py::dict>(&LasytenConfig::update), py::arg("options"))
      .def("update",
           py::overload_cast<const LasytenConfig&>(&LasytenConfig::update),
           py::arg("other"))
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
        py::arg("yaml_text"))
      .def("update_from_file", &LasytenConfig::update_from_file, py::arg("filename"))
      .def("get_option", &LasytenConfig::get_option, py::arg("key"))
      .def("str", &LasytenConfig::str)
      .def("__str__", &LasytenConfig::str)
      .def("__repr__", &LasytenConfig::str)
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<LasytenConfig>(),
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"),
           "Export config to hdf5 such that it can be re-imported with from_hdf5")
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<LasytenConfig>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"),
                  "Load config from hdf5");

    py::class_<TemporaryOptions>(m, "TemporaryOptions")
      .def(py::init<py::dict>(), py::arg("options"))
      .def("__enter__", &TemporaryOptions::enter)
      .def("__exit__", &TemporaryOptions::exit);

    m.def("get_config",
          &get_config,
          py::return_value_policy::reference,
          "Get the global configuration object");
    m.def("get_option", &get_option, py::arg("key"), "Get a config option by name");
    m.def(
      "set_option",
      [](const std::string& key, py::handle value) { set_option(key, value); },
      py::arg("key"),
      py::arg("value"),
      "Set a single config option on the global config");
    m.def(
      "set_options",
      [](py::kwargs kwargs) { set_options(py::dict(kwargs)); },
      "Set config options on the global config");
    m.def(
      "temporary_options",
      [](py::kwargs kwargs) { return TemporaryOptions(py::dict(kwargs)); },
      "Context manager for temporary config overrides");
    m.def("restore_defaults",
          &restore_defaults,
          py::arg("use_user_file") = true,
          py::arg("use_local_file") = true,
          py::arg("use_env_vars") = true,
          "Reset config to defaults and optionally reload files/env");
}

} // namespace lasyten
