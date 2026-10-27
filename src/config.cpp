#include <algorithm>
#include <lasyten/config.h>
#include <lasyten/tools/warn.h>

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <hdf5_io/h5_ops.h>
#include <lasyten/tools/hdf5.h>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lasyten {

namespace {

namespace fs = std::filesystem;

bool g_config_initialized = false;

std::string
to_upper(std::string s)
{
    for (char& c : s)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

bool
coerce_bool(const std::string& value)
{
    std::string lower = value;
    for (char& c : lower)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return lower == "true" || lower == "1" || lower == "y" || lower == "yes";
}

int64
parse_int64(const std::string& value)
{
    try {
        size_t idx = 0;
        long long v = std::stoll(value, &idx);
        if (idx != value.size())
            throw std::invalid_argument("trailing characters");
        return static_cast<int64>(v);
    } catch (const std::exception& e) {
        throw py::value_error(std::string("Invalid integer config value: ") + value + " (" +
                              e.what() + ")");
    }
}

float64
parse_float64(const std::string& value)
{
    try {
        size_t idx = 0;
        double v = std::stod(value, &idx);
        if (idx != value.size())
            throw std::invalid_argument("trailing characters");
        return static_cast<float64>(v);
    } catch (const std::exception& e) {
        throw py::value_error(std::string("Invalid float config value: ") + value + " (" +
                              e.what() + ")");
    }
}

void
check_min(int64 value, int64 min_value, const std::string& key)
{
    if (value < min_value) {
        throw py::value_error("Config option '" + key + "' must be >= " +
                              std::to_string(min_value) + ", got " + std::to_string(value));
    }
}

void
check_min(float64 value, float64 min_value, const std::string& key)
{
    if (value < min_value) {
        throw py::value_error("Config option '" + key + "' must be >= " +
                              std::to_string(min_value) + ", got " + std::to_string(value));
    }
}

bool
is_allowed(const std::string& value, const std::vector<std::string>& allowed)
{
    for (const auto& a : allowed) {
        if (value == a)
            return true;
    }
    return false;
}

std::string
home_directory()
{
    if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0')
        return home;
#ifdef _WIN32
    if (const char* profile = std::getenv("USERPROFILE"); profile != nullptr && *profile != '\0')
        return profile;
    const char* drive = std::getenv("HOMEDRIVE");
    const char* path = std::getenv("HOMEPATH");
    if (drive != nullptr && path != nullptr)
        return std::string(drive) + path;
#endif
    return {};
}

/// Login name, mirroring the lookup order of Python's ``getpass.getuser()``. Used only to build
/// the default SU(N) data path, which is defined (by the external data-generating repo) in terms
/// of ``getpass.getuser()`` -- matching the order keeps the two in agreement whenever several of
/// these are set to different values (e.g. under ``sudo -E``, or in some CI/container setups).
std::string
login_name()
{
    for (const char* var : { "LOGNAME", "USER", "LNAME", "USERNAME" }) {
        if (const char* v = std::getenv(var); v != nullptr && *v != '\0')
            return v;
    }
    return "unknown";
}

/// Empty string if no usable user config path.
std::string
resolve_user_config_path()
{
    if (const char* override = std::getenv("LASYTEN_CONFIG_FILE")) {
        const fs::path p(expand_user(override));
        if (!fs::exists(p)) {
            throw py::value_error(
              std::string("User config file read from LASYTEN_CONFIG_FILE does not exist: ") +
              p.string());
        }
        return p.string();
    }
    const std::string home = home_directory();
    if (home.empty())
        return {};
    const fs::path p = fs::path(home) / ".lasytenconfig.yaml";
    if (!fs::exists(p))
        return {};
    return p.string();
}

/// Empty string if no local config file exists.
std::string
resolve_local_config_path()
{
    const fs::path p = fs::current_path() / ".lasytenconfig.yaml";
    if (!fs::exists(p))
        return {};
    return p.string();
}

void
try_update_from_file(LasytenConfig& config, const std::string& path)
{
    if (path.empty())
        return;
    try {
        config.update_from_file(path);
    } catch (py::error_already_set& e) {
        std::string msg = e.what();
        e.discard_as_unraisable(__func__);
        warn(std::format("Invalid config in {}. Ignoring the file. Reason: {}", path, msg));
    } catch (const std::exception& e) {
        warn(std::format("Invalid config in {}. Ignoring the file. Reason: {}", path, e.what()));
    }
}

} // namespace

std::string
expand_user(std::string path)
{
    if (path.empty() || path[0] != '~')
        return path;
    const std::string home = home_directory();
    if (home.empty())
        return path;
    if (path.size() == 1)
        return home;
    if (path[1] == '/' || path[1] == '\\')
        return home + path.substr(1);
    return path; // ~user forms are not expanded
}

std::string
default_su_n_data_path()
{
    // Deliberately the literal POSIX form on all platforms -- see the doc comment in config.h.
    // Touches only std::getenv (via login_name()), never py:: calls: this runs during dynamic
    // initialization of _global_config, before the Python interpreter state can be relied on.
    return "/home/" + login_name() + "/.tenpy/su_n_symmetry_data";
}

LasytenConfig _global_config;

const std::vector<std::string>&
LasytenConfig::all_option_keys()
{
    static const std::vector<std::string> keys = {
        "print_linewidth",
        "print_indent",
        "maxlines_spaces",
        "maxlines_tensors",
        "check_fusion",
        "implicit_scalar_conversion",
        "default_tensor_backend",
        "default_block_backend",
        "fusion_tree_eps",
        "su_n_data_path",
        "su_n_data_filename_base",
        "su_n_symbol_cache_mb",
        "tree_plan_cache_mb",
        "su2_cg_hweight",
        "su2_f_hweight",
        "su2_r_hweight",
        "coupling_cutoff",
    };
    return keys;
}

std::string
LasytenConfig::env_var_name(const std::string& key)
{
    return "LASYTEN_" + to_upper(key);
}

void
LasytenConfig::set_option(const std::string& key, int64 value)
{
    if (key == "print_linewidth") {
        check_min(value, 10, key);
        print_linewidth = value;
    } else if (key == "print_indent") {
        check_min(value, 0, key);
        print_indent = value;
    } else if (key == "maxlines_spaces") {
        check_min(value, 0, key);
        maxlines_spaces = value;
    } else if (key == "maxlines_tensors") {
        check_min(value, 0, key);
        maxlines_tensors = value;
    } else if (key == "su_n_symbol_cache_mb") {
        check_min(value, 0, key);
        su_n_symbol_cache_mb = value;
    } else if (key == "tree_plan_cache_mb") {
        check_min(value, 0, key);
        tree_plan_cache_mb = value;
    } else if (key == "su2_cg_hweight") {
        check_min(value, 0, key);
        su2_cg_hweight = value;
    } else if (key == "su2_f_hweight") {
        check_min(value, 0, key);
        su2_f_hweight = value;
    } else if (key == "su2_r_hweight") {
        check_min(value, 0, key);
        su2_r_hweight = value;
    } else if (key == "fusion_tree_eps" || key == "coupling_cutoff") {
        set_option(key, static_cast<float64>(value));
    } else if (std::ranges::contains(all_option_keys(), key)) {
        throw py::type_error("Config option '" + key + "' is not an int");
    } else {
        throw py::key_error("Invalid config option: " + key);
    }
}

void
LasytenConfig::set_option(const std::string& key, float64 value)
{
    if (key == "fusion_tree_eps") {
        check_min(value, 0.0, key);
        fusion_tree_eps = value;
    } else if (key == "coupling_cutoff") {
        check_min(value, 0.0, key);
        coupling_cutoff = value;
    } else if (std::ranges::contains(all_option_keys(), key)) {
        throw py::type_error("Config option '" + key + "' is not a float");
    } else {
        throw py::key_error("Invalid config option: " + key);
    }
}

void
LasytenConfig::set_option(const std::string& key, bool value)
{
    if (key == "check_fusion") {
        check_fusion = value;
    } else if (key == "implicit_scalar_conversion") {
        implicit_scalar_conversion = value;
    } else if (std::ranges::contains(all_option_keys(), key)) {
        throw py::type_error("Config option '" + key + "' is not a bool");
    } else {
        throw py::key_error("Invalid config option: " + key);
    }
}

void
LasytenConfig::set_option(const std::string& key, const std::string& value)
{
    if (key == "print_linewidth" || key == "print_indent" || key == "maxlines_spaces" ||
        key == "maxlines_tensors" || key == "su_n_symbol_cache_mb" ||
        key == "tree_plan_cache_mb" || key == "su2_cg_hweight" || key == "su2_f_hweight" ||
        key == "su2_r_hweight") {
        set_option(key, parse_int64(value));
    } else if (key == "check_fusion" || key == "implicit_scalar_conversion") {
        set_option(key, coerce_bool(value));
    } else if (key == "fusion_tree_eps" || key == "coupling_cutoff") {
        set_option(key, parse_float64(value));
    } else if (key == "default_tensor_backend") {
        static const std::vector<std::string> allowed = { "no_symmetry",
                                                          "abelian",
                                                          "fusion_tree" };
        if (!is_allowed(value, allowed))
            throw py::value_error("Invalid default_tensor_backend: " + value);
        default_tensor_backend = value;
    } else if (key == "default_block_backend") {
        static const std::vector<std::string> allowed = {
            "numpy", "torch", "cpu", "gpu", "apple_silicon"
        };
        if (!is_allowed(value, allowed))
            throw py::value_error("Invalid default_block_backend: " + value);
        default_block_backend = value;
    } else if (key == "su_n_data_path") {
        su_n_data_path = value;
    } else if (key == "su_n_data_filename_base") {
        if (value.empty())
            throw py::value_error("Config option 'su_n_data_filename_base' must not be empty");
        su_n_data_filename_base = value;
    } else {
        throw py::key_error("Invalid config option: " + key);
    }
}

void
LasytenConfig::update(py::dict options)
{
    for (auto item : options) {
        std::string key = py::cast<std::string>(item.first);
        py::handle val = item.second;
        // bool is a subclass of int in Python; check bool first.
        if (py::isinstance<py::bool_>(val)) {
            set_option(key, py::cast<bool>(val));
        } else if (py::isinstance<py::int_>(val)) {
            set_option(key, py::cast<int64>(val));
        } else if (py::isinstance<py::float_>(val)) {
            set_option(key, py::cast<float64>(val));
        } else if (py::isinstance<py::str>(val)) {
            set_option(key, py::cast<std::string>(val));
        } else {
            set_option(key, std::string(py::str(val)));
        }
    }
}

void
LasytenConfig::update(const LasytenConfig& other)
{
    *this = other;
}

void
LasytenConfig::update_from_env()
{
    for (const auto& key : all_option_keys()) {
        const char* val = std::getenv(env_var_name(key).c_str());
        if (val == nullptr)
            continue;
        try {
            set_option(key, std::string(val));
        } catch (py::error_already_set& e) {
            std::string msg = e.what();
            e.discard_as_unraisable(__func__);
            warn(std::format(
              "Invalid config option in envvar {}. Reason {}", env_var_name(key), msg));
        } catch (const std::exception& e) {
            warn(std::format(
              "Invalid config option in envvar {}. Reason {}", env_var_name(key), e.what()));
        }
    }
}

void
LasytenConfig::update_from_mapping(py::dict options)
{
    update(std::move(options));
}

namespace {

[[nodiscard]] std::string
trim_ws(std::string s)
{
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

[[nodiscard]] std::string
strip_quotes(std::string s)
{
    if (s.size() >= 2) {
        char a = s.front();
        char b = s.back();
        if ((a == '"' && b == '"') || (a == '\'' && b == '\'')) {
            return s.substr(1, s.size() - 2);
        }
    }
    return s;
}

/// Minimal flat ``key: value`` parser for ``.lasytenconfig.yaml`` files.
/// Nested YAML / multi-line values are not supported; use pybind ``update_from_yaml``.
void
apply_flat_yaml_text(LasytenConfig& config, std::string const& text)
{
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        line = trim_ws(line);
        if (line.empty()) {
            continue;
        }
        auto colon = line.find(':');
        if (colon == std::string::npos) {
            throw py::value_error("Invalid config line (expected key: value): " + line);
        }
        auto key = trim_ws(line.substr(0, colon));
        auto value = strip_quotes(trim_ws(line.substr(colon + 1)));
        if (key.empty()) {
            throw py::value_error("Invalid config line (empty key): " + line);
        }
        config.set_option(key, value);
    }
}

} // namespace

void
LasytenConfig::update_from_file(const std::string& filename)
{
    // Chosen approach: C++ config file loading uses a minimal flat key:value parser so
    // ``src/`` does not import PyYAML. Full YAML (anchors, nested maps, …) is loaded only
    // from pybind via ``yaml.safe_load`` → ``update_from_mapping`` / ``update``.
    const std::filesystem::path path(filename);
    if (!std::filesystem::exists(path))
        return;
    std::ifstream in(path);
    if (!in) {
        throw py::value_error("Could not open config file: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    apply_flat_yaml_text(*this, ss.str());
}

py::object
LasytenConfig::get_option(const std::string& key) const
{
    if (key == "print_linewidth")
        return py::cast(print_linewidth);
    if (key == "print_indent")
        return py::cast(print_indent);
    if (key == "maxlines_spaces")
        return py::cast(maxlines_spaces);
    if (key == "maxlines_tensors")
        return py::cast(maxlines_tensors);
    if (key == "check_fusion")
        return py::cast(check_fusion);
    if (key == "implicit_scalar_conversion")
        return py::cast(implicit_scalar_conversion);
    if (key == "default_tensor_backend")
        return py::cast(default_tensor_backend);
    if (key == "default_block_backend")
        return py::cast(default_block_backend);
    if (key == "fusion_tree_eps")
        return py::cast(fusion_tree_eps);
    if (key == "su_n_data_path")
        return py::cast(su_n_data_path);
    if (key == "su_n_data_filename_base")
        return py::cast(su_n_data_filename_base);
    if (key == "su_n_symbol_cache_mb")
        return py::cast(su_n_symbol_cache_mb);
    if (key == "tree_plan_cache_mb")
        return py::cast(tree_plan_cache_mb);
    if (key == "su2_cg_hweight")
        return py::cast(su2_cg_hweight);
    if (key == "su2_f_hweight")
        return py::cast(su2_f_hweight);
    if (key == "su2_r_hweight")
        return py::cast(su2_r_hweight);
    if (key == "coupling_cutoff")
        return py::cast(coupling_cutoff);
    throw py::key_error("Invalid option name: " + key);
}

std::string
LasytenConfig::str() const
{
    std::ostringstream ss;
    ss << "LasytenConfig(";
    bool first = true;
    for (const auto& key : all_option_keys()) {
        if (!first)
            ss << ", ";
        first = false;
        ss << key << "=";
        py::object val = get_option(key);
        if (py::isinstance<py::str>(val))
            ss << "'" << std::string(py::cast<std::string>(val)) << "'";
        else
            ss << std::string(py::str(val));
    }
    ss << ")";
    return ss.str();
}

void
LasytenConfig::save_hdf5(lasyten::hdf5::Saver& saver,
                         HighFive::Group& /*h5gr*/,
                         const std::string& subpath) const
{
    saver.save_int64(subpath + "print_linewidth", print_linewidth);
    saver.save_int64(subpath + "print_indent", print_indent);
    saver.save_int64(subpath + "maxlines_spaces", maxlines_spaces);
    saver.save_int64(subpath + "maxlines_tensors", maxlines_tensors);
    saver.save_bool(subpath + "check_fusion", check_fusion);
    saver.save_bool(subpath + "implicit_scalar_conversion", implicit_scalar_conversion);
    saver.save_string(subpath + "default_tensor_backend", default_tensor_backend);
    saver.save_string(subpath + "default_block_backend", default_block_backend);
    saver.save_float64(subpath + "fusion_tree_eps", fusion_tree_eps);
    saver.save_string(subpath + "su_n_data_path", su_n_data_path);
    saver.save_string(subpath + "su_n_data_filename_base", su_n_data_filename_base);
    saver.save_int64(subpath + "su_n_symbol_cache_mb", su_n_symbol_cache_mb);
    saver.save_int64(subpath + "tree_plan_cache_mb", tree_plan_cache_mb);
    saver.save_int64(subpath + "su2_cg_hweight", su2_cg_hweight);
    saver.save_int64(subpath + "su2_f_hweight", su2_f_hweight);
    saver.save_int64(subpath + "su2_r_hweight", su2_r_hweight);
    saver.save_float64(subpath + "coupling_cutoff", coupling_cutoff);
}

LasytenConfig
LasytenConfig::from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath)
{
    LasytenConfig obj;
    auto load_opt = [&](std::string const& key, auto&& apply) {
        if (!hdf5_io::h5_contains(loader.root(), subpath + key))
            return;
        hid_t id = loader.open(subpath + key);
        apply(id);
        H5Idec_ref(id);
    };
    load_opt("print_linewidth", [&](hid_t id) { obj.print_linewidth = loader.load_int64(id); });
    load_opt("print_indent", [&](hid_t id) { obj.print_indent = loader.load_int64(id); });
    load_opt("maxlines_spaces", [&](hid_t id) { obj.maxlines_spaces = loader.load_int64(id); });
    load_opt("maxlines_tensors", [&](hid_t id) { obj.maxlines_tensors = loader.load_int64(id); });
    load_opt("check_fusion", [&](hid_t id) { obj.check_fusion = loader.load_bool(id); });
    load_opt("implicit_scalar_conversion",
             [&](hid_t id) { obj.implicit_scalar_conversion = loader.load_bool(id); });
    load_opt("default_tensor_backend",
             [&](hid_t id) { obj.default_tensor_backend = loader.load_string(id); });
    load_opt("default_block_backend",
             [&](hid_t id) { obj.default_block_backend = loader.load_string(id); });
    load_opt("fusion_tree_eps", [&](hid_t id) { obj.fusion_tree_eps = loader.load_float64(id); });
    load_opt("su_n_data_path", [&](hid_t id) { obj.su_n_data_path = loader.load_string(id); });
    load_opt("su_n_data_filename_base",
             [&](hid_t id) { obj.su_n_data_filename_base = loader.load_string(id); });
    load_opt("su_n_symbol_cache_mb",
             [&](hid_t id) { obj.su_n_symbol_cache_mb = loader.load_int64(id); });
    load_opt("tree_plan_cache_mb",
             [&](hid_t id) { obj.tree_plan_cache_mb = loader.load_int64(id); });
    load_opt("su2_cg_hweight", [&](hid_t id) { obj.su2_cg_hweight = loader.load_int64(id); });
    load_opt("su2_f_hweight", [&](hid_t id) { obj.su2_f_hweight = loader.load_int64(id); });
    load_opt("su2_r_hweight", [&](hid_t id) { obj.su2_r_hweight = loader.load_int64(id); });
    load_opt("coupling_cutoff", [&](hid_t id) { obj.coupling_cutoff = loader.load_float64(id); });
    (void)h5gr;
    return obj;
}

const LasytenConfig&
get_config()
{
    if (!g_config_initialized)
        restore_defaults();
    return _global_config;
}

void
set_option(const std::string& key, const std::string& value)
{
    get_config(); // ensure initialized
    _global_config.set_option(key, value);
}

void
set_option(const std::string& key, int64 value)
{
    get_config();
    _global_config.set_option(key, value);
}

void
set_option(const std::string& key, bool value)
{
    get_config();
    _global_config.set_option(key, value);
}

void
set_option(const std::string& key, float64 value)
{
    get_config();
    _global_config.set_option(key, value);
}

void
set_option(const std::string& key, py::handle value)
{
    get_config();
    py::dict d;
    d[py::str(key)] = value;
    _global_config.update(d);
}

void
set_options(py::dict options)
{
    get_config();
    _global_config.update(options);
}

py::object
get_option(const std::string& key)
{
    return get_config().get_option(key);
}

void
restore_defaults(bool use_user_file, bool use_local_file, bool use_env_vars)
{
    _global_config = LasytenConfig{}; // default values

    // Precedence (later wins): defaults -> user file -> local file -> env
    if (use_user_file)
        try_update_from_file(_global_config, resolve_user_config_path());
    if (use_local_file)
        try_update_from_file(_global_config, resolve_local_config_path());
    if (use_env_vars)
        _global_config.update_from_env();

    g_config_initialized = true;
}

} // namespace lasyten
