#pragma once

#include <array>
#include <string>
#include <vector>

#include <lasyten/lasyten.h>
#include <lasyten/tools/hdf5.h>

namespace lasyten {

/// Default directory for SU(N) symmetry data files.
///
/// Deliberately the literal POSIX form ``/home/<login-name>/.tenpy/su_n_symmetry_data`` on *all*
/// platforms, matching the layout produced by the external ``clebsch_gordan_coefficients``
/// generator (which computes ``f"/home/{getpass.getuser()}/.tenpy/su_n_symmetry_data"``).
/// On Windows this path will usually not exist; override it via the ``su_n_data_path`` config
/// option, the ``LASYTEN_SU_N_DATA_PATH`` envvar, or ``.lasytenconfig.yaml``.
std::string default_su_n_data_path();

/// Expand a leading ``~`` (or ``~/...``) to the user's home directory. ``~user`` forms and paths
/// without a resolvable home directory are returned unchanged.
std::string expand_user(std::string path);

/// The weights ``{cg, f, r}`` pinned by the config for the symmetry SU(N); an entry is 0 if it is
/// not pinned. Sources: the option ``su_n_hweights`` and, for N = 2, ``su2_{cg,f,r}_hweight``
/// (which take precedence).
std::array<int64, 3> su_n_pinned_hweights(int N);

// NOLINTBEGIN(readability-magic-numbers)

class LasytenConfig
{
  public:
    int64 print_linewidth = 100;
    int64 print_indent = 2;
    int64 maxlines_spaces = 15;
    int64 maxlines_tensors = 30;
    /// If the symmetry methods should check their inputs are valid
    bool check_fusion = true;
    /// If Python may implicitly convert a `BlockBackend::Scalar` to float / complex / numpy.
    /// Default True for interactive use; tests force this False. Explicit accessors (`as_float64`,
    /// `to_numpy`, …) are never gated. Later this can track autograd (off when conversion would
    /// drop the graph).
    bool implicit_scalar_conversion = true;
    std::string default_tensor_backend = "abelian";
    std::string default_block_backend = "numpy";
    /// Threshold for discarding near-zero fusion-tree blocks after topological moves.
    /// Default is based on tests for 4-leg tensors: smaller values produced extra blocks from
    /// numerical noise when bending legs and restoring the original configuration.
    float64 fusion_tree_eps = 5.0e-14;
    /// Directory holding the SU(N) Clebsch-Gordan / F-symbol / R-symbol HDF5 files.
    /// See :cpp:func:`default_su_n_data_path`.
    std::string su_n_data_path = default_su_n_data_path();
    /// Stem of the SU(N) data file names. The full name is
    /// ``<base>_N{N}_{CG|F|R}_hweight{H}.hdf5``.
    std::string su_n_data_filename_base = "su_n_clebsch_gordan_data";
    /// Memory budget in MB of the RAM cache for SU(N) F symbols, R symbols and fusion outcomes
    /// (least-recently-used entries are evicted when the estimated size exceeds it).
    /// ``0`` disables the cache. The value is read whenever an entry is inserted, so it can be
    /// changed at runtime (e.g. with ``temporary_options``).
    int64 su_n_symbol_cache_mb = 256;
    /// Memory budget in MB of the RAM cache for fusion-tree mappings ("tree plans") and the
    /// ``fusion_trees::size`` memo. ``0`` disables the caches. Read at insertion time.
    int64 tree_plan_cache_mb = 256;
    /// Highest weight of the Clebsch-Gordan / F / R files of the production SU(2) symmetry
    /// (``conserve='SU2'``). ``0`` (default) auto-detects the largest weight available in
    /// ``su_n_data_path``; a positive value pins it.
    /// Pins the data weights of SU(N) symmetries that are created without explicit weights
    /// (``SUN(N)``): a ``;``-separated list of ``N:cg[,f[,r]]`` entries, e.g. ``"3:5;4:3,3,3"``.
    /// ``f`` and ``r`` default to ``cg``; ``0`` or a missing entry means "largest weight found in
    /// ``su_n_data_path``". For N = 2 the options ``su2_{cg,f,r}_hweight`` take precedence.
    std::string su_n_hweights = "";
    int64 su2_cg_hweight = 0;
    int64 su2_f_hweight = 0;
    int64 su2_r_hweight = 0;
    /// Default singular-value cutoff when factorizing a `Coupling` (`from_dense_block` /
    /// `from_tensor`). Singular values below this threshold are discarded.
    float64 coupling_cutoff = 1.0e-13;

    LasytenConfig() = default;

    /// Names of all recognized config options.
    static const std::vector<std::string>& all_option_keys();

    /// Environment variable name for a config option (``LASYTEN_`` + uppercased key).
    static std::string env_var_name(const std::string& key);

    void set_option(const std::string& key, const std::string& value);
    void set_option(const std::string& key, int64 value);
    void set_option(const std::string& key, bool value);
    void set_option(const std::string& key, float64 value);

    void update(py::dict options);
    void update(const LasytenConfig& other);
    void update_from_env();
    /// Apply options from an already-parsed mapping (no YAML parsing in C++).
    /// Python bindings load YAML via ``yaml.safe_load`` then call this / ``update``.
    void update_from_mapping(py::dict options);
    /// Load a flat ``key: value`` YAML-ish config file with a minimal C++ parser
    /// (no PyYAML). Full YAML documents should go through the pybind ``update_from_yaml``.
    void update_from_file(const std::string& filename);

    py::object get_option(const std::string& key) const;

    std::string str() const;

    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;
    static LasytenConfig from_hdf5(lasyten::hdf5::Loader& loader,
                                   HighFive::Group& h5gr,
                                   std::string const& subpath);
};

// NOLINTEND(readability-magic-numbers)

/// Global mutable config (initialized on first get_config / restore_defaults).
/// Prefer get_config() for reads; use set_option / set_options / temporary_options to mutate.
extern LasytenConfig _global_config;

const LasytenConfig& get_config();

void set_option(const std::string& key, const std::string& value);
void set_option(const std::string& key, int64 value);
void set_option(const std::string& key, bool value);
void set_option(const std::string& key, float64 value);
void set_option(const std::string& key, py::handle value);
void set_options(py::dict options);

py::object get_option(const std::string& key);

void restore_defaults(bool use_user_file = true,
                      bool use_local_file = true,
                      bool use_env_vars = true);

} // namespace lasyten
