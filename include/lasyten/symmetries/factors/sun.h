#pragma once

#include "../group.h"

#include <highfive/highfive.hpp>
#include <lasyten/tools/hdf5.h>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lasyten {

/// Standard file name for SU(N) symmetry data, per the SU(N) data convention:
/// ``"<base>_N{N}_{kind}_hweight{hweight}.hdf5"``.
///
/// \param N  Rank of the group, ``SU(N)``.
/// \param kind  One of ``"CG"``, ``"F"`` or ``"R"`` (case-insensitive on input, normalized to
/// upper
///              case in the result).
/// \param hweight  Highest weight stored in the file.
/// \param filename_base  Defaults to the ``su_n_data_filename_base`` config option.
std::string su_n_data_filename(int N,
                               std::string const& kind,
                               int64 hweight,
                               std::optional<std::string> filename_base = std::nullopt);

/// Full path to a standard SU(N) data file: ``{path}`` joined with ``su_n_data_filename(...)``.
///
/// \param N  Rank of the group, ``SU(N)``.
/// \param kind  One of ``"CG"``, ``"F"`` or ``"R"`` (case-insensitive on input, normalized to
/// upper
///              case in the result).
/// \param hweight  Highest weight stored in the file.
/// \param path  Defaults to the ``su_n_data_path`` config option. A leading ``~`` is expanded.
/// \param filename_base  Defaults to the ``su_n_data_filename_base`` config option.
std::string su_n_data_file_path(int N,
                                std::string const& kind,
                                int64 hweight,
                                std::optional<std::string> path = std::nullopt,
                                std::optional<std::string> filename_base = std::nullopt);

/// SU(N) group symmetry
///
/// The sectors are arrays of length N which correspond to first rows of normalized Gelfand-Tsetlin
/// patterns (see https://arxiv.org/pdf/1009.0437 ).
/// E.g. for SU(3) the 8 dimensional irreducible representation is labeled by [2,1,0]
///
/// Clebsch-Gordan coefficients and F/R symbols need to be calculated with the
/// clebsch_gordan_coefficients package and exported as HDF5 files.
///
/// The F and R files may be in the folder layout (one dataset per key) or in the packed layout
/// (root attribute ``Layout = "packed_v1"``, the keys and blocks stored in four arrays); the
/// layout is detected for each file and does not change the results.
///
/// Construct from filenames (or via :func:`from_config`, which resolves standard paths)::
///
///     SUN(N, cg_path, f_path, r_path, descriptive_name=None)
///     SUN.from_config(N, hweight, *, cg_hweight=None, f_hweight=None, r_hweight=None,
///                     path=None, filename_base=None, descriptive_name=None)
///
/// ``from_config`` looks up
/// ``{su_n_data_path}/{su_n_data_filename_base}_N{N}_{CG|F|R}_hweight{H}.hdf5``.
/// ``hweight`` sets all three highest weights; ``cg_hweight`` / ``f_hweight`` /
/// ``r_hweight`` override them individually. The CG highest weight must be >= the
/// F and R highest weights. Use :func:`su_n_data_file_path` to see where lasyten will look.
class SUN : public Group
{
  public:
    using Ptr = std::shared_ptr<SUN>;
    using CPtr = std::shared_ptr<const SUN>;

    int N;
    /// Absolute / resolved paths of the CG / F / R data files.
    std::string CGpath;
    std::string Fpath;
    std::string Rpath;

    /// Construct from paths to the three SU(N) data files (opened read-only via HighFive).
    SUN(int N,
        std::string cg_path,
        std::string f_path,
        std::string r_path,
        std::optional<std::string> descriptive_name = std::nullopt);
    ~SUN() override = default;

    /// Construct from the standard data files, resolved via ``su_n_data_file_path`` (i.e. via the
    /// ``su_n_data_path`` / ``su_n_data_filename_base`` config options, unless overridden here).
    ///
    /// ``hweight`` sets the highest weight for all three files; ``cg_hweight`` / ``f_hweight`` /
    /// ``r_hweight`` override it individually. The CG highest weight must be >= the F and R
    /// highest weights. Without ``hweight`` every weight that is not given is taken from the
    /// config option ``su_n_hweights`` or else is the largest
    /// weight for which a data file exists (F and R not above the CG weight); a missing file
    /// raises ``FileNotFoundError``.
    static Ptr from_config(int N,
                           std::optional<int64> hweight,
                           std::optional<int64> cg_hweight = std::nullopt,
                           std::optional<int64> f_hweight = std::nullopt,
                           std::optional<int64> r_hweight = std::nullopt,
                           std::optional<std::string> path = std::nullopt,
                           std::optional<std::string> filename_base = std::nullopt,
                           std::optional<std::string> descriptive_name = std::nullopt);

    bool is_valid_sector(Sector a) const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;
    int64 sector_dim(Sector a) const override;
    std::string repr() const override;
    Sector dual_sector(Sector a) const override;

    int64 hweight_from_CG_hdf5() const;
    int64 hweight_from_F_hdf5() const;
    int64 hweight_from_R_hdf5() const;

    bool can_fuse_to(Sector a, Sector b, Sector c) const override;
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;

    /// To every SU(N) irrep, labeled by the first row of a GT pattern, we can assign an integer S.
    int64 S_index_irrep_weight(Sector a) const;
    /// Returns the highest irrep which appears in the decomposition of a x b.
    Sector highest_irrep_in_decomp(Sector a, Sector b) const;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;

    /// Dimensions of irreps appearing in the decomposition of a x b (no multiplicities).
    std::map<Sector, int64> dims_of_irreps(Sector a, Sector b) const;
    /// Outer multiplicities for irreps in the decomposition of a x b.
    std::map<Sector, int64> outer_multiplicity_from_CG(Sector a, Sector b) const;

    /// Evaluate a single Clebsch-Gordan coefficient.
    ///
    /// @param a,b,c Sector for the fusion @f$ a \otimes b \mapsto c @f$.
    /// @param q_a,q_b,q_c Indices of the Gelfand Tsetlin pattern
    /// @param mu multiplicity index 1 <= mu
    /// @returns The CG coefficient for the given input
    float64 clebschgordan(Sector a, int64 q_a, Sector b, int64 q_b, Sector c, int64 q_c, int64 mu)
      const;

    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override;
    /// Returns the F symbol for the specified input irreps calculated from CG coefficients.
    ///
    /// a,b,c,d,e,f are irrep labels, i.e. first rows of GT patterns
    /// output is the conjugated F symbol [F^{abc}_{def}]^*_{mu,nu,kappa, lambda}
    /// where a x b = mu c, c x d =nu e, b x d= kappa f and a x f =lambda e
    ///
    /// @param a,b,c,d,e,f Irreps specifying the CG coefficient.
    FusionSymbol _f_symbol_from_CG(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const;
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    /// Returns the R symbol for the specified input irreps calculated from CG coefficients.
    ///
    /// @param a,b,c Irreps specifying the R symbol.
    FusionSymbol _r_symbol_from_CG(Sector a, Sector b, Sector c) const;
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override;
    int64 frobenius_schur(Sector a) const override;

    bool has_data_in_group(hid_t loc) const;
    /// Sanity check for Hdf5 files containing CG-coefficients, F-symbols or R-symbols.
    ///
    /// This method takes an open HighFive file and checks if it has the required structure and if
    /// the necessary data has been saved to it. This excludes the possibility of using
    /// incompletely generated files, but cannot guarantee completeness of the file and correctness
    /// of the data in the file. In particular, consistency of the data in the file should be
    /// checked by the lasyten tests for SU(N) symmetry.
    void sanity_check_hdf5(HighFive::File const& file) const;

    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const override;
    static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath);

  private:
    std::shared_ptr<HighFive::File> CGfile_;
    std::shared_ptr<HighFive::File> Ffile_;
    std::shared_ptr<HighFive::File> Rfile_;
    std::shared_ptr<void> state_; // per-object cache state, defined in sun.cpp (SunState)

    HighFive::Group cg_root() const;
    HighFive::Group f_root() const;
    HighFive::Group r_root() const;
};

/// Weights ``{cg, f, r}`` of the data files that ``SUN::from_config`` would use for N without
/// explicit weights; entries that cannot be determined (nothing pinned, no file found) are empty.
std::array<std::optional<int64>, 3> su_n_detect_hweights(
  int N,
  std::optional<std::string> path = std::nullopt,
  std::optional<std::string> filename_base = std::nullopt,
  std::optional<int64> cg_hweight = std::nullopt,
  std::optional<int64> f_hweight = std::nullopt,
  std::optional<int64> r_hweight = std::nullopt);

} // namespace lasyten
