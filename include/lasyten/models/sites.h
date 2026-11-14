#pragma once

#include <lasyten/models/degrees_of_freedom.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace lasyten {

/// Class for sites that have a single spin degree of freedom.
///
/// TODO find a good format to doc the onsite operators that exist in a site
///
/// Attributes:
///
/// S : float
///     The total spin.
/// double_total_spin : int
///     Twice the `S`. We store this in addition because it is an integer.
/// conserve : Literal['SU(2)', 'Sz', 'parity', 'None']
///     The symmetry to be conserved. We can conserve::
///
///         - SU(2), the full spin rotation symmetry (production ``SUN(N=2)`` by default;
///           pass ``use_test_su2=True`` for the test-only ``_SU2`` factor).
///         - Sz (= U(1) symmetry), with sector labels corresponding to ``2 * Sz``.
///         - Sz parity (= Z_2 symmetry), with sector labels corresponding to ``(Sz + S_tot) % 2``.
///         - nothing.
///
///     Conserves nothing by default.
/// use_test_su2 : bool
///     If True and ``conserve`` is SU(2), use test-only ``_SU2`` instead of ``SUN(N=2)``.
///
/// The spin operators live on ``dofs["spin"]`` (`SpinDOF`).
class SpinSite : public Site
{
  public:
    using Ptr = std::shared_ptr<SpinSite>;

    float64 S{};
    int64 double_total_spin{};
    std::optional<std::string> conserve;
    bool use_test_su2{ false };

    SpinSite(float64 S = 0.5,
             std::optional<std::string> conserve = std::nullopt,
             TensorBackend::Ptr backend = nullptr,
             std::optional<std::string> default_device = std::nullopt,
             bool use_test_su2 = false);

    /// Perform sanity checks.
    void test_sanity() override;
    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        ElementarySpace::Ptr leg;
        py::array spin_vector;
        std::map<std::string, int64> state_labels;
        Symmetry::Ptr sym;
        int64 two_S;
    };
    static Prepared prepare(float64 S,
                            std::optional<std::string> conserve,
                            bool use_test_su2 = false);
    SpinSite(Prepared&& prepared,
             float64 S,
             std::optional<std::string> conserve,
             TensorBackend::Ptr backend,
             std::optional<std::string> default_device,
             bool use_test_su2);
};

/// Site for (possibly multiple) spinless bosons.
///
/// Occupation data lives on ``dofs["occupation"]`` (`BosonicDOF`).
class SpinlessBosonSite : public Site
{
  public:
    using Ptr = std::shared_ptr<SpinlessBosonSite>;

    py::object conserve;
    std::optional<float64> filling;

    SpinlessBosonSite(py::object Nmax = py::int_(1),
                      py::object conserve = py::none(),
                      std::optional<float64> filling = std::nullopt,
                      TensorBackend::Ptr backend = nullptr,
                      std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        ElementarySpace::Ptr leg;
        py::array Nmax_arr;
        py::array creators;
        py::array annihilators;
        std::map<std::string, int64> state_labels;
        int64 total_dim;
    };
    static Prepared prepare(py::object Nmax, py::object conserve);
    SpinlessBosonSite(Prepared&& prepared,
                      py::object conserve,
                      std::optional<float64> filling,
                      TensorBackend::Ptr backend,
                      std::optional<std::string> default_device);
};

/// Site for (possibly multiple) spinless fermions.
///
/// Occupation data lives on ``dofs["occupation"]`` (`FermionicDOF`).
class SpinlessFermionSite : public Site
{
  public:
    using Ptr = std::shared_ptr<SpinlessFermionSite>;

    int64 num_species{};
    py::object conserve;
    std::optional<float64> filling;

    SpinlessFermionSite(int64 num_species = 1,
                        py::object conserve = py::str("parity"),
                        std::optional<float64> filling = std::nullopt,
                        TensorBackend::Ptr backend = nullptr,
                        std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        ElementarySpace::Ptr leg;
        py::array creators;
        py::array annihilators;
        std::map<std::string, int64> state_labels;
    };
    static Prepared prepare(int64 num_species, py::object conserve);
    SpinlessFermionSite(Prepared&& prepared,
                        int64 num_species,
                        py::object conserve,
                        std::optional<float64> filling,
                        TensorBackend::Ptr backend,
                        std::optional<std::string> default_device);
};

/// Site for spin-1/2 fermions.
///
/// Registers both ``dofs["spin"]`` (`SpinDOF`) and ``dofs["occupation"]`` (`FermionicDOF`).
class SpinHalfFermionSite : public Site
{
  public:
    using Ptr = std::shared_ptr<SpinHalfFermionSite>;

    std::string conserve_N;
    std::optional<std::string> conserve_S;
    std::optional<float64> filling;
    bool use_test_su2{ false };

    SpinHalfFermionSite(std::string conserve_N = "parity",
                        std::optional<std::string> conserve_S = std::nullopt,
                        std::optional<float64> filling = std::nullopt,
                        TensorBackend::Ptr backend = nullptr,
                        std::optional<std::string> default_device = std::nullopt,
                        bool use_test_su2 = false);

    void test_sanity() override;

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        ElementarySpace::Ptr leg;
        py::array spin_vector;
        py::array creators;
        py::array annihilators;
        std::map<std::string, int64> state_labels;
        SymmetryFactor::Ptr sym_S_factor;
    };
    static Prepared prepare(std::string const& conserve_N,
                            std::optional<std::string> const& conserve_S,
                            bool use_test_su2 = false);
    SpinHalfFermionSite(Prepared&& prepared,
                        std::string conserve_N,
                        std::optional<std::string> conserve_S,
                        std::optional<float64> filling,
                        TensorBackend::Ptr backend,
                        std::optional<std::string> default_device,
                        bool use_test_su2);
};

/// Class for sites that have a single quantum clock degree of freedom.
///
/// Clock data lives on ``dofs["clock"]`` (`ClockDOF`).
class ClockSite : public Site
{
  public:
    using Ptr = std::shared_ptr<ClockSite>;

    int64 q;
    std::optional<std::string> conserve;

    ClockSite(int64 q,
              std::optional<std::string> conserve = std::nullopt,
              TensorBackend::Ptr backend = nullptr,
              std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        ElementarySpace::Ptr leg;
        py::array clock_operators;
        std::map<std::string, int64> state_labels;
    };
    static Prepared prepare(int64 q, std::optional<std::string> conserve);
    ClockSite(Prepared&& prepared,
              int64 q,
              std::optional<std::string> conserve,
              TensorBackend::Ptr backend,
              std::optional<std::string> default_device);
};

/// Class for anyon models where the local Hilbert space contains all sectors once.
///
/// Anyon data lives on ``dofs["anyon"]`` (`AnyonDOF`).
class AnyonSite : public Site
{
  public:
    using Ptr = std::shared_ptr<AnyonSite>;

    AnyonSite(Symmetry::Ptr symmetry,
              TensorBackend::Ptr backend = nullptr,
              std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  protected:
    /// Initialize from an already-built local space.
    AnyonSite(ElementarySpace::Ptr leg,
              TensorBackend::Ptr backend,
              std::optional<std::string> default_device);
};

/// Class for sites containing the trivial and the Fibonacci / tau sectors.
class FibonacciAnyonSite : public AnyonSite
{
  public:
    using Ptr = std::shared_ptr<FibonacciAnyonSite>;

    FibonacciAnyonSite(TensorBackend::Ptr backend = nullptr,
                       std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        Symmetry::Ptr symmetry;
        ElementarySpace::Ptr leg;
    };
    static Prepared prepare();
    FibonacciAnyonSite(Prepared&& prepared,
                       TensorBackend::Ptr backend,
                       std::optional<std::string> default_device);
};

/// Class for sites containing the trivial, the Ising / sigma, and the fermion / psi sectors.
class IsingAnyonSite : public AnyonSite
{
  public:
    using Ptr = std::shared_ptr<IsingAnyonSite>;

    IsingAnyonSite(int nu = 1,
                   TensorBackend::Ptr backend = nullptr,
                   std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        Symmetry::Ptr symmetry;
        ElementarySpace::Ptr leg;
    };
    static Prepared prepare(int nu);
    IsingAnyonSite(Prepared&& prepared,
                   TensorBackend::Ptr backend,
                   std::optional<std::string> default_device);
};

/// Class for Fibonacci anyon models where the local Hilbert space only contains the tau sector.
class GoldenSite : public Site
{
  public:
    using Ptr = std::shared_ptr<GoldenSite>;

    GoldenSite(std::string handedness = "left",
               TensorBackend::Ptr backend = nullptr,
               std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    GoldenSite(ElementarySpace::Ptr leg,
               TensorBackend::Ptr backend,
               std::optional<std::string> default_device);
};

/// Class for SU(2)_k anyon models where the local Hilbert space only contains the spin-1 sector.
class SU2kSpin1Site : public Site
{
  public:
    using Ptr = std::shared_ptr<SU2kSpin1Site>;

    int64 k;

    SU2kSpin1Site(int64 k,
                  TensorBackend::Ptr backend = nullptr,
                  std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    SU2kSpin1Site(ElementarySpace::Ptr leg,
                  int64 k,
                  TensorBackend::Ptr backend,
                  std::optional<std::string> default_device);
};

/// Group several sites into one larger local Hilbert space (pipe physical leg).
///
/// Onsite operators of constituent site ``i`` are Kronecker-embedded as ``opname + labels[i]``.
/// Degrees of freedom are registered under ``"{label}.{kind}"`` (e.g. ``"0.spin"``) with dense
/// arrays embedded into the grouped dimension. Abelian symmetries only (``AbelianLegPipe``).
///
/// For ``symmetry_combine='same'``, constituents must already share a compatible symmetry
/// (call `set_common_symmetry` first if needed). ``'drop'`` / ``'independent'`` clone and remap
/// legs before piping; the caller's original sites are not mutated.
class GroupedSite : public Site
{
  public:
    using Ptr = std::shared_ptr<GroupedSite>;

    int64 n_sites{};
    std::vector<Site::Ptr> sites;
    std::vector<std::string> labels;
    std::string symmetry_combine;

    GroupedSite(std::vector<Site::Ptr> sites,
                std::optional<std::vector<std::string>> labels = std::nullopt,
                std::string symmetry_combine = "same");

    /// Kronecker product of one operator per ungrouped site (labels ``p``, ``p*``).
    [[nodiscard]] SymmetricTensorPtr kroneckerproduct(std::vector<SymmetricTensorPtr> ops) const;

    [[nodiscard]] std::string repr() const;

  private:
    struct Prepared
    {
        AbelianLegPipe::Ptr pipe;
        std::map<std::string, int64> state_labels;
        std::vector<Site::Ptr> working_sites;
        std::vector<std::string> labels;
        std::string symmetry_combine;
        TensorBackend::Ptr backend;
        std::string default_device;
    };
    static Prepared prepare(std::vector<Site::Ptr> sites,
                            std::optional<std::vector<std::string>> labels,
                            std::string symmetry_combine);
    GroupedSite(Prepared&& prepared);
    void add_kronecker_onsite_ops();
    void register_embedded_dofs();
};

/// Group each ``n`` consecutive sites into a `GroupedSite`.
[[nodiscard]] std::vector<GroupedSite::Ptr> group_sites(
  std::vector<Site::Ptr> const& sites,
  int64 n = 2,
  std::optional<std::vector<std::string>> labels = std::nullopt,
  std::string symmetry_combine = "same");

} // namespace lasyten
