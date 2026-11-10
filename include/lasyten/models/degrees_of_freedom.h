#pragma once

#include <lasyten/backends/backend_factory.h>
#include <lasyten/backends/tensor_backend.h>
#include <lasyten/block_backend/block_backend.h>
#include <lasyten/lasyten.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/symmetry.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/forward_declare.h>
#include <lasyten/tensors/symmetric_tensor.h>

#include <lasyten/tools/hdf5.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lasyten {

/// Sentinel for summing over all species in fermion/boson couplings.
/// Python exposes ``ALL_SPECIES = object()``; `all_species_sentinel` is the unique
/// identity used for ``is`` comparison and pybind default arguments (process-lifetime).
/// Binding sites use ``py::arg_v(..., "ALL_SPECIES")`` so signatures / stubgen show a
/// valid expression instead of ``<object object at 0x...>``.
struct AllSpeciesTag
{};
inline AllSpeciesTag const ALL_SPECIES{};

[[nodiscard]] py::object& all_species_sentinel();
[[nodiscard]] bool is_all_species(py::object const& species);

/// Collects necessary information about a local site of a lattice model.
///
/// A site defines the local Hilbert space in terms of its `leg`.
/// This involves a choice for the local basis.
/// Moreover, it exposes the symmetric single-site operators.
/// Multi-site operators, on the other hand, are represented by `Coupling` s.
///
/// Attributes:
///
/// leg : ElementarySpace
///     The local physical Hilbert space.
/// state_labels : {str: int}
///     Optional labels for the local basis states. Any state may have multiple labels, or none.
/// onsite_operators : {str: SymmetricTensor}
///     The available on-site operators. Note: which operators are available typically depends
///     on what symmetry is enforced. Operators that are symmetric under a small symmetry may
///     not be symmetric under a larger symmetry, and are thus not available as `onsite_operators`.
///     Each must have the `leg` of the site as the only factor in its domain and codomain.
/// hc_ops : {str: str}
///     Mapping from each known onsite operator name to the name of its hermitian conjugate.
///
/// TODO put some
class Site : public virtual std::enable_shared_from_this<Site>
{
  public:
    using Ptr = std::shared_ptr<Site>;
    using CPtr = std::shared_ptr<const Site>;

    ElementarySpace::Ptr leg;
    std::map<std::string, int64> state_labels;
    TensorBackend::Ptr backend;
    std::string default_device;
    std::map<std::string, SymmetricTensorPtr> onsite_operators;
    std::map<std::string, std::string> hc_ops;

    Site(ElementarySpace::Ptr leg,
         std::map<std::string, int64> state_labels = {},
         std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
         TensorBackend::Ptr backend = nullptr,
         std::optional<std::string> default_device = std::nullopt);

    virtual ~Site() = default;

    /// Perform sanity checks.
    virtual void test_sanity();

    [[nodiscard]] Symmetry::Ptr symmetry() const;
    [[nodiscard]] float64 dim() const;

    /// Add an operator to the `onsite_operators`.
    ///
    /// If `hc` is not specified, infer the hermitian-conjugate name by comparing this operator's
    /// adjoint with onsite operators already registered on the site.
    void add_onsite_operator(std::string const& name,
                             py::object op,
                             std::optional<bool> is_diagonal = std::nullopt,
                             bool understood_braiding = false,
                             std::optional<std::string> hc = std::nullopt);

    /// Whether `name` labels a valid onsite operator of this site.
    [[nodiscard]] bool valid_opname(std::string const& name) const;
    /// Validate an onsite operator name or whitespace-separated operator product.
    ///
    /// Raises `SymmetryError` for built-in operators unavailable under this site's symmetry and
    /// `std::invalid_argument` for names that are not known to the site.
    void test_op(std::string const& opname) const;
    /// Rename an onsite operator while preserving its hermitian-conjugate mapping.
    void rename_op(std::string const& old_name, std::string const& new_name);
    /// Return the name of the hermitian conjugate of an operator or operator product.
    ///
    /// For a product of onsite operator names separated by whitespace, reverse their order and
    /// replace each name with its registered hermitian conjugate.
    [[nodiscard]] std::string get_hc_op_name(std::string const& name) const;
    /// Return operator of given name.
    ///
    /// @param name The name of the operator to be returned. In case of multiple operator names
    /// separated by whitespace, we multiply them together to a single on-site operator (with the
    /// one on the right acting first).
    /// @returns op : `SymmetricTensor` The operator given by `name`, with labels ``'p', 'p*'``. If
    /// name already was an onsite operator, it's directly returned.
    [[nodiscard]] SymmetricTensorPtr get_op(std::string const& name);
    /// Multiply operator names together.
    ///
    /// Join the operator names in `names` such that `get_op` returns the product of the
    /// corresponding operators.
    ///
    /// @param names List of valid operator labels.
    /// @returns combined_opname : str A valid operator name Operator name representing the product
    /// of operators in `names`.
    [[nodiscard]] std::string multiply_op_names(std::vector<std::string> const& names) const;
    /// Multiply local operators (possibly given by their names) together.
    ///
    /// @param operators List of valid operator names (to be translated with `get_op`) or directly
    /// on-site operators in the form of tensors with ``'p', 'p*'`` labels. The operators are
    /// multiplied left-to-right.
    /// @returns combined_operator : `SymmetricTensor` The product of the given `operators` in a
    /// left-to-right multiplication following the usual mathematical convention. For example, if
    /// ``operators=['Sz', 'Sp', 'Sx']``, the final operator is equivalent to ``site.get_op('Sz Sp
    /// Sx')``, with the ``'Sx'`` operator acting first on any physical state.
    [[nodiscard]] SymmetricTensorPtr multiply_operators(std::vector<py::object> const& operators);
    /// Build an identity MPO tensor for this site with the given virtual legs.
    ///
    /// Returns a 4-leg tensor with legs ``[wL, p, wR, p*]``; the physical legs carry `leg`.
    /// This tensor acts as identity map between wL and wR, and is symmetric under the symmetry of
    /// this site.
    ///
    /// @param w Virtual leg for the `wL` and `wR` legs (they are the same) of the returned tensor.
    /// @param overbraid Braiding direction when the virtual leg is permuted past the physical leg.
    /// ``True`` (default) uses an over-braid (``bend_right=False`` in `permute_legs`); ``False``
    /// uses an under-braid (``bend_right=True``).
    /// @returns Identity tensor with legs ``[wL, p, wR, p*]``.
    [[nodiscard]] SymmetricTensorPtr identity_tensor(ElementarySpace::Ptr w,
                                                     bool overbraid = true);
    /// Build an MPO tensor that applies an on-site operator and acts as identity on a virtual leg.
    ///
    /// The result is ``outer(op, id_w)`` with ``op = get_op(opname)`` (legs ``p, p*``) and
    /// ``id_w`` the identity on the virtual leg ``w`` (legs ``w, w*``), with the legs then
    /// permuted to the order ``[wL, p, wR, p*]``. Compared to `identity_tensor`, the identity on
    /// the physical leg is replaced by the operator.
    ///
    /// Braiding: in ``outer(op, id_w)`` the codomain is ``[p, w]``. Moving the virtual leg ``w``
    /// to the left of ``p`` means that the virtual line and the operator's physical line cross.
    /// This is only relevant for symmetries with non-symmetric braids (anyons).
    /// - ``overbraid=True``: the *virtual* line passes **over** the physical line, i.e. the
    ///   virtual leg is in front of the operator. This is ``bend_right=False`` in `permute_legs`.
    /// - ``overbraid=False``: the virtual line passes **under** the physical line, i.e. the
    ///   operator is in front of the virtual leg (``bend_right=True``).
    /// The same convention as in `identity_tensor` is used.
    ///
    /// @param opname Name of the operator, translated with `get_op`.
    /// @param w Virtual leg for the `wL` and `wR` legs (they are the same).
    /// @param overbraid If the virtual leg passes over (``True``, default) or under (``False``)
    /// the physical legs, see above.
    /// @param wL_label Label of the left virtual leg of the result.
    /// @param wR_label Label of the right virtual leg of the result.
    /// @returns Tensor with legs ``[wL, p, wR, p*]`` (using the given virtual labels).
    [[nodiscard]] SymmetricTensorPtr get_onsite_W_block(std::string const& opname,
                                                        ElementarySpace::Ptr w,
                                                        bool overbraid = true,
                                                        std::string const& wL_label = "wL",
                                                        std::string const& wR_label = "wR");
    /// The index of a basis state.
    [[nodiscard]] int64 state_index(py::object label) const;
    /// The indices of multiple basis states
    [[nodiscard]] std::vector<int64> state_indices(std::vector<py::object> const& labels) const;
    [[nodiscard]] std::string repr() const;

    /// Export `self` into a HDF5 file.
    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;
    static py::object from_hdf5(py::object cls,
                                lasyten::hdf5::Loader& loader,
                                HighFive::Group& h5gr,
                                std::string const& subpath);

    /// Constructor keyword arguments used by `save_hdf5` / `from_hdf5`.
    [[nodiscard]] virtual py::dict hdf5_init_kwargs() const;

  protected:
    [[nodiscard]] py::dict hdf5_backend_kwargs() const;
};

/// Common base class for sites that have a spin degree of freedom.
///
/// Attributes:
///
/// spin_vector : 3D array
///     The vector of spin operators as a numpy array with axes ``[p, p*, i]`` and shape
///     ``(dim, dim, 3)``. These operators include the factor of the total spin,
///     e.g. for spin-1/2, these are ``.5`` times the pauli matrices.
class SpinDOF : public virtual Site
{
  public:
    using Ptr = std::shared_ptr<SpinDOF>;

    py::array spin_vector;

    SpinDOF(ElementarySpace::Ptr leg,
            py::array spin_vector,
            std::map<std::string, int64> state_labels = {},
            std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
            TensorBackend::Ptr backend = nullptr,
            std::optional<std::string> default_device = std::nullopt);

    /// Perform sanity checks.
    void test_sanity() override;

    /// Build the spin_vector from ``Sz`` and ``Sp = Sx + i Sy``
    [[nodiscard]] static py::array spin_vector_from_Sp(py::array Sz, py::array Sp);
    /// Translate conservation law for a spin to a symmetry.
    ///
    /// For ``'SU(2)'`` / ``'SU2'`` / ``'Stot'``, builds production ``SUN(N=2, ...)`` by default
    /// (via ``SUN::from_config`` and ``su_n_data_path``). Pass ``use_test_su2=True`` to fall
    /// back to the test-only ``_SU2`` factor.
    [[nodiscard]] static Symmetry::Ptr conservation_law_to_symmetry(
      std::optional<std::string> conserve,
      bool use_test_su2 = false);
};

/// Common base class for sites that have a quantum clock degree of freedom.
class ClockDOF : public virtual Site
{
  public:
    using Ptr = std::shared_ptr<ClockDOF>;

    py::array clock_operators;

    ClockDOF(ElementarySpace::Ptr leg,
             py::array clock_operators,
             std::map<std::string, int64> state_labels = {},
             std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
             TensorBackend::Ptr backend = nullptr,
             std::optional<std::string> default_device = std::nullopt);

    /// Perform sanity checks.
    void test_sanity() override;

    /// Translate conservation law for a clock to a symmetry.
    ///
    /// Forwards to ``SpinDOF::conservation_law_to_symmetry`` (including ``use_test_su2``).
    [[nodiscard]] static Symmetry::Ptr conservation_law_to_symmetry(
      std::optional<std::string> conserve,
      bool use_test_su2 = false);
};

/// Common base class for sites that have an anyonic degree of freedom.
class AnyonDOF : public virtual Site
{
  public:
    using Ptr = std::shared_ptr<AnyonDOF>;

    std::vector<std::string> sector_names;

    AnyonDOF(ElementarySpace::Ptr leg,
             std::vector<std::string> sector_names = {},
             std::map<std::string, int64> state_labels = {},
             std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
             TensorBackend::Ptr backend = nullptr,
             std::optional<std::string> default_device = std::nullopt);

    /// Perform sanity checks.
    void test_sanity() override;
};

/// Common base class for sites that have a bosonic or fermionic degree of freedom.
///
/// Requires that the local basis is such that the `number_operators` of all species
/// are diagonal.
///
/// Attributes:
///
/// num_species : int
///     Number of boson species.
/// creators : 3D array
///     The vector of creation operators as a numpy array with shape ``(dim, dim, num_species)``
///     and axes ``[p, p*, i]``, where `i` corresponds to the different species of bosons (i.e.,
///     ``[Bd0, Bd1`, ...]`` stacked along axis 2).
/// annihilators : 3D array
///     The vector of annihilation operators as a numpy array with shape ``(dim, dim,
///     num_species)`` and axes ``[p, p*, i]``, where `i` corresponds to the different species of
///     bosons (i.e.,
///     ``[B0, B1`, ...]`` stacked along axis 2).
/// anti_commute_sign : float
///     ``+1`` for bosons, ``-1`` for fermions.
/// species_names : list of (str | None)
///     Names for each of the species.
/// number_operators : 3D array
///     The vector of occupation number operators with shape ``(dim, dim, num_species)``.
/// n_tot : 2D array
///     The total occupation number operator with shape ``(dim, dim)``.
class OccupationDOF : public virtual Site
{
  public:
    using Ptr = std::shared_ptr<OccupationDOF>;

    int64 num_species{};
    py::array creators;
    py::array annihilators;
    int64 anti_commute_sign{};
    std::vector<std::optional<std::string>> species_names;
    py::array number_operators;
    py::array n_tot;

    OccupationDOF(ElementarySpace::Ptr leg,
                  py::array creators,
                  py::array annihilators,
                  int64 anti_commute_sign,
                  std::vector<std::optional<std::string>> species_names = {},
                  std::map<std::string, int64> state_labels = {},
                  std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
                  TensorBackend::Ptr backend = nullptr,
                  std::optional<std::string> default_device = std::nullopt);

    void test_sanity() override;

    /// Add occupation and parity operators for each species.
    virtual void add_individual_occupation_ops();
    void add_total_occupation_ops();

    /// Wrapper around ``annihilators[:, :, species]``, optionally including JW strings.
    ///
    /// If `include_JW`, we include the ``(-1) ** n_k`` from all ``k < species``.
    [[nodiscard]] virtual py::array get_annihilator_numpy(py::object species,
                                                          bool include_JW = false) = 0;
    /// Wrapper around ``creators[:, :, species]``, optionally including JW strings.
    ///
    /// If `include_JW`, we include the ``(-1) ** n_k`` from all ``k < species``.
    [[nodiscard]] virtual py::array get_creator_numpy(py::object species,
                                                      bool include_JW = false) = 0;

    /// Get the occupation number operator for some or multiple species as a numpy array.
    [[nodiscard]] py::array get_occupation_numpy(py::object species = py::none());
    [[nodiscard]] int64 get_species_idx(py::object species) const;

  protected:
    std::map<std::string, int64> species_name_to_idx;
};

/// Common base class for sites that have a bosonic degree of freedom.
///
/// Requires that the local basis is such that the `number_operators` of all species
/// are diagonal.
///
/// Mutually exclusive with `FermionicDOF`. Sites containing both bosonic and fermionic
/// degrees of freedom can be realized by grouping of a bosonic site with a fermionic one.
///
/// Attributes:
///
/// Nmax : 1D array of int
///     Cutoff defining the maximum number of bosons per species and site. ``Nmax[i]`` corresponds
///     to the cutoff for the `i`th species; a value of ``Nmax[i] = 1`` describes hard-core bosons.
class BosonicDOF : public OccupationDOF
{
  public:
    using Ptr = std::shared_ptr<BosonicDOF>;

    py::array Nmax;
    py::array JW;

    BosonicDOF(ElementarySpace::Ptr leg,
               py::array Nmax,
               py::array creators,
               py::array annihilators,
               std::vector<std::optional<std::string>> species_names = {},
               std::map<std::string, int64> state_labels = {},
               std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
               TensorBackend::Ptr backend = nullptr,
               std::optional<std::string> default_device = std::nullopt);

    void test_sanity() override;

    void add_individual_occupation_ops() override;

    [[nodiscard]] py::array get_annihilator_numpy(py::object species,
                                                  bool include_JW = false) override;
    [[nodiscard]] py::array get_creator_numpy(py::object species,
                                              bool include_JW = false) override;

    [[nodiscard]] static Symmetry::Ptr conservation_law_to_symmetry(py::object conserve);

    [[nodiscard]] static std::pair<py::array, py::array> creation_annihilation_op_from_single_Nmax(
      int64 Nmax,
      int64 dim);

    [[nodiscard]] static std::pair<py::array, py::array> creation_annihilation_ops_from_Nmax(
      py::array Nmax,
      int64 dim);

    [[nodiscard]] static std::pair<py::array, py::array>
    creation_annihilation_ops(int64 num_species, py::array Nmax, int64 dim);
};

/// Common base class for sites that have a fermionic degree of freedom.
///
/// Requires that the local basis is such that the `number_operators` of all species
/// are diagonal.
///
/// Mutually exclusive with `BosonicDOF`. Sites containing both bosonic and fermionic
/// degrees of freedom can be realized by grouping of a bosonic site with a fermionic one.
class FermionicDOF : public OccupationDOF
{
  public:
    using Ptr = std::shared_ptr<FermionicDOF>;

    py::array partial_JWs;
    py::array JW;

    FermionicDOF(ElementarySpace::Ptr leg,
                 py::array creators,
                 py::array annihilators,
                 std::vector<std::optional<std::string>> species_names = {},
                 std::map<std::string, int64> state_labels = {},
                 std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
                 TensorBackend::Ptr backend = nullptr,
                 std::optional<std::string> default_device = std::nullopt);

    void test_sanity() override;

    [[nodiscard]] py::array get_annihilator_numpy(py::object species,
                                                  bool include_JW = false) override;
    [[nodiscard]] py::array get_creator_numpy(py::object species,
                                              bool include_JW = false) override;

    [[nodiscard]] static Symmetry::Ptr conservation_law_to_symmetry(
      std::optional<std::string> conserve);

    [[nodiscard]] static std::pair<py::array, py::array> creation_annihilation_ops(
      int64 num_species);
};

[[nodiscard]] py::array as_immutable_array(py::object a, py::object dtype = py::none());

} // namespace lasyten
