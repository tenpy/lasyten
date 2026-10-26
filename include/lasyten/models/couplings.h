#pragma once

#include <lasyten/models/degrees_of_freedom.h>
#include <lasyten/symmetries/sector.h>
#include <lasyten/tensors/ops_legs.h>
#include <lasyten/tensors/symmetric_tensor.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace lasyten {

[[nodiscard]] std::map<std::string, py::object> space_to_dict(ElementarySpace::Ptr space);
/// Recursively turn the output of the ``_*_to_dict`` helpers into hashable nested tuples.
[[nodiscard]] py::object freeze(py::object obj);

/// A coupling is an operator on a few `Site` s, factorized as one tensor per site.
///
/// A coupling represents an operator of the following form::
///
///     |        p0*  p1*  ..  pN*
///     |        │    │    │    │
///     |       ┏┷━━━━┷━━━━┷━━━━┷┓
///     |       ┃       h        ┃
///     |       ┗┯━━━━┯━━━━┯━━━━┯┛
///     |        │    │    │    │
///     |        p0   p1   ..   pN
///
/// The intended use case is to build tensor network representations (e.g. MPOs) of Hamiltonians.
///
/// Attributes:
///
/// sites : list of `Site`
///     The sites that the operators act on.
/// factorization : list of `SymmetricTensor`
///     A list of tensors that, if contracted, give the operator that is represented.
///     Each tensor ``factorization[i]`` has legs ``[wL, p, wR, p*]``, where ``p`` and ``p*``
///     are the physical `leg` of the corresponding ``sites[i]``, and where contracting
///     the ``wL`` and ``wR`` legs in an MPO-like geometry gives the multi-site operator.
/// name : str, optional
///     A descriptive name that can be used when pretty-printing, to identify the coupling.
///     For example, a Heisenberg coupling is usually initialized with name ``'S.S'``.
class Coupling
{
  public:
    std::vector<Site::Ptr> sites;
    std::vector<SymmetricTensorPtr> factorization;
    std::optional<std::string> name;

    Coupling(std::vector<Site::Ptr> sites,
             std::vector<SymmetricTensorPtr> factorization,
             std::optional<std::string> name = std::nullopt);

    /// Convert a dense block to a `Coupling`.
    ///
    /// @param operator_ The data to be converted to a Coupling as a backend-specific block or some
    /// data that can be converted using `as_block`. The order of axes must match the `sites`, that
    /// is, the axes correspond to ``[p0, p1, ..., p1*, p0*]`` (codomain legs ascending, domain
    /// legs descending), where ``pi`` corresponds to site ``sites[i]``. The block should be given
    /// in the "public" basis order of the sites, i.e., according to ``sites[i].sectors_of_basis``.
    /// @param sites The sites that the operators act on.
    /// @param name A descriptive name that can be used when pretty-printing, to identify the
    /// coupling.
    /// @param dtype If given, the block is converted to that dtype and the resulting tensors in
    /// the factorization will have that dtype. By default, we detect the dtype from the block.
    /// @param understood_braiding Set if the caller has accounted for non-trivial braiding of the
    /// dense block.
    /// @param cutoff_singular_values If given, truncate singular values (see
    /// `horizontal_factorization`) below this threshold. If omitted, the `coupling_cutoff` config
    /// option is used.
    [[nodiscard]] static Coupling from_dense_block(
      py::object operator_,
      std::vector<Site::Ptr> sites,
      std::optional<std::string> name = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      bool understood_braiding = false,
      std::optional<float64> cutoff_singular_values = std::nullopt);

    /// Convert an operator / tensor to a `Coupling`.
    ///
    /// Decomposes an operator into factors using `horizontal_factorization` to
    /// obtain the `factorization` of the coupling.
    ///
    /// @param operator_ Operator to be converted to a coupling. The legs should be ordered as
    /// ``[p0, p1, ..., p1*, p0*]``, where ``pi`` and ``pi*`` correspond to the legs associated
    /// with site ``sites[i]``.
    /// @param sites The sites that the operator acts on.
    /// @param name A descriptive name that can be used when pretty-printing, to identify the
    /// coupling. For example, a Heisenberg coupling is usually initialized with name ``'S.S'``.
    /// @param cutoff If given, truncate singular values (see `horizontal_factorization`) below
    /// this threshold. If omitted, the `coupling_cutoff` config option is used.
    [[nodiscard]] static Coupling from_tensor(SymmetricTensorPtr operator_,
                                              std::vector<Site::Ptr> sites,
                                              std::optional<std::string> name = std::nullopt,
                                              std::optional<float64> cutoff = std::nullopt);

    /// Convert to a tensor.
    [[nodiscard]] SymmetricTensorPtr to_tensor() const;
    /// Convert to a numpy array.
    [[nodiscard]] py::array to_numpy(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      py::object dtype = py::none(),
      bool understood_braiding = false) const;

    /// Place this coupling's tensors among `all_sites`, filling the gaps with identities.
    ///
    /// Returns a new coupling spanning `all_sites` from the first to the last of
    /// `coupling_positions` (inclusive). `self`'s tensors sit at those positions and every site in
    /// between gets an identity tensor independent of what site it is.
    ///
    /// @param all_sites The sites the returned coupling should be based on. Only the positions
    /// defined by `coupling_positions` are used.
    /// @param coupling_positions Strictly ascending, one entry per `factorization` tensor: the
    /// index into `all_sites` where that tensor should sit.
    /// @returns Spans all_sites[coupling_positions[0] to coupling_positions[-1] + 1].
    [[nodiscard]] Coupling stretch_with_identities(
      std::vector<Site::Ptr> const& all_sites,
      std::vector<int64> const& coupling_positions) const;

    /// Decompose into a sum of "elementary" couplings.
    ///
    /// A coupling is elementary if each virtual leg (``wL``, ``wR``) has total multiplicity one
    /// (``total_multiplicity == 1``), i.e. one sector appearing exactly once. The decomposition
    /// is obtained by projecting every virtual bond onto each of its sector copies. Terms that
    /// vanish identically are omitted.
    ///
    /// @returns Couplings on the same sites (and with the same name) whose sum is this coupling.
    [[nodiscard]] std::vector<Coupling> elementary_decomposition() const;

    /// Permute the sites of this coupling, braiding through the (possibly anyonic) legs.
    ///
    /// Realizes `permutation` as a sequence of elementary adjacent-site transpositions and
    /// applies each as a local swap gate to only the two `factorization` tensors it touches
    /// (contract the pair, braid their physical ``(p, p*)`` legs past each other as a single
    /// unit -- the two legs of one site never cross each other -- and re-split the pair via
    /// `horizontal_factorization`), leaving every other tensor in `factorization` untouched.
    /// This is analogous to how `PermuteLegsInstructionEngine` realizes a leg permutation as a
    /// sequence of elementary swaps, tracking a `levels` list that is itself reordered as legs
    /// move.
    ///
    /// Results are cached on `self` (not shared with `self`'s other permutations, or with the
    /// result of this call): calling ``self.permute(permutation, ...)`` twice with the same
    /// `permutation` returns the same (cached) result, using `levels` only the first time; a
    /// different `permutation` triggers a new computation.
    ///
    /// @param permutation A permutation of ``range(len(self.sites))``. ``permutation[k]`` is the
    /// index (in `self`'s current order) of the site that ends up at new position `k`.
    /// @param levels One entry per site of `self` (in `self`'s current order): its "height", used
    /// to derive the braid chirality (over/under) for each elementary transposition, the same way
    /// `Symmetry` legs with a higher level braid over those with a lower one. Only needed for
    /// symmetries without a symmetric braid (see `braiding_style`); ignored otherwise.
    /// @returns A new coupling with `sites` (and the represented operator) reordered according to
    /// `permutation`.
    [[nodiscard]] Coupling permute(std::vector<int64> const& permutation,
                                   std::optional<LevelsSpec> levels = std::nullopt) const;

    [[nodiscard]] std::tuple<py::object, std::vector<Site::Ptr>, std::vector<SymmetricTensorPtr>>
    key() const;

    [[nodiscard]] bool operator==(Coupling const& other) const;
    [[nodiscard]] size_t hash() const;

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] int64 num_sites() const { return static_cast<int64>(sites.size()); }

    /// Perform sanity checks.
    void test_sanity() const;

    std::vector<int64> _levels;
    mutable std::vector<std::pair<std::vector<int64>, Coupling>> _permuted;
    mutable std::vector<std::pair<std::vector<int64>, py::object>> _permuted_py;

  private:
};

/// Two-site coupling between spins.
///
/// \f[
///     h_{ij} = \mathtt{Jx} S_i^x S_j^x + \mathtt{Jy} S_i^y S_j^y + \mathtt{Jz} S_i^z S_j^z
/// \f]
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param Jx, Jy, Jz Prefactor, as given above. By default, all prefactors vanish.
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling spin_spin_coupling(std::vector<Site::Ptr> sites,
                                          float64 Jx = 0,
                                          float64 Jy = 0,
                                          float64 Jz = 0,
                                          py::object backend = py::none(),
                                          py::object device = py::none(),
                                          py::object name = py::none());

/// Single-site coupling of a spin to an external field.
///
/// \f[
///     h_i = \mathtt{hx} S_i^x + \mathtt{hy} S_i^y + \mathtt{hz} S_i^z
/// \f]
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param hx, hy, hz Prefactor, as given above. By default, all prefactors vanish.
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling spin_field_coupling(std::vector<Site::Ptr> sites,
                                           float64 hx = 0,
                                           float64 hy = 0,
                                           float64 hz = 0,
                                           py::object backend = py::none(),
                                           py::object device = py::none(),
                                           py::object name = py::none());

/// Two-site AKLT coupling between spins.
///
/// \f[
///     h_{ij} = \mathtt{J} [\vec{S}_i \cdot \vec{S}_j + \frac{1}{3} (\vec{S}_i \cdot
///     \vec{S}_j)^2]
/// \f]
///
/// This is the coupling originally defined by Affleck, Kennedy, Lieb, Tasaki
/// in [Affleck1987](https://doi.org/10.1103/PhysRevLett.59.799), except we drop the constant part
/// of 1/3 per bond and rescale with a factor of 2, i.e. @f$ h_{ij} = 2 P^{S=2}_{i, j} + const.
/// @f$.
///
/// It was defined for spin-1 degrees of freedom in the original work, but we allow any site
/// with a spin DOF. Note that the coupling simplifies to a Heisenberg coupling for spin-1/2.
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param J Prefactor, as given above. By default use ``1``.
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling aklt_coupling(std::vector<Site::Ptr> sites,
                                     float64 J = 1,
                                     py::object backend = py::none(),
                                     py::object device = py::none(),
                                     py::object name = py::none());

/// Two-site Heisenberg coupling between spins.
///
/// \f[
///     h_{ij} = \mathtt{J} \vec{S}_i \cdot \vec{S}_j
/// \f]
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param J Prefactor, as given above. By default use ``1``, i.e. an anti-ferromagnetic coupling.
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling heisenberg_coupling(std::vector<Site::Ptr> sites,
                                           float64 J = 1,
                                           py::object backend = py::none(),
                                           py::object device = py::none(),
                                           py::object name = py::none());

/// Chiral coupling of three spins.
///
/// \f[
///     h_{ijk} = \mathtt{chi} \vec{S}_i \cdot ( \vec{S}_j \times \vec{S}_k )
/// \f]
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param chi Prefactor, as given above. By default use ``1``.
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling chiral_3spin_coupling(std::vector<Site::Ptr> sites,
                                             float64 chi = 1,
                                             py::object backend = py::none(),
                                             py::object device = py::none(),
                                             py::object name = py::none());

/// Chemical potential for bosons or fermions. Single-site coupling.
///
/// \f[
///     h_i = -\mathtt{mu} \sum_{k \in \mathtt{species}} n_{i, k}
/// \f]
///
/// where @f$ n_{i, k} @f$ is the occupation number of species @f$ k @f$ on site @f$ i @f$.
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param mu Chemical potential, as defined above.
/// @param species If given, the chemical potential only couples to the occupation of this species.
/// By default, it couples to the total occupation of all species.
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling chemical_potential(std::vector<Site::Ptr> sites,
                                          float64 mu,
                                          py::object species = py::none(),
                                          py::object backend = py::none(),
                                          py::object device = py::none(),
                                          py::object name = py::none());

/// Onsite interaction for bosons or fermions. Single-site coupling.
///
/// \f[
///     h_i = \frac{U}{2} n_i^2
/// \f]
///
/// where @f$ n_i @f$ is the total occupation number, or the occupation of a single `species`.
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param U Prefactor, as defined above. By default, use ``1``, i.e. a repulsive interaction.
/// @param species If given, we use only the occupation of this one species as the density @f$ n_i
/// @f$. By default, we use the total occupation of all species.
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling onsite_interaction(std::vector<Site::Ptr> sites,
                                          float64 U = 1,
                                          py::object species = py::none(),
                                          py::object backend = py::none(),
                                          py::object device = py::none(),
                                          py::object name = py::none());

/// Density-density interaction. Two-site coupling.
///
/// \f[
///     h_{ij} = \mathtt{V} n_i n_j
/// \f]
///
/// where @f$ n_i @f$ is the total occupation number.
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param V Prefactor, as defined above. By default, use ``1``, i.e. a repulsive interaction.
/// @param species_i, species_j If given, we use only the occupation of this one species as the
/// density @f$ n_{i/j} @f$. By default, we use the total occupation of all species. Note that if
/// the two species are different, this coupling alone is not hermitian!
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling density_density_interaction(std::vector<Site::Ptr> sites,
                                                   float64 V = 1,
                                                   py::object species_i = py::none(),
                                                   py::object species_j = py::none(),
                                                   py::object backend = py::none(),
                                                   py::object device = py::none(),
                                                   py::object name = py::none());

/// Hopping of fermions or bosons. Two-site coupling.
///
/// \f[
///     h_{ij} = -\mathtt{t} \sum_{k \in \mathtt{species}} a_{i, k_i}^\dagger a_{j, k_j} +
///     h.c.
/// \f]
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param t Prefactor, as given above. By default ``1``.
/// @param species Which species should participate (the sum above goes over ``k_i, k_j in
/// zip(*species)``). By default, we let @f$ k_i = k_j @f$ go over all species, i.e. include all
/// "species preserving" hoppings.
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling hopping(std::vector<Site::Ptr> sites,
                               float64 t = 1,
                               py::object species = py::none(),
                               py::object backend = py::none(),
                               py::object device = py::none(),
                               py::object name = py::none());

/// Superconducting pairing of fermions or bosons. Two-site coupling.
///
/// \f[
///     h_{ij} = \mathtt{Delta} \sum_{k\in\mathtt{species}} a_{i, k_i}^\dagger a_{j,
///     k_j}^\dagger + h.c.
/// \f]
///
/// .. note ::
///     This coupling assumes distinct sites @f$ i \neq j @f$.
///     Use `onsite_pairing` for @f$ i = j @f$.
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param Delta Prefactor, as given above. By default ``1``.
/// @param species Which species should participate (the sum above goes over ``k_i, k_j in
/// zip(*species)``). By default, we let @f$ k_i = k_j @f$ go over all species, i.e. include all
/// "same-species" pairings. onsite_pairing
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling pairing(std::vector<Site::Ptr> sites,
                               float64 Delta = 1,
                               py::object species = py::none(),
                               py::object backend = py::none(),
                               py::object device = py::none(),
                               py::object name = py::none());

/// Superconducting pairing of fermions or bosons. Single-site coupling.
///
/// \f[
///     h_i = \mathtt{Delta} \sum_{k\in\mathtt{species}} a_{i, k_1}^\dagger a_{i,
///     k_2}^\dagger + h.c.
/// \f]
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param Delta Prefactor, as given above. By default ``1``.
/// @param species Which species should participate (the sum above goes over ``k_1, k_2 in
/// zip(*species)``). By default, we let @f$ k_1 = k_2 @f$ go over all species, i.e. include all
/// "same-species" pairings. pairing
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling onsite_pairing(std::vector<Site::Ptr> sites,
                                      float64 Delta = 1,
                                      py::object species = py::none(),
                                      py::object backend = py::none(),
                                      py::object device = py::none(),
                                      py::object name = py::none());

/// Two-site coupling between quantum clocks.
///
/// \f[
///     h_{ij} = \mathtt{Jx} X_i X_j^\dagger + \mathtt{Jz} Z_i Z_j^\dagger + h.c.
/// \f]
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param Jx, Jz Prefactor, as given above. By default, all prefactors vanish.
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling clock_clock_coupling(std::vector<Site::Ptr> sites,
                                            float64 Jx = 0,
                                            float64 Jz = 0,
                                            py::object backend = py::none(),
                                            py::object device = py::none(),
                                            py::object name = py::none());

/// Single-site coupling of a quantum clock to an external field.
///
/// \f[
///     h_i = \mathtt{hx} X_i + \mathtt{hz} Z_i + h.c.
/// \f]
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param hx, hz Prefactor, as given above. By default, all prefactors vanish.
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling clock_field_coupling(std::vector<Site::Ptr> sites,
                                            std::optional<float64> hx = std::nullopt,
                                            std::optional<float64> hz = std::nullopt,
                                            py::object backend = py::none(),
                                            py::object device = py::none(),
                                            py::object name = py::none());

/// Coupling that is given by the projector onto a single sector
///
/// The number of sites is arbitrary and the operator @f$ h_{ij...} @f$ is given
/// by `from_sector_projection`, with prefactor `J`.
/// Note that positive `J` mean that states that fuse to the given `sector` are energetically
/// *disfavored*.
[[nodiscard]] Coupling sector_projection_coupling(std::vector<Site::Ptr> sites,
                                                  float64 J,
                                                  Sector sector,
                                                  py::object name,
                                                  py::object backend = py::none(),
                                                  py::object device = py::none());

/// Two-site coupling of Fibonacci anyons that energy splits fusion to vacuum or tau.
///
/// \f[
///     h_{ij} = -J P^\text{vac}_{i, j}
/// \f]
///
/// @param sites The sites that the coupling acts on. Note that the order matters for the final leg
/// order.
/// @param J Prefactor, as given above. By default ``1``. Positive `J` energetically favor the
/// trivial fusion channel, i.e. they are the "antiferromagnetic" analog.
/// @param backend, device, name Optional tensor backend, device, and coupling name.
[[nodiscard]] Coupling gold_coupling(std::vector<Site::Ptr> sites,
                                     float64 J = 1,
                                     py::object backend = py::none(),
                                     py::object device = py::none(),
                                     py::object name = py::none());

} // namespace lasyten
