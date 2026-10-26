"""A collection of tests for :mod:`lasyten.models.couplings`."""
# Copyright (C) TeNPy Developers, Apache license

import itertools as it
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pytest
from numpy import testing as npt

import lasyten
from lasyten import SymmetryError, backends, tensors
from lasyten.models import couplings, degrees_of_freedom, sites
from lasyten.models.couplings import heisenberg_coupling, spin_field_coupling
from lasyten.models.sites import SpinSite
from lasyten.symmetries import BraidChiralityUnspecifiedError
from lasyten.tensors import permute_legs


def check_coupling(coupling_cls, site_num: int, invalid_site_nums: list[int], boson_fermion_mixing: bool, **kwargs):
    """Perform common checks that make sense for any coupling"""
    # it does not matter what site we use since the number of sites is checked first
    site = sites.SpinlessBosonSite([1])
    for n in invalid_site_nums:
        with pytest.raises(ValueError, match='Invalid number of sites.'):
            _ = coupling_cls([site] * n, **kwargs)
    if boson_fermion_mixing:
        site_list = [site, sites.SpinlessFermionSite(1)]
        site_list.extend([site] * (site_num - 2))
        msg = 'Bosonic and fermionic sites are incompatible and cannot be combined for constructing couplings.'
        with pytest.raises(SymmetryError, match=msg):
            _ = coupling_cls(site_list, **kwargs)


def generate_spin_dofs(backend: backends.TensorBackend) -> list[degrees_of_freedom.SpinDOF]:
    """Return a list of `SpinDOF` sites whose symmetries are consistent with `backend`."""
    site_list = []
    for spin in [0.5, 1, 1.5, 2]:
        site_list.append(sites.SpinSite(S=spin, conserve='None', backend=backend))
        if not isinstance(backend, backends.NoSymmetryBackend):
            site_list.append(sites.SpinSite(S=spin, conserve='parity', backend=backend))
            site_list.append(sites.SpinSite(S=spin, conserve='Sz', backend=backend))
        if isinstance(backend, backends.FusionTreeBackend):
            site_list.append(sites.SpinSite(S=spin, conserve='SU(2)', backend=backend, use_test_su2=True))
    if isinstance(backend, backends.FusionTreeBackend):
        all_conserve_N = ['N', 'parity']
        all_conserve_S = ['SU(2)', 'Sz', 'parity', 'None']
        for conserve_N, conserve_S in it.product(all_conserve_N, all_conserve_S):
            site_list.append(
                sites.SpinHalfFermionSite(
                    conserve_N,
                    conserve_S,
                    backend=backend,
                    use_test_su2=(conserve_S == 'SU(2)'),
                )
            )
    return site_list


def generate_bosonic_dofs(
    backend: backends.TensorBackend, conserve: Sequence[Literal['N', 'parity', 'None']] = ['N', 'parity', 'None']
) -> list[degrees_of_freedom.BosonicDOF]:
    """Return a list of `BosonicDOF` sites whose symmetries are consistent with `backend`."""
    site_list = []
    for Nmax in [[3], [3, 2]]:
        if isinstance(backend, backends.NoSymmetryBackend):
            all_conserve = ['None']
        else:
            all_conserve = conserve[:]
            if len(Nmax) > 1:
                all_conserve.extend(it.product(all_conserve, repeat=len(Nmax)))
        for cons in all_conserve:
            site_list.append(sites.SpinlessBosonSite(Nmax, cons, backend=backend))
    return site_list


def generate_fermionic_dofs(
    backend: backends.TensorBackend, conserve: Sequence[Literal['N', 'parity']] = ['N', 'parity']
) -> list[degrees_of_freedom.FermionicDOF]:
    """Return a list of `FermionicDOF` sites whose symmetries are consistent with `backend`."""
    site_list = []
    if isinstance(backend, (backends.NoSymmetryBackend, backends.AbelianBackend)):
        # fermionic exchange cannot be encoded
        # do it like this (rather than fixing the backend from the start) such that
        # a potential extension of the ablian backend to fermions automatically works
        with pytest.raises(ValueError):
            _ = sites.SpinlessFermionSite(num_species=1, backend=backend)
        return site_list
    for num_species in [1, 2]:
        all_conserve = conserve[:]
        individual_conserve = conserve + ['None']
        if num_species > 1:
            all_conserve.extend(it.product(individual_conserve, repeat=num_species))
        for cons in all_conserve:
            site_list.append(sites.SpinlessFermionSite(num_species, cons, backend=backend))
    all_conserve_N = conserve
    all_conserve_S = ['Sz', 'parity', 'None']
    if isinstance(backend, backends.FusionTreeBackend):
        all_conserve_S.append('SU(2)')
    for conserve_N, conserve_S in it.product(all_conserve_N, all_conserve_S):
        site_list.append(
            sites.SpinHalfFermionSite(
                conserve_N,
                conserve_S,
                backend=backend,
                use_test_su2=(conserve_S == 'SU(2)'),
            )
        )
    return site_list


def generate_clock_dofs(backend: backends.TensorBackend) -> list[degrees_of_freedom.ClockDOF]:
    """Return a list of `ClockDOF` sites whose symmetries are consistent with `backend`."""
    site_list = []
    for q in [2, 3, 4]:
        site_list.append(sites.ClockSite(q, conserve='None', backend=backend))
        if not isinstance(backend, backends.NoSymmetryBackend):
            site_list.append(sites.ClockSite(q, conserve='Z_N', backend=backend))
    return site_list


def generate_anyon_dofs(block_backend: lasyten.block_backends.BlockBackend) -> list[degrees_of_freedom.AnyonDOF]:
    """Return a list of `AnyonDOF` sites."""
    backend = backends.get_backend('fusion_tree', block_backend=block_backend)
    site_list = [
        sites.FibonacciAnyonSite(backend=backend),
        sites.IsingAnyonSite(nu=1, backend=backend),
        sites.IsingAnyonSite(nu=3, backend=backend),
        sites.GoldenSite(backend=backend),
        sites.SU2kSpin1Site(k=4, backend=backend),
        sites.SU2kSpin1Site(k=5, backend=backend),
    ]
    return site_list


@pytest.mark.parametrize('codom', [1, 2, 3])
def test_coupling(codom, make_compatible_space):
    # Keep spaces small: SU2 FusionTree with codom=3 is otherwise very expensive.
    max_sectors = 3 if codom < 3 else 2
    max_mult = 3 if codom < 3 else 2
    legs = [make_compatible_space(max_sectors=max_sectors, max_mult=max_mult) for _ in range(codom)]
    labels = [f'p{i}' for i in range(codom)]
    labels = [*labels, *[l + '*' for l in labels[::-1]]]
    T = tensors.SymmetricTensor.from_random_normal(codomain=legs, domain=legs, labels=labels)
    site_list = [degrees_of_freedom.Site(leg) for leg in legs]
    coupling = couplings.Coupling.from_tensor(T, site_list, name='name')
    coupling.test_sanity()
    assert coupling.name == 'name'
    assert coupling.num_sites == codom
    assert tensors.almost_equal(coupling.to_tensor(), T)
    if T.symmetry.can_be_dropped:
        coupling_to_numpy = coupling.to_numpy(understood_braiding=True)
        npt.assert_almost_equal(coupling_to_numpy, T.to_numpy(understood_braiding=True))
        coupling2 = couplings.Coupling.from_dense_block(coupling_to_numpy, site_list, understood_braiding=True)
        coupling2.test_sanity()
        npt.assert_array_equal(coupling2.sites, coupling.sites)
        # Factorizations need not match site-by-site (QR/SVD gauge on virtual legs).
        assert tensors.almost_equal(coupling2.to_tensor(), T)
        npt.assert_almost_equal(coupling2.to_numpy(understood_braiding=True), T.to_numpy(understood_braiding=True))


# TEST SPIN COUPLINGS


def test_spin_spin_coupling(any_backend, np_random):
    site_list = generate_spin_dofs(any_backend)
    num_sites = min(5, len(site_list))
    site_list = np_random.choice(site_list, size=num_sites, replace=False)
    for i, site1 in enumerate(site_list):
        check_evs = False
        Jx, Jy, Jz = np_random.random(3)
        # either SpinSite or SpinHalfFermionSite
        conserve = site1.conserve if isinstance(site1, sites.SpinSite) else site1.conserve_S
        if conserve in ['Sz']:
            check_evs = True
            Jx = Jy = 0
        elif conserve in ['SU(2)']:
            check_evs = True
            Jx = Jy = Jz

        # test different site combinations
        for site2 in site_list[: i + 1]:
            # Note: is_equivalent_to does not work here since it does not distinguish
            # between U(1) fermion number symmetry and Sz spin symmetry for fermions
            if not site1.symmetry == site2.symmetry:
                continue
            coupling = couplings.spin_spin_coupling([site1, site2], Jx=Jx, Jy=Jy, Jz=Jz)
            coupling.test_sanity()
            tensor = coupling.to_tensor()
            # hermiticity
            assert tensors.almost_equal(tensor.hc, tensor)
            # trace is zero
            npt.assert_almost_equal(tensors.trace(tensor).to_numpy(), 0)
            if site1 == site2:
                # commutation relation
                tensor_commuted = tensors.permute_legs(tensor, codomain=[1, 0], domain=[2, 3])
                tensor_commuted.relabel({'p0': 'p1', 'p1': 'p0', 'p0*': 'p1*', 'p1*': 'p0*'})
                assert tensors.almost_equal(tensor_commuted, tensor)

            # check eigenvalues of special cases
            if check_evs:
                if conserve in ['Sz']:
                    if isinstance(site1, sites.SpinSite):
                        expect_evs = np.arange(-site1.S, site1.S + 1)[:, None]
                        expect_evs = expect_evs @ np.arange(-site2.S, site2.S + 1)[None, :]
                        expect_evs = expect_evs.flatten()
                    else:
                        # spin-1/2 fermions
                        expect_evs = np.array([0] * 12 + [-0.25, 0.25] * 2)
                elif conserve in ['SU(2)']:
                    if isinstance(site1, sites.SpinSite):
                        double_spin = site1.double_total_spin + site2.double_total_spin
                        lower_limit = abs(site1.double_total_spin - site2.double_total_spin)
                        spin_tots = site1.S * (site1.S + 1) + site2.S * (site2.S + 1)
                        expect_evs = [[s * (s + 2) / 4] * (s + 1) for s in range(double_spin, lower_limit - 1, -2)]
                        expect_evs = (np.concatenate(expect_evs) - spin_tots) / 2.0
                    else:
                        expect_evs = np.array([0] * 12 + [0.25] * 3 + [-0.75])
                evs = tensor.to_numpy(leg_order=[0, 1, 3, 2], understood_braiding=True)
                evs = np.reshape(evs, (np.prod(evs.shape[:2]), -1))
                evs = np.sort(np.linalg.eigvalsh(evs))
                npt.assert_almost_equal(evs, np.sort(Jz * expect_evs))

    check_coupling(couplings.spin_spin_coupling, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=False)


def test_spin_field_coupling(any_backend, np_random):
    site_list = generate_spin_dofs(any_backend)
    num_sites = min(5, len(site_list))
    site_list = np_random.choice(site_list, size=num_sites, replace=False)
    for site in site_list:
        hx, hy, hz = np_random.random(3)
        # either SpinSite or SpinHalfFermionSite
        conserve = site.conserve if isinstance(site, sites.SpinSite) else site.conserve_S
        if conserve in ['Sz', 'parity']:
            hx = hy = 0
        elif conserve in ['SU(2)']:
            # coupling not allowed
            continue
        coupling = couplings.spin_field_coupling([site], hx=hx, hy=hy, hz=hz)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is zero
        npt.assert_almost_equal(tensors.trace(tensor).to_numpy(), 0)
        # check eigenvalues
        h = np.sqrt(hx**2 + hy**2 + hz**2)
        if isinstance(site, sites.SpinSite):
            expect_evs = np.arange(-site.S, site.S + 1)
        else:
            # spin-1/2 fermions
            expect_evs = np.array([0, 0, -0.5, 0.5])
        evs = tensor.to_numpy(understood_braiding=True)
        evs = np.sort(np.linalg.eigvalsh(evs))
        npt.assert_almost_equal(evs, np.sort(h * expect_evs))

    check_coupling(couplings.spin_field_coupling, site_num=1, invalid_site_nums=[2], boson_fermion_mixing=False)


def test_aklt_coupling(any_backend, np_random):
    site_list = generate_spin_dofs(any_backend)
    num_sites = min(5, len(site_list))
    site_list = np_random.choice(site_list, size=num_sites, replace=False)
    for i, site1 in enumerate(site_list):
        J = np_random.random()
        # test different site combinations
        for site2 in site_list[: i + 1]:
            if not site1.symmetry == site2.symmetry:
                continue
            coupling = couplings.aklt_coupling([site1, site2], J=J)
            coupling.test_sanity()
            tensor = coupling.to_tensor()
            # hermiticity
            assert tensors.almost_equal(tensor.hc, tensor)
            if site1 == site2:
                # commutation relation
                tensor_commuted = tensors.permute_legs(tensor, codomain=[1, 0], domain=[2, 3])
                tensor_commuted.relabel({'p0': 'p1', 'p1': 'p0', 'p0*': 'p1*', 'p1*': 'p0*'})
                assert tensors.almost_equal(tensor_commuted, tensor)

            if isinstance(site1, sites.SpinSite):
                double_spin = site1.double_total_spin + site2.double_total_spin
                lower_limit = abs(site1.double_total_spin - site2.double_total_spin)
                spin_tots = site1.S * (site1.S + 1) + site2.S * (site2.S + 1)
                expect_evs = [[s * (s + 2) / 4] * (s + 1) for s in range(double_spin, lower_limit - 1, -2)]
                expect_evs = (np.concatenate(expect_evs) - spin_tots) / 2.0
            else:
                expect_evs = np.array([0] * 12 + [0.25] * 3 + [-0.75])
            expect_evs += expect_evs**2 / 3.0
            evs = tensor.to_numpy(leg_order=[0, 1, 3, 2], understood_braiding=True)
            evs = np.reshape(evs, (np.prod(evs.shape[:2]), -1))
            evs = np.sort(np.linalg.eigvalsh(evs))
            npt.assert_almost_equal(evs, np.sort(J * expect_evs))
            if site1 == site2 and isinstance(site1, sites.SpinSite) and site1.double_total_spin == 2:
                # actual AKLT case
                npt.assert_almost_equal(evs, J * np.array([-2.0 / 3.0] * 4 + [4.0 / 3.0] * 5))

    check_coupling(couplings.aklt_coupling, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=False)


@pytest.mark.slow  # TODO can we speed it up?
def test_chiral_3spin_coupling(any_backend, np_random):
    site_list = generate_spin_dofs(any_backend)
    num_sites = min(3, len(site_list))
    site_list = np_random.choice(site_list, size=num_sites, replace=False)
    for i, site1 in enumerate(site_list):
        chi = np_random.random()
        # test different site combinations
        for site2 in site_list[: i + 1]:
            if not site1.symmetry == site2.symmetry:
                continue
            site3 = np_random.choice([site1, site2])
            coupling = couplings.chiral_3spin_coupling([site1, site2, site3], chi=chi)
            coupling.test_sanity()
            tensor = coupling.to_tensor()
            # hermiticity
            assert tensors.almost_equal(tensor.hc, tensor)
            # trace is zero
            npt.assert_almost_equal(tensors.trace(tensor).to_numpy(), 0)
            if site1 == site2:
                # cyclic permutation relation
                tensor_commuted = tensors.permute_legs(tensor, codomain=[2, 0, 1], domain=[3, 5, 4])
                relabel = {'p2': 'p0', 'p1': 'p2', 'p0': 'p1', 'p2*': 'p0*', 'p1*': 'p2*', 'p0*': 'p1*'}
                tensor_commuted.relabel(relabel)
                assert tensors.almost_equal(tensor_commuted, tensor)

    check_coupling(couplings.chiral_3spin_coupling, site_num=3, invalid_site_nums=[1, 2], boson_fermion_mixing=False)


# TEST BOSON AND FERMION COUPLINGS


def test_chemical_potential(any_backend, np_random):
    bosonic_sites = generate_bosonic_dofs(any_backend)
    num_sites = min(3, len(bosonic_sites))
    bosonic_sites = np_random.choice(bosonic_sites, size=num_sites, replace=False)
    fermionic_sites = generate_fermionic_dofs(any_backend)
    num_sites = min(3, len(fermionic_sites))
    fermionic_sites = np_random.choice(fermionic_sites, size=num_sites, replace=False)
    all_sites = [*bosonic_sites, *fermionic_sites]

    for site in all_sites:
        mu = np_random.random()
        species = np_random.integers(1, site.num_species + 1)
        species = np_random.choice(range(site.num_species), size=species, replace=False)
        if isinstance(site, sites.SpinHalfFermionSite):
            if site.conserve_S in ['SU(2)'] and len(species) == 1:
                species = np.append(species, 1 - species[0])
        coupling = couplings.chemical_potential([site], mu=mu, species=species)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # check eigenvalues
        Nmax = site.Nmax if isinstance(site, sites.SpinlessBosonSite) else [1] * site.num_species
        expect_evs = []
        for occupations in it.product(*[list(range(n + 1)) for n in Nmax]):
            expect_evs.append(-mu * sum([occupations[k] for k in species]))
        evs = tensor.to_numpy(understood_braiding=True)
        evs = np.sort(np.linalg.eigvalsh(evs))
        npt.assert_almost_equal(evs, np.sort(expect_evs))

    check_coupling(couplings.chemical_potential, site_num=1, invalid_site_nums=[2], boson_fermion_mixing=False, mu=1.0)


def test_onsite_interaction(any_backend, np_random):
    bosonic_sites = generate_bosonic_dofs(any_backend)
    num_sites = min(3, len(bosonic_sites))
    bosonic_sites = np_random.choice(bosonic_sites, size=num_sites, replace=False)
    fermionic_sites = generate_fermionic_dofs(any_backend)
    num_sites = min(3, len(fermionic_sites))
    fermionic_sites = np_random.choice(fermionic_sites, size=num_sites, replace=False)
    all_sites = [*bosonic_sites, *fermionic_sites]

    for site in all_sites:
        U = np_random.random()
        species = np_random.integers(1, site.num_species + 1)
        species = np_random.choice(range(site.num_species), size=species, replace=False)
        if isinstance(site, sites.SpinHalfFermionSite):
            if site.conserve_S in ['SU(2)'] and len(species) == 1:
                species = np.append(species, 1 - species[0])
        coupling = couplings.onsite_interaction([site], U=U, species=species)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # check eigenvalues
        Nmax = site.Nmax if isinstance(site, sites.SpinlessBosonSite) else [1] * site.num_species
        expect_evs = []
        for occupations in it.product(*[list(range(n + 1)) for n in Nmax]):
            n = sum([occupations[k] for k in species])
            expect_evs.append(U * n**2 / 2.0)
        evs = tensor.to_numpy(understood_braiding=True)
        evs = np.sort(np.linalg.eigvalsh(evs))
        npt.assert_almost_equal(evs, np.sort(expect_evs))

    check_coupling(couplings.onsite_interaction, site_num=1, invalid_site_nums=[2], boson_fermion_mixing=False)


@pytest.mark.slow  # TODO can we speed it up?
def test_density_density_interaction(any_backend, np_random):
    bosonic_sites = generate_bosonic_dofs(any_backend)
    num_sites = min(2, len(bosonic_sites))
    bosonic_sites = np_random.choice(bosonic_sites, size=num_sites, replace=False)
    fermionic_sites = generate_fermionic_dofs(any_backend)
    num_sites = min(2, len(fermionic_sites))
    fermionic_sites = np_random.choice(fermionic_sites, size=num_sites, replace=False)
    all_sites = [*bosonic_sites, *fermionic_sites]

    for site in all_sites:
        V = np_random.random()
        species1 = np_random.integers(1, site.num_species + 1)
        species1 = np_random.choice(range(site.num_species), size=species1, replace=False)
        species2 = np_random.integers(1, site.num_species + 1)
        species2 = np_random.choice(range(site.num_species), size=species2, replace=False)
        if isinstance(site, sites.SpinHalfFermionSite) and site.conserve_S in ['SU(2)']:
            species1 = species2 = [0, 1]
        coupling = couplings.density_density_interaction([site] * 2, V, species1, species2)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        if all(species1 == species2):
            # commutation relation
            tensor_commuted = tensors.permute_legs(tensor, codomain=[1, 0], domain=[2, 3])
            tensor_commuted.relabel({'p0': 'p1', 'p1': 'p0', 'p0*': 'p1*', 'p1*': 'p0*'})
            assert tensors.almost_equal(tensor_commuted, tensor)
        # check eigenvalues
        Nmax = site.Nmax if isinstance(site, sites.SpinlessBosonSite) else [1] * site.num_species
        n1 = []
        n2 = []
        for occupations in it.product(*[list(range(n + 1)) for n in Nmax]):
            n1.append(sum([occupations[k] for k in species1]))
            n2.append(sum([occupations[k] for k in species2]))
        expect_evs = V * np.outer(n1, n2).flatten()
        evs = tensor.to_numpy(leg_order=[0, 1, 3, 2], understood_braiding=True)
        evs = np.reshape(evs, (np.prod(evs.shape[:2]), -1))
        evs = np.sort(np.linalg.eigvalsh(evs))
        npt.assert_almost_equal(evs, np.sort(expect_evs))

    check_coupling(
        couplings.density_density_interaction, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=True
    )


@pytest.mark.slow  # TODO can we speed it up?
def test_hopping(any_backend, np_random):
    bosonic_sites = generate_bosonic_dofs(any_backend)
    num_sites = min(2, len(bosonic_sites))
    bosonic_sites = np_random.choice(bosonic_sites, size=num_sites, replace=False)
    fermionic_sites = generate_fermionic_dofs(any_backend)
    num_sites = min(2, len(fermionic_sites))
    fermionic_sites = np_random.choice(fermionic_sites, size=num_sites, replace=False)
    all_sites = [*bosonic_sites, *fermionic_sites]

    for site in all_sites:
        t = np_random.random()
        species1 = np_random.integers(1, site.num_species + 1)
        species1 = np_random.choice(range(site.num_species), size=species1, replace=False)
        species2 = np_random.integers(1, site.num_species + 1)
        species2 = np_random.choice(range(site.num_species), size=species2, replace=False)
        if len(species1) != len(species2):
            limit = min(len(species1), len(species2))
            species1 = species1[:limit]
            species2 = species2[:limit]

        if isinstance(site, (sites.SpinlessBosonSite, sites.SpinlessFermionSite)):
            if not isinstance(site.conserve, str):
                # easiest way to deal with symmetries on the individual species
                species2 = species1
        if isinstance(site, sites.SpinHalfFermionSite):
            if site.conserve_S in ['Sz']:
                species2 = species1
            elif site.conserve_S in ['SU(2)']:
                species1 = species2 = degrees_of_freedom.ALL_SPECIES

        coupling = couplings.hopping([site] * 2, t, species=(species1, species2))
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is zero
        npt.assert_almost_equal(tensors.trace(tensor).to_numpy(), 0)
        # if there is a permutation s.t. species1 <-> species2, we can commute the legs
        symmetric = False
        for perm in it.permutations(range(len(species1))):
            if np.all(species1[list(perm)] == species2) and np.all(species2[list(perm)] == species1):
                symmetric = True
        if symmetric:
            # commutation relation; this does commute for fermions since
            # a_0_k^\dagger a_1_l + hc -> (exchange legs) -> -1 * a_0_l a_1_k^\dagger + hc
            # = a_1_k^\dagger a_0_l + hc = a_0_l^\dagger a_1_k + hc
            tensor_commuted = tensors.permute_legs(tensor, codomain=[1, 0], domain=[2, 3])
            tensor_commuted.relabel({'p0': 'p1', 'p1': 'p0', 'p0*': 'p1*', 'p1*': 'p0*'})
            assert tensors.almost_equal(tensor_commuted, tensor)

    check_coupling(couplings.hopping, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=True)


def test_pairing(any_backend, np_random):
    bosonic_sites = generate_bosonic_dofs(any_backend, conserve=['parity', 'None'])
    num_sites = min(3, len(bosonic_sites))
    bosonic_sites = np_random.choice(bosonic_sites, size=num_sites, replace=False)
    fermionic_sites = generate_fermionic_dofs(any_backend, conserve=['parity'])
    num_sites = min(3, len(fermionic_sites))
    fermionic_sites = np_random.choice(fermionic_sites, size=num_sites, replace=False)
    all_sites = [*bosonic_sites, *fermionic_sites]

    for site in all_sites:
        Delta = np_random.random()
        species1 = np_random.integers(1, site.num_species + 1)
        species1 = np_random.choice(range(site.num_species), size=species1, replace=False)
        species2 = np_random.integers(1, site.num_species + 1)
        species2 = np_random.choice(range(site.num_species), size=species2, replace=False)
        if len(species1) != len(species2):
            limit = min(len(species1), len(species2))
            species1 = species1[:limit]
            species2 = species2[:limit]

        if isinstance(site, (sites.SpinlessBosonSite, sites.SpinlessFermionSite)):
            if not isinstance(site.conserve, str):
                # easiest way to deal with symmetries on the individual species
                species2 = species1
        if isinstance(site, sites.SpinHalfFermionSite):
            if site.conserve_S in ['Sz']:
                for i, k in enumerate(species1):
                    species2[i] = 1 - k
            elif site.conserve_S in ['SU(2)']:
                species1 = species2 = []

        coupling = couplings.pairing([site] * 2, Delta, species=(species1, species2))
        coupling.test_sanity()
        if len(species1) == 0:
            continue
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is zero
        npt.assert_almost_equal(tensors.trace(tensor).to_numpy(), 0)
        # if there is a permutation s.t. species1 <-> species2, we can commute the legs
        symmetric = False
        for perm in it.permutations(range(len(species1))):
            if np.all(species1[list(perm)] == species2) and np.all(species2[list(perm)] == species1):
                symmetric = True
        if symmetric:
            # commutation relation
            tensor_commuted = tensors.permute_legs(tensor, codomain=[1, 0], domain=[2, 3])
            tensor_commuted.relabel({'p0': 'p1', 'p1': 'p0', 'p0*': 'p1*', 'p1*': 'p0*'})
            assert tensors.almost_equal(tensor_commuted, site.anti_commute_sign * tensor)

    check_coupling(couplings.pairing, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=True)


def test_onsite_pairing(any_backend, np_random):
    bosonic_sites = generate_bosonic_dofs(any_backend, conserve=['parity', 'None'])
    num_sites = min(3, len(bosonic_sites))
    bosonic_sites = np_random.choice(bosonic_sites, size=num_sites, replace=False)
    fermionic_sites = generate_fermionic_dofs(any_backend, conserve=['parity'])
    num_sites = min(3, len(fermionic_sites))
    fermionic_sites = np_random.choice(fermionic_sites, size=num_sites, replace=False)
    all_sites = [*bosonic_sites, *fermionic_sites]

    for site in all_sites:
        Delta = np_random.random()
        species1 = np_random.integers(1, site.num_species + 1)
        species1 = np_random.choice(range(site.num_species), size=species1, replace=False)
        species2 = np_random.integers(1, site.num_species + 1)
        species2 = np_random.choice(range(site.num_species), size=species2, replace=False)
        if len(species1) != len(species2):
            limit = min(len(species1), len(species2))
            species1 = species1[:limit]
            species2 = species2[:limit]

        if isinstance(site, (sites.SpinlessBosonSite, sites.SpinlessFermionSite)):
            if not isinstance(site.conserve, str):
                # easiest way to deal with symmetries on the individual species
                species2 = species1
        if isinstance(site, sites.SpinHalfFermionSite):
            if site.conserve_S in ['Sz', 'SU(2)']:
                species1 = [0]
                species2 = [1]

        coupling = couplings.onsite_pairing([site], Delta, species=(species1, species2))
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is zero
        npt.assert_almost_equal(tensors.trace(tensor).to_numpy(), 0)
        if isinstance(site, degrees_of_freedom.FermionicDOF):
            # default case is trivial for fermions
            coupling = couplings.onsite_pairing([site], Delta=1)
            coupling.test_sanity()
            npt.assert_almost_equal(tensors.norm(coupling.to_tensor()).to_numpy(), 0)

    check_coupling(couplings.onsite_pairing, site_num=1, invalid_site_nums=[2], boson_fermion_mixing=False)


# TEST CLOCK COUPLINGS


def test_clock_clock_coupling(any_backend, np_random):
    site_list = generate_clock_dofs(any_backend)
    for site in site_list:
        Jx, Jz = np_random.random(2)
        coupling = couplings.clock_clock_coupling([site] * 2, Jx=Jx, Jz=Jz)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is zero
        npt.assert_almost_equal(tensors.trace(tensor).to_numpy(), 0)
        # commutation relation
        tensor_commuted = tensors.permute_legs(tensor, codomain=[1, 0], domain=[2, 3])
        tensor_commuted.relabel({'p0': 'p1', 'p1': 'p0', 'p0*': 'p1*', 'p1*': 'p0*'})
        assert tensors.almost_equal(tensor_commuted, tensor)

    check_coupling(couplings.clock_clock_coupling, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=False)


def test_clock_field_coupling(any_backend, np_random):
    site_list = generate_clock_dofs(any_backend)
    for site in site_list:
        hx, hz = np_random.random(2)
        if isinstance(site.leg.symmetry.factors[0], lasyten.ZN):
            hx = 0
        coupling = couplings.clock_field_coupling([site], hx=hx, hz=hz)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is zero
        npt.assert_almost_equal(tensors.trace(tensor).to_numpy(), 0)
        # check eigenvalues
        if isinstance(site.leg.symmetry.factors[0], lasyten.ZN):
            expect_evs = 2 * np.cos(np.linspace(0, 2 * np.pi, site.q, endpoint=False))
            evs = tensor.to_numpy(understood_braiding=True)
            evs = np.sort(np.linalg.eigvalsh(evs))
            npt.assert_almost_equal(evs, np.sort(hz * expect_evs))

    check_coupling(couplings.clock_field_coupling, site_num=1, invalid_site_nums=[2], boson_fermion_mixing=False)


# TEST ANYONIC COUPLINGS


def test_sector_projection_coupling(block_backend):
    site_list = generate_anyon_dofs(block_backend)
    num_sites = [3, 2, 1, 2, 2, 1]
    sectors = np.asarray([[1], [2], [1], [0], [2], [2]], dtype=int)
    for site, num, sector in zip(site_list, num_sites, sectors):
        coupling = couplings.sector_projection_coupling([site] * num, J=1.0, sector=sector, name='')
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace is integer * dim(sector)
        dim_sec = site.symmetry.qdim(sector)
        tr = tensors.trace(tensor).to_numpy()
        npt.assert_almost_equal(np.round(tr / dim_sec, 0), tr / dim_sec)


def test_gold_coupling(block_backend):
    backend = backends.get_backend('fusion_tree', block_backend=block_backend)
    site_list = [sites.GoldenSite(backend=backend), sites.FibonacciAnyonSite(backend=backend)]
    for i, site in enumerate(site_list):
        coupling = couplings.gold_coupling([site] * 2, J=1.0)
        coupling.test_sanity()
        tensor = coupling.to_tensor()
        # hermiticity
        assert tensors.almost_equal(tensor.hc, tensor)
        # trace
        npt.assert_almost_equal(tensors.trace(tensor).to_numpy(), [-1, -2][i])

    coupling = couplings.gold_coupling(site_list, J=1.0)
    coupling.test_sanity()
    tensor = coupling.to_tensor()
    # hermiticity
    assert tensors.almost_equal(tensor.hc, tensor)
    # trace
    npt.assert_almost_equal(tensors.trace(tensor).to_numpy(), -1)

    check_coupling(couplings.gold_coupling, site_num=2, invalid_site_nums=[1, 3], boson_fermion_mixing=False)


def test_coupling_hash_different_for_different_couplings(block_backend):
    """Test that different couplings compare unequal.

    Note: __hash__ is deliberately based on cheap *structural* metadata only (site types, tensor
    shapes/legs/dtype -- see Coupling._key()), not on the tensors' floating-point values, so
    couplings that only differ in their numeric content (like these two, same sites/shapes,
    different J) are a legitimate same-hash collision; __eq__ (not __hash__) is what
    distinguishes them, via the numeric almost_equal check. See test_coupling_eq_hash_numeric_closeness.
    """
    backend = backends.get_backend(block_backend=block_backend)
    site = sites.SpinSite(S=0.5, conserve='None', backend=backend)

    coupling1 = couplings.heisenberg_coupling([site, site], J=1.0)
    coupling2 = couplings.heisenberg_coupling([site, site], J=2.0)

    assert coupling1 != coupling2


def test_coupling_eq_hash_independent_construction():
    """Test that Coupling equality/hashing correctly identifies structurally-equal couplings,
    even when they are distinct Python objects built independently."""

    spin_site1 = SpinSite(S=0.5, conserve='Sz')
    spin_site2 = SpinSite(S=0.5, conserve='Sz')

    coupling_J1 = heisenberg_coupling([spin_site1, spin_site2], J=1.0)
    coupling_J1p = heisenberg_coupling([spin_site1, spin_site2], J=1.0)

    coupling_J2 = heisenberg_coupling([spin_site1, spin_site2], J=2.0)
    coupling_J2p = heisenberg_coupling([spin_site1, spin_site2], J=2.0)
    coupling_diff_J = spin_field_coupling([spin_site1], hz=0.5)

    assert coupling_J1 != coupling_J2
    assert coupling_J1 != coupling_diff_J
    assert coupling_J2 != coupling_diff_J

    # distinct objects, identical content -> equal, and equal hash (required by the hash contract)
    assert coupling_J1p == coupling_J1
    assert coupling_J1p is not coupling_J1
    assert hash(coupling_J1p) == hash(coupling_J1)
    assert coupling_J2p == coupling_J2
    assert hash(coupling_J2p) == hash(coupling_J2)


def test_coupling_hashing():
    """Test that Coupling equality/hashing correctly distinguishes different couplings."""

    spin_site = SpinSite(S=0.5, conserve='Sz')

    coupling_J1 = heisenberg_coupling([spin_site, spin_site], J=1.0)
    coupling_J2 = heisenberg_coupling([spin_site, spin_site], J=2.0)
    coupling_spin_field = spin_field_coupling([spin_site], hz=0.5)

    assert coupling_J1 != coupling_J2
    assert coupling_J1 != coupling_spin_field
    assert coupling_J2 != coupling_spin_field


def test_coupling_eq_hash_numeric_closeness():
    """Coupling equality is exact structural + almost_equal numeric comparison of the
    factorization tensors, so couplings must have both matching structure AND matching tensor
    values (up to almost_equal's tolerance) to compare equal / share a hash bucket."""

    site = SpinSite(S=0.5, conserve='None')

    c1 = heisenberg_coupling([site, site], J=1.0)
    c2 = heisenberg_coupling([site, site], J=1.0)
    c3 = heisenberg_coupling([site, site], J=1.0 + 1e-3)  # different tensor values
    c4 = spin_field_coupling([site], hz=0.5)  # different sites (1-site vs 2-site coupling)

    # same object always equals itself
    assert c1 == c1
    # distinct objects, numerically identical (not merely `is`-equal) tensors -> equal
    assert c1 == c2 and c1 is not c2
    assert hash(c1) == hash(c2)

    # same structure (site types, tensor shapes/legs), different tensor *values* -> not equal;
    # this is a legitimate same-`_key()` collision that only the numeric `almost_equal` check in
    # __eq__ (not `_key()`/`__hash__` alone) can distinguish
    assert c1._key() == c3._key()
    assert c1 != c3

    # different sites/factorization shape entirely -> not equal (and _key() differs too)
    assert c1._key() != c4._key()
    assert c1 != c4

    # usable as dict keys: structurally+numerically-equal couplings collapse to one entry
    d = {c1: 'first'}
    d[c2] = 'second'
    assert len(d) == 1 and d[c1] == 'second'
    d[c3] = 'third'
    d[c4] = 'fourth'
    assert len(d) == 3


@pytest.mark.parametrize(
    'coupling_factory,site_args,coupling_kwargs,valid_positions',
    [
        # 2-site Heisenberg
        (
            couplings.heisenberg_coupling,
            [lambda backend: [sites.SpinSite(S=0.5, conserve='None', backend=backend)] * 2],
            {'J': 1.0},
            [1],
        ),
        # 3-site chiral
        (
            couplings.chiral_3spin_coupling,
            [lambda backend: [sites.SpinSite(S=0.5, conserve='None', backend=backend)] * 3],
            {'chi': 1.0},
            [1, 2],
        ),
        # 2-site AKLT
        (
            couplings.aklt_coupling,
            [lambda backend: [sites.SpinSite(S=1, conserve='None', backend=backend)] * 2],
            {'J': 1.0},
            [1],
        ),
        # 2-site clock
        # (couplings.clock_clock_coupling,
        #  [lambda backend: [sites.ClockSite(3, conserve='None', backend=backend)] * 2],
        #  {"Jx": 1.0, "Jz": 1.0},
        #  [1]),
    ],
)
def test_stretch_with_identities_parametrized(
    block_backend, coupling_factory, site_args, coupling_kwargs, valid_positions
):
    backend = backends.get_backend(block_backend=block_backend)
    sites_list = site_args[0](backend)
    coupling = coupling_factory(sites_list, **coupling_kwargs)
    orig_num_sites = len(coupling.sites)
    orig_num_factors = len(coupling.factorization)

    for pos in valid_positions:
        # insert a gap right before original index `pos`, filled by a copy of its left neighbor
        all_sites = sites_list[:pos] + [sites_list[pos - 1]] + sites_list[pos:]
        coupling_positions = [*range(pos), *range(pos + 1, orig_num_sites + 1)]
        new_coupling = coupling.stretch_with_identities(all_sites, coupling_positions)
        # Number of sites should increase by 1 (a copy of the left neighbor is inserted)
        assert len(new_coupling.sites) == orig_num_sites + 1
        # Number of factors should increase by 1
        assert len(new_coupling.factorization) == orig_num_factors + 1
        # By __eq__/__hash__ distinct from the original
        assert new_coupling != coupling
        new_coupling.test_sanity()

    # too few / too many positions
    with pytest.raises(ValueError):
        coupling.stretch_with_identities(sites_list, list(range(orig_num_factors - 1)))
    with pytest.raises(ValueError):
        coupling.stretch_with_identities(sites_list, list(range(orig_num_factors + 1)))
    # not strictly ascending
    if orig_num_factors > 1:
        with pytest.raises(ValueError):
            coupling.stretch_with_identities(sites_list, list(range(orig_num_factors))[::-1])


def test_identity_tensor_site():
    """Test Site.identity_tensor: structural correctness and ValueError guard."""

    site = SpinSite(S=0.5, conserve='Sz')
    coupling = heisenberg_coupling([site, site])

    wL = coupling.factorization[0].get_leg_co_domain('wR')
    wR = coupling.factorization[1].get_leg_co_domain('wL')
    assert wL == wR, 'coupling virtual bonds must match'

    # overbraid=True
    tensor = site.identity_tensor(wL, overbraid=True)

    assert tensor.labels == ['wL', 'p', 'wR', 'p*']
    assert tensor.num_codomain_legs == 2
    assert tensor.num_domain_legs == 2
    assert tensor.get_leg_co_domain('p') == site.leg
    assert tensor.get_leg_co_domain('p*') == site.leg
    assert tensor.get_leg_co_domain('wL') == wL
    assert tensor.get_leg_co_domain('wR') == wR
    tensor.test_sanity()

    # overbraid=False
    # For a group symmetry (U(1)) braiding is symmetric, so the result is the same tensor.
    tensor_under = site.identity_tensor(wL, overbraid=False)
    assert tensor_under.labels == ['wL', 'p', 'wR', 'p*']
    tensor_under.test_sanity()

    # Non-trivial physical leg: spin-1 site, same bond
    site_s1 = SpinSite(S=1.0, conserve='Sz')
    coupling_s1 = heisenberg_coupling([site_s1, site_s1])
    wL_s1 = coupling_s1.factorization[0].get_leg_co_domain('wR')
    wR_s1 = coupling_s1.factorization[1].get_leg_co_domain('wL')
    assert wL_s1 == wR_s1, 'wL and wR must be the same legs, since they get contracted'
    tensor_s1 = site_s1.identity_tensor(wL_s1)
    assert tensor_s1.get_leg_co_domain('p') == site_s1.leg
    tensor_s1.test_sanity()


def test_get_onsite_W_block():
    site = SpinSite(S=0.5, conserve='Sz')
    coupling = heisenberg_coupling([site, site])
    w = coupling.factorization[0].get_leg_co_domain('wR')

    for overbraid in [True, False]:
        ident = site.get_onsite_W_block('Id', w, overbraid=overbraid)
        expect = site.identity_tensor(w, overbraid=overbraid)
        assert ident.labels == ['wL', 'p', 'wR', 'p*']
        assert abs(ident.to_numpy() - expect.to_numpy()).max() < 1e-14

    block = site.get_onsite_W_block('Sz', w, wL_label='a', wR_label='b')
    assert block.labels == ['a', 'p', 'b', 'p*']
    assert block.get_leg_co_domain('a') == w
    assert block.get_leg_co_domain('b') == w
    block.test_sanity()


def test_stretch_with_identities():
    # stretch_with_identities(all_sites, coupling_positions) places this coupling's own tensors
    # at `coupling_positions` within `all_sites`, filling any gaps with identities on whatever
    # site actually lives there -- which need not be one of the coupling's own sites.

    # ------------------------------------------------------------------ structure
    site_a = SpinSite(S=0.5, conserve='Sz')
    site_b = SpinSite(S=0.5, conserve='Sz')
    original = heisenberg_coupling([site_a, site_b])

    result = original.stretch_with_identities([site_a, site_a, site_b], [0, 2])

    assert len(result.sites) == 3
    assert len(result.factorization) == 3
    assert result.sites[0] is site_a
    assert result.sites[1] is site_a  # fill with an identity
    assert result.sites[2] is site_b
    assert result.factorization[1].labels == ['wL', 'p', 'wR', 'p*']
    result.test_sanity()

    with pytest.raises(ValueError):
        original.stretch_with_identities([site_a, site_b], [0])  # wrong number of positions
    with pytest.raises(ValueError):
        original.stretch_with_identities([site_b, site_a], [1, 0])  # not strictly ascending

    # a coupling placed at a position whose site doesn't match the coupling's own site is invalid
    site_half_ns = SpinSite(S=0.5, conserve='None')
    site_one_ns = SpinSite(S=1.0, conserve='None')
    original_ns = heisenberg_coupling([site_half_ns, site_half_ns])
    with pytest.raises(ValueError):
        original_ns.stretch_with_identities([site_half_ns, site_one_ns], [0, 1])

    # ------------------------------------------------------------------ content check (NoSymmetry)
    # For a coupling C2 on [s0, s1] (same site s0 == s1 == site_half_ns), stretching in a gap
    # site at position 1 produces a 3-site coupling C3 satisfying:
    #   C3[p0, pi, p1, p1*, pi*, p0*] = C2[p0, p1, p1*, p0*] * delta(pi, pi*)
    #
    # Numpy leg order (domain labels are stored reversed in the label list):
    #   C2: [p0, p1, p1*, p0*]  → shape [d0, d1, d1, d0]
    #   C3: [p0, pi, p1, p1*, pi*, p0*] → shape [d0, di, d1, d1, di, d0]
    result_ns = original_ns.stretch_with_identities([site_half_ns, site_half_ns, site_half_ns], [0, 2])
    assert result_ns.sites[1] is site_half_ns
    result_ns.test_sanity()

    C2 = original_ns.to_numpy(understood_braiding=True)  # [d0, d1, d1, d0]
    C3 = result_ns.to_numpy(understood_braiding=True)  # [d0, di, d1, d1, di, d0]

    di = site_half_ns.dim  # 2 for spin-1/2
    assert C3.shape == (C2.shape[0], di, C2.shape[1], C2.shape[2], di, C2.shape[3])

    for pi in range(di):
        # Diagonal block matches original coupling.
        np.testing.assert_allclose(
            C3[:, pi, :, :, pi, :], C2, atol=1e-13, err_msg=f'diagonal block pi={pi} does not match original coupling'
        )
    for pi in range(di):
        for pi_star in range(di):
            if pi != pi_star:
                # Off-diagonal blocks: must vanish (identity in physical space).
                np.testing.assert_allclose(
                    C3[:, pi, :, :, pi_star, :],
                    0,
                    atol=1e-13,
                    err_msg=f'off-diagonal block pi={pi}, pi*={pi_star} is non-zero',
                )

    # gap site with a different physical leg than the coupling's own sites
    site_one = SpinSite(S=1.0, conserve='Sz')
    stretched = original.stretch_with_identities([site_a, site_one, site_one, site_b], [0, 3])
    stretched.test_sanity()
    assert [s.S for s in stretched.sites] == [0.5, 1.0, 1.0, 0.5]
    for factor, site in zip(stretched.factorization[1:-1], stretched.sites[1:-1]):
        assert factor.get_leg_co_domain('p') == site.leg


def _to_matrix(dense, dims):
    """Convert a dense block with axes [p0,...,p(n-1), p(n-1)*,...,p0*] (bra reversed, as
    returned by Coupling.to_tensor().to_numpy()) into a plain (prod(dims), prod(dims)) matrix
    with row = ket multi-index, column = bra multi-index, both in normal (non-reversed) order.
    """
    n = len(dims)
    bra_axes_reversed = list(range(n, 2 * n))
    normal_order = list(range(n)) + bra_axes_reversed[::-1]
    dense = np.transpose(dense, normal_order)
    dim = int(np.prod(dims))
    return dense.reshape(dim, dim)


def _permute_matrix(mat, dims, permutation):
    """Conjugate a matrix (as returned by `_to_matrix`) by the basis permutation that reorders
    the `len(dims)` tensor factors according to `permutation`."""
    n = len(dims)
    new_dims = [dims[i] for i in permutation]
    mat_tensor = mat.reshape(tuple(dims) + tuple(dims))
    axes = list(permutation) + [n + i for i in permutation]
    mat_tensor = np.transpose(mat_tensor, axes)
    dim = int(np.prod(new_dims))
    return mat_tensor.reshape(dim, dim)


def _random_hermitian_coupling(sites, seed):
    """A coupling with a random Hermitian dense block, for NoSymmetry sites."""
    dims = [s.dim for s in sites]
    rng = np.random.default_rng(seed)
    shape = tuple(dims) + tuple(dims[::-1])
    block = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    dim = int(np.prod(dims))
    mat = block.reshape(dim, dim)
    mat = mat + mat.conj().T
    return couplings.Coupling.from_dense_block(mat.reshape(shape), sites, name='random', understood_braiding=True)


def test_coupling_permute():
    """Coupling.permute should reorder the sites/operator like conjugating by a basis
    permutation, cache repeated requests, and correctly track `_levels`."""
    site_dims = [0.5, 1.0, 1.5, 2.0]
    sites = [SpinSite(S=S, conserve=None) for S in site_dims]
    dims = [s.dim for s in sites]
    coupling = _random_hermitian_coupling(sites, seed=1234)

    assert coupling._levels == [1, 2, 3, 4]
    assert coupling._permuted == []

    permutation = [2, 3, 0, 1]
    levels = [1, 2, 3, 4]

    result = coupling.permute(permutation, levels)
    result.test_sanity()

    # structure: sites reordered as expected
    assert [s.dim for s in result.sites] == [dims[i] for i in permutation]
    # _levels: tracks which original level ended up where
    assert result._levels == [coupling._levels[i] for i in permutation]
    # a freshly permuted coupling starts with its own, empty cache
    assert result._permuted == []

    # value check: equivalent to conjugating the dense operator by the basis permutation
    H_mat = _to_matrix(coupling.to_tensor().to_numpy(understood_braiding=True), dims)
    expected_mat = _permute_matrix(H_mat, dims, permutation)
    new_dims = [dims[i] for i in permutation]
    result_mat = _to_matrix(result.to_tensor().to_numpy(understood_braiding=True), new_dims)
    np.testing.assert_allclose(result_mat, expected_mat, atol=1e-10)

    # caching: same permutation returns the cached object, even with different levels
    result_again = coupling.permute(permutation, [9, 9, 9, 9])
    assert result_again is result
    assert len(coupling._permuted) == 1

    # a different permutation triggers a new computation and a new cache entry
    other_permutation = [1, 0, 2, 3]
    other_result = coupling.permute(other_permutation, levels)
    assert other_result is not result
    assert len(coupling._permuted) == 2


def test_coupling_permute_identity():
    """Permuting with the identity permutation (0 swaps) should reproduce the same operator."""
    sites = [SpinSite(S=S, conserve=None) for S in (0.5, 1.0, 1.5)]
    dims = [s.dim for s in sites]
    coupling = _random_hermitian_coupling(sites, seed=5)

    result = coupling.permute([0, 1, 2], [1, 2, 3])
    assert [s is s2 for s, s2 in zip(result.sites, coupling.sites)] == [True, True, True]
    H_mat = _to_matrix(coupling.to_tensor().to_numpy(understood_braiding=True), dims)
    result_mat = _to_matrix(result.to_tensor().to_numpy(understood_braiding=True), dims)
    np.testing.assert_allclose(result_mat, H_mat, atol=1e-10)


def test_coupling_permute_errors():
    """Coupling.permute should raise clear errors for invalid input."""
    sites = [SpinSite(S=S, conserve=None) for S in (0.5, 1.0, 1.5, 2.0)]
    coupling = _random_hermitian_coupling(sites, seed=99)
    levels = [1, 2, 3, 4]

    with pytest.raises(ValueError):
        coupling.permute([0, 1, 2, 2], levels)  # not a valid permutation

    with pytest.raises(BraidChiralityUnspecifiedError):
        # two sites that must braid (adjacent swap) with equal levels: chirality is ambiguous
        coupling.permute([1, 0, 3, 2], [5, 5, 7, 7])


def _asym_hopping_dense_block(site):
    """Dense block [p0,p1,p1*,p0*] for the (non-Hermitian) 2-site term ``Cd_0 C_1`` (or the
    bosonic analogue ``Bd_0 B_1``), correctly JW-dressed as in
    :func:`~lasyten.models.couplings._quadratic_coupling_numpy`: the *first* (left) operand carries
    the JW-string factor.
    """
    creator = site.get_creator_numpy(species=0, include_JW=True)
    annihilator = site.get_annihilator_numpy(species=0, include_JW=True)
    return (creator @ site._JW)[:, None, None, :] * annihilator[None, :, :, None]


@pytest.mark.parametrize(
    'site_factory,label',
    [
        (lambda: sites.SpinlessBosonSite(Nmax=1, conserve='N'), 'boson'),
        (lambda: sites.SpinlessFermionSite(num_species=1, conserve='N'), 'fermion'),
    ],
)
def test_coupling_permute_matches_direct_permute_legs(site_factory, label):
    """Verify that `Coupling.permute` produces the exact same results as permuting the
    fully-contracted tensor directly.

    Internally, `Coupling.permute` applies each elementary transposition as a local swap gate
    to only the two `factorization` tensors it touches, rather than permuting the fully
    contracted tensor. This test ensures that this local-swap / re-factorization round-trip
    reproduces the same result as the direct, whole-tensor `permute_legs`, without losing or
    duplicating any data. It checks this behavior for both fermionic and bosonic sites.
    """
    site = site_factory()
    coupling = couplings.Coupling.from_dense_block(
        _asym_hopping_dense_block(site), [site, site], understood_braiding=True
    )

    levels = [2, 1]
    over = levels[0] > levels[1]  # higher level braids over the lower one
    permuted = coupling.permute([1, 0], levels=levels)

    codomain_labels, domain_labels = ['p0', 'p1'], ['p0*', 'p1*']
    level_dict = {
        codomain_labels[0]: 1 if over else 0,
        domain_labels[0]: 1 if over else 0,
        codomain_labels[1]: 0 if over else 1,
        domain_labels[1]: 0 if over else 1,
    }
    tensor_direct = permute_legs(coupling.to_tensor(), codomain=['p1', 'p0'], domain=['p1*', 'p0*'], levels=level_dict)
    # relabel to the same p{new_pos} convention Coupling.permute uses (site formerly at 1 is now p0)
    tensor_direct = tensor_direct.relabel({'p0': 'q1', 'p1': 'q0', 'p0*': 'q1*', 'p1*': 'q0*'})
    tensor_direct = tensor_direct.relabel({'q0': 'p0', 'q1': 'p1', 'q0*': 'p0*', 'q1*': 'p1*'})

    labels = ['p0', 'p1', 'p0*', 'p1*']
    dim_permuted = permuted.to_tensor().to_numpy(labels, understood_braiding=True)
    dim_direct = tensor_direct.to_numpy(labels, understood_braiding=True)
    np.testing.assert_allclose(dim_permuted, dim_direct, atol=1e-10, err_msg=label)


@pytest.mark.parametrize(
    'site_factory,expected_sign,label',
    [
        (lambda: sites.SpinlessBosonSite(Nmax=1, conserve='N'), +1, 'boson'),
        (lambda: sites.SpinlessFermionSite(num_species=1, conserve='N'), -1, 'fermion'),
    ],
)
def test_coupling_permute_exchange_sign(site_factory, expected_sign, label):
    """
    Returns the exchange sign for Coupling.permute (negative for fermions, positive for bosons).

    This checks the physical sign by comparing the permuted coupling against a new
    coupling built from scratch with reversed site order.
    """

    site = site_factory()
    creator = site.get_creator_numpy(species=0, include_JW=True)
    annihilator = site.get_annihilator_numpy(species=0, include_JW=True)
    JW = site._JW

    # forward: coupling on [site, site] representing Cd_0 C_1 (JW dressing on the first operand)
    h_fwd = (creator @ JW)[:, None, None, :] * annihilator[None, :, :, None]
    coupling_fwd = couplings.Coupling.from_dense_block(h_fwd, [site, site], understood_braiding=True)

    # independently-built coupling for the *reversed* site order, representing C_1 Cd_0: same
    # construction principle, but now the annihilator is the first operand and gets the JW dressing
    h_rev = (annihilator @ JW)[:, None, None, :] * creator[None, :, :, None]
    coupling_rev = couplings.Coupling.from_dense_block(h_rev, [site, site], understood_braiding=True)

    permuted = coupling_fwd.permute([1, 0], levels=[1, 2])

    labels = ['p0', 'p1', 'p0*', 'p1*']
    dim_permuted = permuted.to_tensor().to_numpy(labels, understood_braiding=True)
    dim_rev = coupling_rev.to_tensor().to_numpy(labels, understood_braiding=True)
    np.testing.assert_allclose(dim_permuted, expected_sign * dim_rev, atol=1e-10, err_msg=label)
    # sanity: the *wrong* sign should NOT match (guards against a vacuously-passing all-zero case)
    assert np.max(np.abs(dim_rev)) > 1e-10
    assert not np.allclose(dim_permuted, -expected_sign * dim_rev, atol=1e-10)
