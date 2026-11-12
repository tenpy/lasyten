# Copyright (C) TeNPy Developers, Apache license
import os.path
from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy import testing as npt

from lasyten import symmetries

default_rng = np.random.default_rng()

no_symmetry = symmetries.NoSymmetry().as_Symmetry()
u1_symmetry = symmetries.U1().as_Symmetry()
z2_symmetry = symmetries.ZN(N=2).as_Symmetry()
z3_symmetry = symmetries.ZN(N=3).as_Symmetry()
fermion_parity = symmetries.FermionParity().as_Symmetry()
fermion_number = symmetries.FermionNumber().as_Symmetry()


def shuffled(x, axis=0, np_rng=default_rng):
    """Like ``np.random.Generator.shuffle``, but returns a shuffled copy instead of acting in-place."""
    x = np.copy(x)
    np_rng.shuffle(x, axis=axis)
    return x


def sampled_zip(sequence, num_copies: int, num_samples: int, np_rng=default_rng, accept_fewer=True):
    """Generate a given number of random samples from the zip of multiple sequences"""
    len_sequence = len(sequence)
    num_combinations = len_sequence**num_copies
    if num_samples > num_combinations:
        if accept_fewer:
            num_samples = num_combinations
        else:
            raise ValueError(f'Can not generate {num_samples} samples.')
    for idx in np_rng.choice(num_combinations, size=num_samples, replace=False):
        sample = []
        for _ in range(num_copies):
            sample.append(sequence[idx % len_sequence])
            idx = idx // len_sequence
        assert idx == 0
        yield tuple(sample)


def sample_from(sequence, num_samples: int, accept_fewer: bool = False, np_rng=default_rng):
    if num_samples > len(sequence):
        if accept_fewer:
            num_samples = len(sequence)
        else:
            raise ValueError(f'Can not generate {num_samples} samples from sequence of len {len(sequence)}')
    for n in np_rng.choice(len(sequence), size=num_samples, replace=False):
        yield sequence[n]


def sample_offdiagonal(sequence, num_samples: int, accept_fewer: bool = False, np_rng=default_rng):
    """Sample pairs ``(sequence[n], sequence[m])`` with ``n != m``."""
    len_sequence = len(sequence)
    num_combinations = len_sequence * (len_sequence - 1)
    if num_samples > num_combinations:
        if accept_fewer:
            num_samples = num_combinations
        else:
            raise ValueError(f'Can not generate {num_samples} samples.')
    for i in np_rng.choice(num_combinations, size=num_samples, replace=False):
        idx1, idx2 = divmod(i, len_sequence)
        # now   0 <= idx1 < len-1   and 0 <= idx2 < len
        if idx1 >= idx2:
            idx1 += 1
        yield sequence[idx1], sequence[idx2]


def sample_sector_triplets(
    symmetry: symmetries.Symmetry, sectors, num_samples: int, accept_fewer: bool = True, np_rng=default_rng
):
    """Yield samples ``(a, b, c)`` such that ``a x b -> c`` is an allowed fusion"""
    a_b_list = list(sampled_zip(sectors, num_copies=2, num_samples=num_samples, np_rng=np_rng, accept_fewer=True))
    fusion_outcomes = [symmetry.fusion_outcomes(a, b) for a, b in a_b_list]

    if len(a_b_list) >= num_samples:  # it is enough to select one fusion outcome c per a_b
        for (a, b), outcomes in zip(a_b_list, fusion_outcomes):
            yield a, b, np_rng.choice(outcomes)
        return

    num_outcomes = [len(outcomes) for outcomes in fusion_outcomes]
    total_number = sum(num_outcomes)
    if total_number < num_samples:
        if not accept_fewer:
            raise ValueError('Could not generate enough samples')
        # just yield all of them
        for (a, b), outcomes in zip(a_b_list, fusion_outcomes):
            for c in outcomes:
                yield a, b, c
        return

    sample_percentage = num_samples / total_number
    # divide evenly as much as possible
    num_samples_per_a_b = [int(sample_percentage * num) for num in num_outcomes]
    # distribute the remaining samples randomly
    num_missing = num_samples - sum(num_samples_per_a_b)
    for idx in np_rng.choice(len(a_b_list), size=num_missing, replace=False):
        num_samples_per_a_b[idx] += 1
    assert sum(num_samples_per_a_b) == num_samples
    for (a, b), outcomes, num in zip(a_b_list, fusion_outcomes, num_samples_per_a_b):
        for c in sample_from(outcomes, num_samples=num, accept_fewer=False, np_rng=np_rng):
            yield a, b, c


def sample_sector_sextets(
    symmetry: symmetries.Symmetry, sectors, num_samples: int, accept_fewer: bool = True, np_rng=default_rng
):
    """Yield samples ``(a, b, c, d, e, f)`` that are valid F/C symbol inputs.

    The constraint is that both ``(a x b) x c -> f x c -> d`` and ``a x (b x c) -> a x e -> d``
    are allowed.
    """
    abc_list = list(sampled_zip(sectors, num_copies=3, num_samples=num_samples, np_rng=np_rng, accept_fewer=True))

    if len(abc_list) >= num_samples:  # it is enough to select one fusion channel per a, b, c
        for a, b, c in abc_list:
            f = np_rng.choice(symmetry.fusion_outcomes(a, b))
            d = np_rng.choice(symmetry.fusion_outcomes(f, c))
            # need to find an f from the fusion products of b x c such that a x f -> d is allowed
            for e in shuffled(symmetry.fusion_outcomes(b, c), axis=0, np_rng=np_rng):
                if symmetry.can_fuse_to(a, e, d):
                    yield a, b, c, d, e, f
                    break
        return

    # TODO can do something analogous to `sample_sector_triplets` to efficiently sample without
    #      duplicates. I am lazy now and just return fewer samples...
    assert accept_fewer
    yield from sample_sector_sextets(symmetry=symmetry, sectors=sectors, num_samples=len(abc_list), np_rng=np_rng)


def sample_sector_nonets(
    symmetry: symmetries.Symmetry, sectors, num_samples: int, accept_fewer: bool = True, np_rng=default_rng
):
    """Yield samples ``(a, b, c, d, e, f, g, l, k)`` that are valid inputs to test the pentagon equation.

    The constraint is that both ``((a x b) x c) x d -> (f x c) x d -> g x d -> e``
    and ``a x (b x (c x d)) -> a x (b x l) -> a x k -> e`` are allowed.
    """
    abcd_list = list(sampled_zip(sectors, num_copies=4, num_samples=num_samples, np_rng=np_rng, accept_fewer=True))

    if len(abcd_list) >= num_samples:  # it is enough to select one fusion channel per a, b, c, d
        for a, b, c, d in abcd_list:
            f = np_rng.choice(symmetry.fusion_outcomes(a, b))
            g = np_rng.choice(symmetry.fusion_outcomes(f, c))
            e = np_rng.choice(symmetry.fusion_outcomes(g, d))
            # need to find l, k such that a x k -> e is allowed;
            # there may be choices for l such that all possible k are inconsistent
            for l in shuffled(symmetry.fusion_outcomes(c, d), axis=0, np_rng=np_rng):
                if symmetry.can_fuse_to(f, l, e):
                    for k in shuffled(symmetry.fusion_outcomes(b, l), axis=0, np_rng=np_rng):
                        if symmetry.can_fuse_to(a, k, e):
                            yield a, b, c, d, e, f, g, l, k
                            break
        return

    # TODO do something analogous to `sample_sector_triplets`?
    assert accept_fewer
    yield from sample_sector_nonets(symmetry=symmetry, sectors=sectors, num_samples=len(abcd_list), np_rng=np_rng)


def sample_sector_unitarity_test(
    symmetry: symmetries.Symmetry,
    sectors_low_qdim,
    num_samples: int,
    accept_fewer: bool = True,
    np_rng=default_rng,
):
    """Yield samples ``(a, b, c, d, e, g)`` that can be used for the F/C symbol unitarity tests.

    The constraint is that both ``a x (b x c) -> a x e -> d`` and ``a x (b x c) -> a x g -> d``
    are allowed. The appropriate charges `f` for which ``(a x b) x c -> f x c -> d`` are summed
    over in the unitarity test.
    """
    abc_list = list(
        sampled_zip(sectors_low_qdim, num_copies=3, num_samples=num_samples, np_rng=np_rng, accept_fewer=True)
    )

    if len(abc_list) >= num_samples:  # it is enough to select one fusion channel per a, b, c
        for a, b, c in abc_list:
            e = np_rng.choice(symmetry.fusion_outcomes(b, c))
            d = np_rng.choice(symmetry.fusion_outcomes(a, e))
            # need to find an g from the fusion products of b x c such that a x g -> d is allowed
            for g in shuffled(symmetry.fusion_outcomes(b, c), axis=0, np_rng=np_rng):
                if symmetry.can_fuse_to(a, g, d):
                    yield a, b, c, d, e, g
                    break
        return

    # TODO do something analoguous to `sample_sector_triplets`?
    assert accept_fewer
    yield from sample_sector_unitarity_test(
        symmetry=symmetry, sectors_low_qdim=sectors_low_qdim, num_samples=len(abc_list), np_rng=np_rng
    )


def common_checks(
    sym: symmetries.Symmetry, example_sectors, example_sectors_low_qdim, np_random, skip_fusion_tensor=False
):
    """Common consistency checks to be performed on a symmetry instance.

    Assumes example_sectors are duplicate free.
    example_sectors_low_qdim is like example_sectors but for charges with small quantum dimension such
    that we do not need to sum over many charges when verifying the unitarity of the F and C symbols.

    TODO: The fusion consistency check right now is not elegant and should be revisited at some point.
          To make things more efficient, we should check for consistency only once, i.e., when we check,
          e.g., the unitarity of the F-moves, we can use the same consistency check as for the C symbols
    """
    example_sectors = np.unique(example_sectors, axis=0)
    example_sectors_low_qdim = np.unique(example_sectors_low_qdim, axis=0)

    # generate a few samples of sectors that fulfill fusion rules, used to check the symbols
    sector_triplets = list(
        sample_sector_triplets(sym, example_sectors, num_samples=10, accept_fewer=True, np_rng=np_random)
    )
    sector_sextets = list(
        sample_sector_sextets(sym, example_sectors, num_samples=10, accept_fewer=True, np_rng=np_random)
    )
    sector_nonets = list(
        sample_sector_nonets(sym, example_sectors, num_samples=10, accept_fewer=True, np_rng=np_random)
    )
    sector_unitarity_test = list(
        sample_sector_unitarity_test(sym, example_sectors_low_qdim, num_samples=10, accept_fewer=True, np_rng=np_random)
    )

    assert sym.trivial_sector.shape == (sym.sector_ind_len,)
    assert sym.is_valid_sector(sym.trivial_sector)
    assert not sym.is_valid_sector(np.zeros(shape=(sym.sector_ind_len + 2), dtype=int))
    # objects which are not 1D arrays should not be valid sectors
    for invalid_sector in [0, 1, 42.0, None, False, 'foo', [0], ['foo'], [None], (), [], np.zeros((1, 1), dtype=int)]:
        assert not sym.is_valid_sector(invalid_sector)
    assert sym.qdim(sym.trivial_sector) in [1, 1.0]
    assert sym.num_sectors == np.inf or (isinstance(sym.num_sectors, int) and sym.num_sectors > 0)

    # check all_sectors
    if sym.num_sectors < np.inf:
        all_sectors = sym.all_sectors()
        assert all_sectors.shape == (sym.num_sectors, sym.sector_ind_len)
        for s in all_sectors:
            assert sym.is_valid_sector(s)

    # string representations : just check if they run
    _ = sym.sector_str(sym.trivial_sector)
    _ = repr(sym)
    _ = str(sym)

    # get_sector is the inverse of sector_str
    for a in example_sectors:
        npt.assert_array_equal(sym.get_sector(sym.sector_str(a)).to_numpy(), a)

    # defining property of trivial sector
    for a in example_sectors:
        fusion = sym.fusion_outcomes(a, sym.trivial_sector)
        assert len(fusion) == 1
        npt.assert_array_equal(fusion[0], a)

    # trivial sector is its own dual
    npt.assert_array_equal(sym.dual_sector(sym.trivial_sector), sym.trivial_sector)

    # check fusion tensors, if available
    if sym.can_be_dropped and not skip_fusion_tensor:
        check_fusion_tensor(sym, example_sectors, np_random)
        check_symbols_via_fusion_tensors(sym, sector_triplets, sector_sextets, np_random)

    # check N symbol
    for a in example_sectors:
        # duality (diagonal part)
        assert sym.n_symbol(a, sym.dual_sector(a), sym.trivial_sector) == 1
        # left and right unitor (diagonal part)
        assert sym.n_symbol(a, sym.trivial_sector, a) == 1
        assert sym.n_symbol(sym.trivial_sector, a, a) == 1
    for a, b in sample_offdiagonal(example_sectors, num_samples=10, accept_fewer=True, np_rng=np_random):
        # duality (off-diagonal part)
        b_dual = sym.dual_sector(b)
        if not np.all(a == b_dual):
            assert sym.n_symbol(a, b_dual, sym.trivial_sector) == 0
        # left and right unitor (off-diagonal part)
        assert sym.n_symbol(a, sym.trivial_sector, b) == 0
        assert sym.n_symbol(sym.trivial_sector, a, b) == 0
    for a, b, c, d, _, _ in sector_sextets:
        # TODO associativity constraint \sum_e N(a, b, e) N(e, c, d) == \sum_f N(b, c, d) N(a, f, d)
        pass

    # check F symbol
    check_F_symbols(sym, sector_sextets, sector_unitarity_test)
    check_pentagon_equation(sym, sector_nonets)

    # check R symbol
    check_R_symbols(sym, sector_triplets, example_sectors_low_qdim)
    check_hexagon_equation(sym, sector_sextets, True)

    # check C symbol
    check_C_symbols(sym, sector_sextets, sector_unitarity_test)

    # check B symbol
    check_B_symbols(sym, sector_triplets)

    # check derived topological data vs the fallback implementations.
    for a in example_sectors:
        if sym.can_be_dropped:
            npt.assert_almost_equal(
                sym.Z_iso(a), symmetries.SymmetryFactor.Z_iso(sym, a), err_msg='Z_iso does not match fallback'
            )
        assert sym.frobenius_schur(a) == symmetries.SymmetryFactor.frobenius_schur(sym, a), (
            'frobenius_schur does not match fallback'
        )
        # note: need almost equal for non-integer qdim
        npt.assert_almost_equal(
            sym.qdim(a), symmetries.SymmetryFactor.qdim(sym, a), err_msg='qdim does not match fallback'
        )
        npt.assert_almost_equal(
            sym.topological_twist(a),
            symmetries.SymmetryFactor.topological_twist(sym, a),
            err_msg='topological_twist does not match fallback',
        )
    for a, b, c in sector_triplets:
        npt.assert_almost_equal(
            sym._b_symbol(a, b, c),
            symmetries.SymmetryFactor._b_symbol(sym, a, b, c),
            err_msg='B symbol does not match fallback',
        )
    for c, a, b, d, e, f in sector_sextets:
        npt.assert_almost_equal(
            sym._c_symbol(a, b, c, d, e, f),
            symmetries.SymmetryFactor._c_symbol(sym, a, b, c, d, e, f),
            err_msg='C symbol does not match fallback',
        )

    # check braiding style
    for a in example_sectors:  # check topological twist
        if sym.braiding_style == symmetries.BraidingStyle.bosonic:
            npt.assert_almost_equal(sym.topological_twist(a), 1)
        elif sym.braiding_style == symmetries.BraidingStyle.fermionic:
            npt.assert_almost_equal(sym.topological_twist(a) ** 2, 1)

    if sym.braiding_style.value <= symmetries.BraidingStyle.fermionic.value:  # check R symbols
        for a, b, c in sector_triplets:
            npt.assert_almost_equal(sym.r_symbol(a, b, c) ** 2, np.ones(sym.n_symbol(a, b, c)))

    # check fusion style
    if sym.is_abelian:
        for a in example_sectors:
            for b in example_sectors:
                assert len(sym.fusion_outcomes(a, b)) == 1

    if sym.has_unique_fusion:
        for a, b, c in sector_triplets:
            # we check `== 1` and not `in [0, 1]` here since we iterate over sector_triplets
            assert sym.n_symbol(a, b, c) == 1


def check_fusion_tensor(sym: symmetries.Symmetry, example_sectors, np_random):
    """Checks if the fusion tensor of a given symmetry behaves as expected.

    Subroutine of `common_checks`.
    The ``example_sectors`` should be duplicate free.

    We check:
    - correct shape
    - orthonormality
    - completeness
    - relationship to left/right unitor
    - relationship to cup & Z
    - unitarity of Z, transpose relations of Z
    """
    for a, b in sampled_zip(example_sectors, num_copies=2, num_samples=10, np_rng=np_random):
        d_a = sym.sector_dim(a)
        d_b = sym.sector_dim(b)
        fusion_outcomes = sym.fusion_outcomes(a, b)

        assert len(fusion_outcomes) < 20  # otherwise we should not loop over all i guess?

        completeness_res = np.zeros((d_a, d_b, d_a, d_b))
        # iterate over coupled sectors and combine several checks
        for c in fusion_outcomes:
            d_c = sym.sector_dim(c)
            X_abc = sym.fusion_tensor(a, b, c)
            assert X_abc.dtype == sym.fusion_tensor_dtype.to_numpy_dtype()
            N_abc = sym.n_symbol(a, b, c)
            Y_abc = np.conj(X_abc)

            # correct shape?
            assert X_abc.shape == (N_abc, d_a, d_b, d_c)

            # orthonormal?
            res = np.tensordot(Y_abc, X_abc, [[1, 2], [1, 2]])  # [mu', m_c', mu, m_c]
            expect = np.eye(N_abc)[:, None, :, None] * np.eye(sym.sector_dim(c))[None, :, None, :]
            npt.assert_almost_equal(res, expect)

            # accumulate projector for checking completeness after the loop
            completeness_res += np.tensordot(X_abc, Y_abc, [[0, 3], [0, 3]])

        # complete?
        eye_a = np.eye(d_a)[:, None, :, None]
        eye_b = np.eye(d_b)[None, :, None, :]
        npt.assert_almost_equal(completeness_res, eye_a * eye_b)

        # remains: orthonormality for different sectors
        for c, d in sample_offdiagonal(fusion_outcomes, num_samples=5, accept_fewer=True, np_rng=np_random):
            d_c = sym.sector_dim(c)
            d_d = sym.sector_dim(d)
            N_abc = sym.n_symbol(a, b, c)
            N_abd = sym.n_symbol(a, b, d)
            Y_abc = np.conj(sym.fusion_tensor(a, b, c))
            X_abd = sym.fusion_tensor(a, b, d)
            res = np.tensordot(Y_abc, X_abd, [[1, 2], [1, 2]])  # [mu', m_c', mu, m_d]
            expect = np.zeros((N_abc, d_c, N_abd, d_d), dtype=res.dtype)
            npt.assert_almost_equal(res, expect)

    for a in example_sectors:
        d_a = sym.sector_dim(a)
        a_bar = sym.dual_sector(a)
        Z_a = sym.Z_iso(a)  # [m_a, m_abar]
        Z_a_hc = Z_a.conj().T
        Z_a_bar = sym.Z_iso(a_bar)  # [m_abar, m_a]

        # Z iso unitary?
        npt.assert_almost_equal(Z_a @ Z_a_hc, np.eye(d_a))
        npt.assert_almost_equal(Z_a_hc @ Z_a, np.eye(d_a))

        # defining property of frobenius schur?
        npt.assert_almost_equal(Z_a.T, sym.frobenius_schur(a) * Z_a_bar)

        # reduces to left/right unitor if one input is trivial?  [Jakob thesis, (5.63)
        X_aua = sym.fusion_tensor(a, sym.trivial_sector, a)
        npt.assert_almost_equal(X_aua, np.eye(d_a, dtype=X_aua.dtype)[None, :, None, :])
        X_uaa = sym.fusion_tensor(sym.trivial_sector, a, a)
        npt.assert_almost_equal(X_uaa, np.eye(d_a, dtype=X_uaa.dtype)[None, None, :, :])

        # relationship to cup  [Jakob thesis, (5.84)]
        Y = sym.fusion_tensor(a, a_bar, sym.trivial_sector).conj()[0, :, :, 0]  # [m_a, m_abar]
        expect_1 = Z_a.T / np.sqrt(d_a)  # transpose [m_a, m_abar] -> [m_a, m_abar]
        expect_2 = sym.frobenius_schur(a) / np.sqrt(d_a) * Z_a_bar
        npt.assert_almost_equal(Y, expect_1)
        npt.assert_almost_equal(Y, expect_2)


def check_symbols_via_fusion_tensors(sym: symmetries.Symmetry, sector_triplets, sector_sextets, np_random):
    """Check the defining properties of the F, R, C, B symbols with explicit fusion tensors.

    Subroutine of `common_checks`.
    The ``example_sectors`` should be duplicate free.
    """
    # ================
    # F symbol
    # ================
    # [Jakob thesis (5.74)]
    for a, b, c, d, e, f in sector_sextets:
        # need (a x b x c) -> (a x e) -> d AND (a x b x c) -> (f x c) -> d
        res = sym.fusion_tensor(f, c, d).conj()  # [lambda, f, c, d]
        # [lambda, (f), c, d] @ [kappa, a, b, (f)] -> [lambda, c, d, kappa, a, b]
        res = np.tensordot(res, sym.fusion_tensor(a, b, f).conj(), (1, 3))
        # [lambda, (c), d, kappa, a, (b)] @ [mu, (b), (c), e] -> [lambda, d, kappa, a, mu, e]
        res = np.tensordot(res, sym.fusion_tensor(b, c, e), ([1, 5], [2, 1]))
        # [lambda, d, kappa, (a), mu, (e)] @ [nu, (a), (e), d] -> [lambda, d, kappa, mu, nu, d]
        res = np.tensordot(res, sym.fusion_tensor(a, e, d), ([3, 5], [1, 2]))
        # [lambda, d, kappa, mu, nu, d] -> [mu, nu, kappa, lambda, d, d]
        res = np.transpose(res, [3, 4, 2, 0, 1, 5])

        F = sym.f_symbol(a, b, c, d, e, f)
        id_d = np.eye(sym.sector_dim(d))
        expect = F[..., None, None] * id_d[None, None, None, None, :, :]
        npt.assert_almost_equal(res, expect)

    # ================
    # R symbol
    # ================
    # [Jakob thesis (5.76)]
    for a, b, c in sector_triplets:
        # need a x b -> c
        res = sym.fusion_tensor(b, a, c).conj()  # [nu, b, a, c]
        # [nu, (b), (a), c] @ [a, b, (a*), (b*)] -> [nu, c, a, b]
        res = np.tensordot(res, sym.swap_gate(b, a), ([2, 1], [2, 3]))
        # [nu, c, (a), (b)] @ [mu, (a), (b), c] -> [nu, c, mu, c]
        res = np.tensordot(res, sym.fusion_tensor(a, b, c), ([2, 3], [1, 2]))
        # [nu, c, mu, c] -> [mu, nu, c, c]
        res = np.transpose(res, [2, 0, 1, 3])

        R = sym.r_symbol(a, b, c)  # [mu, nu]
        id_c = np.eye(sym.sector_dim(c))  # [c, c]
        expect = np.diag(R)[:, :, None, None] * id_c[None, None, :, :]
        npt.assert_almost_equal(res, expect)

    # ================
    # C symbol
    # ================
    # [Jakob thesis (5.88)]
    for c, a, b, d, e, f in sector_sextets:
        # need (a x b x c) -> (e x c) -> f  AND  (a x c x b) -> (f x b) -> d
        res = sym.fusion_tensor(f, b, d).conj()  # [lambda, f, b, d]
        # [lambda, (f), b, d] @ [kappa, a, c, (f)] -> [lambda, b, d, kappa, a, c]
        res = np.tensordot(res, sym.fusion_tensor(a, c, f).conj(), (1, -1))
        # [lambda, (b), d, kappa, a, (c)] @ [b, c, (b*), (c*)] -> [lambda, d, kappa, a, b, c]
        res = np.tensordot(res, sym.swap_gate(c, b), ([1, 5], [2, 3]))
        # [lambda, d, kappa, (a), (b), c] @ [mu, (a), (b), e] -> [lambda, d, kappa, c, mu, e]
        res = np.tensordot(res, sym.fusion_tensor(a, b, e), ([3, 4], [1, 2]))
        # [lambda, d, kappa, (c), mu, (e)] @ [nu, (e), (c), d] -> [lambda, d, kappa, mu, nu, d]
        res = np.tensordot(res, sym.fusion_tensor(e, c, d), ([3, 5], [2, 1]))
        # [lambda, d, kappa, mu, nu, d] -> [mu, nu, kappa, lambda, d, d]
        res = np.transpose(res, [3, 4, 2, 0, 1, 5])

        C = sym.c_symbol(a, b, c, d, e, f)
        id_d = np.eye(sym.sector_dim(d))
        expect = C[..., None, None] * id_d[None, None, None, None, :, :]
        npt.assert_almost_equal(res, expect)

    # ================
    # B symbol
    # ================
    # [Jakob thesis (5.94)]
    for a, b, c in sector_triplets:
        bbar = sym.dual_sector(b)
        # need (a x b) -> c
        res = sym.fusion_tensor(a, b, c)  # [mu, a, b, c]
        # note: cup is taken care of trivially
        # [mu, a, (b), c] @ [bbar, (b)] -> [mu, a, c, bbar]
        res = np.tensordot(res, sym.Z_iso(b), (2, 1))
        # [mu, a, (c), (bbar)] @ [nu, (c), (bbar), a] -> [mu, a, nu, a]
        res = np.tensordot(res, sym.fusion_tensor(c, bbar, a), ([2, 3], [1, 2]))
        # [mu, a, nu, a] -> [mu, nu, a, a]
        res = np.transpose(res, [0, 2, 1, 3])

        B = sym.b_symbol(a, b, c)
        id_a = np.eye(sym.sector_dim(a))
        expect = B[:, :, None, None] * id_a[None, None, :, :]
        npt.assert_almost_equal(res, expect)


def check_F_symbols(sym: symmetries.Symmetry, sector_sextets, sector_unitarity_test):
    """Check correct shape and unitarity of F symbols."""
    for charges in sector_sextets:
        a, b, c, d, e, f = charges
        shape = (sym.n_symbol(b, c, e), sym.n_symbol(a, e, d), sym.n_symbol(a, b, f), sym.n_symbol(f, c, d))
        F = sym.f_symbol(a, b, c, d, e, f)

        if not sym.has_complex_topological_data:
            npt.assert_almost_equal(F, np.real(F))

        assert F.shape == shape  # shape
        if np.any([np.array_equal(charge, sym.trivial_sector) for charge in [a, b, c]]):
            npt.assert_almost_equal(F, np.eye(shape[0] * shape[1]).reshape(shape))  # for trivial sector

    for charges in sector_unitarity_test:  # unitarity
        a, b, c, d, e, g = charges
        shape = (sym.n_symbol(b, c, e), sym.n_symbol(a, e, d), sym.n_symbol(b, c, g), sym.n_symbol(a, g, d))

        res = np.zeros(shape, dtype=complex)
        for f in sym.fusion_outcomes(a, b):
            if sym.can_fuse_to(f, c, d):
                F1 = sym.f_symbol(a, b, c, d, e, f)
                F2 = sym.f_symbol(a, b, c, d, g, f).conj()
                res += np.tensordot(F1, F2, axes=[[2, 3], [2, 3]])
        if np.array_equal(e, g):
            npt.assert_almost_equal(res, np.eye(shape[0] * shape[1]).reshape(shape))
        else:
            npt.assert_almost_equal(res, np.zeros(shape))


def check_R_symbols(sym: symmetries.Symmetry, sector_triplets, example_sectors_low_qdim):
    """Check correct shape and unitarity of R symbols."""
    for charges in sector_triplets:
        a, b, c = charges
        shape = (sym.n_symbol(a, b, c),)
        R = sym.r_symbol(a, b, c)

        if not sym.has_complex_topological_data:
            npt.assert_almost_equal(R, np.real(R))

        assert R.shape == shape  # shape
        npt.assert_almost_equal(np.abs(R), np.ones(shape))  # unitarity

        if np.any([np.array_equal(charge, sym.trivial_sector) for charge in [a, b]]):
            npt.assert_almost_equal(R, np.ones_like(R))  # exchange with trivial sector


def check_C_symbols(sym: symmetries.Symmetry, sector_sextets, sector_unitarity_test):
    """Check correct shape and unitarity of C symbols."""
    for charges in sector_sextets:
        c, a, b, d, e, f = charges
        shape = (sym.n_symbol(a, b, e), sym.n_symbol(e, c, d), sym.n_symbol(a, c, f), sym.n_symbol(f, b, d))
        C = sym.c_symbol(a, b, c, d, e, f)

        if not sym.has_complex_topological_data:
            npt.assert_almost_equal(C, np.real(C))

        assert C.shape == shape  # shape
        if np.any([np.array_equal(charge, sym.trivial_sector) for charge in [b, c]]):
            npt.assert_almost_equal(C, np.eye(shape[0] * shape[1]).reshape(shape))  # for trivial sector

    for charges in sector_unitarity_test:  # unitarity
        c, a, b, d, e, g = charges
        shape = (sym.n_symbol(a, b, e), sym.n_symbol(e, c, d), sym.n_symbol(a, b, g), sym.n_symbol(g, c, d))

        res = np.zeros(shape, dtype=complex)
        for f in sym.fusion_outcomes(a, c):
            if sym.can_fuse_to(f, b, d):
                C1 = sym.c_symbol(a, b, c, d, e, f)
                C2 = sym.c_symbol(a, b, c, d, g, f).conj()
                res += np.tensordot(C1, C2, axes=[[2, 3], [2, 3]])
        if np.array_equal(e, g):
            npt.assert_almost_equal(res, np.eye(shape[0] * shape[1]).reshape(shape))
        else:
            npt.assert_almost_equal(res, np.zeros(shape))


def check_B_symbols(sym: symmetries.Symmetry, sector_triplets):
    """Check correct shape, normalization and snake equation of B symbols."""
    for charges in sector_triplets:
        a, b, c = charges
        shape = (sym.n_symbol(a, b, c), sym.n_symbol(a, b, c))
        B = sym.b_symbol(a, b, c)

        if not sym.has_complex_topological_data:
            npt.assert_almost_equal(B, np.real(B))

        assert B.shape == shape  # shape

        norm = np.diag(np.ones(shape[0])) * sym.qdim(c) / sym.qdim(a)
        npt.assert_almost_equal(np.tensordot(B, B.conj(), axes=[1, 1]), norm)  # normalization

        snake = np.tensordot(B, sym.b_symbol(c, sym.dual_sector(b), a), axes=[1, 1])  # snake eq.
        npt.assert_almost_equal(snake, sym.frobenius_schur(b) * np.diag(np.ones(shape[0])))


def check_pentagon_equation(sym: symmetries.Symmetry, sector_nonets):
    r"""Check consistency of the F symbols using the pentagon equation.

    :math:`\sum_{σ} [F^{fcd}_e]^{gνρ}_{jγσ} [F^{abj}_e]^{fμσ}_{iδκ}
    = \sum_{h,σ,λ,ω} [F^{abc}_g]^{fμν}_{hσλ} [F^{ahd}_e]^{gλρ}_{iωκ} [F^{bcd}_i]^{hσω}_{jγδ}`

    This is Eq. (33) in https://arxiv.org/pdf/1511.08090.
    Compared to our convention, we have to exchange the outer indices of the F symbols.
    """
    for charges in sector_nonets:
        a, b, c, d, e, f, g, j, i = charges

        lhs = sym.f_symbol(f, c, d, e, j, g)  # [γ, σ, ν, ρ]
        lhs = np.tensordot(lhs, sym.f_symbol(a, b, j, e, i, f), axes=[1, 3])  # [γ, ν, ρ, δ, κ, μ]
        lhs = lhs.transpose([5, 1, 4, 2, 0, 3])  # [μ, ν, κ, ρ, γ, δ]

        rhs = np.zeros_like(lhs, dtype=complex)
        for h in sym.fusion_outcomes(b, c):
            if sym.can_fuse_to(h, a, g) and sym.can_fuse_to(h, d, i):
                rhs_ = sym.f_symbol(a, b, c, g, h, f)  # [σ, λ, μ, ν]
                rhs_ = np.tensordot(rhs_, sym.f_symbol(a, h, d, e, i, g), axes=[1, 2])  # [σ, μ, ν, ω, κ, ρ]
                rhs_ = np.tensordot(rhs_, sym.f_symbol(b, c, d, i, j, h), axes=([0, 3], [2, 3]))  # [μ, ν, κ, ρ, γ, δ]
                rhs += rhs_

        npt.assert_almost_equal(lhs, rhs)


def check_hexagon_equation(sym: symmetries.Symmetry, sector_sextets, check_both_versions: bool = True):
    r"""Check consistency of the R symbols using the hexagon equations.
    There are two versions of the hexagon equation that are both checked by default.

    :math:`\sum_{λ,γ} [R^{ca}_e]_{αλ} [F^{acb}_d]^{eλβ}_{gγν} [R^{cb}_g]_{γμ}
    = \sum_{f,σ,δ,ψ} [F^{cab}_d]^{eαβ}_{fδσ} [R^{cf}_d]_{σψ} [F^{abc}_d]^{fδψ}_{gμν}`

    The hexagon equation above is taken from Bonderson's PhD thesis
    (https://thesis.library.caltech.edu/2447/2/thesis.pdf).
    Compared to our convention, we have to exchange the outer indices of the F symbols.
    Further, there are some minor errors in the multiplicity indices in Bonderson's thesis,
    we have corrected these errors in the equation above.

    The second hexagon equation is obtained by letting all R symbols
    :math:`[R^{ab}_c]_{αβ} -> [(R^{ba}_c)^{-1}]_{αβ} = [R^{ab}_c]_{βα}*`
    """

    def _return_r(a, b, c, conj=False):
        if conj:
            return np.diag(sym.r_symbol(a, b, c)).conj()
        return np.diag(sym.r_symbol(a, b, c))

    conjugate = [False]
    if check_both_versions:
        conjugate.append(True)

    for charges in sector_sextets:
        # this charge labeling should be consistent with sample_sector_sextets
        a, c, b, d, g, e = charges

        for conj in conjugate:
            lhs = _return_r(c, a, e, conj)  # [α, λ]
            lhs = np.tensordot(lhs, sym.f_symbol(a, c, b, d, g, e), axes=[1, 2])  # [α, γ, ν, β]
            lhs = np.tensordot(lhs, _return_r(c, b, g, conj), axes=[1, 0])  # [α, ν, β, μ]
            lhs = lhs.transpose([0, 2, 3, 1])  # [α, β, μ, ν]

            rhs = np.zeros_like(lhs, dtype=complex)
            for f in sym.fusion_outcomes(a, b):
                if sym.can_fuse_to(c, f, d):  # this is not given
                    _rhs = sym.f_symbol(c, a, b, d, f, e)  # [δ, σ, α, β]
                    _rhs = np.tensordot(_rhs, _return_r(c, f, d, conj), axes=[1, 0])  # [δ, α, β, ψ]
                    _rhs = np.tensordot(_rhs, sym.f_symbol(a, b, c, d, g, f), axes=([0, 3], [2, 3]))  # [α, β, μ, ν]
                    rhs += _rhs

            npt.assert_almost_equal(lhs, rhs)


def test_no_symmetry(np_random):
    sym = symmetries.NoSymmetry()
    s = np.array([0])
    common_checks(sym, example_sectors=s[np.newaxis, :], example_sectors_low_qdim=s[np.newaxis, :], np_random=np_random)

    assert sym.is_abelian
    assert sym.has_unique_fusion
    assert sym.has_trivial_braid
    assert sym.has_symmetric_braid

    print('checking valid sectors')
    assert sym.is_valid_sector(s)
    assert not sym.is_valid_sector(np.array([1]))
    assert not sym.is_valid_sector(np.array([0, 0]))

    print('checking fusion_outcomes')
    npt.assert_array_equal(sym.fusion_outcomes(s, s), np.array([[0]]))

    print('checking fusion_outcomes_broadcast')
    many_s = np.stack([s, s, s])
    npt.assert_array_equal(sym.fusion_outcomes_broadcast(many_s, many_s), many_s)

    # print('checking sector dimensions')
    # nothing to do, the dimension of the only sector (trivial) is checked in common_checks

    print('checking equality')
    assert sym == sym
    assert sym == symmetries.NoSymmetry()
    assert sym != symmetries.U1()
    assert sym != symmetries._SU2() * u1_symmetry

    print('checking is_equivalent_to')
    assert sym.is_equivalent_to(symmetries.NoSymmetry())
    assert not sym.is_equivalent_to(symmetries.U1())
    assert not sym.is_equivalent_to(symmetries._SU2() * u1_symmetry)

    print('checking dual_sector')
    npt.assert_array_equal(sym.dual_sector(s), s)

    print('checking dual_sectors')
    npt.assert_array_equal(sym.dual_sectors(many_s), many_s)


def test_get_sector():
    # Single-factor FermionParity: brackets optional
    fp = fermion_parity
    odd = symmetries.FermionParity.odd
    even = symmetries.FermionParity.even
    assert fp.get_sector('odd') == odd
    assert fp.get_sector('[odd]') == odd
    assert fp.get_sector('even') == even
    assert fp.factors[0].get_sector('odd') == odd

    # U1 charge string without brackets
    assert u1_symmetry.get_sector('1') == symmetries.Sector([1])
    assert u1_symmetry.get_sector('[1]') == symmetries.Sector([1])
    assert u1_symmetry.get_sector('-3') == symmetries.Sector([-3])

    # FermionNumber uses default "[n]" sector_str
    assert fermion_number.get_sector('[2]') == symmetries.Sector([2])
    assert fermion_number.get_sector('2') == symmetries.Sector([2])

    # Multi-factor product requires brackets
    product = u1_symmetry * fermion_parity
    assert product.get_sector('[1, odd]') == symmetries.Sector([1, 1])
    assert product.get_sector(product.sector_str(symmetries.Sector([1, 1]))) == symmetries.Sector([1, 1])
    with pytest.raises(ValueError):
        product.get_sector('odd')

    # Site state labels are not sector names
    with pytest.raises(ValueError):
        fp.get_sector('up')
    with pytest.raises(ValueError):
        fermion_number.get_sector('empty')

    # _SU2 accepts bare jj or full sector_str
    su2 = symmetries._SU2().as_Symmetry()
    assert su2.get_sector('1') == symmetries.Sector([1])
    assert su2.get_sector('1 (J=1/2)') == symmetries.Sector([1])
    assert su2.get_sector(su2.sector_str(symmetries.Sector([2]))) == symmetries.Sector([2])


# @pytest.mark.xfail(reason='Topological data not implemented.')
def test_product_symmetry(np_random):
    doubleFibo = symmetries.Symmetry(
        [symmetries.FibonacciAnyonCategory('left'), symmetries.FibonacciAnyonCategory('right')]
    )
    common_checks(
        doubleFibo,
        example_sectors=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        example_sectors_low_qdim=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        np_random=np_random,
    )
    assert doubleFibo._f_symbol([0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]) == 1
    npt.assert_almost_equal(
        doubleFibo._f_symbol([1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1])[0, 0, 0, 0],
        1 / ((0.5 * (1 + np.sqrt(5))) ** 2),
    )

    for k in range(1, 16, 2):
        doubleIsing = symmetries.Symmetry([symmetries.IsingAnyonCategory(k), symmetries.IsingAnyonCategory(-k)])
        common_checks(
            doubleIsing,
            example_sectors=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [1, 2], [2, 2], [0, 2], [2, 0]]),
            example_sectors_low_qdim=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [1, 2], [2, 2], [0, 2], [2, 0]]),
            np_random=np_random,
        )
        npt.assert_almost_equal(doubleIsing._r_symbol([1, 1], [1, 1], [0, 0]), 1)
        npt.assert_almost_equal(doubleIsing._r_symbol([1, 1], [2, 2], [1, 1]), 1)
        npt.assert_almost_equal(doubleIsing._r_symbol([2, 2], [1, 1], [1, 1]), 1)

    sym = symmetries.Symmetry([symmetries._SU2(), symmetries.U1(), symmetries.FermionParity()])
    sym_with_name = symmetries.Symmetry([symmetries._SU2('foo'), symmetries.U1('bar'), symmetries.FermionParity()])
    s1 = np.array([5, 3, 1])  # e.g. spin 5/2 , 3 particles , odd parity ("fermionic")
    s2 = np.array([3, 2, 0])  # e.g. spin 3/2 , 2 particles , even parity ("bosonic")
    sectors = np.array([s1, s2])
    common_checks(
        sym, example_sectors=sectors, example_sectors_low_qdim=sectors, np_random=np_random, skip_fusion_tensor=False
    )

    u1_z3 = u1_symmetry * z3_symmetry
    common_checks(
        u1_z3,
        example_sectors=np.array([[42, 1], [-1, 2], [-2, 0]]),
        example_sectors_low_qdim=np.array([[42, 1], [-1, 2], [-2, 0]]),
        np_random=np_random,
        skip_fusion_tensor=False,
    )

    assert not sym.is_abelian
    assert sym.has_unique_fusion
    assert not sym.has_trivial_braid
    assert sym.has_symmetric_braid

    assert u1_z3.is_abelian
    assert u1_z3.has_unique_fusion
    assert u1_z3.has_trivial_braid
    assert u1_z3.has_symmetric_braid

    print('checking creation via __mul__')
    sym2 = symmetries._SU2() * u1_symmetry * fermion_parity
    assert sym2 == sym

    print('checking valid sectors')
    assert sym.is_valid_sector(s1)
    assert sym.is_valid_sector(s2)
    assert not sym.is_valid_sector(np.array([-1, 2, 0]))  # negative spin is invalid
    assert not sym.is_valid_sector(np.array([3, 2, 42]))  # parity not in [0, 1] is invalid
    assert not sym.is_valid_sector(np.array([3, 2, 0, 1]))  # too many entries

    print('checking fusion_outcomes')
    outcomes = sym.fusion_outcomes(s1, s2)
    # spin 3/2 and 5/2 can fuse to [1, 2, 3, 4]  ;  U(1) charges  3 + 2 = 5  ;  fermion charges 1 + 0 = 1
    expect = np.array([[2, 5, 1], [4, 5, 1], [6, 5, 1], [8, 5, 1]])
    npt.assert_array_equal(outcomes, expect)

    print('checking fusion_outcomes_broadcast')
    with pytest.raises(ValueError):
        # sym is not abelian, so this should raise
        _ = sym.fusion_outcomes_broadcast(s1[None, :], s2[None, :])
    outcomes = u1_z3.fusion_outcomes_broadcast(np.array([[42, 2], [-2, 0]]), np.array([[1, 1], [2, 1]]))
    npt.assert_array_equal(outcomes, np.array([[43, 0], [0, 1]]))

    print('checking sector dimensions')
    assert sym.sector_dim(s1) == 6
    assert sym.sector_dim(s2) == 4

    print('checking equality')
    assert sym == sym
    assert sym != sym_with_name
    assert sym != symmetries._SU2() * u1_symmetry
    assert sym != no_symmetry

    print('checking is_equivalent_to')
    assert sym.is_equivalent_to(sym)
    assert sym.is_equivalent_to(sym_with_name)
    assert not sym.is_equivalent_to(symmetries._SU2() * u1_symmetry)
    assert not sym.is_equivalent_to(no_symmetry)

    print('checking dual_sector')
    npt.assert_array_equal(sym.dual_sector(s1), np.array([5, -3, 1]))
    npt.assert_array_equal(sym.dual_sector(s2), np.array([3, -2, 0]))

    print('checking dual_sectors')
    npt.assert_array_equal(sym.dual_sectors(np.stack([s1, s2])), np.array([[5, -3, 1], [3, -2, 0]]))


def test_u1_symmetry(np_random):
    sym = symmetries.U1()
    sym_with_name = symmetries.U1('foo')
    s_0 = np.array([0])
    s_1 = np.array([1])
    s_neg1 = np.array([-1])
    s_2 = np.array([2])
    s_42 = np.array([42])
    sectors = np.array([s_0, s_1, s_neg1, s_2, s_42])
    common_checks(sym, example_sectors=sectors, example_sectors_low_qdim=sectors, np_random=np_random)

    assert sym.is_abelian
    assert sym.has_unique_fusion
    assert sym.has_trivial_braid
    assert sym.has_symmetric_braid

    print('checking valid sectors')
    for s in [s_0, s_1, s_neg1, s_2, s_42]:
        assert sym.is_valid_sector(s)
    assert not sym.is_valid_sector(np.array([0, 0]))

    print('checking fusion_outcomes')
    npt.assert_array_equal(sym.fusion_outcomes(s_1, s_1), s_2[None, :])
    npt.assert_array_equal(sym.fusion_outcomes(s_neg1, s_1), s_0[None, :])

    print('checking fusion_outcomes_broadcast')
    outcomes = sym.fusion_outcomes_broadcast(np.stack([s_0, s_1, s_0]), np.stack([s_neg1, s_1, s_2]))
    npt.assert_array_equal(outcomes, np.stack([s_neg1, s_2, s_2]))

    print('checking sector dimensions')
    for s in [s_0, s_1, s_neg1, s_2, s_42]:
        assert sym.sector_dim(s) == 1

    print('checking equality')
    assert sym == sym
    assert sym != sym_with_name
    assert sym == symmetries.U1()
    assert sym != symmetries.NoSymmetry()
    assert sym != symmetries._SU2() * u1_symmetry

    print('checking is_equivalent_to')
    assert sym.is_equivalent_to(sym)
    assert sym.is_equivalent_to(sym_with_name)
    assert sym.is_equivalent_to(symmetries.U1('even_with_name'))
    assert not sym.is_equivalent_to(symmetries.NoSymmetry())
    assert not sym.is_equivalent_to(symmetries._SU2() * u1_symmetry)

    print('checking dual_sector')
    npt.assert_array_equal(sym.dual_sector(s_1), s_neg1)

    print('checking dual_sectors')
    npt.assert_array_equal(sym.dual_sectors(np.stack([s_0, s_1, s_42, s_2])), np.array([0, -1, -42, -2])[:, None])


@pytest.mark.parametrize('N', [2, 3, 4, 42])
def test_ZN_symmetry(N, np_random):
    sym = symmetries.ZN(N=N)
    sym_with_name = symmetries.ZN(N, descriptive_name='foo')
    sectors_a = np.array([0, 1, 2, 10])[:, None] % N
    sectors_b = np.array([0, 1, 3, 11])[:, None] % N
    common_checks(sym, example_sectors=sectors_a, example_sectors_low_qdim=sectors_a, np_random=np_random)

    assert sym.is_abelian
    assert sym.has_unique_fusion
    assert sym.has_trivial_braid
    assert sym.has_symmetric_braid

    print('checking valid sectors')
    for s in sectors_a:
        assert sym.is_valid_sector(s)
    assert not sym.is_valid_sector(np.array([0, 0]))
    assert not sym.is_valid_sector(np.array([N]))
    assert not sym.is_valid_sector(np.array([-1]))

    print('checking fusion_outcomes')
    for a in sectors_a:
        for b in sectors_b:
            expect = (a + b)[None, :] % N
            npt.assert_array_equal(sym.fusion_outcomes(a, b), expect)

    print('checking fusion_outcomes_broadcast')
    expect = (sectors_a + sectors_b) % N
    npt.assert_array_equal(sym.fusion_outcomes_broadcast(sectors_a, sectors_b), expect)

    print('checking sector dimensions')
    for s in sectors_a:
        assert sym.sector_dim(s) == 1

    print('checking equality')
    other = {N: symmetries.ZN(N) for N in [2, 3, 4, 5, 42, 43]}
    assert sym == sym
    assert sym != sym_with_name
    assert sym == other[N]
    assert sym != other[N + 1]
    assert sym != u1_symmetry

    print('checking is_equivalent_to')
    assert sym.is_equivalent_to(sym)
    assert sym.is_equivalent_to(sym_with_name)
    assert sym.is_equivalent_to(other[N])
    assert not sym.is_equivalent_to(other[N + 1])
    assert not sym.is_equivalent_to(u1_symmetry)

    print('checking dual_sector')
    for s in sectors_a:
        npt.assert_array_equal(sym.dual_sector(s), (-s) % N)

    print('checking dual_sectors')
    npt.assert_array_equal(sym.dual_sectors(sectors_a), (-sectors_a) % N)


def test_su2_symmetry(np_random):
    sym = symmetries._SU2()
    common_checks(
        sym,
        example_sectors=np.array([[0], [3], [5], [2], [1], [23]]),
        example_sectors_low_qdim=np.array([[0], [2], [5], [3], [4]]),
        np_random=np_random,
    )

    spin_1 = np.array([2])
    spin_3_half = np.array([3])
    sym_with_name = symmetries._SU2('foo')

    assert not sym.is_abelian
    assert sym.has_unique_fusion
    assert sym.has_trivial_braid
    assert sym.has_symmetric_braid

    print('checking valid sectors')
    for valid in [[0], [1], [2], [42]]:
        assert sym.is_valid_sector(np.array(valid))
    for invalid in [[-1], [0, 0]]:
        assert not sym.is_valid_sector(np.array(invalid))

    print('checking fusion_outcomes')
    # 1 x 3/2 = 1/2 + 3/2 + 5/2
    npt.assert_array_equal(sym.fusion_outcomes(spin_1, spin_3_half), np.array([[1], [3], [5]]))

    print('checking fusion_outcomes_broadcast')
    with pytest.raises(ValueError):
        # sym is not abelian, so this should raise
        _ = sym.fusion_outcomes_broadcast(spin_1[None, :], spin_3_half[None, :])

    print('checking sector dimensions')
    assert sym.sector_dim(spin_1) == 3
    assert sym.sector_dim(spin_3_half) == 4

    print('checking equality')
    assert sym == sym
    assert sym != sym_with_name
    assert sym == symmetries._SU2()
    assert sym != fermion_parity

    print('checking dual_sector')
    npt.assert_array_equal(sym.dual_sector(spin_1), spin_1)
    npt.assert_array_equal(sym.dual_sector(spin_3_half), spin_3_half)

    print('checking dual_sectors')
    npt.assert_array_equal(sym.dual_sectors(np.stack([spin_1, spin_3_half])), np.stack([spin_1, spin_3_half]))


_SUN_SYMBOLS_SUBMODULE_DIR = str(Path(__file__).resolve().parents[2] / 'external' / 'SUN_symbols')


def _locate_su_n_files(N, cg_h, f_h, r_h):
    """Locate (CGfile, Ffile, Rfile) for the given (N, hweights).

    Tries, in order: the configured ``su_n_data_path`` (standard naming convention), the
    ``external/SUN_symbols`` git submodule (same convention), then legacy hand-made test file
    names (kept so existing local data still works).

    Returns ``(files, path)`` where ``path`` is the directory to pass as
    ``symmetries.SUN(..., path=path)`` -- ``None`` meaning "use the configured su_n_data_path" --
    or ``False`` if only the legacy names were found (no path-based constructor available for
    those). Returns ``(None, None)`` if nothing was found.
    """
    for path in (None, _SUN_SYMBOLS_SUBMODULE_DIR):
        standard = [
            symmetries.su_n_data_file_path(N, kind, h, path=path) for kind, h in [('CG', cg_h), ('F', f_h), ('R', r_h)]
        ]
        if all(os.path.exists(f) for f in standard):
            return standard, path
    legacy = [
        f'Test_N_{N}_HWeight_{cg_h}.hdf5',
        f'Test_Fsymb_{N}_HWeight_{f_h}.hdf5',
        f'Test_Rsymb_{N}_HWeight_{r_h}.hdf5',
    ]
    if all(os.path.exists(f) for f in legacy):
        return legacy, False
    return None, None


@pytest.mark.parametrize('N, cg_h, f_h, r_h', [(2, 20, 6, 6), (3, 7, 4, 4)])
def test_suN_symmetry(N, cg_h, f_h, r_h, np_random):
    files, _ = _locate_su_n_files(N, cg_h, f_h, r_h)
    if files is None:
        pytest.skip(
            f'Need to provide files for SU(N) data! Expected e.g. {symmetries.su_n_data_file_path(N, "CG", cg_h)}'
        )
    CGfile, Ffile, Rfile = files

    def gen_irrepsTEST(N, k):
        """generates a list of all possible irreps for given N and highest weight k"""
        if N <= 0:
            return [[]]
        r = []
        for i in range(k, -1, -1):
            for comb in gen_irrepsTEST(N - 1, i):
                a = [i] + comb
                if a[-1] == 0:
                    r.append(a[:])
        return r

    CGfile = h5py.File(CGfile, 'r')
    Ffile = h5py.File(Ffile, 'r')
    Rfile = h5py.File(Rfile, 'r')
    sym = symmetries.SUN(N, CGfile, Ffile, Rfile)
    sym_with_name = symmetries.SUN(N, CGfile, Ffile, Rfile, 'Some SU(N)')
    exsectors = np.array(gen_irrepsTEST(N, 2))
    common_checks(
        sym,
        example_sectors=exsectors,
        example_sectors_low_qdim=np.array([[0] * N, [1] + [0] * (N - 1), [2] + [0] * (N - 1)]),
        np_random=np_random,
    )

    # spin_1 = np.array([2])
    # spin_3_half = np.array([3])
    # sym_with_name = symmetries._SU2('foo')

    assert not sym.is_abelian
    # SUN always uses FusionStyle::general (outer multiplicities can exceed 1 for N >= 3, via the
    # "Outer Multiplicity" field in the CG data), so has_unique_fusion is False for every N,
    # including N=2 even though SU(2) fusion itself has no multiplicities.
    assert not sym.has_unique_fusion
    assert sym.has_trivial_braid
    assert sym.has_symmetric_braid

    print('checking valid sectors')
    for valid in gen_irrepsTEST(N, 2):
        assert sym.is_valid_sector(np.array(valid))
    for invalid in [[-1], [-1, 0], [1, 2]]:
        assert not sym.is_valid_sector(np.array(invalid))

    print('checking sector dimensions')
    assert sym.sector_dim([0] * N) == 1

    print('checking equality')
    assert sym == sym
    assert sym != sym_with_name
    assert sym != symmetries._SU2()
    assert sym != fermion_parity


@pytest.mark.parametrize('N, cg_h, f_h, r_h', [(2, 20, 6, 6), (3, 7, 4, 4)])
def test_suN_from_config(N, cg_h, f_h, r_h):
    files, path = _locate_su_n_files(N, cg_h, f_h, r_h)
    if files is None or path is False:
        pytest.skip('Need SU(N) data files following the standard naming convention!')

    sym = symmetries.SUN(N, cg_h, f_hweight=f_h, r_hweight=r_h, path=path)
    assert sym.N == N
    assert sym.hweight_from_CG_hdf5() == cg_h
    assert sym.hweight_from_F_hdf5() == f_h
    assert sym.hweight_from_R_hdf5() == r_h

    handles = [h5py.File(f, 'r') for f in files]
    assert sym == symmetries.SUN(N, *handles)

    named = symmetries.SUN(N, cg_h, f_hweight=f_h, r_hweight=r_h, path=path, descriptive_name='foo')
    assert named != sym


def test_suN_missing_data_error(tmp_path):
    from lasyten.config import temporary_options

    with temporary_options(su_n_data_path=str(tmp_path)):
        with pytest.raises(FileNotFoundError, match=r'su_n_clebsch_gordan_data_N3_CG_hweight7\.hdf5'):
            symmetries.SUN(3, 7)
        with pytest.raises(FileNotFoundError, match='LASYTEN_SU_N_DATA_PATH'):
            symmetries.SUN(3, 7)

    # the path/filename_base overrides are a per-call escape hatch too
    with pytest.raises(FileNotFoundError, match=r'nope[/\\]base_N3_CG_hweight7\.hdf5'):
        symmetries.SUN(3, 7, path=str(tmp_path / 'nope'), filename_base='base')


def test_suN_hweight_validation(tmp_path):
    with pytest.raises(ValueError, match='CG'):
        symmetries.SUN(3, 4, cg_hweight=2, f_hweight=4, path=str(tmp_path))


def test_suN_overload_dispatch():
    """(N, hweight) and (N, CGfile, Ffile, Rfile) must not shadow each other.

    cg/f/r hweight overrides are keyword-only, so ``SUN(3, 7, 4, 4)`` cannot hit the
    ``(N, hweight, ...)`` overload. It is accepted by the ``(N, CGfile, Ffile, Rfile, ...)``
    overload instead -- ``py::object`` parameters match any Python object, ints included -- and
    fails once the body tries to treat the ints as HDF5 file handles.
    """
    with pytest.raises(AttributeError):
        symmetries.SUN(3, 7, 4, 4)


def test_fermion_parity(np_random):
    sym = symmetries.FermionParity()
    even = np.array([0])
    odd = np.array([1])
    common_checks(
        sym, example_sectors=np.array([even, odd]), example_sectors_low_qdim=np.array([even, odd]), np_random=np_random
    )

    assert sym.is_abelian
    assert sym.has_unique_fusion
    assert not sym.has_trivial_braid
    assert sym.has_symmetric_braid

    print('checking valid sectors')
    assert sym.is_valid_sector(odd)
    assert not sym.is_valid_sector(np.array([2]))
    assert not sym.is_valid_sector(np.array([-1]))
    assert not sym.is_valid_sector(np.array([0, 0]))

    print('checking fusion_outcomes')
    npt.assert_array_equal(sym.fusion_outcomes(odd, odd), even[None, :])
    npt.assert_array_equal(sym.fusion_outcomes(odd, even), odd[None, :])

    print('checking fusion_outcomes_broadcast')
    npt.assert_array_equal(
        sym.fusion_outcomes_broadcast(np.stack([even, even, odd]), np.stack([even, odd, odd])),
        np.stack([even, odd, even]),
    )

    print('checking equality')
    assert sym == sym
    assert sym == symmetries.FermionParity()
    assert sym != symmetries.FermionNumber()
    assert sym != symmetries.NoSymmetry()
    assert sym != symmetries._SU2()
    assert sym != fermion_parity * z2_symmetry

    print('checking is_equivalent_to')
    assert sym.is_equivalent_to(sym)
    assert not sym.is_equivalent_to(no_symmetry)
    assert not sym.is_equivalent_to(symmetries._SU2())

    print('checking dual_sector')
    npt.assert_array_equal(sym.dual_sector(odd), odd)

    print('checking dual_sectors')
    npt.assert_array_equal(sym.dual_sectors(np.stack([odd, even, odd])), np.stack([odd, even, odd]))


def test_fermion_number(np_random):
    sym = symmetries.FermionNumber()
    example_sectors = np.array([0, 1, -1, 2, 42, -123])[:, None]
    common_checks(sym, example_sectors=example_sectors, example_sectors_low_qdim=example_sectors, np_random=np_random)

    assert sym.is_abelian
    assert sym.has_unique_fusion
    assert not sym.has_trivial_braid
    assert sym.has_symmetric_braid

    print('checking valid sectors')
    assert sym.is_valid_sector(example_sectors[0])
    assert sym.is_valid_sector(np.array([2]))
    assert sym.is_valid_sector(np.array([-1]))
    assert not sym.is_valid_sector(np.array([0, 0]))

    print('checking fusion_outcomes')
    npt.assert_array_equal(sym.fusion_outcomes(np.array([1]), np.array([1])), np.array([[2]]))
    npt.assert_array_equal(sym.fusion_outcomes(np.array([1]), np.array([-1])), np.array([[0]]))

    print('checking fusion_outcomes_broadcast')
    npt.assert_array_equal(sym.fusion_outcomes_broadcast(example_sectors, 2 * example_sectors), 3 * example_sectors)

    print('checking equality')
    assert sym == sym
    assert sym == symmetries.FermionNumber()
    assert sym != symmetries.FermionParity()
    assert sym != symmetries.NoSymmetry()
    assert sym != symmetries._SU2()
    assert sym != fermion_number * z2_symmetry

    print('checking is_equivalent_to')
    assert sym.is_equivalent_to(sym)
    assert not sym.is_equivalent_to(symmetries.NoSymmetry())
    assert not sym.is_equivalent_to(symmetries._SU2())

    print('checking dual_sector')
    npt.assert_array_equal(sym.dual_sector(np.array([2])), np.array([-2]))

    print('checking dual_sectors')
    npt.assert_array_equal(sym.dual_sectors(example_sectors), -example_sectors)


@pytest.mark.parametrize('handedness', ['left', 'right'])
def test_fibonacci_grading(handedness, np_random):
    other_handedness = 'right' if handedness == 'left' else 'left'
    sym = symmetries.FibonacciAnyonCategory(handedness)
    vac = np.array([0])
    tau = np.array([1])
    common_checks(
        sym, example_sectors=sym.all_sectors(), example_sectors_low_qdim=sym.all_sectors(), np_random=np_random
    )

    assert not sym.is_abelian
    assert sym.has_unique_fusion
    assert not sym.has_trivial_braid
    assert not sym.has_symmetric_braid

    print('checking valid sectors')
    assert sym.is_valid_sector(tau)
    assert not sym.is_valid_sector(np.array([3]))
    assert not sym.is_valid_sector(np.array([-1]))
    assert not sym.is_valid_sector(np.array([0, 0]))

    print('checking fusion rules')
    npt.assert_array_equal(sym.fusion_outcomes(vac, tau), tau[None, :])
    npt.assert_array_equal(sym.fusion_outcomes(tau, tau), np.stack([vac, tau]))

    print('checking equality')
    assert sym == sym
    assert sym == symmetries.FibonacciAnyonCategory(handedness)
    assert sym != symmetries.FibonacciAnyonCategory(other_handedness)
    assert sym != no_symmetry
    assert sym != symmetries._SU2()

    print('checking is_equivalent_to')
    assert sym.is_equivalent_to(sym)
    assert sym.is_equivalent_to(symmetries.FibonacciAnyonCategory(handedness))
    assert not sym.is_equivalent_to(symmetries.FibonacciAnyonCategory(other_handedness))
    assert not sym.is_equivalent_to(no_symmetry)
    assert not sym.is_equivalent_to(symmetries._SU2())

    print('checking dual_sector')
    npt.assert_array_equal(sym.dual_sector(tau), tau)


@pytest.mark.parametrize('nu', [*range(1, 16, 2)])
def test_ising_grading(nu, np_random):
    sym = symmetries.IsingAnyonCategory(nu)
    vac = np.array([0])
    anyon = np.array([1])
    fermion = np.array([2])
    common_checks(
        sym, example_sectors=sym.all_sectors(), example_sectors_low_qdim=sym.all_sectors(), np_random=np_random
    )

    assert not sym.is_abelian
    assert sym.has_unique_fusion
    assert not sym.has_trivial_braid
    assert not sym.has_symmetric_braid

    print('checking valid sectors')
    assert sym.is_valid_sector(anyon)
    assert sym.is_valid_sector(fermion)
    assert not sym.is_valid_sector(np.array([3]))
    assert not sym.is_valid_sector(np.array([-1]))
    assert not sym.is_valid_sector(np.array([0, 0]))

    print('checking fusion rules')
    npt.assert_array_equal(sym.fusion_outcomes(vac, anyon), anyon[None, :])
    npt.assert_array_equal(sym.fusion_outcomes(vac, fermion), fermion[None, :])
    npt.assert_array_equal(sym.fusion_outcomes(anyon, fermion), anyon[None, :])
    npt.assert_array_equal(sym.fusion_outcomes(anyon, anyon), np.stack([vac, fermion]))

    print('checking equality')
    assert sym == sym
    assert sym == symmetries.IsingAnyonCategory(nu)
    assert sym != symmetries.IsingAnyonCategory(nu + 2)
    assert sym != symmetries.NoSymmetry()
    assert sym != symmetries._SU2()

    print('checking is_equivalent_to')
    assert sym.is_equivalent_to(sym)
    assert sym.is_equivalent_to(symmetries.IsingAnyonCategory(nu))
    assert not sym.is_equivalent_to(symmetries.IsingAnyonCategory(nu + 2))
    assert not sym.is_equivalent_to(symmetries.NoSymmetry())
    assert not sym.is_equivalent_to(symmetries._SU2())

    print('checking dual_sector')
    npt.assert_array_equal(sym.dual_sector(anyon), anyon)
    npt.assert_array_equal(sym.dual_sector(fermion), fermion)


def test_SU3_3AnyonCategory(np_random):
    a = np.array([0])
    b = np.array([1])
    c = np.array([2])
    d = np.array([3])
    sym = symmetries.SU3_3AnyonCategory()
    common_checks(
        sym, example_sectors=sym.all_sectors(), example_sectors_low_qdim=sym.all_sectors(), np_random=np_random
    )

    assert not sym.is_abelian
    assert not sym.has_unique_fusion
    assert not sym.has_trivial_braid
    assert not sym.has_symmetric_braid

    print('checking valid sectors')
    for valid in [a, b, c, d]:
        assert sym.is_valid_sector(valid)
    for invalid in [[-1], [4], [0, 0]]:
        assert not sym.is_valid_sector(np.array(invalid))

    print('checking fusion_outcomes')
    npt.assert_array_equal(sym.fusion_outcomes(b, b), np.stack([a, b, c, d]))
    npt.assert_array_equal(sym.fusion_outcomes(b, c), b[None, :])
    npt.assert_array_equal(sym.fusion_outcomes(b, d), b[None, :])
    npt.assert_array_equal(sym.fusion_outcomes(c, c), d[None, :])
    npt.assert_array_equal(sym.fusion_outcomes(c, d), a[None, :])
    npt.assert_array_equal(sym.fusion_outcomes(d, d), c[None, :])

    print('checking equality')
    assert sym == sym
    assert sym != no_symmetry
    assert sym != symmetries._SU2()

    print('checking is_equivalent_to')
    assert sym.is_equivalent_to(sym)
    assert not sym.is_equivalent_to(no_symmetry)
    assert not sym.is_equivalent_to(symmetries._SU2())

    print('checking dual_sector')
    npt.assert_array_equal(sym.dual_sector(b), b)
    npt.assert_array_equal(sym.dual_sector(c), d)
    npt.assert_array_equal(sym.dual_sector(d), c)

    print('checking dual_sectors')
    npt.assert_array_equal(sym.dual_sectors(np.stack([a, b, c, d])), np.stack([a, b, d, c]))


@pytest.mark.parametrize('N', [4, 7, 36])
@pytest.mark.parametrize('n', [1, 3, 4])
@pytest.mark.parametrize('cls', [symmetries.ZNAnyonCategory, symmetries.ZNAnyonCategory2])
def test_ZNAnyonCategories(cls, N, n, np_random):
    if cls is symmetries.ZNAnyonCategory2 and N % 2 != 0:
        with pytest.raises(ValueError):
            _ = cls(N, n)
        return  # error is expected behavior

    sym = cls(N, n)
    sectors_a = np.array([0, 1, 2, 10])[:, None] % N
    sectors_b = np.array([0, 1, 3, 11])[:, None] % N
    common_checks(sym, example_sectors=sectors_a, example_sectors_low_qdim=sectors_a, np_random=np_random)

    assert sym.is_abelian
    assert sym.has_unique_fusion
    assert not sym.has_trivial_braid
    assert not sym.has_symmetric_braid

    print('checking valid sectors')
    for s in sectors_a:
        assert sym.is_valid_sector(s)
    assert not sym.is_valid_sector(np.array([0, 0]))
    assert not sym.is_valid_sector(np.array([N]))
    assert not sym.is_valid_sector(np.array([-1]))

    print('checking fusion_outcomes')
    for a in sectors_a:
        for b in sectors_b:
            expect = (a + b)[None, :] % N
            npt.assert_array_equal(sym.fusion_outcomes(a, b), expect)

    print('checking fusion_outcomes_broadcast')
    expect = (sectors_a + sectors_b) % N
    npt.assert_array_equal(sym.fusion_outcomes_broadcast(sectors_a, sectors_b), expect)

    print('checking sector dimensions')
    for s in sectors_a:
        assert sym.sector_dim(s) == 1

    print('checking equality')
    cls = sym.__class__
    other = [cls(N, n), cls(N, n + N), cls(N, n - 1), cls(N, n + 1), cls(N + 2, n)]
    assert sym == sym
    assert sym == cls(N, n)
    assert sym == cls(N, n + N)
    assert sym != cls(N, n - 1)
    assert sym != cls(N, n + 1)
    assert sym != cls(N + 2, n)
    if cls is symmetries.ZNAnyonCategory and N % 2 == 0:
        assert sym != symmetries.ZNAnyonCategory2(N, n)
    if cls is symmetries.ZNAnyonCategory2:
        assert sym != symmetries.ZNAnyonCategory(N, n)

    print('checking is_equivalent_to')
    assert sym.is_equivalent_to(sym)
    assert sym.is_equivalent_to(other[0])
    assert sym.is_equivalent_to(other[1])
    for i in range(2, 5):
        assert not sym.is_equivalent_to(other[i])
    assert not sym.is_equivalent_to(u1_symmetry)

    print('checking dual_sector')
    for s in sectors_a:
        npt.assert_array_equal(sym.dual_sector(s), (-s) % N)

    print('checking dual_sectors')
    npt.assert_array_equal(sym.dual_sectors(sectors_a), (-sectors_a) % N)


def test_semion_constructors():
    s = symmetries.semion_category()
    assert s == symmetries.ZNAnyonCategory2(2, 0).as_Symmetry()
    d = symmetries.double_semion_category()
    assert d == symmetries.ZNAnyonCategory2(2, 0) * symmetries.ZNAnyonCategory2(2, 1)


@pytest.mark.parametrize('N', [3, 8, 31])
def test_QuantumDoubleZNAnyonCategory(N, np_random):
    sym = symmetries.QuantumDoubleZNAnyonCategory(N)
    sectors_a = np.array([[0, 0], [1, 2], [2, 1], [10, 21]]) % N
    sectors_b = np.array([[0, 2], [1, 1], [3, 8], [11, 4]]) % N
    common_checks(sym, example_sectors=sectors_a, example_sectors_low_qdim=sectors_a, np_random=np_random)

    assert sym.is_abelian
    assert sym.has_unique_fusion
    assert not sym.has_trivial_braid
    assert not sym.has_symmetric_braid

    print('checking valid sectors')
    for s in np.append(sectors_a, sectors_b, axis=0):
        assert sym.is_valid_sector(s)
    assert not sym.is_valid_sector(np.array([0, 0, 0]))
    assert not sym.is_valid_sector(np.array([0]))
    assert not sym.is_valid_sector(np.array([N, 0]))
    assert not sym.is_valid_sector(np.array([0, N]))

    print('checking fusion_outcomes')
    for a in sectors_a:
        for b in sectors_b:
            expect = (a + b)[None, :] % N
            npt.assert_array_equal(sym.fusion_outcomes(a, b), expect)

    print('checking fusion_outcomes_broadcast')
    expect = (sectors_a + sectors_b) % N
    npt.assert_array_equal(sym.fusion_outcomes_broadcast(sectors_a, sectors_b), expect)

    print('checking sector dimensions')
    for s in sectors_a:
        assert sym.sector_dim(s) == 1

    print('checking equality')
    other = [
        symmetries.QuantumDoubleZNAnyonCategory(N),
        symmetries.QuantumDoubleZNAnyonCategory(N + 1),
        symmetries.ZNAnyonCategory(N, 1),
        symmetries.Symmetry([symmetries.ZNAnyonCategory(N, 1), symmetries.ZNAnyonCategory(N, 1)]),
    ]
    assert sym == sym
    assert sym == other[0]
    assert sym != other[1]
    assert sym != other[2]
    assert sym != other[3]
    assert sym != no_symmetry

    print('checking is_equivalent_to')
    assert sym.is_equivalent_to(sym)
    assert sym.is_equivalent_to(other[0])
    for i in range(1, 4):
        assert not sym.is_equivalent_to(other[i])
    assert not sym.is_equivalent_to(u1_symmetry)

    print('checking dual_sector')
    for s in sectors_a:
        npt.assert_array_equal(sym.dual_sector(s), (-s) % N)

    print('checking dual_sectors')
    npt.assert_array_equal(sym.dual_sectors(sectors_a), (-sectors_a) % N)


@pytest.mark.parametrize('k', [3, 4, 6])
@pytest.mark.parametrize('handedness', ['left', 'right'])
def test_SU2_kAnyonCategory(k, handedness, np_random):
    sym = symmetries.SU2_kAnyonCategory(k, handedness)
    sectors_a = np.array([[i] for i in range(min(k + 1, 10))])
    sectors_b = np.array([[i] for i in range(min(k + 1, 5))])
    common_checks(sym, example_sectors=sectors_a, example_sectors_low_qdim=sectors_b, np_random=np_random)

    assert not sym.is_abelian
    assert sym.has_unique_fusion
    assert not sym.has_trivial_braid
    assert not sym.has_symmetric_braid

    print('checking valid sectors')
    for valid in [[0], [1], [k]]:
        assert sym.is_valid_sector(np.array(valid))
    for invalid in [[-1], [0, 0], [k + 1]]:
        assert not sym.is_valid_sector(np.array(invalid))

    print('checking fusion_outcomes')
    npt.assert_array_equal(sym.fusion_outcomes(sectors_a[-1], sectors_a[-1]), sectors_a[0][None, :])
    npt.assert_array_equal(sym.fusion_outcomes(sectors_a[-1], sectors_a[-2]), sectors_a[1][None, :])
    npt.assert_array_equal(sym.fusion_outcomes(sectors_a[-2], sectors_a[-2]), sectors_a[0:4:2])
    a, b = sectors_a[2][0], sectors_a[-3][0]
    limit = min(a + b, 2 * k - a - b)
    npt.assert_array_equal(sym.fusion_outcomes(sectors_a[2], sectors_a[-3]), sectors_a[abs(a - b) : limit + 2 : 2])

    print('checking equality')
    assert sym == sym
    assert (sym == symmetries.SU2_kAnyonCategory(k, 'right')) == (handedness == 'right')
    assert sym != symmetries.SU2_kAnyonCategory(k + 1, handedness)
    assert sym != symmetries._SU2()

    print('checking dual_sector')
    npt.assert_array_equal(sym.dual_sector(sectors_a[-1]), sectors_a[-1])
    npt.assert_array_equal(sym.dual_sector(sectors_b[-1]), sectors_b[-1])

    print('checking dual_sectors')
    npt.assert_array_equal(sym.dual_sectors(sectors_a), sectors_a)


# test_suN_symmetry(3,'/space/ge36xeh/TenpyV2a/Test_N_3_HWeight_7.hdf5', '/space/ge36xeh/TenpyV2a/Test_Fsymb_3_HWeight_3.hdf5', '/space/ge36xeh/TenpyV2a/Test_Rsymb_3_HWeight_4.hdf5', default_rng)
# test_suN_symmetry(2,'/space/ge36xeh/TenpyV2a/Test_N_2_HWeight_20.hdf5', '/space/ge36xeh/TenpyV2a/Test_Fsymb_2_HWeight_6.hdf5', '/space/ge36xeh/TenpyV2a/Test_Rsymb_2_HWeight_6.hdf5', default_rng)
