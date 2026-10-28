# Copyright (C) TeNPy Developers, Apache license
import numpy as np
import pytest
from numpy import testing as npt

from lasyten import SymmetryError, get_backend, symmetries, tensors
from lasyten.block_backends import NumpyBlockBackend
from lasyten.symmetries import (
    U1,
    FermionParity,
    NoSymmetry,
    SectorArray,
    as_sector_array,
    assert_sectors_equal,
    spaces,
    trees,
)
from lasyten.tensors import Mask, SymmetricTensor
from lasyten.testing import random_ElementarySpace, random_LegPipe
from lasyten.tools import is_permutation, make_grid

fermion_parity = FermionParity().as_Symmetry()
no_symmetry = NoSymmetry().as_Symmetry()
u1_symmetry = U1().as_Symmetry()

# TODO test all cases of Space.as_ElementarySpace


def test_ElementarySpace(any_symmetry, make_any_sectors, np_random):
    sectors = as_sector_array(make_any_sectors(10))
    sectors = sectors[sectors.lexsort_indices()]
    dual_sectors = any_symmetry.dual_sectors(sectors)
    dual_sectors_sort = dual_sectors.lexsort_indices()
    mults = np_random.integers(1, 10, size=len(sectors))

    s1 = spaces.ElementarySpace(symmetry=any_symmetry, defining_sectors=sectors, multiplicities=mults)
    s2 = spaces.ElementarySpace.from_trivial_sector(dim=8)

    print('checking ElementarySpace.sector_decomposition')
    assert_sectors_equal(s2.sector_decomposition, SectorArray.from_sector(no_symmetry.trivial_sector))

    print('checking str and repr')
    _ = str(s1)
    _ = str(s2)
    _ = repr(s1)
    _ = repr(s2)

    print('checking duality and equality')
    assert_spaces_equal(s1, s1)
    s1_dual = s1.dual
    assert s1 != s1_dual
    assert s1 != s2
    wrong_mults = mults.copy()
    if len(mults) > 2:
        wrong_mults[-2] += 1
    else:
        wrong_mults[0] += 1
    assert s1 != spaces.ElementarySpace(symmetry=any_symmetry, defining_sectors=sectors, multiplicities=wrong_mults)
    assert_sectors_equal(s1_dual.defining_sectors, s1.defining_sectors)
    assert_sectors_equal(s1_dual.sector_decomposition, dual_sectors)
    npt.assert_array_equal(s1_dual.multiplicities, s1.multiplicities)
    assert s1_dual.symmetry == s1.symmetry
    assert s1_dual.is_dual is True

    print('checking is_trivial')
    assert not s1.is_trivial
    assert not s2.is_trivial
    assert spaces.ElementarySpace.from_trivial_sector(dim=1).is_trivial
    assert spaces.ElementarySpace(
        symmetry=any_symmetry, defining_sectors=SectorArray.from_sector(any_symmetry.trivial_sector)
    ).is_trivial

    print('checking is_subspace_of')
    same_sectors_less_mults = spaces.ElementarySpace(
        symmetry=any_symmetry, defining_sectors=sectors, multiplicities=[max(1, m - 1) for m in mults]
    )
    same_sectors_different_mults = spaces.ElementarySpace(
        symmetry=any_symmetry,
        defining_sectors=sectors,
        multiplicities=[max(1, m + (+1 if i % 2 == 0 else -1)) for i, m in enumerate(mults)],
    )  # but at least one mult is larger than for s1
    if len(sectors) > 2:
        which1 = [0, -1]
        which2 = [1, -2]
    else:
        # if there are only two sectors, we cant have different sets of sectors,
        # both of which have multiple entries
        which1 = [0]
        which2 = [-1]
    fewer_sectors1 = spaces.ElementarySpace(
        symmetry=any_symmetry, defining_sectors=[sectors[i] for i in which1], multiplicities=[mults[i] for i in which1]
    )
    fewer_sectors2 = spaces.ElementarySpace(
        symmetry=any_symmetry, defining_sectors=[sectors[i] for i in which2], multiplicities=[mults[i] for i in which2]
    )
    assert s1.is_subspace_of(s1)
    expect_dual_is_subspace = np.array_equal(s1.sector_decomposition.to_numpy(), dual_sectors.to_numpy())
    assert s1_dual.is_subspace_of(s1) == expect_dual_is_subspace

    assert same_sectors_less_mults.is_subspace_of(s1)
    assert not s1.is_subspace_of(same_sectors_less_mults)
    assert not same_sectors_different_mults.is_subspace_of(s1)
    assert len(sectors) == 1 or not s1.is_subspace_of(same_sectors_different_mults)
    assert fewer_sectors1.is_subspace_of(s1)
    if len(sectors) == 1:
        # if there is only one sector, the "fewer_sectors*" spaces dont actually have fewer sectors
        # and are both equal to s1
        assert s1.is_subspace_of(fewer_sectors1)
        assert fewer_sectors1.is_subspace_of(fewer_sectors2)
        assert fewer_sectors2.is_subspace_of(fewer_sectors1)
    else:
        assert not s1.is_subspace_of(fewer_sectors1)
        assert not fewer_sectors1.is_subspace_of(fewer_sectors2)
        assert not fewer_sectors2.is_subspace_of(fewer_sectors1)

    print('check idx_to_sector and parse_idx')
    if any_symmetry.can_be_dropped:
        idx = 0  # will step this up during the loop
        for n_sector, sector in enumerate(s1.sector_decomposition):
            d = any_symmetry.sector_dim(sector)
            for m in range(s1.multiplicities[n_sector]):
                for mu in range(d):
                    sector_idx, mult_idx = s1.parse_index(idx)
                    assert sector_idx == n_sector
                    assert mult_idx == m * d + mu
                    assert_sectors_equal(s1.idx_to_sector(idx), sector)
                    idx += 1

    print('check sector lookup')
    for expect in [2, 3, 4]:
        expect = expect % s1.num_sectors
        assert s1.sector_decomposition_where(s1.sector_decomposition[expect]) == expect
        assert s1.sector_decomposition_where(s1.sector_decomposition[expect]) == expect
        assert s1.sector_multiplicity(s1.sector_decomposition[expect]) == s1.multiplicities[expect]
    assert s1.total_multiplicity == sum(s1.multiplicities)

    print('check from_basis')
    if any_symmetry.can_be_dropped:
        if any_symmetry.has_factor(symmetries._SU2):
            with pytest.raises(ValueError, match='Sectors must appear in whole multiplets'):
                bad_sectors = np.array([0, 1, 1, 1, 2, 2, 2])[:, None]
                # have three basis vectors for 2-dimensional spin-1/2
                _ = spaces.ElementarySpace.from_basis(symmetry=any_symmetry, sectors_of_basis=bad_sectors)

            # spins 0, 1/2 and 1, each two times
            #                         0  1  2  3  4  5  6  7  8  9  10 11
            sectors_of_basis = np.array([0, 2, 2, 1, 2, 1, 2, 2, 0, 2, 1, 1])[:, None]
            expect_basis_perm = np.array([0, 8, 3, 5, 10, 11, 1, 2, 4, 6, 7, 9])
            expect_sectors = np.array([0, 1, 2])[:, None]
            expect_mults = np.array([2, 2, 2])
        else:
            if any_symmetry.num_sectors == 1:
                which_sectors = np.array([0] * 9)
                expect_basis_perm = np.arange(9)
                expect_sectors = sectors[:1]
            elif any_symmetry.num_sectors == 2:
                #                         0  1  2  3  4  5  6  7  8
                which_sectors = np.array([1, 0, 0, 1, 1, 0, 1, 1, 1])
                expect_basis_perm = np.array([1, 2, 5, 0, 3, 4, 6, 7, 8])
                expect_sectors = sectors[:2]
            else:
                assert len(np.unique(sectors[:3], axis=0)) == 3
                #                         0  1  2  3  4  5  6  7  8  9
                which_sectors = np.array([2, 0, 1, 2, 2, 2, 0, 0, 1, 2])
                expect_basis_perm = np.array([1, 6, 7, 2, 8, 0, 3, 4, 5, 9])
                expect_sectors = sectors[:3]
            expect_mults = np.sum(which_sectors[:, None] == np.arange(len(expect_sectors))[None, :], axis=0)
            sectors_of_basis = sectors[which_sectors]
        space = spaces.ElementarySpace.from_basis(symmetry=any_symmetry, sectors_of_basis=sectors_of_basis)
        assert_sectors_equal(space.sector_decomposition, expect_sectors)
        npt.assert_array_equal(space.multiplicities, expect_mults)
        npt.assert_array_equal(space.basis_perm, expect_basis_perm)
        # also check sectors_of_basis property
        assert_sectors_equal(space.sectors_of_basis, sectors_of_basis)


def test_ElementarySpace_from_defining_sectors(any_symmetry, make_any_sectors, np_random):
    # TODO also test from_sector_decomposition
    sectors = np.concatenate([make_any_sectors(5) for _ in range(5)])
    multiplicities = np_random.integers(1, 5, size=len(sectors))
    if any_symmetry.can_be_dropped:
        dim = np.sum(multiplicities * any_symmetry.batch_sector_dim(sectors))
        basis_perm = np_random.permutation(dim)
    else:
        basis_perm = None
    #
    # call from_defining_sectors
    res = spaces.ElementarySpace.from_defining_sectors(
        symmetry=any_symmetry, defining_sectors=sectors, multiplicities=multiplicities, basis_perm=basis_perm
    )
    res.test_sanity()
    #
    # check sectors and multiplicities
    sectors_np = as_sector_array(sectors).to_numpy()
    expect_sectors = np.unique(sectors_np, axis=0)
    expect_sectors = expect_sectors[as_sector_array(expect_sectors).lexsort_indices()]
    mult_contributions = np.where(
        np.all(sectors_np[None, :, :] == expect_sectors[:, None, :], axis=2), multiplicities[None, :], 0
    )
    expect_mults = np.sum(mult_contributions, axis=1)
    assert_sectors_equal(res.sector_decomposition, expect_sectors)
    npt.assert_array_equal(res.multiplicities, expect_mults)
    #
    # check basis perm
    if any_symmetry.can_be_dropped:
        expect_internal_basis = []
        for s, m in zip(sectors_np, multiplicities):
            expect_internal_basis.extend([s] * m * any_symmetry.sector_dim(s))
        expect_internal_basis = np.array(expect_internal_basis)
        expect_public_basis = np.zeros_like(expect_internal_basis)
        expect_public_basis[basis_perm] = expect_internal_basis
        #
        internal_basis = []
        for s, m in zip(res.sector_decomposition, res.multiplicities):
            internal_basis.extend([s] * m * any_symmetry.sector_dim(s))
        internal_basis = np.array(internal_basis)
        public_basis = np.zeros_like(internal_basis)
        public_basis[res.basis_perm] = internal_basis
        #
        npt.assert_array_equal(public_basis, expect_public_basis)


def test_take_slice(make_any_space, any_symmetry, np_random):
    if not any_symmetry.can_be_dropped:
        space: spaces.ElementarySpace = make_any_space()
        with pytest.raises(SymmetryError, match='take_slice is meaningless for .*.'):
            _ = space.take_slice([True])
        return

    if any_symmetry.has_factor(symmetries._SU2):
        sectors = np.array([0, 1, 2, 4])[:, None]
        mults = np.array([3, 1, 2, 2])
        basis_perm = np.array([19, 20, 17, 2, 9, 16, 8, 3, 0, 4, 11, 13, 5, 15, 12, 14, 10, 7, 1, 18, 6])
        space = spaces.ElementarySpace(
            any_symmetry, defining_sectors=sectors, multiplicities=mults, basis_perm=basis_perm
        )

        # build an allowed and an illegal mask in the internal basis order
        keep_states = []
        illegal_keep_states = []
        for sect, mult in zip(sectors, mults):
            dim = (sect + 1).item()
            for keep in np_random.choice([True, False], size=mult):
                keep_states.extend([keep] * dim)
                print(f'{keep=} {keep_states=}')
                illegal_keep_states.extend([keep] * (dim // 2) + [not keep] * (dim - dim // 2))
        mask = np.array(keep_states)[space.inverse_basis_perm]
        illegal_mask = np.array(illegal_keep_states)[space.inverse_basis_perm]

        with pytest.raises(ValueError, match='Multiplets need to be kept or discarded as a whole.'):
            _ = space.take_slice(illegal_mask)
    else:
        assert any_symmetry.is_abelian, 'Need to design test differently for non-abelian symm'
        space: spaces.ElementarySpace = make_any_space()
        mask = np_random.choice([True, False], size=space.dim)

    small = space.take_slice(mask)
    npt.assert_array_equal(small.sectors_of_basis, space.sectors_of_basis[mask])
    #
    internal_mask = mask[space.basis_perm]
    x = np.arange(space.dim)
    npt.assert_array_equal(x[mask][small.basis_perm], x[space.basis_perm][internal_mask])


def check_basis_perm(perm, inv_perm=None):
    """Check that `perm` is a valid permutation, and optionally that `inv_perm` is its inverse."""
    perm = np.asarray(perm)
    assert perm.ndim == 1
    assert is_permutation(perm)

    if inv_perm is not None:
        check_basis_perm(perm=inv_perm, inv_perm=None)
        npt.assert_array_equal(perm[inv_perm], np.arange(len(perm)))
        npt.assert_array_equal(inv_perm[perm], np.arange(len(perm)))


def test_basis_perm_rejects_wrong_length():
    with pytest.raises(ValueError, match='basis_perm'):
        spaces.ElementarySpace.from_trivial_sector(dim=4, basis_perm=[0, 1, 2])


def test_LegPipe_basis_perm(np_random):
    block_backend = NumpyBlockBackend.from_factory('cpu')
    a = spaces.ElementarySpace.from_trivial_sector(dim=5, is_dual=True, basis_perm=np_random.permutation(5))
    b = spaces.ElementarySpace.from_trivial_sector(dim=6, is_dual=False, basis_perm=np_random.permutation(6))
    c = spaces.ElementarySpace.from_trivial_sector(dim=7, is_dual=True, basis_perm=np_random.permutation(7))
    d = spaces.ElementarySpace.from_trivial_sector(dim=8, is_dual=False, basis_perm=np_random.permutation(8))

    data_numpy = np.arange(a.dim * b.dim * c.dim * d.dim).reshape([a.dim, b.dim, c.dim, d.dim])
    data = block_backend.block_from_numpy(data_numpy)

    pipe_ab = spaces.LegPipe([a, b])
    check_basis_perm(pipe_ab.basis_perm, pipe_ab.inverse_basis_perm)
    data1 = block_backend.to_numpy(
        block_backend.apply_leg_permutations(data, [a.basis_perm, b.basis_perm, c.basis_perm, d.basis_perm])
    ).reshape((a.dim * b.dim, c.dim, d.dim))
    data2 = block_backend.to_numpy(
        block_backend.apply_leg_permutations(
            block_backend.block_from_numpy(data_numpy.reshape((a.dim * b.dim, c.dim, d.dim))),
            [pipe_ab.basis_perm, c.basis_perm, d.basis_perm],
        )
    )
    npt.assert_array_equal(data1, data2)

    pipe_abc = spaces.LegPipe([a, b, c])
    check_basis_perm(pipe_abc.basis_perm, pipe_abc.inverse_basis_perm)
    data1 = block_backend.to_numpy(
        block_backend.apply_leg_permutations(data, [a.basis_perm, b.basis_perm, c.basis_perm, d.basis_perm])
    ).reshape((a.dim * b.dim * c.dim, d.dim))
    data2 = block_backend.to_numpy(
        block_backend.apply_leg_permutations(
            block_backend.block_from_numpy(data_numpy.reshape((a.dim * b.dim * c.dim, d.dim))),
            [pipe_abc.basis_perm, d.basis_perm],
        )
    )
    npt.assert_array_equal(data1, data2)

    pipe_nested = spaces.LegPipe([pipe_ab, c])
    check_basis_perm(pipe_nested.basis_perm, pipe_nested.inverse_basis_perm)
    data1 = block_backend.to_numpy(
        block_backend.apply_leg_permutations(data, [a.basis_perm, b.basis_perm, c.basis_perm, d.basis_perm])
    ).reshape((a.dim * b.dim * c.dim, d.dim))
    data2 = block_backend.to_numpy(
        block_backend.apply_leg_permutations(
            block_backend.block_from_numpy(data_numpy.reshape((a.dim * b.dim * c.dim, d.dim))),
            [pipe_nested.basis_perm, d.basis_perm],
        )
    )
    npt.assert_array_equal(data1, data2)


@pytest.mark.parametrize('num_spaces', [3, 4, 5])
def test_TensorProduct(any_symmetry, make_any_space, make_any_sectors, num_spaces):
    domain = spaces.TensorProduct([make_any_space() for _ in range(num_spaces)], symmetry=any_symmetry)
    domain.test_sanity()

    for coupled in make_any_sectors(10):
        expect1 = sum(
            len(trees.fusion_trees(any_symmetry, uncoupled, coupled)) * np.prod(mults)
            for uncoupled, mults in domain.iter_uncoupled()
        )
        expect2 = sum(domain.forest_block_size(uncoupled, coupled) for uncoupled, _ in domain.iter_uncoupled())
        res = domain.block_size(coupled)
        assert res == expect1
        assert res == expect2


def test_TensorProduct_SU2():
    sym = symmetries._SU2().as_Symmetry()
    a = spaces.ElementarySpace(sym, [[0], [3], [2]], [2, 3, 4])
    b = spaces.ElementarySpace(sym, [[1], [4]], [5, 6])
    c = spaces.ElementarySpace(sym, [[0], [3], [1]], [3, 1, 2])

    ab = spaces.TensorProduct([a, b])
    # a     b           fusion
    #                   0 1/2   1 3/2   2 5/2   3 7/2
    # 0     1/2            10
    # 0     2                          12
    # 3/2   1/2                15      15
    # 3/2   2              18      18      18      18
    # 1     1/2            20      20
    # 1     2                  24      24      24
    npt.assert_array_equal(ab.sector_decomposition, np.array([1, 2, 3, 4, 5, 6, 7])[:, None])
    npt.assert_array_equal(ab.multiplicities, np.array([48, 39, 38, 51, 18, 24, 18]))

    bc = spaces.TensorProduct([b, c])
    # c     b      mult     fusion
    #                       0 1/2   1 3/2   2 5/2   3 7/2
    # 0     1/2    3*5         15
    # 0     2      3*6                     18
    # 3/2   1/2    1*5              5       5
    # 3/2   2      1*6          6       6       6       6
    # 1/2   1/2    2*5     10      10
    # 1/2   2      2*6                 12      12
    npt.assert_array_equal(bc.sector_decomposition, np.array([0, 1, 2, 3, 4, 5, 7])[:, None])
    npt.assert_array_equal(bc.multiplicities, np.array([10, 21, 15, 18, 23, 18, 6]))

    abc = spaces.TensorProduct([a, b, c])
    # ab    c   mult    fusion
    #                   0   1/2   1   3/2   2   5/2   3   7/2   4   9/2   5
    # 1/2   0   48*3        144
    # 1/2 3/2   48*1             48        48
    # 1/2 1/2   48*2   96        96
    # 1     0   39*3            117
    # 1   3/2   39*1         39        39        39
    # 1   1/2   39*2         78        78
    # 3/2   0   38*3                  114
    # 3/2 3/2   38*1   38        38        38        38
    # 3/2 1/2   38*2             76        76
    # 2     0   51*3                      153
    # 2   3/2   51*1         51        51        51        51
    # 2   1/2   51*2                  102       102
    # 5/2   0   18*3                             54
    # 5/2 3/2   18*1             18        18        18        18
    # 5/2 1/2   18*2                       36        36
    # 3     0   24*3                                 72
    # 3   3/2   24*1                   24        24        24       24
    # 3   1/2   24*2                             48        48
    # 7/2   0   18*3                                       54
    # 7/2 3/2   18*1                       18        18        18        18
    # 7/2 1/2   18*2                                 36        36
    expect_mults = [
        96 + 38,
        144 + 39 + 78 + 51,
        48 + 96 + 117 + 38 + 76 + 18,
        39 + 78 + 114 + 51 + 102 + 24,
        48 + 38 + 76 + 153 + 18 + 36 + 18,
        39 + 51 + 102 + 54 + 24 + 48,
        38 + 18 + 36 + 72 + 18 + 36,
        51 + 24 + 48 + 54,
        18 + 18 + 36,
        24,
        18,
    ]
    npt.assert_array_equal(abc.sector_decomposition, np.arange(11)[:, None])
    npt.assert_array_equal(abc.multiplicities, np.array(expect_mults))


@pytest.mark.parametrize('combine_cstyle', [True, False])
@pytest.mark.parametrize('pipe_dual', [False, True])
def test_AbelianLegPipe(abelian_group_symmetry, combine_cstyle, pipe_dual, np_random):
    def iter_combinations(s1, s2):
        # iterate combinations in either C-style or F-style
        if combine_cstyle:
            for a in s1:
                for b in s2:
                    yield a, b
        else:
            for b in s2:
                for a in s1:
                    yield a, b

    leg_1: spaces.ElementarySpace = random_ElementarySpace(symmetry=abelian_group_symmetry, np_random=np_random)
    leg_2: spaces.ElementarySpace = random_ElementarySpace(symmetry=abelian_group_symmetry, np_random=np_random)
    leg_1.basis_perm = np_random.permutation(leg_1.dim)
    leg_2.basis_perm = np_random.permutation(leg_2.dim)

    pipe = spaces.AbelianLegPipe([leg_1, leg_2], is_dual=pipe_dual, combine_cstyle=combine_cstyle)
    pipe.test_sanity()

    # Setup
    # =======================================
    # for each basis element, store its sector and an int as a unique identifier
    public_basis_1 = [(s, i) for i, s in enumerate(leg_1.sectors_of_basis)]
    public_basis_2 = [(s, i) for i, s in enumerate(leg_2.sectors_of_basis)]
    internal_basis_1 = [public_basis_1[n] for n in leg_1.basis_perm]
    internal_basis_2 = [public_basis_2[n] for n in leg_2.basis_perm]

    # make sure we built them correctly
    start = 0
    for sector, mult in zip(leg_1.sector_decomposition, leg_1.multiplicities):
        for b in internal_basis_1[start : start + mult]:
            assert_sectors_equal(b[0], sector)
        start = start + mult
    assert start == leg_1.dim

    # Misc properties of the pipe
    # =======================================
    assert pipe.is_isomorphic_to(spaces.TensorProduct([leg_1, leg_2]))

    # check fusion_outcomes_sort
    # =======================================
    fusion_outcomes = [
        abelian_group_symmetry.fusion_outcomes(s_1, s_2)[0]
        for s_1, s_2 in iter_combinations(leg_1.sector_decomposition, leg_2.sector_decomposition)
    ]
    fusion_outcomes_sorted, expect = _sort_sectors(fusion_outcomes, abelian_group_symmetry, by_duals=pipe.is_dual)
    npt.assert_array_equal(pipe.fusion_outcomes_sort, expect)

    # check block_ind_map_slices
    # =======================================
    for n, sector in enumerate(pipe.sector_decomposition):
        start = pipe.block_ind_map_slices[n]
        stop = pipe.block_ind_map_slices[n + 1]
        assert all(row == sector for row in fusion_outcomes_sorted[start:stop])

    # check block_ind_map
    # =======================================
    pass  # completely checked by pipe.test_sanity()

    # check _get_fusion_outcomes_perm()
    # =======================================
    internal_fusion_outcomes = [
        (abelian_group_symmetry.fusion_outcomes(b_1[0], b_2[0])[0], b_1[1], b_2[1])
        for b_1, b_2 in iter_combinations(internal_basis_1, internal_basis_2)
    ]
    _, fusion_outcomes_perm = _sort_sectors(
        [b[0] for b in internal_fusion_outcomes], abelian_group_symmetry, by_duals=pipe.is_dual
    )

    npt.assert_array_equal(pipe._get_fusion_outcomes_perm(pipe.multiplicities), fusion_outcomes_perm)

    # check basis_perm
    # =======================================
    assert pipe.basis_perm.shape == (pipe.dim,)
    npt.assert_array_equal(np.sort(pipe.basis_perm), np.arange(pipe.dim))
    public_basis_pipe = [
        (abelian_group_symmetry.fusion_outcomes(b_1[0], b_2[0])[0], b_1[1], b_2[1])
        for b_1, b_2 in iter_combinations(public_basis_1, public_basis_2)
    ]
    internal_basis_pipe = [internal_fusion_outcomes[n] for n in fusion_outcomes_perm]

    # want to do ``expect_perm = [public_basis_pipe.index(i) for i in internal_basis_pipe]``
    # but need to deal with array comparison
    expect_perm = []
    for i in internal_basis_pipe:
        for j, p in enumerate(public_basis_pipe):
            if i[0] == p[0] and i[1:] == p[1:]:
                expect_perm.append(j)
                break
        else:  # else == "no break occurred"
            raise RuntimeError  # should not happen

    npt.assert_array_equal(pipe.basis_perm, np.array(expect_perm))


@pytest.mark.parametrize('is_dual', [True, False])
def test_direct_sum(is_dual, make_any_space, max_mult=5, max_sectors=5):
    a = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=is_dual)
    b = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=is_dual)
    c = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=is_dual)
    single = spaces.ElementarySpace.direct_sum(a)
    assert isinstance(single, spaces.DirectSumSpace)
    assert single.spaces == [a]
    d = spaces.ElementarySpace.direct_sum(a, b, c)
    d.test_sanity()
    assert isinstance(d, spaces.DirectSumSpace)
    assert d.is_direct_sum_space
    assert d.spaces == [a, b, c]
    assert d.is_dual == is_dual
    if a.symmetry.can_be_dropped:
        expect = np.concatenate([leg.sectors_of_basis for leg in [a, b, c]], axis=0)
        npt.assert_array_equal(d.sectors_of_basis, expect)
    sector2mult = {}
    for leg in [a, b, c]:
        for s, m in zip(leg.sector_decomposition, leg.multiplicities):
            key = tuple(s)
            sector2mult[key] = sector2mult.get(key, 0) + m
    sectors = np.array(list(sector2mult.keys()))
    mults = np.array(list(sector2mult.values()))
    sort = as_sector_array(sectors).lexsort_indices()
    sectors = sectors[sort]
    mults = mults[sort]
    if is_dual:
        expected_order = d.sector_decomposition.lexsort_indices()
    else:
        expected_order = slice(None, None, None)
    assert_sectors_equal(d.sector_decomposition[expected_order], sectors)
    npt.assert_array_equal(d.multiplicities[expected_order], mults)
    # Structural equality: not equal to the collapsed plain ElementarySpace
    plain = d.as_plain_ElementarySpace()
    assert isinstance(plain, spaces.ElementarySpace)
    assert not isinstance(plain, spaces.DirectSumSpace)
    assert plain != d
    assert plain.num_sectors == d.num_sectors
    npt.assert_array_equal(plain.multiplicities, d.multiplicities)


@pytest.mark.parametrize('is_dual', [True, False])
def test_DirectSumSpace(is_dual, make_any_space, max_mult=3, max_sectors=3):
    a = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=is_dual)
    b = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=is_dual)
    c = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=is_dual)
    d = spaces.DirectSumSpace([a, b, c], is_dual=is_dual)
    d.test_sanity()

    # nesting flattens
    nested = spaces.DirectSumSpace([a, spaces.DirectSumSpace([b, c], is_dual=is_dual)], is_dual=is_dual)
    assert nested.spaces == [a, b, c]
    assert nested == d

    # dual / with_opposite_duality preserve structure
    dual = d.dual
    assert isinstance(dual, spaces.DirectSumSpace)
    assert dual.is_dual == (not is_dual)
    assert len(dual.spaces) == 3
    dual.test_sanity()
    assert dual.dual == d
    # must agree with the dual of the equivalent plain ElementarySpace
    assert dual.as_plain_ElementarySpace() == d.as_plain_ElementarySpace().dual
    opp = d.with_opposite_duality()
    assert isinstance(opp, spaces.DirectSumSpace)
    assert opp.is_dual == (not is_dual)
    opp.test_sanity()
    assert opp.with_opposite_duality() == d

    # take_slice collapses
    if d.symmetry.can_be_dropped and d.dim > 0:
        import warnings

        mask = np.ones(int(d.dim), dtype=bool)
        mask[-1] = False if d.dim > 1 else True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            sliced = d.take_slice(mask)
            assert any('direct-sum structure' in str(x.message) for x in w)
        assert isinstance(sliced, spaces.ElementarySpace)
        assert not isinstance(sliced, spaces.DirectSumSpace)


def test_DirectSumSpace_summand_labels(make_any_space, max_mult=3, max_sectors=3):
    a = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=False)
    b = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=False)
    c = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=False)

    # unlabeled by default
    d = spaces.DirectSumSpace([a, b, c])
    d.test_sanity()
    assert d.summand_labels == ['', '', '']

    d = spaces.DirectSumSpace([a, b, c], summand_labels=['IdL', '', 'IdR'])
    d.test_sanity()
    assert d.summand_labels == ['IdL', '', 'IdR']
    assert d.has_summand_label('IdL')
    assert not d.has_summand_label('other')
    assert d.get_summand_idx('IdL') == 0
    assert d.get_summand_idx('IdR') == 2
    assert d.get_summand_idx(1) == 1
    assert d.get_summand_idx(-1) == 2

    with pytest.raises(ValueError):
        d.get_summand_idx('missing')

    # duplicate labels are rejected
    with pytest.raises(ValueError):
        spaces.DirectSumSpace([a, b, c], summand_labels=['x', 'x', ''])

    # length mismatch is rejected
    with pytest.raises(ValueError):
        spaces.DirectSumSpace([a, b, c], summand_labels=['x', ''])

    # structural equality ignores labels
    unlabeled = spaces.DirectSumSpace([a, b, c])
    assert d == unlabeled

    # labels are preserved through structure-preserving ops
    dual = d.dual
    assert dual.summand_labels == ['IdL', '', 'IdR']
    opp = d.with_opposite_duality()
    assert opp.summand_labels == ['IdL', '', 'IdR']

    # nested DirectSumSpace summands propagate their own labels
    nested_inner = spaces.DirectSumSpace([b, c], summand_labels=['x', 'y'])
    nested = spaces.DirectSumSpace([a, nested_inner], summand_labels=['IdL', ''])
    assert nested.spaces == [a, b, c]
    assert nested.summand_labels == ['IdL', 'x', 'y']

    # an explicit label for a nested-DirectSumSpace slot is ambiguous
    with pytest.raises(ValueError):
        spaces.DirectSumSpace([a, nested_inner], summand_labels=['IdL', 'not-allowed'])


def test_DirectSumSpace_inclusion_unit_vector(compatible_symmetry, compatible_backend):
    """Inclusions / unit vectors without requiring a dense basis."""
    sym = compatible_symmetry
    backend = compatible_backend
    I = spaces.ElementarySpace.from_trivial_sector(1, symmetry=sym)
    d = spaces.DirectSumSpace([I, I, I], summand_labels=['left', 'middle', 'right'])
    d.test_sanity()

    for i, label in ((0, 'left'), (1, 'middle'), (-1, 'right')):
        proj = d.projection_onto_summand(i, backend=backend)
        incl = d.inclusion_of_summand(i, backend=backend)
        assert (d.projection_onto_summand(label, backend=backend) == proj).all()
        assert (d.inclusion_of_summand(label, backend=backend) == incl).all()
        proj.test_sanity()
        incl.test_sanity()
        assert proj.is_projection
        assert not incl.is_projection
        assert (incl.dagger == proj).all()

    mps = spaces.ElementarySpace.from_trivial_sector(2, symmetry=sym)
    eye = SymmetricTensor.from_eye([mps], backend=backend, labels=['vL', 'vL*'])
    unit = d.unit_vector_of_summand(0, backend=backend, labels=['wL'], dtype=eye.dtype)
    assert unit.num_legs == 1
    lp = tensors.outer(eye, tensors.dagger(unit))
    lp.test_sanity()
    assert lp.labels == ['vL', 'wL*', 'vL*']
    rp_unit = d.unit_vector_of_summand(-1, backend=backend, labels=['wR'], dtype=eye.dtype)
    assert rp_unit.num_legs == 1
    rp = tensors.outer(tensors.dagger(rp_unit), eye)
    rp.test_sanity()
    assert 'vL' in rp.labels and 'wR*' in rp.labels


def test_DirectSumSpace_projection_onto_summands(compatible_symmetry, compatible_backend):
    """Multi-summand projection keeps ascending union of multiplicity slices."""
    sym = compatible_symmetry
    backend = compatible_backend
    I = spaces.ElementarySpace.from_trivial_sector(1, symmetry=sym)
    other = spaces.ElementarySpace.from_trivial_sector(2, symmetry=sym)
    d = spaces.DirectSumSpace([I, other, I], summand_labels=['left', 'middle', 'right'])  # IdL ⊕ other ⊕ IdR
    d.test_sanity()

    # complement of IdL/IdR = middle summand only
    proj_other = d.projection_onto_summands(keep_indices=[1], backend=backend)
    proj_ends = d.projection_onto_summands(keep_indices=[0, -1], backend=backend)
    proj_all = d.projection_onto_summands(keep_indices=[0, 1, 2], backend=backend)
    proj_other_by_discard = d.projection_onto_summands(discard_indices=['left', 'right'], backend=backend)
    proj_all_by_empty_discard = d.projection_onto_summands(discard_indices=[], backend=backend)
    proj_right_by_mixed_refs = d.projection_onto_summands(discard_indices=['left', 1], backend=backend)
    assert proj_other.is_projection
    assert proj_ends.is_projection
    assert (proj_ends.dagger.dagger == proj_ends).all()
    assert (proj_other_by_discard == proj_other).all()
    assert (proj_all_by_empty_discard == proj_all).all()
    assert (proj_right_by_mixed_refs == d.projection_onto_summand('right', backend=backend)).all()

    # single-summand wrappers match multi-summand
    assert (d.projection_onto_summand(1, backend=backend) == proj_other).all()
    assert (d.inclusion_of_summands(keep_indices=[1], backend=backend).dagger == proj_other).all()
    assert (d.inclusion_of_summands(discard_indices=['left', 'right'], backend=backend).dagger == proj_other).all()

    # projecting onto all summands is the identity Mask on d
    eye = Mask.from_eye(d, is_projection=True, backend=backend)
    assert (proj_all == eye).all()

    with pytest.raises(ValueError, match='exactly one'):
        d.projection_onto_summands(backend=backend)
    with pytest.raises(ValueError, match='exactly one'):
        d.inclusion_of_summands(keep_indices=[0], discard_indices=[1], backend=backend)
    with pytest.raises(ValueError, match='at least one'):
        d.projection_onto_summands(keep_indices=[], backend=backend)

    with pytest.raises(TypeError, match='basis_perm'):
        d.set_basis_perm([0, 1, 2, 3])


def test_str_repr(make_any_space, any_symmetry, str_max_lines=20, repr_max_lines=20):
    """Check if str and repr work. Automatically, we can only check if they run at all.
    To check if the output is sensible and useful, a human should look at it.
    Run ``pytest -rP -k test_str_repr`` to see the output.
    """
    terminal_width = 80
    str_max_len = terminal_width * str_max_lines
    repr_max_len = terminal_width * str_max_lines
    # TODO output is a bit long, should we force shorter?

    instances = {
        'ElementarySpace (short)': make_any_space(max_sectors=3, is_dual=True),
        'ElementarySpace (med)': make_any_space(max_sectors=10, is_dual=False),
        'ElementarySpace (long)': make_any_space(max_sectors=100, is_dual=True),
        'LegPipe (1)': spaces.LegPipe([make_any_space(max_sectors=5)]),
        'LegPipe (3)': spaces.LegPipe([make_any_space(max_sectors=5) for _ in range(3)], is_dual=True),
        'LegPipe (5)': spaces.LegPipe([make_any_space(max_sectors=3) for _ in range(5)]),
        'TensorProduct (1)': spaces.TensorProduct(
            [make_any_space(max_sectors=5)],
        ),
        'TensorProduct (3)': spaces.TensorProduct([make_any_space(max_sectors=5) for _ in range(3)]),
        'TensorProduct (5)': spaces.TensorProduct([make_any_space(max_sectors=3) for _ in range(5)]),
    }
    if any_symmetry.is_abelian and any_symmetry.can_be_dropped:
        more = {
            'AbelianLegPipe (1)': spaces.AbelianLegPipe([make_any_space(max_sectors=5)]),
            'AbelianLegPipe (1F)': spaces.AbelianLegPipe([make_any_space(max_sectors=5)], combine_cstyle=False),
            'AbelianLegPipe (3)': spaces.AbelianLegPipe(
                [make_any_space(max_sectors=5) for _ in range(3)], is_dual=True
            ),
            'AbelianLegPipe (5)': spaces.AbelianLegPipe(
                [make_any_space(max_sectors=3) for _ in range(5)],
            ),
        }
        instances.update(more)

    for name, space in instances.items():
        space.test_sanity()
        print()
        print()
        print('-' * 40)
        print(name)
        print('-' * 40)
        res = repr(space)
        assert len(res) <= repr_max_len
        assert res.count('\n') <= repr_max_lines
        print(res)

        print()
        res = str(space)
        assert len(res) <= str_max_len
        assert res.count('\n') <= str_max_lines
        print(res)


@pytest.mark.parametrize('symm', [no_symmetry, u1_symmetry, fermion_parity], ids=['no_symm', 'U1', 'fermion'])
@pytest.mark.parametrize('V_dual', [True, False], ids=['Vbra', 'Vket'])
@pytest.mark.parametrize('W_dual', [True, False], ids=['Wbra', 'Wket'])
@pytest.mark.parametrize('basis_perm', [True, False], ids=['perm', 'noperm'])
@pytest.mark.parametrize('V_pipe', [3, 2, 1, False], ids=['Vpipe3', 'Vpipe2', 'Vpipe1', 'Vnopipe'])
@pytest.mark.parametrize('W_pipe', [3, 2, 1, False], ids=['Wpipe3', 'Wpipe2', 'Wpipe1', 'Wnopipe'])
@pytest.mark.parametrize('abelian_pipes', [True, False], ids=['abelianPipes', 'plainPipes'])
def test_swap_gate(symm, basis_perm, abelian_pipes, np_random, V_pipe, W_pipe, V_dual, W_dual):
    pipe_backend = get_backend('abelian' if abelian_pipes else 'fusion_tree')
    if not pipe_backend.supports_symmetry(symm):
        return

    if V_pipe is False:
        V = random_ElementarySpace(symm, allow_basis_perm=basis_perm, np_random=np_random, is_dual=V_dual)
    else:
        V = random_LegPipe(
            symmetry=symm,
            backend=pipe_backend,
            allow_basis_perm=basis_perm,
            num_legs=V_pipe,
            np_random=np_random,
            is_dual=V_dual,
        )
    if W_pipe is False:
        W = random_ElementarySpace(symm, allow_basis_perm=basis_perm, np_random=np_random, is_dual=W_dual)
    else:
        W = random_LegPipe(
            symmetry=symm,
            backend=pipe_backend,
            allow_basis_perm=basis_perm,
            num_legs=W_pipe,
            np_random=np_random,
            is_dual=W_dual,
        )

    if not symm.can_be_dropped:
        with pytest.raises(SymmetryError, match='braid can not be written as array'):
            _ = spaces.swap_gate(V, W)
        return  # error is expected behavior

    dV = int(V.dim)
    dW = int(W.dim)

    swap = spaces.swap_gate(V, W)
    assert swap.shape == (dW, dV, dW, dV)

    # must be "diagonal", in the swapped basis swap[i, j, k, l] = phases[i, j] δ_ik δ_jl
    phases = np.diag(swap.reshape(dW * dV, dW * dV)).reshape(dW, dV)
    npt.assert_almost_equal(swap, np.diag(phases.reshape(-1)).reshape(dW, dV, dW, dV))
    # -> it is now enough to check that the phases are as expected

    # they must be complex phases
    npt.assert_almost_equal(np.abs(phases), 1)

    if symm.has_trivial_braid:
        npt.assert_almost_equal(phases, +1)

    elif symm is fermion_parity:
        if V_pipe is False:
            V_parity = np.reshape(V.sectors_of_basis % 2, -1)
        else:
            # implementation of this test assumes no nested pipes...
            V_sectors = [Vi.sectors_of_basis for Vi in V.legs]
            V_parity = np.array(
                [
                    sum(sectors[i] for sectors, i in zip(V_sectors, i_idcs)) % 2
                    for i_idcs in make_grid([int(Vi.dim) for Vi in V.legs], cstyle=V.combine_cstyle)
                ]
            ).reshape(-1)
        if W_pipe is False:
            W_parity = np.reshape(W.sectors_of_basis % 2, -1)
        else:
            W_sectors = [Wi.sectors_of_basis for Wi in W.legs]
            W_parity = np.array(
                [
                    sum(sectors[j] for sectors, j in zip(W_sectors, j_idcs)) % 2
                    for j_idcs in make_grid([int(Wi.dim) for Wi in W.legs], cstyle=W.combine_cstyle)
                ]
            ).reshape(-1)

        for j in range(dW):
            for i in range(dV):
                npt.assert_almost_equal(phases[j, i], 1 - 2 * V_parity[i] * W_parity[j])
        npt.assert_almost_equal(phases, 1 - 2 * V_parity[None, :] * W_parity[:, None])

    else:
        # would need to redesign the tests
        raise NotImplementedError


# TODO move to some testing tools module?
def assert_spaces_equal(space1: spaces.Space, space2: spaces.Space):
    # TODO review in light of new spaces classes
    if isinstance(space1, spaces.ElementarySpace):
        assert isinstance(space2, spaces.ElementarySpace), 'mismatching types'
        assert space1.is_dual == space2.is_dual, 'mismatched is_dual'
        assert space1.symmetry == space2.symmetry, 'mismatched symmetry'
        assert space1.num_sectors == space2.num_sectors, 'mismatched num_sectors'
        npt.assert_array_equal(space1.multiplicities, space2.multiplicities), 'mismatched multiplicities'
        assert_sectors_equal(space1.sector_decomposition, space2.sector_decomposition, 'mismatched sectors')
        if (space1._basis_perm is not None) or (space2._basis_perm is not None):
            # otherwise both are trivial and this match
            npt.assert_array_equal(space1.basis_perm, space2.basis_perm), 'mismatched basis_perm'
    elif isinstance(space1, spaces.TensorProduct):
        assert isinstance(space2, spaces.TensorProduct), 'mismatching types'
        assert space1.num_factors == space2.num_factors
        for n, (s1, s2) in enumerate(zip(space1.factors, space2.factors)):
            try:
                assert_spaces_equal(s1, s2)
            except AssertionError as e:
                raise AssertionError(f'Mismatched spaces[{n}]: {str(e)}') from None
    else:
        raise ValueError('Not a known Space type')

    # should have checked all conditions already, but just to be sure do this again
    assert space1 == space2


def _sort_sectors(sectors, sym: symmetries.Symmetry, by_duals: bool = False):
    sectors = as_sector_array(sectors)
    sort_by = sym.dual_sectors(sectors) if by_duals else sectors
    perm = sort_by.lexsort_indices()
    return sectors[perm], perm
