"""A collection of tests for lasyten.trees."""

# Copyright (C) TeNPy Developers, Apache license
from collections.abc import Callable

import numpy as np
import pytest
from numpy import testing as npt

from lasyten.backends import get_backend
from lasyten.block_backends import Block
from lasyten.block_backends.dtypes import Dtype
from lasyten.symmetries import (
    U1,
    ZN,
    ElementarySpace,
    Sector,
    SectorArray,
    Symmetry,
    SymmetryError,
    TensorProduct,
    as_sector_array,
    assert_sectors_equal,
    trees,
)
from lasyten.testing import random_symmetry_sectors

u1_symmetry = U1().as_Symmetry()
z3_symmetry = ZN(N=3).as_Symmetry()


def random_fusion_tree(
    symmetry: Symmetry, num_uncoupled: int, sector_rng: Callable[[], Sector], np_random: np.random.Generator
) -> trees.FusionTree:
    if num_uncoupled == 0:
        return trees.FusionTree.from_empty(symmetry=symmetry)
    if num_uncoupled == 1:
        sector = sector_rng()
        is_dual = bool(np_random.choice([True, False]))
        return trees.FusionTree.from_sector(symmetry, sector, is_dual=is_dual)
    fusion_outcomes = []
    multiplicities = []
    left = sector_rng()
    uncoupled = [left]
    for _ in range(num_uncoupled - 1):
        right = sector_rng()
        uncoupled.append(right)
        outcome = np_random.choice(symmetry.fusion_outcomes(left, right))
        fusion_outcomes.append(outcome)
        multiplicities.append(np_random.choice(symmetry.n_symbol(left, right, outcome)))
        left = outcome
    coupled = fusion_outcomes[-1]
    are_dual = np_random.choice([True, False], size=num_uncoupled)
    inner_sectors = fusion_outcomes[:-1]
    res = trees.FusionTree(
        symmetry,
        uncoupled=uncoupled,
        coupled=coupled,
        are_dual=are_dual,
        inner_sectors=inner_sectors,
        multiplicities=multiplicities,
    )
    res.test_sanity()
    return res


def random_tree_pair(
    symmetry: Symmetry,
    num_uncoupled_in: int,
    num_uncoupled_out: int,
    sector_rng: Callable[[], Sector],
    np_random: np.random.Generator,
) -> tuple[trees.FusionTree, trees.FusionTree]:
    X = random_fusion_tree(symmetry, num_uncoupled_in, sector_rng, np_random)
    root = X.coupled
    uncoupled_out_reversed = []
    multiplicities_reversed = []
    inner_sectors_reversed = []
    for _ in range(num_uncoupled_out - 1):
        right = sector_rng()
        uncoupled_out_reversed.append(right)
        outcome = np_random.choice(symmetry.fusion_outcomes(root, symmetry.dual_sector(right)))
        multiplicities_reversed.append(np_random.choice(symmetry.n_symbol(right, outcome, root)))
        inner_sectors_reversed.append(outcome)
        root = outcome
    uncoupled_out_reversed.append(inner_sectors_reversed.pop(-1))
    are_dual = np_random.choice([True, False], num_uncoupled_out)
    Y = trees.FusionTree(
        symmetry,
        uncoupled_out_reversed[::-1],
        X.coupled,
        are_dual,
        inner_sectors_reversed[::-1],
        multiplicities_reversed[::-1],
    )
    Y.test_sanity()
    return X, Y


@pytest.mark.xfail(reason='Test not implemented yet')
def test_FusionTree_class():
    # TODO test hash, eq, str, repr, copy
    raise NotImplementedError


@pytest.mark.parametrize('overbraid', [True, False])
@pytest.mark.parametrize('j', [0, 1, 2])
def test_FusionTree_braid(overbraid, j, any_symmetry, make_any_sectors, np_random):
    num_uncoupled = 5
    tree = random_fusion_tree(
        symmetry=any_symmetry,
        num_uncoupled=num_uncoupled,
        sector_rng=lambda: make_any_sectors(1)[0],
        np_random=np_random,
    )
    braided1 = list(tree.braid(j, overbraid=overbraid).items())
    for t, _ in braided1:
        assert_sectors_equal(t.uncoupled[:j], tree.uncoupled[:j])
        assert_sectors_equal(t.uncoupled[j], tree.uncoupled[j + 1])
        assert_sectors_equal(t.uncoupled[j + 1], tree.uncoupled[j])
        assert_sectors_equal(t.uncoupled[j + 2 :], tree.uncoupled[j + 2 :])
        #
        npt.assert_array_equal(t.are_dual[:j], tree.are_dual[:j])
        assert t.are_dual[j] == tree.are_dual[j + 1]
        assert t.are_dual[j + 1] == tree.are_dual[j]
        npt.assert_array_equal(t.are_dual[j + 2 :], tree.are_dual[j + 2 :])
        #
        t.test_sanity()

    # for groups: check versus explicit matrix representations
    if any_symmetry.can_be_dropped:
        tree_np = tree.to_dense_block(understood_braiding=True).to_numpy()  # [a1 ... aj aj+1 ... aJ c]
        swap = any_symmetry.swap_gate(tree.uncoupled[j], tree.uncoupled[j + 1])  # [aj+1 aj aj+1* aj*]
        expect = np.tensordot(tree_np, swap, ([j, j + 1], [1, 0]))  # [a1 ... aj-1 aj+2 ... aj c aj+1 aj]
        expect = np.transpose(
            expect, [*range(j), -2, -1, *range(j, num_uncoupled - 1)]
        )  # [a1 ... aj-1 aj+2 ... aj c aj+1 aj]
        res = sum(a * t.to_dense_block(understood_braiding=True).to_numpy() for t, a in braided1)
        npt.assert_almost_equal(res, expect)

    # check if opposite braid undoes it
    res = {}
    for t_in, a in braided1:
        for t_out, b in t_in.braid(j, overbraid=not overbraid).items():
            res[t_out] = res.get(t_out, 0) + a * b
    assert tree in res
    for t, a in res.items():
        npt.assert_almost_equal(a, 1 if t == tree else 0)

    # check yang baxter
    #   \   /   |          |   \   /
    #    \ /    |          |    \ /
    #     X     |          |     X
    #    / \   /            \   / \
    #   |   \ /              \ /   |
    #   |    X        =       X    |
    #   |   / \              / \   |
    #    \ /   \            /   \ /
    #     X     |          |     X
    #    / \    |          |    / \
    #   /   \   |          |   /   \
    #  j   j+1  j+2
    lhs1 = {t: a for t, a in braided1}
    lhs2 = {}  # apply braid (j+1, j+2)
    for t_in, a in lhs1.items():
        for t_out, b in t_in.braid(j + 1, overbraid=overbraid).items():
            lhs2[t_out] = lhs2.get(t_out, 0) + a * b
    lhs = {}  # apply braid (j, j+1)
    for t_in, a in lhs2.items():
        for t_out, b in t_in.braid(j, overbraid=overbraid).items():
            lhs[t_out] = lhs.get(t_out, 0) + a * b

    rhs1 = tree.braid(j + 1, overbraid=overbraid)
    rhs2 = {}
    for t_in, a in rhs1.items():
        for t_out, b in t_in.braid(j, overbraid=overbraid).items():
            rhs2[t_out] = rhs2.get(t_out, 0) + a * b
    rhs = {}
    for t_in, a in rhs2.items():
        for t_out, b in t_in.braid(j + 1, overbraid=overbraid).items():
            rhs[t_out] = rhs.get(t_out, 0) + a * b

    assert lhs.keys() == rhs.keys()
    for t, a_lhs in lhs.items():
        a_rhs = rhs[t]
        npt.assert_almost_equal(a_lhs, a_rhs)


def tree_pair_to_numpy(X: trees.FusionTree, Y: trees.FusionTree):
    X = X.to_dense_block(understood_braiding=True).to_numpy().conj()  # [a1 ... aJ c]
    Y = Y.to_dense_block(understood_braiding=True).to_numpy()  # [b1 ... bK c]
    Y = np.transpose(Y, list(reversed(range(Y.ndim))))  # [c bK .. b1]
    return np.tensordot(X, Y, (-1, 0))  # [a1 ... aJ bK ... b1]


@pytest.mark.parametrize('bend_down', [True, False])
def test_FusionTree_bend_leg(bend_down, any_symmetry, make_any_sectors, np_random):
    X, Y = random_tree_pair(
        symmetry=any_symmetry,
        num_uncoupled_in=4,
        num_uncoupled_out=4,
        sector_rng=lambda: make_any_sectors(1)[0],
        np_random=np_random,
    )
    X.test_sanity()
    Y.test_sanity()
    res = list(trees.FusionTree.bend_leg(X, Y, bend_down).items())

    for (X_i, Y_i), _ in res:
        X_i.test_sanity()
        Y_i.test_sanity()
        assert_sectors_equal(X_i.coupled, Y_i.coupled)
        if bend_down:
            assert_sectors_equal(X_i.uncoupled[:-1], X.uncoupled)
            assert_sectors_equal(Y_i.uncoupled, Y.uncoupled[:-1])
            assert_sectors_equal(X_i.uncoupled[-1], any_symmetry.dual_sector(Y.uncoupled[-1]))
        else:
            assert_sectors_equal(X_i.uncoupled, X.uncoupled[:-1])
            assert_sectors_equal(Y_i.uncoupled[:-1], Y.uncoupled)
            assert_sectors_equal(Y_i.uncoupled[-1], any_symmetry.dual_sector(X.uncoupled[-1]))

    # compare to matrix representation
    if any_symmetry.can_be_dropped:
        # bending leg does nothing in this case
        expect = tree_pair_to_numpy(X, Y)
        res_np = sum(a_i * tree_pair_to_numpy(X_i, Y_i) for (X_i, Y_i), a_i in res)
        npt.assert_almost_equal(res_np, expect)

    # check that bending back gives back the same tree
    res2 = {}
    for (X_i, Y_i), a_i in res:
        for (Y2_i, X2_i), b_i in trees.FusionTree.bend_leg(X_i, Y_i, not bend_down).items():
            res2[Y2_i, X2_i] = res2.get((Y2_i, X2_i), 0) + a_i * b_i
    assert (X, Y) in res2
    for pair, a in res2.items():
        npt.assert_almost_equal(a, 1 if pair == (X, Y) else 0)

    # TODO is there anything else we can check at this level...?


def test_FusionTree_manipulations(compatible_symmetry, compatible_backend, make_compatible_sectors, np_random):
    # TODO add a symmetry that detects the difference between conjugating the F symbols
    # and not conjugating them. SU(3)_3, SU(2) and SU(2)_k are not suitable for this.
    sym = compatible_symmetry
    backend = compatible_backend

    # test insert and split
    num_uncoupled = np_random.integers(4, 8)
    # generate uncoupled sectors like this to allow identical sectors in trees
    uncoupled = np.vstack([np_random.choice(make_compatible_sectors(5)) for _ in range(num_uncoupled)])
    are_dual = np_random.choice([True, False], size=num_uncoupled)
    all_trees = random_trees_from_uncoupled(sym, uncoupled, np_random)
    random_trees = np_random.choice(all_trees, size=10)
    for tree in random_trees:
        n_split = int(np_random.integers(0, num_uncoupled + 1))

        # test errors
        if n_split == num_uncoupled or n_split < 2:
            if n_split == num_uncoupled:
                msg = r'Right tree has no vertices \(n >= num_uncoupled\)'
            else:
                msg = r'Left tree has no vertices \(n < 2\)'
            with pytest.raises(ValueError, match=msg):
                _ = tree.split(n_split)
            continue

        left_tree, right_tree = tree.split(n_split)
        split_sector = tree.inner_sectors[n_split - 2]

        # test left tree
        assert_sectors_equal(left_tree.uncoupled, tree.uncoupled[:n_split])
        npt.assert_array_equal(left_tree.are_dual, tree.are_dual[:n_split])
        assert_sectors_equal(left_tree.inner_sectors, tree.inner_sectors[: n_split - 2])
        assert_sectors_equal(left_tree.coupled, split_sector)
        npt.assert_array_equal(left_tree.multiplicities, tree.multiplicities[: n_split - 1])

        # test right tree
        assert_sectors_equal(right_tree.uncoupled, as_sector_array([split_sector, *tree.uncoupled[n_split:]]))
        npt.assert_array_equal(right_tree.are_dual, np.append([False], tree.are_dual[n_split:]))
        assert_sectors_equal(right_tree.inner_sectors, tree.inner_sectors[n_split - 1 :])
        assert_sectors_equal(right_tree.coupled, tree.coupled)
        npt.assert_array_equal(right_tree.multiplicities, tree.multiplicities[n_split - 1 :])

        # test insert
        assert tree == right_tree.insert(left_tree)

    # test insert_at
    num_uncoupled = np_random.integers(3, 6, size=2)
    uncoupled1 = np.vstack([np_random.choice(make_compatible_sectors(5)) for _ in range(num_uncoupled[0])])
    # no dual sectors here to possibly enable insert_at for every uncoupled sector
    all_trees1 = random_trees_from_uncoupled(sym, uncoupled1, np_random)
    random_trees1 = np_random.choice(all_trees1, size=5)

    uncoupled2 = np.vstack([np_random.choice(make_compatible_sectors(5)) for _ in range(num_uncoupled[1])])
    are_dual = np_random.choice([True, False], size=num_uncoupled[1])
    all_trees2 = random_trees_from_uncoupled(sym, uncoupled2, np_random, are_dual=are_dual)
    coupled2 = all_trees2[0].coupled
    random_trees2 = np_random.choice(all_trees2, size=5)

    for i in range(num_uncoupled[0]):
        if coupled2 != uncoupled1[i]:
            continue  # make sure inserting is possible
        perm = list(range(i))
        perm.extend(list(range(num_uncoupled[0], sum(num_uncoupled))))
        perm.extend(list(range(i, num_uncoupled[0])))
        for tree1 in random_trees1:
            for tree2 in random_trees2:
                # do this check for all symmetries (also checks the sectors in the trees)
                check_insert_at_via_f_symbols(tree1, tree2, i)
                # check with as_block
                if sym.can_be_dropped:
                    block1 = tree1.to_dense_block(backend, understood_braiding=True)
                    block2 = tree2.to_dense_block(backend, understood_braiding=True)
                    expect = backend.block_backend.tdot(block1, block2, [i], [-1])
                    expect = backend.block_backend.permute_axes(expect, perm)
                    combined_tree = tree1.insert_at(i, tree2)
                    combined_block = tree_superposition_as_block(combined_tree, backend)
                    assert backend.block_backend.allclose(combined_block, expect, rtol=1e-8, atol=1e-5)

    # test outer
    uncoupled1 = np.vstack([np_random.choice(make_compatible_sectors(5)) for _ in range(num_uncoupled[0])])
    are_dual1 = np_random.choice([True, False], size=num_uncoupled[0])
    all_trees1 = random_trees_from_uncoupled(sym, uncoupled1, np_random, are_dual=are_dual1)
    random_trees1 = np_random.choice(all_trees1, size=5)

    uncoupled2 = np.vstack([np_random.choice(make_compatible_sectors(5)) for _ in range(num_uncoupled[1])])
    are_dual2 = np_random.choice([True, False], size=num_uncoupled[1])
    all_trees2 = random_trees_from_uncoupled(sym, uncoupled2, np_random, are_dual=are_dual2)
    random_trees2 = np_random.choice(all_trees2, size=5)

    for tree1 in random_trees1:
        for tree2 in random_trees2:
            check_outer_via_f_symbols(tree1, tree2)


def check_insert_at_via_f_symbols(tree1: trees.FusionTree, tree2: trees.FusionTree, i: int):
    """Check correct amplitudes, normalization (sum of amplitudes), uncoupled sectors,
    inner sectors, coupled sectors and multilcities.
    """
    combined_tree = tree1.insert_at(i, tree2)
    uncoupled = as_sector_array([*tree1.uncoupled[:i], *tree2.uncoupled, *tree1.uncoupled[i + 1 :]])
    are_dual = np.concatenate([tree1.are_dual[:i], tree2.are_dual, tree1.are_dual[i + 1 :]])
    coupled = tree1.coupled
    norm = 0
    for tree, amp in combined_tree.items():
        tree.test_sanity()
        assert_sectors_equal(tree.uncoupled, uncoupled)
        npt.assert_array_equal(tree.are_dual, are_dual)
        assert_sectors_equal(tree.coupled, coupled)
        npt.assert_array_equal(tree.multiplicities[i + tree2.num_vertices :], tree1.multiplicities[i:])
        if i > 0:
            assert_sectors_equal(tree.inner_sectors[i - 1 + tree2.num_vertices :], tree1.inner_sectors[i - 1 :])
            assert_sectors_equal(tree.inner_sectors[: i - 1], tree1.inner_sectors[: i - 1])
            npt.assert_array_equal(tree.multiplicities[: i - 1], tree1.multiplicities[: i - 1])

        if i == 0 or tree2.num_uncoupled == 1:
            fs = 1  # no F symbols to apply
        else:
            f_symbols = []
            a = tree.uncoupled[0] if i == 1 else tree.inner_sectors[i - 2]
            for j in range(tree2.num_uncoupled - 1):
                b = tree.uncoupled[i] if j == 0 else tree2.inner_sectors[j - 1]
                c = tree.uncoupled[i + j + 1]
                d = tree.coupled if i + j + 1 == tree.num_uncoupled - 1 else tree.inner_sectors[i + j]
                e = tree2.coupled if j == tree2.num_inner_edges else tree2.inner_sectors[j]
                f = tree.inner_sectors[i + j - 1]
                f_symbols.append(np.conj(tree1.symmetry.f_symbol(a, b, c, d, e, f)))

            # deal with multiplicities
            kap = tree.multiplicities[i - 1]
            lam = tree.multiplicities[i]
            mu = tree2.multiplicities[0]
            nu = tree1.multiplicities[i - 1]
            fs = f_symbols[0][mu, :, kap, lam]
            for j, f in enumerate(f_symbols[1:]):
                lam = tree.multiplicities[i + j + 1]
                mu = tree2.multiplicities[j + 1]
                fs = np.tensordot(fs, f[mu, :, :, lam], [0, 1])
            fs = fs[nu]
        npt.assert_almost_equal(fs, amp)
        norm += amp * np.conj(amp)
    npt.assert_almost_equal(norm, 1)


def check_outer_via_f_symbols(tree1: trees.FusionTree, tree2: trees.FusionTree):
    """Check correct amplitudes, normalization (sum of amplitudes), uncoupled sectors,
    inner sectors, coupled sectors and multilcities.
    """
    combined_tree = tree1.outer(tree2)
    uncoupled = as_sector_array([*tree1.uncoupled, *tree2.uncoupled])
    are_dual = np.concatenate([tree1.are_dual, tree2.are_dual])
    coupled = tree1.symmetry.fusion_outcomes(tree1.coupled, tree2.coupled)
    # one normalized tree for each new consistent coupled sector
    norm_expect = sum([tree1.symmetry.n_symbol(tree1.coupled, tree2.coupled, c) for c in coupled])
    norm = 0
    for tree, amp in combined_tree.items():
        tree.test_sanity()
        assert_sectors_equal(tree.uncoupled, uncoupled)
        npt.assert_array_equal(tree.are_dual, are_dual)
        assert_sectors_equal(tree.inner_sectors[: tree1.num_inner_edges], tree1.inner_sectors)
        assert_sectors_equal(tree.inner_sectors[tree1.num_inner_edges], tree1.coupled)
        npt.assert_array_equal(tree.multiplicities[: tree1.num_inner_edges], tree1.multiplicities[:-1])

        if tree1.num_uncoupled == 0 or tree2.num_uncoupled <= 1:
            fs = 1
        else:
            f_symbols = []
            a = tree1.coupled
            for j in range(tree2.num_uncoupled - 1):
                b = tree2.uncoupled[j] if j == 0 else tree2.inner_sectors[j - 1]
                c = tree2.uncoupled[j + 1]
                d = (
                    tree.coupled
                    if j + 1 == tree2.num_uncoupled - 1
                    else tree.inner_sectors[tree1.num_inner_edges + j + 2]
                )
                e = tree2.coupled if j == tree2.num_inner_edges else tree2.inner_sectors[j]
                f = tree.inner_sectors[tree1.num_inner_edges + j + 1]
                f_symbols.append(np.conj(tree1.symmetry.f_symbol(a, b, c, d, e, f)))

            # deal with multiplicities
            kap = tree.multiplicities[tree1.num_vertices]
            lam = tree.multiplicities[tree1.num_vertices + 1]
            mu = tree2.multiplicities[0]
            fs = f_symbols[0][mu, :, kap, lam]
            for j, f in enumerate(f_symbols[1:]):
                lam = tree.multiplicities[tree1.num_vertices + j + 2]
                mu = tree2.multiplicities[j + 1]
                fs = np.tensordot(fs, f[mu, :, :, lam], [0, 1])
            # the remaining axis specifies the multiplicity in the vertex where the
            # two coupled sectors fuse -> sum them as we allow all multiplicities
            fs = np.sum(fs[:])

        npt.assert_almost_equal(fs, amp)
        norm += amp * np.conj(amp)
    npt.assert_almost_equal(norm, norm_expect)


def random_trees_from_uncoupled(symmetry, uncoupled, np_random, are_dual=None) -> list[trees.FusionTree]:
    """Choose a random coupled sector consistent with the given uncoupled sectors and
    return all fusion trees with consistent inner sectors and multiplicities as list.
    """
    spaces = [ElementarySpace(symmetry, [a]) for a in uncoupled]
    domain = TensorProduct(spaces)
    coupled = np_random.choice(domain.sector_decomposition)
    return list(trees.fusion_trees(symmetry, uncoupled, coupled, are_dual=are_dual))


def tree_superposition_as_block(superposition, backend, dtype=None) -> Block:
    for i, (tree, amp) in enumerate(superposition.items()):
        if i == 0:
            res = amp * tree.to_dense_block(backend, understood_braiding=True, dtype=dtype)
        else:
            res += amp * tree.to_dense_block(backend, understood_braiding=True, dtype=dtype)
    return res


def check_fusion_trees(it: trees.fusion_trees, expect_len: int = None):
    if expect_len is None:
        expect_len = len(it)
    else:
        assert len(it) == expect_len

    # make sure they run.
    _ = str(it)
    _ = repr(it)

    num_trees = 0
    for tree in it:
        npt.assert_array_equal(tree.are_dual, it.are_dual)
        tree.test_sanity()
        assert it.index(tree) == num_trees
        num_trees += 1
    assert num_trees == expect_len


def test_fusion_trees(any_symmetry: Symmetry, make_any_sectors, np_random):
    """test the ``fusion_trees`` iterator"""
    some_sectors = make_any_sectors(20)  # generates unique sectors
    non_trivial = [sector for sector in some_sectors if sector != any_symmetry.trivial_sector]
    non_trivial_sectors = (
        as_sector_array(non_trivial) if non_trivial else SectorArray.empty(any_symmetry.sector_ind_len)
    )
    i = any_symmetry.trivial_sector

    print('consistent fusion: [] -> i')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [], i), expect_len=1)

    print('consistent fusion: i -> i')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [i], i, [False]), expect_len=1)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [i], i, [True]), expect_len=1)

    print('large consistent fusion')
    uncoupled = some_sectors[:5]
    are_dual = np_random.choice([True, False], size=len(uncoupled), replace=True)
    # find the allowed coupled sectors
    allowed = TensorProduct([ElementarySpace(any_symmetry, [a]) for a in uncoupled]).sector_decomposition
    some_allowed = np_random.choice(allowed, axis=0)
    print(f'  uncoupled={", ".join(map(str, uncoupled))}   coupled={some_allowed}')
    it = trees.fusion_trees(any_symmetry, uncoupled, some_allowed, are_dual=are_dual)
    assert len(it) > 0
    check_fusion_trees(it)

    print('large inconsistent fusion')
    # find a forbidden coupled sector
    are_allowed = np.array([any(sector == candidate for candidate in allowed) for sector in some_sectors])
    forbidden_idcs = np.where(np.logical_not(are_allowed))[0]
    if len(forbidden_idcs) > 0:
        forbidden = some_sectors[np_random.choice(forbidden_idcs)]
        it = trees.fusion_trees(any_symmetry, uncoupled, forbidden, are_dual=are_dual)
        check_fusion_trees(it, expect_len=0)

    # rest of the checks assume we have access to at least one non-trivial sector
    if len(non_trivial_sectors) == 0:
        return
    c = non_trivial_sectors[0]
    c_dual = any_symmetry.dual_sector(c)

    print(f'consistent fusion: c -> c')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c], c, [True]), expect_len=1)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c], c, [False]), expect_len=1)

    print(f'consistent fusion: [c, dual(c)] -> i')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c, c_dual], i, [False, False]), expect_len=1)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c, c_dual], i, [False, True]), expect_len=1)

    # rest of the checks assume we have access to at least two non-trivial sector
    if len(non_trivial_sectors) == 1:
        return
    d = non_trivial_sectors[1]

    print(f'inconsistent fusion: c -> d')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c], d, [True]), expect_len=0)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c], d, [False]), expect_len=0)

    print('consistent fusion: [c, d] -> ?')
    e = any_symmetry.fusion_outcomes(c, d)[0]
    N = any_symmetry.n_symbol(c, d, e)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c, d], e, [False, False]), expect_len=N)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c, d], e, [False, True]), expect_len=N)


def check_to_block(symmetry, backend, uncoupled, np_random, dtype):
    """Common implementation for test_to_block and test_to_block_no_backend"""
    all_trees = random_trees_from_uncoupled(symmetry, uncoupled, np_random)

    if not symmetry.can_be_dropped:
        with pytest.raises(SymmetryError, match='Can not convert to block for symmetry .*'):
            _ = all_trees[0].to_dense_block(backend, dtype)
        # error is expected behavior, nothing else to check
        return
    if not symmetry.has_trivial_braid:
        with pytest.raises(SymmetryError, match='do not consistently reproduce the braiding'):
            _ = all_trees[0].to_dense_block(backend, dtype, understood_braiding=False)
        # error is expected behavior, but we can continue with understood_braiding=True

    coupled_dim = symmetry.sector_dim(all_trees[0].coupled)
    uncoupled_dims = symmetry.batch_sector_dim(uncoupled)
    all_blocks = [t.to_dense_block(backend, dtype, understood_braiding=True) for t in all_trees]
    axes = list(range(len(uncoupled)))
    if symmetry.fusion_tensor_dtype.is_complex:
        expect_dtype = dtype.to_complex
    else:
        expect_dtype = dtype

    if backend is None:
        backend = get_backend()
    coupled_eye = backend.block_backend.eye_block([coupled_dim], dtype)
    coupled_zero = backend.block_backend.zeros([coupled_dim, coupled_dim], dtype)
    for i, X in enumerate(all_blocks):
        assert backend.block_backend.get_shape(X) == (*uncoupled_dims, coupled_dim)
        assert backend.block_backend.get_dtype(X) == expect_dtype
        for j, Y in enumerate(all_blocks):
            if i < j:
                continue  # redundant with (i, j) <-> (j, i)
            X_Y = backend.block_backend.tdot(backend.block_backend.conj(X), Y, axes, axes)
            expect = coupled_eye if i == j else coupled_zero
            assert backend.block_backend.allclose(X_Y, expect, rtol=1e-8, atol=1e-5)


@pytest.mark.parametrize('dtype', [Dtype.float64, Dtype.complex128])
def test_to_block(compatible_symmetry, compatible_backend, make_compatible_sectors, np_random, dtype):
    # need two test_* functions to generate the cases, implement actual test in check_to_block...
    uncoupled = make_compatible_sectors(4)
    check_to_block(compatible_symmetry, compatible_backend, uncoupled, np_random, dtype)


@pytest.mark.parametrize('dtype', [Dtype.float64, Dtype.complex128])
def test_to_block_no_backend(any_symmetry, make_any_sectors, np_random, dtype):
    # need two test_* functions to generate the cases, implement actual test in check_to_block
    coupled = make_any_sectors(4)
    check_to_block(any_symmetry, None, coupled, np_random, dtype)


@pytest.mark.parametrize('num_uncoupled', [0, 1, 2, 5])
@pytest.mark.parametrize('symmetry', [u1_symmetry, u1_symmetry * z3_symmetry])
def test_FusionTree_ascii_diagram(symmetry, num_uncoupled, np_random):
    # run e.g. ``pytest -rP -k test_FusionTree_ascii_diagram`` to see the output
    X = random_fusion_tree(
        symmetry=symmetry,
        num_uncoupled=num_uncoupled,
        sector_rng=lambda: random_symmetry_sectors(symmetry, 1, np_random=np_random)[0],
        np_random=np_random,
    )
    print('>>> X.ascii_diagram(dagger=True)')
    print(X.ascii_diagram(dagger=True))
    print()
    print('>>> X.ascii_diagram()')
    print(X.ascii_diagram())


@pytest.mark.parametrize('num_uncoupled', [0, 1, 2, 5])
@pytest.mark.parametrize('symmetry', [u1_symmetry, u1_symmetry * z3_symmetry])
def test_FusionTree_str(symmetry, num_uncoupled, np_random):
    # run e.g. ``pytest -rP -k test_FusionTree_str`` to see the output
    X = random_fusion_tree(
        symmetry=symmetry,
        num_uncoupled=num_uncoupled,
        sector_rng=lambda: random_symmetry_sectors(symmetry, 1, np_random=np_random)[0],
        np_random=np_random,
    )
    print('>>> str(X)')
    print(str(X))
