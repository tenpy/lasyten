import numpy as np
import numpy.testing as npt
import pytest

import lasyten as lasy
from lasyten.block_backends import Scalar

no_symmetry = lasy.NoSymmetry().as_Symmetry()
u1_symmetry = lasy.U1().as_Symmetry()
fermion_parity = lasy.FermionParity().as_Symmetry()
fibonacci_anyon_category = lasy.FibonacciAnyonCategory(handedness='left').as_Symmetry()


def is_cyclical_perm(seq: list[int]) -> bool:
    if len(seq) == 0:
        return True
    n = seq[0]
    N = len(seq)
    return list(seq) == [*range(n, N), *range(n)]


@pytest.mark.parametrize(
    'legs, num_legs, is_planar',
    [
        ([1, 2, 3], 7, True),
        ([1, 2, 5, 6], 10, False),
        ([], 0, True),
        ([], 6, True),
        ([0, 1, 2, 3], 4, True),
        ([0, 1, 7, 8], 9, True),
        ([0, 1, 5, 6], 10, False),
    ],
)
@pytest.mark.parametrize('shuffle', [True, False])
def test_parse_leg_bipartition(legs, num_legs, is_planar, shuffle, np_random):
    if shuffle:
        np_random.shuffle(legs)

    if not is_planar:
        with pytest.raises(ValueError, match='Not a planar bipartition'):
            _ = lasy.planar.parse_leg_bipartition(legs, num_legs)
        return

    a, b = lasy.planar.parse_leg_bipartition(legs, num_legs)

    assert len(a) == len(legs)
    assert len(b) == num_legs - len(a)
    assert len(set(a) & set(b)) == 0, 'not a bipartition (duplicates)!'
    assert {*a, *b} == {*range(num_legs)}, 'not a bipartition (missing)!'
    assert all(n2 == n1 + 1 or (n2, n1) == (0, num_legs - 1) for n1, n2 in zip(a[:-1], a[1:]))
    assert all(n2 == n1 + 1 or (n2, n1) == (0, num_legs - 1) for n1, n2 in zip(b[:-1], b[1:]))


planar_combine_cases = {
    'trivial': ([['a', 'b'], ['c', 'd']], [['a'], ['c']], ['a', 'b'], ['c', 'd'], None),
    # without bending
    'codomain-1': ([['a', 'b', 'c'], ['d']], [['a', 'b']], ['a', 'b', 'c'], ['d'], None),
    'codomain-2': ([['a', 'b', 'c', 'd'], ['e']], [['a', 'b'], ['d', 'c']], ['a', 'b', 'c', 'd'], ['e'], None),
    'domain-1': ([['a'], ['b', 'c', 'd']], [['b', 'c']], ['a'], ['b', 'c', 'd'], None),
    'domain-2': ([['a'], ['b', 'c', 'd', 'e']], [['c', 'b'], ['d', 'e']], ['a'], ['b', 'c', 'd', 'e'], None),
    # bending right
    'bend-right-down-1': ([['a', 'b'], ['c']], [['b', 'c']], ['a', 'b', 'c'], [], True),
    'bend-right-down-2': ([['a', 'b'], ['c', 'd']], [['b', 'd', 'c']], ['a', 'b', 'd', 'c'], [], True),
    'bend-right-up-1': ([['a', 'b'], ['c']], [['c', 'b']], ['a'], ['c', 'b'], True),
    'bend-right-up-2': ([['a', 'b', 'c'], ['d', 'e']], [['e', 'd', 'b', 'c']], ['a'], ['d', 'e', 'c', 'b'], True),
    # bending left
    'bend-left-down-1': ([['a', 'b'], ['c']], [['a', 'c']], ['c', 'a', 'b'], [], False),
    'bend-left-down-2': ([['a', 'b'], ['c', 'd']], [['a', 'd', 'c']], ['d', 'c', 'a', 'b'], [], False),
    'bend-left-up-1': ([['a', 'b'], ['c']], [['c', 'a']], ['b'], ['a', 'c'], False),
    'bend-left-up-2': ([['a', 'b', 'c'], ['d', 'e']], [['e', 'd', 'b', 'a']], ['c'], ['b', 'a', 'd', 'e'], False),
    # bending left and right
    'bend-down-down': (
        [['a', 'b'], ['c', 'd', 'e']],
        [['a', 'c'], ['b', 'e']],
        ['c', 'a', 'b', 'e'],
        ['d'],
        [None, None, True, None, False],
    ),
    'bend-down-up': (
        [['a', 'b'], ['c', 'd', 'e']],
        [['a', 'c'], ['e', 'b']],
        ['c', 'a'],
        ['d', 'e', 'b'],
        [None, True, None, None, False],
    ),
    'bend-up-down': (
        [['a', 'b', 'c'], ['d', 'e', 'f']],
        [['d', 'a'], ['c', 'f', 'e'], ['b']],
        ['b', 'c', 'f', 'e'],
        ['a', 'd'],
        [False, None, None, True, True, None],
    ),
    'bend-up-up': (
        [['a', 'b', 'c'], ['d', 'e', 'f']],
        [['d', 'e', 'a'], ['f', 'c'], ['b']],
        ['b'],
        ['a', 'd', 'e', 'f', 'c'],
        [False, None, True, None, None, None],
    ),
}


@pytest.mark.parametrize(
    'labels, which_legs, new_codomain, new_domain, bend_right',
    planar_combine_cases.values(),
    ids=planar_combine_cases.keys(),
)
@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (no_symmetry, 'no_symmetry'),
        (u1_symmetry, 'abelian'),
        (u1_symmetry, 'fusion_tree'),
        (fermion_parity, 'fusion_tree'),
        (fibonacci_anyon_category, 'fusion_tree'),
    ],
)
def test_planar_combine_legs(
    labels,
    which_legs,
    new_codomain,
    new_domain,
    bend_right,
    symmetry,
    backend,
    np_random,
):
    backend = lasy.get_backend(backend, 'numpy')
    T: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain=len(labels[0]),
        domain=len(labels[1]),
        labels=[*labels[0], *reversed(labels[1])],
        backend=backend,
        np_random=np_random,
    )
    pipe_dualities = np_random.choice([True, False], size=len(which_legs))

    num_codomain_legs = T.num_codomain_legs
    num_codomain_flat_legs = T.num_codomain_flat_legs
    num_domain_legs = T.num_domain_legs
    num_domain_flat_legs = T.num_domain_flat_legs
    for group in which_legs:
        leg0 = T.labels.index(group[0])
        for i, leg in enumerate(T.labels.indices(group)):
            if i == 0:
                continue
            if leg < T.num_codomain_legs:
                num_codomain_legs -= 1
            else:
                num_domain_legs -= 1
            flat_num = len(T.get_leg(leg)) if isinstance(T.get_leg(leg), lasy.LegPipe) else 1
            if leg0 < T.num_codomain_legs and leg >= T.num_codomain_legs:
                num_codomain_flat_legs += flat_num
                num_domain_flat_legs -= flat_num
            elif leg0 >= T.num_codomain_legs and leg < T.num_codomain_legs:
                num_codomain_flat_legs -= flat_num
                num_domain_flat_legs += flat_num

    T_combined = lasy.planar.planar_combine_legs(T, *which_legs, pipe_dualities=pipe_dualities)
    T_combined.test_sanity()
    assert T_combined.num_codomain_legs == num_codomain_legs
    assert T_combined.num_codomain_flat_legs == num_codomain_flat_legs
    assert T_combined.num_domain_legs == num_domain_legs
    assert T_combined.num_domain_flat_legs == num_domain_flat_legs
    assert lasy.planar.planar_almost_equal(T, lasy.split_legs(T_combined))

    # compare to nonplanar function
    T_combined2 = lasy.permute_legs(T, codomain=new_codomain, domain=new_domain, bend_right=bend_right)
    # make sure which_legs is sorted to avoid braids
    which_legs = [sorted(T_combined2.labels.indices(group)) for group in which_legs]
    T_combined2 = lasy.combine_legs(T_combined2, *which_legs, pipe_dualities=pipe_dualities)
    assert lasy.almost_equal(T_combined, T_combined2)


planar_contraction_cases = {
    '2-0-2-right': ([['a'], ['b']], [['c', 'd'], []], [], [], None, None, True),
    '2-1-2-right': ([['a'], ['b']], [['b', 'c'], []], [1], [0], None, None, True),
    '2-1-2-left': ([['a'], ['b']], [['c', 'b'], []], [1], [1], None, None, False),
    '2-2-2-right': ([['a'], ['b']], [[], ['a', 'b']], [1, 0], [0, 1], None, None, True),
    '3-1-3-right': ([['a'], ['c', 'b']], [['c', 'd'], ['e']], [2], [0], None, None, True),
    '3-1-3-left': ([['a'], ['b', 'c']], [['d', 'c'], ['e']], [1], [1], [2, 0], [0, 2], False),
    '3-2-3-left': ([['b', 'a'], ['c']], [['c'], ['b', 'd']], [0, 2], [2, 0], [1], [1], False),
    '3-3-3-right': ([['a', 'b'], ['c']], [['c'], ['a', 'b']], [2, 1, 0], [0, 1, 2], None, None, True),
    '4-2-3-left': ([['a', 'b', 'c'], ['d']], [['d'], ['a', 'e']], [0, 3], [2, 0], None, None, False),
    '4-2-3-right': ([['a', 'b', 'c'], ['d']], [['d'], ['e', 'c']], [3, 2], [0, 1], None, None, True),
    '4-3-3-right': ([['a', 'b', 'c', 'd'], []], [['d'], ['b', 'c']], [3, 2, 1], [0, 1, 2], None, None, True),
    '4-3-4-right1': ([['a', 'b'], ['d', 'c']], [['d', 'c'], ['e', 'b']], [3, 2, 1], [0, 1, 2], None, None, True),
    '4-3-4-right2': (
        [['b'], ['a', 'd', 'c']],
        [['d', 'c'], ['e', 'b']],
        [2, 1, 0],
        [0, 1, 2],
        None,
        None,
        [True, None, True, False],
    ),
    '4-3-4-left1': ([['a', 'b'], ['d', 'c']], [['d', 'c'], ['a', 'e']], [0, 3, 2], [3, 0, 1], None, None, False),
    '4-3-4-left2': (
        [['a'], ['d', 'c', 'b']],
        [['d', 'c'], ['a', 'e']],
        [0, 3, 2],
        [3, 0, 1],
        None,
        None,
        [False, True, None, False],
    ),
}


@pytest.mark.parametrize(
    'labels_A, labels_B, contr_A, contr_B, A_codomain, B_domain, bend_right',
    planar_contraction_cases.values(),
    ids=planar_contraction_cases.keys(),
)
@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (no_symmetry, 'no_symmetry'),
        (u1_symmetry, 'abelian'),
        (u1_symmetry, 'fusion_tree'),
        (fermion_parity, 'fusion_tree'),
        (fibonacci_anyon_category, 'fusion_tree'),
    ],
)
def test_planar_contraction(
    labels_A: list[list[str]],
    labels_B: list[list[str]],
    contr_A: list[int],
    contr_B: list[int],
    A_codomain,
    B_domain,
    bend_right: bool,
    symmetry,
    backend,
    np_random,
):
    backend = lasy.get_backend(backend, 'numpy')
    # same construction as in test_tdot in test_tensors.py
    A: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain=len(labels_A[0]),
        domain=len(labels_A[1]),
        labels=[*labels_A[0], *reversed(labels_A[1])],
        backend=backend,
        np_random=np_random,
    )

    B: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain=[A._as_domain_leg(l) if A.labels.has(l) else None for l in labels_B[0]],
        domain=[A._as_codomain_leg(l) if A.labels.has(l) else None for l in labels_B[1]],
        labels=[*labels_B[0], *reversed(labels_B[1])],
        backend=backend,
        np_random=np_random,
    )

    # make sure we defined compatible legs
    for ia, ib in zip(contr_A, contr_B):
        assert A._as_domain_leg(ia) == B._as_codomain_leg(ib)

    res = lasy.planar.planar_contraction(A, B, contr_A, contr_B)

    A_permuted = lasy.permute_legs(A, codomain=A_codomain, domain=contr_A, bend_right=bend_right)
    B_permuted = lasy.permute_legs(B, codomain=contr_B, domain=B_domain, bend_right=bend_right)
    contr_A = list(reversed(range(A_permuted.num_codomain_legs, A_permuted.num_legs)))
    contr_B = list(range(B_permuted.num_codomain_legs))
    expect = lasy.tdot(A_permuted, B_permuted, contr_A, contr_B)

    if isinstance(res, lasy.Tensor):
        assert isinstance(expect, lasy.Tensor)
        assert lasy.planar.planar_almost_equal(res, expect)
    else:
        assert isinstance(res, Scalar)
        assert isinstance(expect, Scalar)
        assert len(contr_A) == A.num_legs and len(contr_B) == B.num_legs
        npt.assert_almost_equal(res.to_numpy(), expect.to_numpy())


@pytest.mark.parametrize(
    'cls, dom, cod, dom_cut, cod_cut, new_leg_dual',
    [
        pytest.param(lasy.SymmetricTensor, 1, 1, 0, 1, False, id='Sym-1-1-False'),
        pytest.param(lasy.SymmetricTensor, 2, 0, 1, 0, True, id='Sym-2-0-True'),
        pytest.param(lasy.SymmetricTensor, 3, 1, 2, 0, False, id='Sym-3-1-False'),
        pytest.param(lasy.SymmetricTensor, 1, 3, 1, 1, True, id='Sym-1-3-True'),
        pytest.param(lasy.SymmetricTensor, 2, 2, 1, 1, False, id='Sym-2-2-False'),
        pytest.param(lasy.SymmetricTensor, 2, 2, 0, 2, True, id='Sym-2-2-True'),
        pytest.param(lasy.DiagonalTensor, 1, 1, 1, 0, False, id='Diag-False'),
        pytest.param(lasy.DiagonalTensor, 1, 1, 0, 1, True, id='Diag-True'),
    ],
)
def test_planar_eigh(cls, dom, cod, dom_cut, cod_cut, new_leg_dual, make_compatible_tensor):
    # prepare hermitian tensor, do this as in test_eigh, then bend the legs
    num_dom_legs = (cod + dom) // 2
    T2: lasy.Tensor = make_compatible_tensor(num_dom_legs, num_dom_legs, cls=cls, max_blocks=3, max_block_size=3)
    T2: lasy.Tensor = make_compatible_tensor(T2.domain, T2.domain, cls=cls)
    T2 = T2 + T2.hc
    T2_labels = list('efghijk')[:num_dom_legs]
    T2_labels.extend(reversed([f'{l}*' for l in T2_labels]))
    T2.labels.set(T2_labels)
    T2.test_sanity()

    left_bends = dom_cut
    right_bends = cod - cod_cut
    cod_legs = [*range(left_bends, num_dom_legs + right_bends)]
    dom_legs = [*reversed(range(left_bends)), *reversed(range(num_dom_legs + right_bends, 2 * num_dom_legs))]
    bend_right = [False] * left_bends + [True] * (2 * num_dom_legs - left_bends)
    T = lasy.permute_legs(T2, cod_legs, dom_legs, bend_right=bend_right)

    sort = ['<', '>', 'm<', 'm>'][dom_cut - cod_cut]
    W, V = lasy.planar.planar_eigh(
        T, codomain_cut=cod_cut, domain_cut=dom_cut, new_labels=['a', 'b', 'c'], new_leg_dual=new_leg_dual, sort=sort
    )
    W.test_sanity()
    V.test_sanity()
    assert isinstance(W, lasy.DiagonalTensor)
    assert W.labels == ['b', 'c']
    assert V.num_codomain_legs == cod_cut
    assert V.num_domain_legs == dom_cut + 1
    assert V.labels == [*T.labels.as_list()[:cod_cut], 'a', *T.labels.as_list()[T.num_legs - dom_cut :]]

    assert lasy.planar.planar_almost_equal(
        lasy.planar_contraction(lasy.planar_contraction(V, W, 'a', 'b'), V.hc, 'c', 'a*'), T
    )
    eye1 = lasy.SymmetricTensor.from_eye(T2.domain, T.backend, labels=T2_labels[: T2.num_domain_legs])
    eye2 = lasy.SymmetricTensor.from_eye([V.get_leg('a')], T.backend, labels=['a'])
    eye2_contr = [l for l in V.labels.as_list() if l != 'a']
    assert lasy.planar.planar_almost_equal(lasy.planar_contraction(V, V.hc, 'a', 'a*'), eye1)
    assert lasy.planar.planar_almost_equal(
        lasy.planar_contraction(V, V.hc, eye2_contr, [f'{l}*' for l in eye2_contr]), eye2
    )

    # compare to non-planar result
    W2, V2 = lasy.eigh(T2, new_labels=['a', 'b', 'c'], new_leg_dual=new_leg_dual, sort=sort)
    W2.test_sanity()
    V2.test_sanity()
    assert lasy.almost_equal(W, W2)
    assert lasy.planar.planar_almost_equal(V, V2)


@pytest.mark.parametrize(
    'cls, dom, cod, dom_cut, cod_cut, new_leg_dual',
    [
        pytest.param(lasy.SymmetricTensor, 1, 1, 0, 1, False, id='Sym-1-1-False'),
        pytest.param(lasy.SymmetricTensor, 2, 0, 1, 0, True, id='Sym-2-0-True'),
        pytest.param(lasy.SymmetricTensor, 3, 1, 2, 0, False, id='Sym-3-1-False'),
        pytest.param(lasy.SymmetricTensor, 1, 3, 1, 1, True, id='Sym-1-3-True'),
        pytest.param(lasy.SymmetricTensor, 2, 2, 1, 1, False, id='Sym-2-2-False'),
        pytest.param(lasy.SymmetricTensor, 2, 2, 0, 2, True, id='Sym-2-2-True'),
        pytest.param(lasy.DiagonalTensor, 1, 1, 1, 0, False, id='Diag-False'),
        pytest.param(lasy.DiagonalTensor, 1, 1, 0, 1, True, id='Diag-True'),
    ],
)
def test_planar_eig(cls, dom, cod, dom_cut, cod_cut, new_leg_dual, make_compatible_tensor):
    num_dom_legs = (cod + dom) // 2
    T2: lasy.Tensor = make_compatible_tensor(num_dom_legs, num_dom_legs, cls=cls, max_blocks=3, max_block_size=3)
    T2: lasy.Tensor = make_compatible_tensor(T2.domain, T2.domain, cls=cls)
    T2_labels = list('efghijk')[:num_dom_legs]
    T2_labels.extend(reversed([f'{l}*' for l in T2_labels]))
    T2.labels.set(T2_labels)
    T2.test_sanity()

    left_bends = dom_cut
    right_bends = cod - cod_cut
    cod_legs = [*range(left_bends, num_dom_legs + right_bends)]
    dom_legs = [*reversed(range(left_bends)), *reversed(range(num_dom_legs + right_bends, 2 * num_dom_legs))]
    bend_right = [False] * left_bends + [True] * (2 * num_dom_legs - left_bends)
    T = lasy.permute_legs(T2, cod_legs, dom_legs, bend_right=bend_right)

    sort = ['<', '>', 'm<', 'm>'][dom_cut - cod_cut]
    W, V = lasy.planar.planar_eig(
        T, codomain_cut=cod_cut, domain_cut=dom_cut, new_labels=['a', 'b', 'c'], new_leg_dual=new_leg_dual, sort=sort
    )
    W.test_sanity()
    V.test_sanity()
    assert isinstance(W, lasy.DiagonalTensor)
    assert W.labels == ['b', 'c']
    assert V.num_codomain_legs == cod_cut
    assert V.num_domain_legs == dom_cut + 1
    assert V.labels == [*T.labels.as_list()[:cod_cut], 'a', *T.labels.as_list()[T.num_legs - dom_cut :]]

    W2, V2 = lasy.eig(T2, new_labels=['a', 'b', 'c'], new_leg_dual=new_leg_dual, sort=sort)
    W2.test_sanity()
    V2.test_sanity()
    assert lasy.almost_equal(W, W2)
    assert lasy.planar.planar_almost_equal(V, V2)


@pytest.mark.parametrize(
    'cls, dom, cod, dom_cut, cod_cut, new_leg_dual',
    [
        pytest.param(lasy.SymmetricTensor, 1, 1, 0, 1, False, id='Sym-1-1-False'),
        pytest.param(lasy.SymmetricTensor, 2, 0, 1, 0, True, id='Sym-2-0-True'),
        pytest.param(lasy.SymmetricTensor, 3, 1, 2, 0, False, id='Sym-3-1-False'),
        pytest.param(lasy.SymmetricTensor, 1, 3, 1, 1, True, id='Sym-1-3-True'),
        pytest.param(lasy.SymmetricTensor, 2, 2, 1, 1, False, id='Sym-2-2-False'),
        pytest.param(lasy.SymmetricTensor, 2, 2, 0, 2, True, id='Sym-2-2-True'),
        pytest.param(lasy.DiagonalTensor, 1, 1, 1, 0, False, id='Diag-False'),
        pytest.param(lasy.DiagonalTensor, 1, 1, 0, 1, True, id='Diag-True'),
    ],
)
def test_planar_eigvals(cls, dom, cod, dom_cut, cod_cut, new_leg_dual, make_compatible_tensor):
    num_dom_legs = (cod + dom) // 2
    T2: lasy.Tensor = make_compatible_tensor(num_dom_legs, num_dom_legs, cls=cls, max_blocks=3, max_block_size=3)
    T2: lasy.Tensor = make_compatible_tensor(T2.domain, T2.domain, cls=cls)
    T2_labels = list('efghijk')[:num_dom_legs]
    T2_labels.extend(reversed([f'{l}*' for l in T2_labels]))
    T2.labels.set(T2_labels)

    left_bends = dom_cut
    right_bends = cod - cod_cut
    cod_legs = [*range(left_bends, num_dom_legs + right_bends)]
    dom_legs = [*reversed(range(left_bends)), *reversed(range(num_dom_legs + right_bends, 2 * num_dom_legs))]
    bend_right = [False] * left_bends + [True] * (2 * num_dom_legs - left_bends)
    T = lasy.permute_legs(T2, cod_legs, dom_legs, bend_right=bend_right)

    sort = ['<', '>', 'm<', 'm>'][dom_cut - cod_cut]
    W = lasy.planar.planar_eigvals(
        T, codomain_cut=cod_cut, domain_cut=dom_cut, new_labels=['b', 'c'], new_leg_dual=new_leg_dual, sort=sort
    )
    W.test_sanity()
    assert isinstance(W, lasy.DiagonalTensor)
    assert W.labels == ['b', 'c']

    W_eig, _V = lasy.planar.planar_eig(
        T, codomain_cut=cod_cut, domain_cut=dom_cut, new_labels=['a', 'b', 'c'], new_leg_dual=new_leg_dual, sort=sort
    )
    assert lasy.almost_equal(W, W_eig)
    W2 = lasy.eigvals(T2, new_labels=['b', 'c'], new_leg_dual=new_leg_dual, sort=sort)
    assert lasy.almost_equal(W, W2)


planar_partial_trace_cases = {
    # traces in codomain
    'codomain-aab': (['a', 'a', 'b'], []),
    'codomain-aabbc': (['a', 'a', 'b', 'b', 'c'], []),
    'codomain-abba-c': (['a', 'b', 'b', 'a'], ['c']),
    # traces in domain
    'domain-b-aa': (['b'], ['a', 'a']),
    'domain-c-aabb': (['c'], ['a', 'a', 'b', 'b']),
    'domain-c-abba': (['c'], ['a', 'b', 'b', 'a']),
    # traces in both codomain and domain
    'co_domain-aac-bb': (['a', 'a', 'c'], ['b', 'b']),
    # left and right
    'co_domain-acb-ab': (['a', 'c', 'b'], ['a', 'b']),
    # two left
    'co_domain-abc-ab': (['a', 'b', 'c'], ['a', 'b']),
    # two right
    'co_domain-cab-ab': (['c', 'a', 'b'], ['a', 'b']),
    # winding
    'codomain-aba': (['a', 'b', 'a'], []),
    'codomain-abcba': (['a', 'b', 'c', 'b', 'a'], []),
    'domain--aba': ([], ['a', 'b', 'a']),
    'domain--abcba': ([], ['a', 'b', 'c', 'b', 'a']),
    'co_domain-abcb-a': (['a', 'b', 'c', 'b'], ['a']),
    'co_domain-acab-b': (['a', 'c', 'a', 'b'], ['b']),
}


def trace_partial_np(a: np.ndarray, idcs1: list[int], idcs2: list[int], remaining: list[int]) -> np.ndarray:
    """numpy version of block_backend.trace_partial"""
    a = np.transpose(a, remaining + idcs1 + idcs2)
    trace_dim = np.prod(a.shape[len(remaining) : len(remaining) + len(idcs1)], dtype=int)
    a = np.reshape(a, a.shape[: len(remaining)] + (trace_dim, trace_dim))
    return np.trace(a, axis1=-2, axis2=-1)


@pytest.mark.parametrize('codomain, domain', planar_partial_trace_cases.values(), ids=planar_partial_trace_cases.keys())
@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (no_symmetry, 'no_symmetry'),
        (u1_symmetry, 'abelian'),
        (u1_symmetry, 'fusion_tree'),
        (fermion_parity, 'fusion_tree'),
        (fibonacci_anyon_category, 'fusion_tree'),
    ],
)
def test_planar_partial_trace(codomain, domain, symmetry, backend, np_random):
    # same construction as in test_partial_trace in test_tensors.py
    backend = lasy.get_backend(backend, 'numpy')
    trace_legs = {
        l: lasy.testing.random_leg(symmetry, backend, False, np_random=np_random)
        for l in lasy.tools.misc.duplicate_entries([*codomain, *domain])
    }
    seen_labels = []
    codomain_spaces = []
    codomain_labels = []
    for l in codomain:
        if l in seen_labels:
            codomain_spaces.append(trace_legs[l].dual)
            codomain_labels.append(f'{l}*')
        elif l in trace_legs:
            codomain_spaces.append(trace_legs[l])
            seen_labels.append(l)
            codomain_labels.append(l)
        else:
            codomain_spaces.append(lasy.testing.random_leg(symmetry, backend, False, np_random=np_random))
            codomain_labels.append(l)
    domain_spaces = []
    domain_labels = []
    for l in domain:
        if l in seen_labels:
            domain_spaces.append(trace_legs[l])
            domain_labels.append(f'{l}*')
        elif l in trace_legs:
            domain_spaces.append(trace_legs[l].dual)
            domain_labels.append(l)
            seen_labels.append(l)
        else:
            domain_spaces.append(lasy.testing.random_leg(symmetry, backend, False, np_random=np_random))
            domain_labels.append(l)

    T: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain_spaces,
        domain_spaces,
        labels=[*codomain_labels, *reversed(domain_labels)],
        backend=backend,
        np_random=np_random,
    )

    pairs = [(T.labels.index(l), T.labels.index(f'{l}*')) for l in trace_legs]
    res = lasy.planar.planar_partial_trace(T, *pairs)
    res.test_sanity()
    assert res.labels == [l for l in T.labels.as_list() if l[0] not in trace_legs]
    assert res.legs == [T.get_leg(l) for l in T.labels.as_list() if l[0] not in trace_legs]

    if T.symmetry.has_trivial_braid:
        T_np = T.to_numpy()
        idcs1 = [p[0] for p in pairs]
        idcs2 = [p[1] for p in pairs]
        remaining = [n for n in range(T.num_legs) if n not in idcs1 and n not in idcs2]
        expect = trace_partial_np(T_np, idcs1, idcs2, remaining)
        res_np = res.to_numpy()
        npt.assert_almost_equal(res_np, expect)

    levels = None
    if not T.symmetry.has_symmetric_braid:
        flat_pairs = [p for pair in pairs for p in pair]
        # may need to braid with open legs since we do not trace over the left side
        levels = [flat_pairs.index(n) if n in flat_pairs else 10 for n in range(T.num_legs)]

    expect = lasy.tensors.partial_trace(T, *pairs, levels=levels)
    assert expect.labels == res.labels
    assert expect.legs == res.legs
    assert lasy.almost_equal(res, expect)


planar_permute_legs_cases = {
    'trivial': (3, 2, None, [0, 1, 2][::-1]),
    # same basic case with different input possibilities
    'basic-idcs': (3, 2, None, [3, 4, 0][::-1]),
    'basic-labels': (3, 2, None, ['a', 'e', 'd']),
    'basic-codomain': (3, 2, [1, 2], None),
    # empty codomain/domain
    'empty-codomain': (2, 2, [], [1, 0, 3, 2]),
    'empty-domain': (2, 2, [0, 1, 2, 3], None),
    # input has no codomain
    'J0-empty-domain': (0, 3, [2, 0, 1], []),
    'J0': (0, 3, None, [1]),
    'J0-empty-codomain': (0, 3, None, [0, 1, 2][::-1]),
    # input has no domain
    'K0-empty-domain': (3, 0, [1, 2, 0], []),
    'K0': (3, 0, None, [1]),
    'K0-empty-codomain': (3, 0, [0, 1, 2], None),
}


@pytest.mark.parametrize(
    'J, K, codomain, domain', planar_permute_legs_cases.values(), ids=planar_permute_legs_cases.keys()
)
@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (no_symmetry, 'no_symmetry'),
        (u1_symmetry, 'abelian'),
        (u1_symmetry, 'fusion_tree'),
        (fermion_parity, 'fusion_tree'),
        (fibonacci_anyon_category, 'fusion_tree'),
    ],
)
def test_planar_permute_legs(J, K, codomain, domain, symmetry, backend, np_random):
    backend = lasy.get_backend(backend, 'numpy')
    T_labels = list('abcdefghijk')[: J + K]
    T: lasy.SymmetricTensor = lasy.testing.random_tensor(symmetry, J, K, labels=T_labels, np_random=np_random)

    res = lasy.planar.planar_permute_legs(T, codomain=codomain, domain=domain)
    res.test_sanity()

    # test planar_almost_equal; test both ways
    assert lasy.planar.planar_almost_equal(res, T)
    assert lasy.planar.planar_almost_equal(T, res)

    if codomain is None or len(codomain) == 0:
        domain = T.labels.indices(domain)
        num_codom_legs = T.num_legs - len(domain)
        codomain = [i % T.num_legs for i in range(domain[0] + 1, domain[0] + 1 + num_codom_legs)]
        rev_domain = domain[::-1]
    else:
        codomain = T.labels.indices(codomain)
        num_dom_legs = T.num_legs - len(codomain)
        rev_domain = [i % T.num_legs for i in range(codomain[-1] + 1, codomain[-1] + 1 + num_dom_legs)]
    leg_perm = [*codomain, *rev_domain]
    assert is_cyclical_perm(leg_perm)
    assert res.labels == [T.labels[n] for n in leg_perm]
    assert res.legs == [T.get_leg(n) for n in leg_perm]

    if symmetry.can_be_dropped:
        T_np = T.to_numpy(understood_braiding=True)
        res_np = res.to_numpy(understood_braiding=True)
        expect = np.transpose(T_np, leg_perm)
        if symmetry.has_trivial_braid:
            npt.assert_almost_equal(res_np, expect)
        else:
            # the expect is missing some signs from twists in the diagram.
            # I dont know how to figure them out right now, so we just ignore signs here...
            npt.assert_almost_equal(np.abs(res_np), np.abs(expect))

    if len(T.codomain_labels) > 0:
        permuted_back1 = lasy.planar.planar_permute_legs(res, codomain=T.codomain_labels)
        permuted_back1.test_sanity()
        assert lasy.almost_equal(permuted_back1, T)

    if len(T.domain_labels) > 0:
        permuted_back2 = lasy.planar.planar_permute_legs(res, domain=T.domain_labels)
        permuted_back2.test_sanity()
        assert lasy.almost_equal(permuted_back2, T)


planar_permute_legs_hidden_cases = {
    # h sits between b and c, which stay neighbors in the codomain -> h is forced in between.
    # Planar only if h is hidden, since a public h would have to end up in the codomain too.
    'inner-gap': (3, 2, list('abhcd'), ['b', 'c'], ['a', 'd'], ['b', 'h', 'c'], ['a', 'd'], True),
    # h sits at the (left) boundary between new domain and codomain -> could go on either side.
    # It stays in the codomain, where it is originally, while a public h would be bent.
    'boundary-gap-codomain': (3, 2, list('abhcd'), ['c', 'd'], None, ['h', 'c', 'd'], ['b', 'a'], False),
    # same, but h is originally in the domain and stays there
    'boundary-gap-domain': (2, 3, list('abhcd'), ['c', 'd'], None, ['c', 'd'], ['h', 'b', 'a'], False),
    # all public legs end up in the codomain, h stays alone in the domain
    'wrap-gap-empty-domain': (2, 2, list('abch'), ['a', 'b', 'c'], [], ['a', 'b', 'c'], ['h'], False),
}


@pytest.mark.parametrize(
    'J, K, labels, codomain, domain, expect_codomain, expect_domain, needs_hiding',
    planar_permute_legs_hidden_cases.values(),
    ids=planar_permute_legs_hidden_cases.keys(),
)
@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (no_symmetry, 'no_symmetry'),
        (u1_symmetry, 'abelian'),
        (u1_symmetry, 'fusion_tree'),
        (fermion_parity, 'fusion_tree'),
        (fibonacci_anyon_category, 'fusion_tree'),
    ],
)
def test_planar_permute_legs_hidden(
    J, K, labels, codomain, domain, expect_codomain, expect_domain, needs_hiding, symmetry, backend, np_random
):
    # For a HiddenLegTensor, only the public legs need to be permuted planarly. Hidden legs stay in
    # the gap between their public neighbors (never braid) and are only bent if necessary.
    backend = lasy.get_backend(backend, 'numpy')
    T_sym = lasy.testing.random_tensor(symmetry, J, K, labels=labels, backend=backend, np_random=np_random)
    T = lasy.HiddenLegTensor(T_sym, ['h'])

    if needs_hiding:
        with pytest.raises(ValueError):
            _ = lasy.planar.planar_permute_legs(T_sym, codomain=codomain, domain=domain)
        with pytest.raises(ValueError, match='non-planar'):
            _ = lasy.planar.planar_permute_legs(T_sym, codomain=codomain)

    res = lasy.planar.planar_permute_legs(T, codomain=codomain, domain=domain)
    res.test_sanity()
    assert isinstance(res, lasy.HiddenLegTensor)
    hide = {'h': '!h'}
    assert res.codomain_labels == [hide.get(l, l) for l in expect_codomain]
    assert res.domain_labels == [hide.get(l, l) for l in expect_domain]

    # the same as a planar permutation that specifies the hidden leg explicitly
    expect = lasy.planar.planar_permute_legs(T_sym, codomain=expect_codomain, domain=expect_domain)
    assert lasy.almost_equal(res.as_SymmetricTensor(), expect)
    assert lasy.planar.planar_almost_equal(res.as_SymmetricTensor(), T_sym)


@pytest.mark.parametrize(
    'cls, dom, cod, dom_cut, cod_cut, new_leg_dual',
    [
        pytest.param(lasy.SymmetricTensor, 1, 1, 0, 1, False, id='Sym-1-1-False'),
        pytest.param(lasy.SymmetricTensor, 1, 1, 1, 0, True, id='Sym-1-1-True'),
        pytest.param(lasy.SymmetricTensor, 3, 1, 2, 1, False, id='Sym-3-1-False'),
        pytest.param(lasy.SymmetricTensor, 2, 2, 1, 1, False, id='Sym-2-2-False'),
        pytest.param(lasy.SymmetricTensor, 2, 2, 0, 2, True, id='Sym-2-2-True'),
        pytest.param(lasy.DiagonalTensor, 1, 1, 1, 0, False, id='Diag-False'),
        pytest.param(lasy.DiagonalTensor, 1, 1, 0, 1, True, id='Diag-True'),
        pytest.param(lasy.Mask, 1, 1, 0, 1, False, id='Mask-False'),
        pytest.param(lasy.Mask, 1, 1, 1, 0, True, id='Mask-True'),
    ],
)
def test_planar_qr_lq(cls, dom, cod, dom_cut, cod_cut, new_leg_dual, make_compatible_tensor):
    T_labels = list('abcdef')[: dom + cod]
    T: lasy.Tensor = make_compatible_tensor(cod, dom, cls=cls, labels=T_labels, max_blocks=3, max_block_size=3)

    Q, R = lasy.planar.planar_qr(T, codomain_cut=cod_cut, domain_cut=dom_cut, new_leg_dual=new_leg_dual, new_labels='v')
    Q.test_sanity()
    R.test_sanity()
    assert Q.num_codomain_legs == cod_cut
    assert Q.num_domain_legs == dom_cut + 1
    assert Q.labels == [*T.labels.as_list()[:cod_cut], 'v', *T.labels.as_list()[T.num_legs - dom_cut :]]
    assert R.num_codomain_legs == 1 + (cod - cod_cut)
    assert R.num_domain_legs == dom - dom_cut
    assert R.labels == ['v*', *T.labels.as_list()[cod_cut : T.num_legs - dom_cut]]
    assert lasy.planar.planar_almost_equal(lasy.planar_contraction(Q, R, 'v', 'v*'), T)
    eye = lasy.SymmetricTensor.from_eye([Q.get_leg('v')], T.backend, labels=['v'])
    contr = [l for l in Q.labels.as_list() if l != 'v']
    assert lasy.planar.planar_almost_equal(lasy.planar_contraction(Q, Q.hc, contr, [f'{l}*' for l in contr]), eye)
    # compare to non-planar result
    idcs = [*range(T.num_legs - dom_cut, T.num_legs), *range(cod_cut)]
    T2 = lasy.planar_permute_legs(T, codomain=idcs)
    Q2, R2 = lasy.qr(T2, new_labels='v', new_leg_dual=new_leg_dual)
    Q2.test_sanity()
    R2.test_sanity()
    assert lasy.planar.planar_almost_equal(Q, Q2)
    assert lasy.planar.planar_almost_equal(R, R2)

    L, Q = lasy.planar.planar_lq(
        T, codomain_cut=cod_cut, domain_cut=dom_cut, new_leg_dual=new_leg_dual, new_labels=['v*', 'v']
    )
    L.test_sanity()
    Q.test_sanity()
    assert L.num_codomain_legs == cod_cut
    assert L.num_domain_legs == dom_cut + 1
    assert L.labels == [*T.labels.as_list()[:cod_cut], 'v*', *T.labels.as_list()[T.num_legs - dom_cut :]]
    assert Q.num_codomain_legs == 1 + (cod - cod_cut)
    assert Q.num_domain_legs == dom - dom_cut
    assert Q.labels == ['v', *T.labels.as_list()[cod_cut : T.num_legs - dom_cut]]
    assert lasy.planar.planar_almost_equal(lasy.planar_contraction(L, Q, 'v*', 'v'), T)
    eye = lasy.SymmetricTensor.from_eye([Q.get_leg('v')], T.backend, labels='v')
    contr = [l for l in Q.labels.as_list() if l != 'v']
    assert lasy.planar.planar_almost_equal(lasy.planar_contraction(Q, Q.hc, contr, [f'{l}*' for l in contr]), eye)
    # compare to non-planar result
    L2, Q2 = lasy.lq(T2, new_labels=['v*', 'v'], new_leg_dual=new_leg_dual)
    L2.test_sanity()
    Q2.test_sanity()
    assert lasy.planar.planar_almost_equal(L, L2)
    assert lasy.planar.planar_almost_equal(Q, Q2)


@pytest.mark.parametrize(
    'cls, dom, cod, dom_cut, cod_cut, new_leg_dual',
    [
        pytest.param(lasy.SymmetricTensor, 1, 1, 1, 0, False, id='Sym-1-1-False'),
        pytest.param(lasy.SymmetricTensor, 1, 3, 0, 1, False, id='Sym-1-3-False'),
        pytest.param(lasy.SymmetricTensor, 3, 1, 2, 1, False, id='Sym-3-1-False'),
        pytest.param(lasy.SymmetricTensor, 2, 2, 1, 0, False, id='Sym-2-2-False'),
        pytest.param(lasy.SymmetricTensor, 2, 2, 1, 1, True, id='Sym-2-2-True'),
        pytest.param(lasy.DiagonalTensor, 1, 1, 1, 0, False, id='Diag-False'),
        pytest.param(lasy.DiagonalTensor, 1, 1, 0, 1, True, id='Diag-True'),
        pytest.param(lasy.Mask, 1, 1, 0, 1, False, id='Mask-False'),
        pytest.param(lasy.Mask, 1, 1, 1, 0, True, id='Mask-True'),
    ],
)
def test_planar_svd(cls, dom, cod, dom_cut, cod_cut, new_leg_dual, make_compatible_tensor):
    T_labels = list('efghijklmn')[: dom + cod]
    T: lasy.Tensor = make_compatible_tensor(cod, dom, labels=T_labels, cls=cls, max_blocks=3, max_block_size=3)

    print('Normal (non-truncated) SVD')
    U, S, Vh = lasy.planar.planar_svd(
        T, codomain_cut=cod_cut, domain_cut=dom_cut, new_labels=['a', 'b', 'c', 'd'], new_leg_dual=new_leg_dual
    )
    U.test_sanity()
    S.test_sanity()
    Vh.test_sanity()
    assert U.num_codomain_legs == cod_cut
    assert U.num_domain_legs == dom_cut + 1
    assert U.labels == [*T.labels.as_list()[:cod_cut], 'a', *T.labels.as_list()[T.num_legs - dom_cut :]]
    assert S.labels == ['b', 'c']
    assert Vh.num_codomain_legs == 1 + (cod - cod_cut)
    assert Vh.num_domain_legs == dom - dom_cut
    assert Vh.labels == ['d', *T.labels.as_list()[cod_cut : T.num_legs - dom_cut]]

    assert isinstance(S, lasy.DiagonalTensor)
    assert (S >= 0).all()
    npt.assert_almost_equal(lasy.norm(S).to_numpy(), lasy.norm(T).to_numpy())

    assert lasy.planar.planar_almost_equal(
        lasy.planar_contraction(lasy.planar_contraction(U, S, 'a', 'b'), Vh, 'c', 'd'), T
    )
    eye = lasy.SymmetricTensor.from_eye(S.domain, backend=T.backend, labels=['a*', 'a'])
    contr_U = [l for l in U.labels.as_list() if l != 'a']
    contr_Vh = [l for l in Vh.labels.as_list() if l != 'd']
    assert lasy.planar.planar_almost_equal(lasy.planar_contraction(U, U.hc, contr_U, [f'{l}*' for l in contr_U]), eye)
    eye.labels.set(['d', 'd*'])
    assert lasy.planar.planar_almost_equal(
        lasy.planar_contraction(Vh, Vh.hc, contr_Vh, [f'{l}*' for l in contr_Vh]), eye
    )

    # compare to non-planar result
    idcs = [*range(T.num_legs - dom_cut, T.num_legs), *range(cod_cut)]
    T2 = lasy.planar_permute_legs(T, codomain=idcs)
    U2, S2, Vh2 = lasy.svd(T2, new_labels=['a', 'b', 'c', 'd'], new_leg_dual=new_leg_dual)
    U2.test_sanity()
    S2.test_sanity()
    Vh2.test_sanity()
    assert lasy.planar.planar_almost_equal(U, U2)
    assert lasy.almost_equal(S, S2)
    assert lasy.planar.planar_almost_equal(Vh, Vh2)

    print('Truncated SVD')
    # chi_max=1 forces non-trivial truncation, which is needed to catch bugs in the
    # reported `err` (e.g. confusing it with the squared error).
    for svd_min, normalize_to, chi_max in [
        (1e-14, None, None),
        (1e-4, None, None),
        (1e-4, 2.7, None),
        (0.0, None, 1),
        (0.0, 2.7, 1),
    ]:
        U, S, Vh, err, renormalize = lasy.planar.planar_truncated_svd(
            T,
            codomain_cut=cod_cut,
            domain_cut=dom_cut,
            new_labels=['a', 'b', 'c', 'd'],
            new_leg_dual=new_leg_dual,
            normalize_to=normalize_to,
            svd_min=svd_min,
            chi_max=chi_max,
        )
        U.test_sanity()
        S.test_sanity()
        Vh.test_sanity()
        assert U.num_codomain_legs == cod_cut
        assert U.num_domain_legs == dom_cut + 1
        assert U.labels == [*T.labels.as_list()[:cod_cut], 'a', *T.labels.as_list()[T.num_legs - dom_cut :]]
        assert S.labels == ['b', 'c']
        assert Vh.num_codomain_legs == 1 + (cod - cod_cut)
        assert Vh.num_domain_legs == dom - dom_cut
        assert Vh.labels == ['d', *T.labels.as_list()[cod_cut : T.num_legs - dom_cut]]

        norm_T = lasy.norm(T).to_numpy()
        lambda_kept = S / norm_T if normalize_to is None else (renormalize / norm_T) * S
        npt.assert_almost_equal(err**2, 1 - lasy.norm(lambda_kept).to_numpy() ** 2)
        T_approx = renormalize * lasy.planar_contraction(lasy.planar_contraction(U, S, 'a', 'b'), Vh, 'c', 'd')
        T_approx = lasy.planar_permute_legs(T_approx, codomain=T.codomain_labels, domain=T.domain_labels)
        distance = lasy.norm(T.as_SymmetricTensor() - T_approx).to_numpy()
        npt.assert_almost_equal(err, distance / norm_T)

        # check isometric properties
        eye = lasy.SymmetricTensor.from_eye(S.domain, backend=T.backend, labels=['a*', 'a'])
        assert lasy.planar.planar_almost_equal(
            lasy.planar_contraction(U, U.hc, contr_U, [f'{l}*' for l in contr_U]), eye
        )
        eye.labels.set(['d', 'd*'])
        assert lasy.planar.planar_almost_equal(
            lasy.planar_contraction(Vh, Vh.hc, contr_Vh, [f'{l}*' for l in contr_Vh]), eye
        )

        # compare to non-planar result
        U2, S2, Vh2, _, _ = lasy.truncated_svd(
            T2,
            new_labels=['a', 'b', 'c', 'd'],
            new_leg_dual=new_leg_dual,
            normalize_to=normalize_to,
            svd_min=svd_min,
            chi_max=chi_max,
        )
        U2.test_sanity()
        S2.test_sanity()
        Vh2.test_sanity()
        assert lasy.planar.planar_almost_equal(U, U2)
        assert lasy.almost_equal(S, S2)
        assert lasy.planar.planar_almost_equal(Vh, Vh2)


@pytest.mark.parametrize('symmetry', [no_symmetry, u1_symmetry, fibonacci_anyon_category])
def test_PlanarDiagram(symmetry, np_random):
    # ===========================================
    # define a diagram
    # ===========================================
    density_matrix_mixing_left = lasy.PlanarDiagram(
        tensors='Lp[vR*, wR, vR], Lp_hc[vR*, wR*, vR], W[wL, p, wR, p*], W_hc[p, wR*, p*, wL*], '
        'mixL[wL, wL*], theta[vL, p0, p1, vR], theta_hc[vR*, p1*, p0*, vL*]',
        definition='Lp:vR @ theta:vL, Lp:wR @ W:wL, Lp:vR* -> vL, '
        'theta:p0 @ W:p*, theta:p1 @ theta_hc:p1*, theta:vR @ theta_hc:vR*, '
        'W:p -> p, W:wR @ mixL:wL, '
        'Lp_hc:vR -> vL*, Lp_hc:wR* @ W_hc:wL*, Lp_hc:vR* @ theta_hc:vL*, '
        'W_hc:p @ theta_hc:p0*, W_hc:p* -> p*, W_hc:wR* @ mixL:wL*',
        dims=dict(
            chi=['vR', 'vL', 'vR*', 'vL*'], w=['wL', 'wR', 'wL*', 'wR*'], d=['p', 'p*', 'p0', 'p0*', 'p1', 'p1*']
        ),
    )
    r"""Planar diagram arising when mixing the left site

        |    .---theta*---.
        |    |   |    \   |
        |   LP*--W0*-  \  |
        |    |   |   \  | |
        |          mixL | |
        |    |   |   /  | |
        |   LP---W0--  /  |
        |    |   |    /   |
        |    .---theta----.
    """

    # ===========================================
    # create example tensors
    # ===========================================
    theta = lasy.testing.random_tensor(
        symmetry, codomain=2, domain=2, labels=['vL', 'p0', 'p1', 'vR'], np_random=np_random
    )
    p0 = theta.get_leg('p0')
    vL = theta.get_leg('vL')
    Lp = lasy.testing.random_tensor(symmetry, codomain=[vL], domain=[vL, None], labels=['vR*', 'wR', 'vR'])
    wR = Lp.get_leg('wR')
    W = lasy.testing.random_tensor(symmetry, codomain=[p0, wR], domain=[wR, p0], labels=['p', 'wR', 'p*', 'wL'])
    mixL = lasy.testing.random_tensor(symmetry, codomain=[wR], domain=[wR], labels=['wL*', 'wL'])

    # ===========================================
    # evaluate the diagram
    # ===========================================
    res = density_matrix_mixing_left(Lp=Lp, Lp_hc=Lp.hc, W=W, W_hc=W.hc, mixL=mixL, theta=theta, theta_hc=theta.hc)
    res.test_sanity()
    assert res.labels == ['vL', 'p', 'p*', 'vL*'], 'if cyclical need to redesign test. otherwise wrong!'
    assert res.num_codomain_legs == 4, 'if this fails, just need to redesign tests'

    # ===========================================
    # compare to manual contraction, using planar routines
    # ===========================================
    expect1 = lasy.planar.planar_contraction(theta, theta.hc, ['p1', 'vR'], ['p1*', 'vR*'])
    expect1 = lasy.planar.planar_contraction(expect1, Lp, 'vL', 'vR')
    expect1 = lasy.planar.planar_contraction(expect1, W, ['wR', 'p0'], ['wL', 'p*'])
    expect1 = lasy.planar.planar_contraction(expect1, mixL, 'wR', 'wL')
    expect1 = lasy.planar.planar_contraction(expect1, W.hc, ['p0*', 'wL*'], ['p', 'wR*'])
    expect1 = lasy.planar.planar_contraction(expect1, Lp.hc, ['vL*', 'wL*'], ['vR*', 'wR*'])
    expect1.labels.update({'vR*': 'vL', 'vR': 'vL*'})
    expect1 = lasy.planar.planar_permute_legs(expect1, codomain=['p*', 'vL*'])
    assert expect1.labels == ['p*', 'vL*', 'vL', 'p']
    expect1.test_sanity()
    assert lasy.planar.planar_almost_equal(res, expect1)

    # ===========================================
    # compare to manual contraction, using general (not planar) routines
    # ===========================================
    assert theta.codomain_labels == ['vL', 'p0']
    assert theta.domain_labels == ['vR', 'p1']
    theta_bent = lasy.permute_legs(theta, ['p1', 'vR'], ['p0', 'vL'], bend_right=[True, True, False, False])
    expect2 = lasy.compose(theta_bent.hc, theta_bent)
    expect2 = lasy.compose(
        lasy.permute_legs(expect2, ['p0', 'p0*', 'vL*'], ['vL'], bend_right=[None, None, None, False]),
        lasy.permute_legs(Lp, ['vR'], ['wR', 'vR*'], bend_right=[True, None, False]),
    )
    expect2 = lasy.compose(
        lasy.permute_legs(expect2, ['p0*', 'vL*', 'vR*'], ['p0', 'wR'], bend_right=[False, None, None, True, None]),
        lasy.permute_legs(W, ['p*', 'wL'], ['wR', 'p'], bend_right=[True, True, False, False]),
    )
    expect2 = lasy.compose(
        lasy.permute_legs(expect2, ['p0*', 'vL*', 'vR*', 'p'], ['wR'], bend_right=[None, None, None, True, None]),
        lasy.transpose(mixL),
    )
    expect2 = lasy.compose(
        lasy.permute_legs(expect2, ['vL*', 'vR*', 'p'], ['p0*', 'wL*'], bend_right=[False, None, None, None, None]),
        lasy.permute_legs(W.hc, ['p', 'wR*'], ['wL*', 'p*'], bend_right=[False, None, True, None]),
    )
    expect2 = lasy.compose(
        lasy.permute_legs(expect2, ['vR*', 'p', 'p*'], ['vL*', 'wL*'], bend_right=[False, None, None, True, None]),
        Lp.hc,
    )
    expect2.labels.update({'vR*': 'vL', 'vR': 'vL*'})
    expect2 = lasy.permute_legs(expect2, ['p*', 'vL*'], ['p', 'vL'], bend_right=[False, False, None, True])
    expect2.test_sanity()
    assert lasy.planar.planar_almost_equal(res, expect2)


@pytest.mark.parametrize('symmetry', [no_symmetry, u1_symmetry, fibonacci_anyon_category])
def test_PlanarDiagram_add_remove_tensor(symmetry, np_random):
    # ===========================================
    # define a diagram
    # ===========================================
    partial_diagram = lasy.PlanarDiagram(
        tensors='T1[vL, vR], T2[vL, vR], T3[vL, w, vR]',
        definition=[
            ('T1', 'vL', 'T2', 'vR'),
            ('T1', 'vR', None, 'vR'),
            ('T2', 'vL', 'T3', 'vL'),
            ('T3', 'vR', None, 'vL'),
            ('T3', 'w', None, 'w1'),
        ],
        dims=dict(chi=['vR', 'vL'], w=['w']),
    )
    full_diagram = lasy.PlanarDiagram(
        tensors='T1[vL, vR], T2[vL, vR], T3[vL, w, vR], T4[vL, w, vR]',
        definition='T1:vL @ T2:vR, T1:vR @ T4:vR, T2:vL @ T3:vL, T3:vR @ T4:vL, T3:w -> w1, T4:w -> w2',
        dims=dict(chi=['vR', 'vL'], w=['w']),
    )
    r"""Random planar diagram; T4 is used to test adding and removal

        |   .--T1--.
        |   |      |
        |  T2      T4-
        |   |      |
        |   .--T3--.
        |      |
    """

    # ===========================================
    # create example tensors
    # ===========================================
    T1 = lasy.testing.random_tensor(symmetry, codomain=2, labels=['vL', 'vR'], np_random=np_random)
    T1vL = T1.get_leg('vL')
    T1vR = T1.get_leg('vR')
    T2 = lasy.testing.random_tensor(symmetry, codomain=[None, T1vL.dual], labels=['vL', 'vR'], np_random=np_random)
    T2vL = T2.get_leg('vL')
    T3 = lasy.testing.random_tensor(
        symmetry, codomain=[T2vL.dual, None, None], labels=['vL', 'w', 'vR'], np_random=np_random
    )
    T3vR = T3.get_leg('vR')
    T4 = lasy.testing.random_tensor(
        symmetry, codomain=[T1vR.dual, T3vR.dual, None], labels=['vR', 'vL', 'w'], np_random=np_random
    )

    # ===========================================
    # evaluate the diagrams
    # ===========================================
    partial_res = partial_diagram(T1=T1, T2=T2, T3=T3)
    partial_res.test_sanity()
    assert partial_res.labels == ['w1', 'vL', 'vR']
    assert partial_res.num_codomain_legs == 3

    full_res = full_diagram(T1=T1, T2=T2, T3=T3, T4=T4)
    full_res.test_sanity()
    assert full_res.labels == ['w1', 'w2']
    assert full_res.num_codomain_legs == 2

    # ===========================================
    # transform between the diagrams
    # ===========================================
    full_diagram2 = partial_diagram.add_tensor(
        tensor={'T4': lasy.planar.TensorPlaceholder(['vR', 'vL', 'w'], ['chi', 'chi', 'w'])},
        extra_definition=[('T4', 'vL', 'T3', 'vR'), ('T1', 'vR', 'T4', 'vR'), ('T4', 'w', None, 'w2')],
    )
    full_res2 = full_diagram2(T1=T1, T2=T2, T3=T3, T4=T4)
    full_res2.test_sanity()
    full_res2 = lasy.planar.planar_permute_legs(full_res2, codomain=['w2'])
    assert full_res2.labels == ['w2', 'w1']
    assert full_res2.num_codomain_legs == 1
    assert lasy.planar.planar_almost_equal(full_res, full_res2)

    partial_diagram2 = full_diagram.remove_tensor(
        'T4', extra_definition=[('T3', 'vR', None, 'vL'), ('T1', 'vR', None, 'vR')]
    )
    partial_res2 = partial_diagram2(T1=T1, T2=T2, T3=T3)
    partial_res2.test_sanity()
    assert partial_res2.labels == ['w1', 'vL', 'vR']
    assert partial_res2.num_codomain_legs == 3
    assert lasy.almost_equal(partial_res, partial_res2)


@pytest.mark.parametrize('symmetry', [no_symmetry, u1_symmetry, fibonacci_anyon_category])
def test_PlanarDiagram_contraction_orders(symmetry, np_random):
    # ===========================================
    # define a diagram
    # ===========================================
    diagram1 = lasy.PlanarDiagram(
        tensors='T1[vL, vR], T2[vL, vR], T3[vL, w, vR], T4[vL, w, vR]',
        definition=[
            ('T1', 'vL', 'T2', 'vR'),
            ('T1', 'vR', 'T4', 'vR'),
            ('T2', 'vL', 'T3', 'vL'),
            ('T3', 'vR', 'T4', 'vL'),
            ('T3', 'w', None, 'w1'),
            ('T4', 'w', None, 'w2'),
        ],
        dims=dict(chi=['vR', 'vL'], w=['w']),
    )
    tensors = {
        'T1': lasy.planar.TensorPlaceholder(['vL', 'vR'], ['chi', 'chi']),
        'T2': lasy.planar.TensorPlaceholder(['vR', 'vL'], ['chi', 'chi']),
        'T3': lasy.planar.TensorPlaceholder(['vL', 'w', 'vR'], ['chi', 'w', 'chi']),
        'T4': lasy.planar.TensorPlaceholder(['vR', 'vL', 'w'], ['chi', 'chi', 'w']),
    }
    diagram2 = lasy.PlanarDiagram(
        tensors=tensors,
        definition='T1:vL @ T2:vR, T1:vR @ T4:vR, T2:vL @ T3:vL, T3:vR @ T4:vL, T3:w -> w1, T4:w -> w2',
    )
    r"""Same diagram as for test_PlanarDiagram_add_remove_tensor;
    also test adding and removing tensors in the different ways

        |   .--T1--.
        |   |      |
        |  T2      T4-
        |   |      |
        |   .--T3--.
        |      |
    """

    # ===========================================
    # create example tensors
    # ===========================================
    T1 = lasy.testing.random_tensor(symmetry, codomain=2, labels=['vL', 'vR'], np_random=np_random)
    T1vL = T1.get_leg('vL')
    T1vR = T1.get_leg('vR')
    T2 = lasy.testing.random_tensor(symmetry, codomain=[None, T1vL.dual], labels=['vL', 'vR'], np_random=np_random)
    T2vL = T2.get_leg('vL')
    T3 = lasy.testing.random_tensor(
        symmetry, codomain=[T2vL.dual, None, None], labels=['vL', 'w', 'vR'], np_random=np_random
    )
    T3vR = T3.get_leg('vR')
    T4 = lasy.testing.random_tensor(
        symmetry, codomain=[T1vR.dual, T3vR.dual, None], labels=['vR', 'vL', 'w'], np_random=np_random
    )

    # ===========================================
    # evaluate the diagrams
    # ===========================================
    res1 = diagram1(T1=T1, T2=T2, T3=T3, T4=T4)
    res1.test_sanity()
    assert res1.labels == ['w1', 'w2']
    assert res1.num_codomain_legs == 2

    res2 = diagram2(T1=T1, T2=T2, T3=T3, T4=T4)
    res2.test_sanity()
    assert res2.labels == ['w1', 'w2']
    assert res2.num_codomain_legs == 2

    assert lasy.almost_equal(res1, res2)

    # ===========================================
    # remove T4
    # ===========================================
    partial_diagram1 = diagram1.remove_tensor('T4', extra_definition='T3:vR -> vL, T1:vR -> vR')
    partial_diagram2 = diagram2.remove_tensor(
        'T4', extra_definition=[('T3', 'vR', None, 'vL'), ('T1', 'vR', None, 'vR')]
    )
    partial_res1 = partial_diagram1(T1=T1, T2=T2, T3=T3)
    partial_res1.test_sanity()
    assert partial_res1.labels == ['w1', 'vL', 'vR']
    assert partial_res1.num_codomain_legs == 3

    partial_res2 = partial_diagram2(T1=T1, T2=T2, T3=T3)
    partial_res2.test_sanity()
    assert partial_res2.labels == ['w1', 'vL', 'vR']
    assert partial_res2.num_codomain_legs == 3

    assert lasy.almost_equal(partial_res1, partial_res2)

    # ===========================================
    # add T4 again
    # ===========================================
    diagram1_ = partial_diagram1.add_tensor(
        tensor={'T4': tensors['T4']},
        extra_definition='T4:vL @ T3:vR, T1:vR @ T4:vR, T4:w -> w2',
    )
    diagram2_ = partial_diagram2.add_tensor(
        tensor='T4[vL, w, vR]',
        extra_definition=[('T4', 'vL', 'T3', 'vR'), ('T1', 'vR', 'T4', 'vR'), ('T4', 'w', None, 'w2')],
        extra_dims=dict(chi=['vR', 'vL'], w=['w']),
    )
    res1_ = diagram1_(T1=T1, T2=T2, T3=T3, T4=T4)
    res1_.test_sanity()
    assert res1_.labels == ['w1', 'w2']
    assert res1_.num_codomain_legs == 2

    res2_ = diagram2_(T1=T1, T2=T2, T3=T3, T4=T4)
    res2_.test_sanity()
    assert res2_.labels == ['w1', 'w2']
    assert res2_.num_codomain_legs == 2

    assert lasy.almost_equal(res1_, res2_)
    assert lasy.planar.planar_almost_equal(res1_, res1)


@pytest.mark.parametrize('symmetry', [no_symmetry, u1_symmetry, fibonacci_anyon_category])
def test_PlanarDiagram_with_traces(symmetry, np_random):
    # ===========================================
    # define a diagram
    # ===========================================
    diagram = lasy.PlanarDiagram(
        tensors='T1[vL, vR], T2[w, vL, vR, w*], T3[vR, w1, w1*, w2, w2*, vL], T4[vL, w1, w2, w2*, w1*, vR]',
        definition='T1:vL @ T2:vR, T1:vR @ T3:vL, '
        'T2:w @ T2:w*, T2:vL @ T4:vL,'
        'T3:w1 @ T3:w1*, T3:w2 @ T3:w2*, T3:vR @ T4:vR, '
        'T4:w1 @ T4:w1*, T4:w2* @ T4:w2',
        dims=dict(chi=['vR', 'vL'], w=['w', 'w1', 'w2', 'w*', 'w1*', 'w2*']),
    )
    r"""Random planar diagram with multiple partial traces

        |    .---T1---.
        | .  |        |  .
        | |\ |        | /|
        | | T2        T3-.
        | |/ |        ||\
        | .  |        |.-.
        |    .---T4---.
        |       //\\
        |      /.--.\
        |     .------.
    """

    # ===========================================
    # create example tensors
    # ===========================================
    T1 = lasy.testing.random_tensor(symmetry, codomain=2, labels=['vL', 'vR'], np_random=np_random)
    T1vL = T1.get_leg('vL')
    T1vR = T1.get_leg('vR')
    traced_legs = [lasy.testing.random_ElementarySpace(symmetry, np_random=np_random) for _ in range(5)]
    T2 = lasy.testing.random_tensor(
        symmetry,
        codomain=[traced_legs[0], None, T1vL.dual, traced_legs[0].dual],
        labels=['w', 'vL', 'vR', 'w*'],
        np_random=np_random,
    )
    T3 = lasy.testing.random_tensor(
        symmetry,
        codomain=[None, traced_legs[1].dual, traced_legs[1], traced_legs[2]],
        domain=[T1vR, traced_legs[2]],
        labels=['vR', 'w1', 'w1*', 'w2', 'w2*', 'vL'],
        np_random=np_random,
    )
    T4vL = T2.get_leg('vL').dual
    T4vR = T3.get_leg('vR')
    T4 = lasy.testing.random_tensor(
        symmetry,
        codomain=[T4vL, traced_legs[3], traced_legs[4], traced_legs[4].dual],
        domain=[T4vR, traced_legs[3]],
        labels=['vL', 'w1', 'w2', 'w2*', 'w1*', 'vR'],
        np_random=np_random,
    )

    # ===========================================
    # evaluate the diagram
    # ===========================================
    res = diagram(T1=T1, T2=T2, T3=T3, T4=T4)
    assert isinstance(res, Scalar)
    res = res.to_numpy()

    # ===========================================
    # compare to manual contraction, using planar routines
    # ===========================================
    T2_traced = lasy.planar.planar_partial_trace(T2, ['w', 'w*'])
    T3_traced = lasy.planar.planar_partial_trace(T3, ['w1', 'w1*'], ['w2', 'w2*'])
    T4_traced = lasy.planar.planar_partial_trace(T4, ['w1', 'w1*'], ['w2', 'w2*'])

    expect1 = lasy.planar.planar_contraction(T1, T2_traced, ['vL'], ['vR'])
    expect1 = lasy.planar.planar_contraction(expect1, T3_traced, ['vR'], ['vL'])
    expect1 = lasy.planar.planar_contraction(expect1, T4_traced, ['vL', 'vR'], ['vL', 'vR'])
    assert isinstance(expect1, Scalar)
    npt.assert_almost_equal(expect1.to_numpy(), res)

    # ===========================================
    # compare to manual contraction, using general (not planar) routines
    # ===========================================
    T2_traced_ = lasy.planar.partial_trace(T2, ['w', 'w*'], levels=[3, 1, 2, 3])
    T3_traced_ = lasy.planar.partial_trace(T3, ['w1', 'w1*'], ['w2', 'w2*'])
    T4_traced_ = lasy.planar.partial_trace(T4, ['w1', 'w1*'], ['w2', 'w2*'], levels=[None, 1, 2, 2, 1, None])
    assert lasy.almost_equal(T2_traced, T2_traced_)
    assert lasy.almost_equal(T3_traced, T3_traced_)
    assert lasy.almost_equal(T4_traced, T4_traced_)

    expect2 = lasy.permute_legs(T1, codomain=['vL'], domain=['vR'], bend_right=True)
    expect2 = lasy.compose(lasy.permute_legs(T2_traced_, codomain=['vL'], domain=['vR'], bend_right=True), expect2)
    expect2 = lasy.compose(
        expect2, lasy.permute_legs(T3_traced_, codomain=['vL'], domain=['vR'], bend_right=[True, False])
    )
    expect2 = lasy.compose(
        lasy.permute_legs(expect2, domain=['vL', 'vR'], bend_right=False),
        lasy.permute_legs(T4_traced_, codomain=['vL', 'vR'], bend_right=True),
    )
    assert isinstance(expect2, Scalar)
    npt.assert_almost_equal(expect2.to_numpy(), res)


def test_PlanarDiagram_verify_diagram():
    # only a single tensor
    _ = lasy.PlanarDiagram(
        tensors='T1[l1, l2, l3]',
        definition='T1:l2 @ T1:l1, T1:l3 -> l3',
        dims=dict(chi=['l1', 'l2', 'l3']),
    )
    with pytest.raises(ValueError):
        _.remove_tensor('T1')

    # disconnected tensors
    with pytest.raises(ValueError, match='The planar diagram is disconnected'):
        _ = lasy.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2], T3[l1, l2]',
            definition='T1:l1 @ T2:l1, T1:l2 -> l2, T1:l3 @ T2:l2, T3:l1 @ T3:l2',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )
    with pytest.raises(ValueError, match='The planar diagram is disconnected'):
        _ = lasy.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2]',
            definition='T1:l1 @ T1:l3, T1:l2 -> l2, T2:l1 @ T2:l2',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )

    # add a disconnected tensor
    diagram = lasy.PlanarDiagram(
        tensors='T1[l1, l2, l3]',
        definition='T1:l2 @ T1:l1, T1:l3 -> l3',
        dims=dict(chi=['l1', 'l2', 'l3']),
    )
    with pytest.raises(ValueError, match='The planar diagram is disconnected'):
        _ = diagram.add_tensor('T2[l1, l2]', 'T2:l1 -> l1, T2:l2 -> l2')

    # remove a tensor to make the diagram disconnected
    diagram = lasy.PlanarDiagram(
        tensors='T1[l1, l2], T2[l1, l2, l3], T3[l1, l2]',
        definition='T1:l1 -> l1, T1:l2 @ T2:l1, T2:l2 @ T3:l1, T2:l3 -> l3, T3:l2 -> l2',
        dims=dict(chi=['l1', 'l2', 'l3']),
    )
    with pytest.raises(ValueError):
        _ = diagram.remove_tensor('T2')

    # open leg = contracted leg
    with pytest.raises(ValueError, match='Number of contracted and open legs does not match the total number of legs'):
        _ = lasy.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2]',
            definition='T1:l1 @ T2:l1, T1:l2 -> l2, T1:l3 @ T2:l2, T1:l1 -> l1',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )
    with pytest.raises(ValueError, match='Inconsistent open legs'):
        _ = lasy.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2]',
            definition='T1:l1 @ T2:l1, T1:l3 @ T2:l2, T1:l1 -> l1',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )

    # contract leg twice
    with pytest.raises(ValueError, match='duplicate legs'):
        _ = lasy.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2]',
            definition='T1:l1 -> l1, T1:l2 @ T2:l2, T1:l3 @ T2:l2',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )

    # unspecified action on leg
    with pytest.raises(ValueError, match='Number of contracted and open legs does not match the total number of legs'):
        _ = lasy.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2]',
            definition='T1:l1 @ T2:l1, T1:l3 @ T2:l2',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )

    # unknown tensor
    with pytest.raises(ValueError, match='No tensor with name T2'):
        _ = lasy.PlanarDiagram(
            tensors='T1[l1, l2, l3]',
            definition='T1:l2 @ T2:l1, T1:l1 -> l1, T1:l3 -> l3',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )

    # unknown label
    with pytest.raises(ValueError, match='Tensor T1 has no leg l4'):
        _ = lasy.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2]',
            definition='T1:l1 @ T2:l1, T1:l3 @ T2:l2, T1:l4 -> l4, T1:l3 -> l3',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )

    # non-planar due to partial trace
    with pytest.raises(ValueError, match='Not a planar trace'):
        _ = lasy.PlanarDiagram(
            tensors='T1[l1, l2, l3, l4], T2[l1, l2]',
            definition='T1:l2 @ T1:l4, T1:l1 @ T2:l1, T1:l3 -> l3, T2:l2 -> l2',
            dims=dict(chi=['l1', 'l2', 'l3', 'l4']),
        )

    # non-planar due to contractions
    with pytest.raises(ValueError, match='Not a planar bipartition'):
        _ = lasy.PlanarDiagram(
            tensors='T1[l1, l2, l3, l4], T2[l1, l2]',
            definition='T1:l1 @ T2:l1, T1:l3 @ T2:l2, T1:l3 -> l3, T1:l4 -> l4',
            dims=dict(chi=['l1', 'l2', 'l3', 'l4']),
        )


@pytest.mark.parametrize('symmetry', [no_symmetry, u1_symmetry, fibonacci_anyon_category])
def test_PlanarLinearOperator(symmetry, np_random):
    # ===========================================
    # define an operator
    # ===========================================

    class TwoSiteEffectiveH(lasy.PlanarLinearOperator):
        r"""Effective Hamiltonian during Two-site
        The operator is given by the following network::

            |        .---       ---.
            |        |    |   |    |
            |       LP----W0--W1---RP
            |        |    |   |    |
            |        .---       ---.

        and acts on two-site wavefunctions ``theta`` as::

            |        .---       ---.
            |        |    |   |    |
            |       LP----W0--W1---RP
            |        |    |   |    |
            |        .--- theta ---.
        """

        op_diagram = lasy.PlanarDiagram(
            tensors='Lp[vR*, wR, vR], W0[wL, p, wR, p*], W1[wL, p, wR, p*], Rp[vL*, vL, wL]',
            definition='Lp:vR* -> vL, Lp:wR @ W0:wL, Lp:vR -> vL*, '
            'W0:p -> p0, W0:wR @ W1:wL, W0:p* -> p0*, '
            'W1:p -> p1, W1:wR @ Rp:wL, W1:p* -> p1*, '
            'Rp:vL* -> vR, Rp:vL -> vR*',
            dims=dict(chi=['vR', 'vR*', 'vL', 'vL*'], w=['wL', 'wR'], d=['p', 'p*']),
        )
        matvec_diagram = op_diagram.add_tensor(
            tensor='theta[vL, p0, p1, vR]',
            extra_definition='theta:vL @ Lp:vR, theta:p0 @ W0:p*, theta:p1 @ W1:p*, theta:vR @ Rp:vL',
            extra_dims=dict(chi=['vL', 'vR'], d=['p0', 'p1']),
        )

        def __init__(self, Lp, W0, W1, Rp):
            lasy.planar.PlanarLinearOperator.__init__(
                self,
                op_diagram=self.op_diagram,
                matvec_diagram=self.matvec_diagram,
                op_tensors=dict(Lp=Lp, W0=W0, W1=W1, Rp=Rp),
                vec_name='theta',
            )

    # ===========================================
    # create example tensors
    # ===========================================
    # Fib FusionTree with several multi-sector legs makes to_tensor / matvec / planar
    # rebuild each take ~3s under the default RNG seed. Single-sector legs stay cheap
    # while still exercising non-abelian planar contractions.
    max_mult = 1 if not symmetry.is_abelian else 2
    max_blocks = 1 if not symmetry.is_abelian else 2
    theta = lasy.testing.random_tensor(
        symmetry,
        4,
        labels=['vL', 'p0', 'p1', 'vR'],
        max_multiplicity=max_mult,
        max_blocks=max_blocks,
        np_random=np_random,
    )
    vL, p0, p1, vR = theta.legs
    Lp = lasy.testing.random_tensor(
        symmetry,
        [vL, None, vL.dual],
        labels=['vR*', 'wR', 'vR'],
        max_multiplicity=max_mult,
        max_blocks=max_blocks,
        np_random=np_random,
    )
    W0 = lasy.testing.random_tensor(
        symmetry,
        [p0, None, p0.dual, Lp.get_leg('wR').dual],
        labels=['p', 'wR', 'p*', 'wL'],
        max_multiplicity=max_mult,
        max_blocks=max_blocks,
        np_random=np_random,
    )
    W1 = lasy.testing.random_tensor(
        symmetry,
        [p1, None, p1.dual, W0.get_leg('wR').dual],
        labels=['p', 'wR', 'p*', 'wL'],
        max_multiplicity=max_mult,
        max_blocks=max_blocks,
        np_random=np_random,
    )
    Rp = lasy.testing.random_tensor(
        symmetry,
        [vR, vR.dual, W1.get_leg('wR').dual],
        labels=['vL*', 'vL', 'wL'],
        max_multiplicity=max_mult,
        max_blocks=max_blocks,
        np_random=np_random,
    )

    # ===========================================
    # create an op instance, call to_tensor and matvec
    # ===========================================

    H = TwoSiteEffectiveH(Lp=Lp, W0=W0, W1=W1, Rp=Rp)

    op = H.to_tensor()
    op.test_sanity()
    # get to the correct cyclic permutation
    op = lasy.planar.planar_permute_legs(op, codomain=['vL', 'p0', 'p1', 'vR'])
    assert op.codomain_labels == ['vL', 'p0', 'p1', 'vR']
    assert op.domain_labels == ['vL*', 'p0*', 'p1*', 'vR*']

    H_theta = H.matvec(theta)
    H_theta.test_sanity()
    assert H_theta.codomain_labels == ['vL', 'p0', 'p1', 'vR']
    assert H_theta.domain_labels == []

    # ===========================================
    # compare to manual contraction, using planar routines
    # ===========================================
    op_1 = lasy.planar.planar_contraction(
        Lp, W0, 'wR', 'wL', relabel1={'vR*': 'vL', 'vR': 'vL*'}, relabel2={'p': 'p0', 'p*': 'p0*'}
    )
    op_1 = lasy.planar.planar_contraction(op_1, W1, 'wR', 'wL', relabel2={'p': 'p1', 'p*': 'p1*'})
    op_1 = lasy.planar.planar_contraction(op_1, Rp, 'wR', 'wL', relabel2={'vL*': 'vR', 'vL': 'vR*'})
    op_1 = lasy.planar.planar_permute_legs(op_1, codomain=['vL', 'p0', 'p1', 'vR'])
    assert lasy.almost_equal(op_1, op)

    H_theta_1 = lasy.compose(op_1, theta)
    assert lasy.almost_equal(H_theta_1, H_theta)

    # ===========================================
    # compare to manual contraction, using general (not planar) routines
    # ===========================================
    op_2 = lasy.compose(
        lasy.permute_legs(Rp, ['vL*', 'vL'], ['wL'], bend_right=[None, None, True]),
        lasy.permute_legs(W1, ['wR'], ['p', 'wL', 'p*'], bend_right=[False, None, True, True]),
        relabel1={'vL*': 'vR', 'vL': 'vR*'},
        relabel2={'p': 'p1', 'p*': 'p1*'},
    )
    op_2 = lasy.compose(
        lasy.permute_legs(op_2, ['p1', 'vR', 'vR*', 'p1*'], ['wL'], bend_right=[None, None, True, None, False]),
        lasy.permute_legs(W0, ['wR'], ['p', 'wL', 'p*'], bend_right=[False, None, True, True]),
        relabel2={'p': 'p0', 'p*': 'p0*'},
    )
    bend_right = [None] * 4 + [True, None, False]
    op_2 = lasy.compose(
        lasy.permute_legs(op_2, ['p0', 'p1', 'vR', 'vR*', 'p1*', 'p0*'], ['wL'], bend_right=bend_right),
        lasy.permute_legs(Lp, ['wR'], ['vR*', 'vR'], bend_right=[False, None, True]),
        relabel2={'vR*': 'vL', 'vR': 'vL*'},
    )
    bend_right = [None] * 3 + [True] * 3 + [None, False]
    op_2 = lasy.permute_legs(op_2, ['vL', 'p0', 'p1', 'vR'], ['vL*', 'p0*', 'p1*', 'vR*'], bend_right=bend_right)
    assert lasy.almost_equal(op_2, op)

    H_theta_2 = lasy.compose(op_2, theta)
    assert lasy.almost_equal(H_theta_2, H_theta)


@pytest.mark.parametrize('symmetry', [no_symmetry, u1_symmetry])
def test_PlanarDiagram_single_charged_tensor(symmetry, np_random):
    diagram = lasy.PlanarDiagram(
        tensors='theta[vL, p, vR], op[p, p*]',
        definition='theta:p @ op:p*, theta:vL -> vL, theta:vR -> vR, op:p -> p',
        dims=dict(chi=['vL', 'vR'], d=['p', 'p*']),
    )
    assert set(diagram.open_legs) == {'vL', 'vR', 'p'}

    theta = lasy.testing.random_tensor(symmetry, codomain=2, domain=1, labels=['vL', 'p', 'vR'], np_random=np_random)
    p = theta.get_leg('p')
    op = lasy.testing.random_tensor(
        symmetry, codomain=[p], domain=[p], labels=['p', 'p*'], cls=lasy.ChargedTensor, np_random=np_random
    )

    res = diagram(theta=theta, op=op)
    res.test_sanity()
    assert isinstance(res, lasy.ChargedTensor)
    assert set(res.labels) == {'vL', 'vR', 'p'}
    expect = lasy.tdot(theta, op, ['p'], ['p*'])
    assert lasy.almost_equal(res, expect)

    op_sym = lasy.testing.random_tensor(symmetry, codomain=[p], domain=[p], labels=['p', 'p*'], np_random=np_random)
    res_sym = diagram(theta=theta, op=op_sym)
    res_sym.test_sanity()
    assert isinstance(res_sym, lasy.SymmetricTensor)
    assert set(res_sym.labels) == {'vL', 'vR', 'p'}
    expect_sym = lasy.tdot(theta, op_sym, ['p'], ['p*'])
    # Diagram open-leg order may differ from tdot; compare dense blocks after a shared permute.
    order = ['vL', 'vR', 'p']
    res_np = lasy.permute_legs(res_sym, order, bend_right=True).to_numpy(understood_braiding=True)
    exp_np = lasy.permute_legs(expect_sym, order, bend_right=True).to_numpy(understood_braiding=True)
    npt.assert_almost_equal(res_np, exp_np)


@pytest.mark.parametrize('symmetry', [no_symmetry, u1_symmetry])
def test_PlanarDiagram_transfer_matrix_charged(symmetry, np_random):
    diagram = lasy.PlanarDiagram(
        tensors='LP[vR*, vR], ket[vL, p, vR], bra[vR*, p*, vL*]',
        definition='LP:vR @ ket:vL, ket:p @ bra:p*, LP:vR* @ bra:vL*, ket:vR -> vR, bra:vR* -> vR*',
        dims=dict(chi=['vR', 'vL', 'vR*', 'vL*'], d=['p', 'p*']),
        allow_multiple_charged_tensors=True,
    )
    assert set(diagram.open_legs) == {'vR', 'vR*'}

    ket = lasy.testing.random_tensor(
        symmetry, codomain=2, domain=1, labels=['vL', 'p', 'vR'], cls=lasy.ChargedTensor, np_random=np_random
    )
    bra = ket.hc
    LP = lasy.testing.random_tensor(
        symmetry,
        codomain=[bra.get_leg('vL*').dual],
        domain=[ket.get_leg('vL')],
        labels=['vR*', 'vR'],
        np_random=np_random,
    )

    with pytest.raises(NotImplementedError, match='state_tensor_product'):
        _ = diagram(LP=LP, ket=ket, bra=bra)

    ket_sym = lasy.testing.random_tensor(
        symmetry, codomain=ket.codomain, domain=ket.domain, labels=ket.labels, np_random=np_random
    )
    bra_sym = ket_sym.hc
    LP_sym = lasy.testing.random_tensor(
        symmetry,
        codomain=[bra_sym.get_leg('vL*').dual],
        domain=[ket_sym.get_leg('vL')],
        labels=['vR*', 'vR'],
        np_random=np_random,
    )
    res_sym = diagram(LP=LP_sym, ket=ket_sym, bra=bra_sym)
    res_sym.test_sanity()
    assert isinstance(res_sym, lasy.SymmetricTensor)
    expect_sym = lasy.planar.planar_contraction(LP_sym, ket_sym, 'vR', 'vL')
    expect_sym = lasy.planar.planar_contraction(expect_sym, bra_sym, ['p', 'vR*'], ['p*', 'vL*'])
    assert lasy.planar.planar_almost_equal(res_sym, expect_sym)

    ket_mixed = lasy.testing.random_tensor(
        symmetry, codomain=2, domain=1, labels=['vL', 'p', 'vR'], cls=lasy.ChargedTensor, np_random=np_random
    )
    bra_for_mixed = lasy.testing.random_tensor(
        symmetry, codomain=ket_mixed.codomain, domain=ket_mixed.domain, labels=ket_mixed.labels, np_random=np_random
    ).hc
    LP_mixed = lasy.testing.random_tensor(
        symmetry,
        codomain=[bra_for_mixed.get_leg('vL*').dual],
        domain=[ket_mixed.get_leg('vL')],
        labels=['vR*', 'vR'],
        np_random=np_random,
    )
    res_mixed = diagram(LP=LP_mixed, ket=ket_mixed, bra=bra_for_mixed)
    res_mixed.test_sanity()
    assert isinstance(res_mixed, lasy.ChargedTensor)
    assert set(res_mixed.labels) == {'vR', 'vR*'}


@pytest.mark.parametrize('symmetry', [no_symmetry, u1_symmetry])
def test_PlanarDiagram_two_open_charge_legs(symmetry, np_random):
    contiguous = lasy.PlanarDiagram(
        tensors='A[a, b], B[b, c]',
        definition='A:b @ B:b, A:a -> a, B:c -> c',
        dims=dict(chi=['a', 'b', 'c']),
    )
    assert set(contiguous.open_legs) == {'a', 'c'}

    A_sym = lasy.testing.random_tensor(symmetry, codomain=3, labels=['a', 'b', 'charge'], np_random=np_random)
    A = lasy.HiddenLegTensor(A_sym, ['charge'])
    B_sym = lasy.testing.random_tensor(
        symmetry,
        codomain=[A_sym.get_leg('b').dual, None],
        domain=[None],
        labels=['b', 'c', 'charge*'],
        np_random=np_random,
    )
    B = lasy.HiddenLegTensor(B_sym, ['charge*'])
    res = contiguous(A=A, B=B)
    res.test_sanity()
    assert isinstance(res, lasy.SymmetricTensor)
    assert set(res.labels) == {'a', 'c'}
    expect = lasy.tdot(A, B, ['b'], ['b'])
    assert lasy.planar.planar_almost_equal(res, expect)


def test_PlanarDiagram_charged_flag_and_planarity():
    with pytest.raises(ValueError, match='!'):
        _ = lasy.PlanarDiagram(
            tensors='A[a, !], B[a, !]',
            definition='A:a @ B:a',
            dims=dict(chi=['a']),
        )

    with pytest.raises(ValueError, match='!'):
        _ = lasy.PlanarDiagram(
            tensors='A[a, !]',
            definition='A:a -> a',
            dims=dict(chi=['a']),
        )

    # Public-leg diagram stays planar even when tensors carry hidden legs that would cross.
    diagram = lasy.PlanarDiagram(
        tensors='A[a, x, y], B[a, x, y]',
        definition='A:a @ B:a, A:x -> x1, A:y -> y1, B:x -> x2, B:y -> y2',
        dims=dict(chi=['a', 'x', 'y']),
    )
    assert set(diagram.open_legs) == {'x1', 'x2', 'y1', 'y2'}

    diagram2 = lasy.PlanarDiagram(
        tensors='A[a, b]',
        definition='A:a -> a, A:b -> b',
        dims=dict(chi=['a', 'b']),
    )
    added = diagram2.add_tensor(
        tensor='B[b, c]',
        extra_definition='B:b @ A:b, B:c -> c',
        extra_dims=dict(chi=['b', 'c']),
    )
    assert set(added.open_legs) == {'a', 'c'}

    removed = added.remove_tensor(name='B', extra_definition='A:b -> b', order='definition')
    assert set(removed.open_legs) == {'a', 'b'}


_hidden_planar_diagram_cases = [
    (no_symmetry, 'no_symmetry'),
    (u1_symmetry, 'abelian'),
    (u1_symmetry, 'fusion_tree'),
    (fermion_parity, 'fusion_tree'),
    (fibonacci_anyon_category, 'fusion_tree'),
]


@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (no_symmetry, 'no_symmetry'),
        (u1_symmetry, 'abelian'),
        (u1_symmetry, 'fusion_tree'),
        (fermion_parity, 'fusion_tree'),
    ],
)
def test_PlanarDiagram_hidden_implicit_no_braid(symmetry, backend, np_random):
    # Hidden legs live on a second plane and may pass public legs. Contracting two
    # public legs with a hidden dual pair sitting between them is planar, even though
    # the same contraction would braid if that hidden leg were public.
    backend = lasy.get_backend(backend, 'numpy')
    diagram = lasy.PlanarDiagram(
        tensors='A[a, b, c], B[b, c]',
        definition='A:b @ B:b, A:c @ B:c, A:a -> a',
        dims=dict(chi=['a', 'b', 'c']),
    )
    A_sym = lasy.testing.random_tensor(
        symmetry,
        4,
        0,
        labels=['a', 'b', 'h', 'c'],
        backend=backend,
        np_random=np_random,
    )
    B_sym = lasy.testing.random_tensor(
        symmetry,
        codomain=[A_sym.get_leg('b').dual, A_sym.get_leg('c').dual],
        domain=[A_sym.get_leg('h')],
        labels=[['b', 'c'], ['h*']],
        backend=backend,
        np_random=np_random,
    )
    A = lasy.HiddenLegTensor(A_sym, ['h'])
    B = lasy.HiddenLegTensor(B_sym, ['h*'])
    assert A.labels == ['a', 'b', '!h', 'c']

    res = diagram(A=A, B=B)
    res.test_sanity()
    assert isinstance(res, lasy.SymmetricTensor)
    assert set(res.labels) == {'a'}
    expect = lasy.tdot(A, B, ['b', 'c'], ['b', 'c'])
    assert lasy.planar.planar_almost_equal(res, expect)


@pytest.mark.parametrize('symmetry, backend', _hidden_planar_diagram_cases)
def test_PlanarDiagram_hidden_implicit_contiguous_extras(symmetry, backend, np_random):
    # Two extra hidden legs form a contiguous arc on the hidden circle, so they
    # can be contracted implicitly without braiding past unmatched hidden legs.
    backend = lasy.get_backend(backend, 'numpy')
    diagram = lasy.PlanarDiagram(
        tensors='A[a, b], B[b, c]',
        definition='A:b @ B:b, A:a -> a, B:c -> c',
        dims=dict(chi=['a', 'b', 'c']),
    )
    A_sym = lasy.testing.random_tensor(
        symmetry,
        4,
        0,
        labels=['a', 'h1', 'h2', 'b'],
        backend=backend,
        np_random=np_random,
    )
    B_sym = lasy.testing.random_tensor(
        symmetry,
        codomain=[A_sym.get_leg('b').dual, None],
        domain=[A_sym.get_leg('h1'), A_sym.get_leg('h2')],
        labels=[['b', 'c'], ['h1*', 'h2*']],
        backend=backend,
        np_random=np_random,
    )
    A = lasy.HiddenLegTensor(A_sym, ['h1', 'h2'])
    B = lasy.HiddenLegTensor(B_sym, ['h1*', 'h2*'])

    res = diagram(A=A, B=B)
    res.test_sanity()
    assert isinstance(res, lasy.SymmetricTensor)
    assert set(res.labels) == {'a', 'c'}
    try:
        expect = lasy.tdot(A, B, ['b'], ['b'])
    except (lasy.SymmetryError, lasy.BraidChiralityUnspecifiedError):  # fmt: skip
        # Fusion-tree tdot may refuse the implicit extra permute; the diagram path
        # itself is still planar when extras form a contiguous hidden arc.
        return
    assert lasy.planar.planar_almost_equal(res, expect)


@pytest.mark.parametrize('symmetry, backend', _hidden_planar_diagram_cases)
def test_PlanarDiagram_hidden_implicit_would_braid(symmetry, backend, np_random):
    # Public diagram A[a,b] @ B[b,c] is planar. Implicit hidden contraction is not
    # when extras are not a contiguous arc on the hidden circle, or when extra
    # pairing is not the clockwise reverse of the other tensor's hidden order.
    backend = lasy.get_backend(backend, 'numpy')
    diagram = lasy.PlanarDiagram(
        tensors='A[a, b], B[b, c]',
        definition='A:b @ B:b, A:a -> a, B:c -> c',
        dims=dict(chi=['a', 'b', 'c']),
    )

    A_gap = lasy.testing.random_tensor(
        symmetry,
        6,
        0,
        labels=['a', 'h1', 'u1', 'h2', 'u2', 'b'],
        backend=backend,
        np_random=np_random,
    )
    B_gap = lasy.testing.random_tensor(
        symmetry,
        codomain=[A_gap.get_leg('b').dual, None],
        domain=[A_gap.get_leg('h2'), A_gap.get_leg('h1')],
        labels=[['b', 'c'], ['h2*', 'h1*']],
        backend=backend,
        np_random=np_random,
    )
    A = lasy.HiddenLegTensor(A_gap, ['h1', 'u1', 'h2', 'u2'])
    B = lasy.HiddenLegTensor(B_gap, ['h1*', 'h2*'])
    with pytest.raises(ValueError, match='hidden legs would have to be braided'):
        _ = diagram(A=A, B=B)

    A_perm = lasy.testing.random_tensor(
        symmetry,
        6,
        0,
        labels=['a', 'h1', 'h2', 'h3', 'h4', 'b'],
        backend=backend,
        np_random=np_random,
    )
    B_perm = lasy.testing.random_tensor(
        symmetry,
        codomain=[
            A_perm.get_leg('b').dual,
            None,
            A_perm.get_leg('h1').dual,
            A_perm.get_leg('h3').dual,
            A_perm.get_leg('h2').dual,
            A_perm.get_leg('h4').dual,
        ],
        labels=['b', 'c', 'h1*', 'h3*', 'h2*', 'h4*'],
        backend=backend,
        np_random=np_random,
    )
    A = lasy.HiddenLegTensor(A_perm, ['h1', 'h2', 'h3', 'h4'])
    B = lasy.HiddenLegTensor(B_perm, ['h1*', 'h2*', 'h3*', 'h4*'])
    with pytest.raises(ValueError, match='hidden legs would have to be braided'):
        _ = diagram(A=A, B=B)


@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (no_symmetry, 'no_symmetry'),
        (u1_symmetry, 'abelian'),
        (u1_symmetry, 'fusion_tree'),
        (fermion_parity, 'fusion_tree'),
        (fibonacci_anyon_category, 'fusion_tree'),
    ],
)
def test_issue_270(symmetry, backend, np_random):
    # this is the contraction of a right environment RP with B_ket-W-B_bra,
    # where the B_ket-W-B_bra are already contracted and W has no wR leg
    backend = lasy.get_backend(backend, 'numpy')
    A: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain=4,
        domain=1,
        labels=[['vL', 'wL', 'vL*', 'vR*'], ['vR']],
        backend=backend,
        np_random=np_random,
    )
    RP: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain=[A._as_domain_leg('vR*')],
        domain=[A._as_codomain_leg('vR')],
        labels=[['vL*'], ['vL']],
        backend=backend,
        np_random=np_random,
    )

    # this works
    A_ = lasy.planar.planar_permute_legs(A, domain=['vR', 'vR*'])
    RP_ = lasy.planar.planar_permute_legs(RP, codomain=['vL', 'vL*'])
    expect = lasy.compose(A_, RP_)

    # this does not work
    # NOTE there are currently two distinct ValueError: "Not a planar contraction" and "Inconsistent inner sector."
    res1 = lasy.planar_contraction(A, RP, ['vR', 'vR*'], ['vL', 'vL*'])
    assert lasy.planar.planar_almost_equal(res1, expect)
    res2 = lasy.planar_contraction(RP, A, ['vL', 'vL*'], ['vR', 'vR*'])
    assert lasy.planar.planar_almost_equal(res2, expect)


@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (no_symmetry, 'no_symmetry'),
        (u1_symmetry, 'abelian'),
        (u1_symmetry, 'fusion_tree'),
        (fermion_parity, 'fusion_tree'),
        (fibonacci_anyon_category, 'fusion_tree'),
    ],
)
def test_issue_270_2(symmetry, backend, np_random):
    backend = lasy.get_backend(backend, 'numpy')
    LP: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain=1,
        domain=1,
        labels=[['vR*'], ['vR']],
        backend=backend,
        np_random=np_random,
    )
    RP: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain=[LP._as_domain_leg('vR*')],
        domain=[LP._as_codomain_leg('vR')],
        labels=[['vL*'], ['vL']],
        backend=backend,
        np_random=np_random,
    )

    # this works
    LP_ = lasy.planar.planar_permute_legs(LP, domain=['vR', 'vR*'])
    RP_ = lasy.planar.planar_permute_legs(RP, codomain=['vL', 'vL*'])
    expect = lasy.compose(LP_, RP_).to_numpy()

    # this also works
    res1 = lasy.planar_contraction(LP, RP, ['vR', 'vR*'], ['vL', 'vL*']).to_numpy()
    npt.assert_almost_equal(res1, expect)
    res2 = lasy.planar_contraction(RP, LP, ['vL', 'vL*'], ['vR', 'vR*']).to_numpy()
    npt.assert_almost_equal(res2, expect)

    # this fails
    res3 = lasy.planar_contraction(LP_, RP_, ['vR', 'vR*'], ['vL', 'vL*']).to_numpy()
    npt.assert_almost_equal(res3, expect)
    res4 = lasy.planar_contraction(RP_, LP_, ['vL', 'vL*'], ['vR', 'vR*']).to_numpy()
    npt.assert_almost_equal(res4, expect)


@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (no_symmetry, 'no_symmetry'),
        (u1_symmetry, 'abelian'),
        (u1_symmetry, 'fusion_tree'),
        (fermion_parity, 'fusion_tree'),
        (fibonacci_anyon_category, 'fusion_tree'),
    ],
)
def test_issue_273(symmetry, backend, np_random):
    diagram = lasy.PlanarDiagram(
        tensors='RP[vL*, vL], W[wL, p, p*], ket[vL, p, vR], bra[vR*, p*, vL*]',
        definition=(
            'RP:vL @ ket:vR, ket:p @ W:p*, RP:vL* @ bra:vR*, W:p @ bra:p*, ket:vL -> vL, bra:vL* -> vL*, W:wL -> wL'
        ),
        dims=dict(chi=['vR', 'vL', 'vR*', 'vL*'], d=['p', 'p*'], w=['wL']),
    )
    backend = lasy.get_backend(backend, 'numpy')
    RP: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain=1,
        domain=1,
        labels=[['vL*'], ['vL']],
        backend=backend,
        np_random=np_random,
    )
    ket: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain=2,
        domain=[RP._as_codomain_leg('vL')],
        labels=[['vL', 'p'], ['vR']],
        backend=backend,
        np_random=np_random,
    )
    bra: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain=[RP._as_domain_leg('vL*')],
        domain=2,
        labels=[['vR*'], ['vL*', 'p*']],
        backend=backend,
        np_random=np_random,
    )
    W: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain=[None, bra._as_domain_leg('p*')],
        domain=[ket._as_codomain_leg('p')],
        labels=[['wL', 'p'], ['p*']],
        backend=backend,
        np_random=np_random,
    )

    new_RP = lasy.planar_contraction(ket, RP, ['vR'], ['vL'])
    new_RP = lasy.planar_contraction(new_RP, W, ['p'], ['p*'])
    new_RP = lasy.planar_contraction(new_RP, bra, ['vL*', 'p'], ['vR*', 'p*'])

    new_RP_diagram = diagram.evaluate(dict(RP=RP, W=W, ket=ket, bra=bra))
    assert lasy.planar.planar_almost_equal(new_RP, new_RP_diagram)

    ket: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        codomain=[ket._as_codomain_leg('vL'), ket._as_codomain_leg('p')],
        domain=[None, ket._as_domain_leg('vR')],
        labels=[['vL', 'p'], ['charge', 'vR']],
        backend=backend,
        np_random=np_random,
    )
    bra: lasy.SymmetricTensor = lasy.testing.random_tensor(
        symmetry,
        # tdot/compose dual pairing: bra domain carries ket's charge space
        # (not ChargedTensor-style ket._as_codomain_leg in the bra codomain).
        codomain=[bra._as_codomain_leg('vR*')],
        domain=[ket.get_leg('charge'), bra._as_domain_leg('vL*'), bra._as_domain_leg('p*')],
        labels=[['vR*'], ['charge*', 'vL*', 'p*']],
        backend=backend,
        np_random=np_random,
    )
    ket = lasy.HiddenLegTensor(ket, ['charge'])
    bra = lasy.HiddenLegTensor(bra, ['charge*'])

    new_RP = lasy.planar_contraction(ket, RP, ['vR'], ['vL'])
    new_RP = lasy.planar_contraction(new_RP, W, ['p'], ['p*'])
    new_RP = lasy.planar_contraction(new_RP, bra, ['vL*', 'p'], ['vR*', 'p*'])

    new_RP_diagram = diagram.evaluate(dict(RP=RP, W=W, ket=ket, bra=bra))
    assert lasy.planar.planar_almost_equal(new_RP, new_RP_diagram)
