"""A collection of tests for lasyten.tensors."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

import operator
from contextlib import nullcontext

import numpy as np
import numpy.testing as npt
import pytest

from lasyten import BraidChiralityUnspecifiedError, backends, tensors
from lasyten.backends import conventional_leg_order, get_backend
from lasyten.block_backends import NumpyBlockBackend, Scalar
from lasyten.block_backends.dtypes import Dtype
from lasyten.symmetries import (
    _SU2,
    U1,
    ZN,
    AbelianLegPipe,
    DirectSumSpace,
    ElementarySpace,
    FermionParity,
    FibonacciAnyonCategory,
    LegPipe,
    NoSymmetry,
    SectorArray,
    SymmetryError,
    TensorProduct,
)
from lasyten.tensors import ChargedTensor, DiagonalTensor, HiddenLegTensor, Mask, SymmetricTensor, Tensor
from lasyten.testing import assert_equivalent_legs, assert_tensors_almost_equal, random_tensor, swap_gate_numpy
from lasyten.tools.misc import duplicate_entries, inverse_permutation, iter_common_noncommon_sorted_arrays, to_valid_idx

fermion_parity = FermionParity().as_Symmetry()
fibonacci_anyon_category = FibonacciAnyonCategory(handedness='left').as_Symmetry()
no_symmetry = NoSymmetry().as_Symmetry()
su2_symmetry = _SU2().as_Symmetry()
u1_symmetry = U1().as_Symmetry()
z3_symmetry = ZN(N=3).as_Symmetry()
z4_symmetry = ZN(N=4).as_Symmetry()

# TENSOR CLASSES


class DummyTensor(tensors.Tensor):
    """Want to test the Tensor class directly.

    This overrides the abstractmethods, so we can actually make instances.
    """

    def __init__(self, codomain, domain, backend, labels, dtype):
        super().__init__(codomain, domain, backend, labels, dtype, device='cpu')

    def as_dtype(self, dtype):
        raise NotImplementedError

    def copy(self, deep=True) -> tensors.Tensor:
        raise NotImplementedError

    def to_backend(self, backend, dtype=None, device=None):
        return NotImplementedError

    def to_dense_block(self, leg_order: list[int | str] = None, dtype: Dtype = None, understood_braiding: bool = False):
        raise NotImplementedError

    def as_SymmetricTensor(self) -> SymmetricTensor:
        raise NotImplementedError

    def _get_item(self, idx: list[int]) -> bool | float | complex:
        raise NotImplementedError

    def move_to_device(self, device: str):
        raise NotImplementedError


def test_base_Tensor(make_compatible_space, compatible_backend):
    a, b, c, d, e = [make_compatible_space() for _ in range(5)]

    print('checking different labels input formats')
    tens1 = DummyTensor([a, b, c], [d, e], backend=compatible_backend, labels=None, dtype=Dtype.float64)
    tens1.test_sanity()
    assert tens1._labels == [None] * 5

    tens2 = DummyTensor(
        [a, b, c], [d, e], backend=compatible_backend, labels=['a', 'b', 'c', 'e*', 'd*'], dtype=Dtype.float64
    )
    tens2.test_sanity()
    assert tens2._labels == ['a', 'b', 'c', 'e*', 'd*']

    tens3 = DummyTensor(
        [a, b, c], [d, e], backend=compatible_backend, labels=[['a', None, None], ['d*', 'e*']], dtype=Dtype.float64
    )
    tens3.test_sanity()
    assert tens3._labels == ['a', None, None, 'e*', 'd*']

    tens4 = DummyTensor(
        [a, b, c], [d, e], backend=compatible_backend, labels=[['a', None, None], None], dtype=Dtype.float64
    )
    tens4.test_sanity()
    assert tens4._labels == ['a', None, None, None, None]

    print('checking .legs , .num_(co)domain_legs')
    for t in [tens1, tens2, tens3, tens4]:
        assert t.legs == [a, b, c, e.dual, d.dual]
        assert t.num_legs == 5
        assert t.num_codomain_legs == 3
        assert t.num_domain_legs == 2

        if t.symmetry.can_be_dropped:
            assert t.num_parameters <= t.size
        else:
            with pytest.raises(SymmetryError, match='not defined'):
                _ = t.size

    with pytest.raises(TypeError, match='does not support == comparison'):
        _ = tens1 == tens2

    print('checking .(co)domain_labels')
    assert tens1.codomain_labels == [None] * 3
    assert tens1.domain_labels == [None] * 2
    assert tens2.codomain_labels == ['a', 'b', 'c']
    assert tens2.domain_labels == ['d*', 'e*']
    assert tens3.codomain_labels == ['a', None, None]
    assert tens3.domain_labels == ['d*', 'e*']
    assert tens4.codomain_labels == ['a', None, None]
    assert tens4.domain_labels == [None, None]

    print('checking .is_fully_labelled')
    assert not tens1.is_fully_labelled
    assert tens2.is_fully_labelled
    assert not tens3.is_fully_labelled
    assert not tens4.is_fully_labelled

    print('check setting labels')
    tens1.labels = ['e', 'b', 'f', 'c', 'x']
    assert tens1._labels == ['e', 'b', 'f', 'c', 'x']

    print('check relabel')
    tens1.relabel(dict(e='xx', x='e'))
    assert tens1._labels == ['xx', 'b', 'f', 'c', 'e']

    print('check _parse_leg_idx')
    assert tens1._parse_leg_idx(1) == (False, 1, 1)
    assert tens1._parse_leg_idx(3) == (True, 1, 3)
    assert tens1._parse_leg_idx(-1) == (True, 0, 4)
    assert tens1._parse_leg_idx(-3) == (False, 2, 2)
    assert tens2._parse_leg_idx('a') == (False, 0, 0)
    assert tens2._parse_leg_idx('e*') == (True, 1, 3)

    print('check get_leg')
    assert tens2.get_leg(0) == a
    assert tens2.get_leg('b') == b
    assert tens2.get_leg('e*') == e.dual

    print('check has_label')
    assert tens2.has_label('a')
    assert tens2.has_label('a', 'b', 'e*')
    assert not tens2.has_label('foo')
    assert not tens2.has_label('a', 'b', '42')

    print('check labels_are')
    assert tens2.labels_are('a', 'c', 'b', 'e*', 'd*')
    assert tens2.labels_are('c', 'e*', 'd*', 'a', 'b', planar=True)
    assert not tens2.labels_are('a', 'c', 'b', 'e*', 'd*', planar=True)


@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (3, 0), (0, 3)], ids=['1->1', '1->2', '0->3', '3->0'])
@pytest.mark.parametrize('use_pipes', [True, 0.3])
def test_SymmetricTensor(make_compatible_tensor, leg_nums, use_pipes):
    T: SymmetricTensor = make_compatible_tensor(*leg_nums, use_pipes=use_pipes)

    print(T.ascii_diagram)

    T.test_sanity()
    assert T.num_codomain_legs == leg_nums[0]
    assert T.num_domain_legs == leg_nums[1]

    if not T.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    print('checking to_numpy')
    numpy_block = T.to_numpy(understood_braiding=True)
    dense_block = T.backend.block_backend.block_from_numpy(numpy_block)

    print('checking from_dense_block')
    tens = SymmetricTensor.from_dense_block(
        dense_block, codomain=T.codomain, domain=T.domain, backend=T.backend, understood_braiding=True
    )
    tens.test_sanity()
    tens_np = tens.to_numpy(understood_braiding=True)
    npt.assert_almost_equal(tens_np, numpy_block)

    can_have_non_symmetric_dense_blocks = T.num_parameters < T.size

    if can_have_non_symmetric_dense_blocks:  # otherwise all blocks are symmetric
        pert = tens.backend.block_backend.random_uniform(T.shape, dtype=T.dtype)
        non_symmetric_block = dense_block + pert
        with pytest.raises(ValueError, match='Block is not symmetric'):
            _ = SymmetricTensor.from_dense_block(
                non_symmetric_block, codomain=T.codomain, domain=T.domain, backend=T.backend, understood_braiding=True
            )

    # TODO: missing coverage:
    # - from_block_func / from_sector_block_func
    # - from_sector_projection
    # - diagonal

    print('checking from_zero')
    zero_tens = SymmetricTensor.from_zero(codomain=T.codomain, domain=T.domain, backend=T.backend)
    zero_tens.test_sanity()
    zero_tens_np = zero_tens.to_numpy(understood_braiding=True)
    npt.assert_array_almost_equal_nulp(zero_tens_np, np.zeros(T.shape), 10)

    print('checking repr and str')
    _ = str(T)
    _ = repr(T)
    _ = str(zero_tens)
    _ = repr(zero_tens)

    print('checking as_dtype')
    # use as_dtype and compare to to_backend
    dtype = Dtype.complex64
    T2 = T.as_dtype(dtype)
    T2.test_sanity()
    T3 = T.to_backend(T.backend, dtype)
    T3.test_sanity()
    assert T.dtype != T2.dtype
    assert T2.dtype == dtype
    assert T3.dtype == dtype
    assert tensors.almost_equal(T, T2)
    assert tensors.almost_equal(T, T3)


@pytest.mark.parametrize('leg_num', [1, 2, 3])
def test_SymmetricTensor_from_eye(make_compatible_space, make_compatible_tensor, compatible_backend, leg_num):
    # Shrink spaces for multi-leg eye: dense checks scale poorly with FusionTree/_SU2.
    max_sectors = 3 if leg_num < 3 else 2
    max_mult = 3 if leg_num < 3 else 2
    legs = [make_compatible_space(max_sectors=max_sectors, max_mult=max_mult) for _ in range(leg_num)]
    labels = list('abcdefg')[:leg_num]
    tens = SymmetricTensor.from_eye(legs, backend=compatible_backend, labels=labels)

    T1 = make_compatible_tensor(codomain=legs)
    eye_T1 = tensors.compose(tens, T1)
    assert tensors.almost_equal(eye_T1, T1)

    T2 = make_compatible_tensor(domain=legs)
    T2_eye = tensors.compose(T2, tens)
    assert tensors.almost_equal(T2_eye, T2)

    if tens.symmetry.can_be_dropped:
        expect_from_backend = compatible_backend.block_backend.to_numpy(
            compatible_backend.block_backend.eye_block([leg.dim for leg in legs], dtype=tens.dtype)
        )
        res = tens.to_numpy(understood_braiding=True)
        if leg_num == 1:
            expect_explicit = np.eye(legs[0].dim)
        elif leg_num == 2:
            expect_explicit = np.eye(legs[0].dim)[:, None, None, :] * np.eye(legs[1].dim)[None, :, :, None]
        elif leg_num == 3:
            expect_explicit = (
                np.eye(legs[0].dim)[:, None, None, None, None, :]
                * np.eye(legs[1].dim)[None, :, None, None, :, None]
                * np.eye(legs[2].dim)[None, None, :, :, None, None]
            )
        else:
            raise RuntimeError('Need to adjust test design')
        npt.assert_array_almost_equal_nulp(expect_from_backend, expect_explicit, 10)
        if T2.symmetry.is_abelian:
            # should be exactly the eye blocks
            npt.assert_array_almost_equal_nulp(res, expect_explicit, 100)
        else:
            # composing fusion tensors should introduce some floating point errors
            npt.assert_array_almost_equal(res, expect_explicit, decimal=10)


@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (3, 0), (0, 3)], ids=['1->1', '1->2', '0->3', '3->0'])
@pytest.mark.parametrize('dtype', [Dtype.float64, Dtype.complex128], ids=['real', 'complex'])
def test_SymmetricTensor_from_random_uniform(leg_nums, dtype, make_compatible_tensor):
    # use larger block size to reliably check distributions
    # based on some test runs, this corresponds to up to 9e5 samples
    # but may also only be 5e2 samples for a single leg in codomain and domain
    T: SymmetricTensor = make_compatible_tensor(*leg_nums, max_block_size=80)

    if T.symmetry.has_complex_topological_data and dtype.is_real:
        with pytest.raises(ValueError, match='must have complex dtype'):
            _ = SymmetricTensor.from_random_uniform(
                codomain=T.codomain, domain=T.domain, backend=T.backend, dtype=dtype
            )
        return

    rand_tens = SymmetricTensor.from_random_uniform(
        codomain=T.codomain, domain=T.domain, backend=T.backend, dtype=dtype
    )
    rand_tens.test_sanity()

    if not T.symmetry.can_be_dropped:
        # TODO what can we even check...?
        return

    if isinstance(T.backend, backends.NoSymmetryBackend):
        samples = np.asarray(rand_tens.data).flatten()
    else:
        samples = np.concatenate([np.asarray(block).flatten() for block in rand_tens.data.blocks])
    # test distribution if there are enough samples
    if samples.size > 1e4:
        true_mean = 0
        true_var = 1 / 3
        tol = 5 * samples.size**-0.5
        mean = np.mean(samples)
        if dtype.is_real:
            npt.assert_allclose(mean, true_mean, atol=tol)
            npt.assert_allclose(np.var(samples), true_var, atol=tol)
            assert samples[samples < -1].size == 0
            assert samples[samples > 1].size == 0
            # probability of happening: 0.999 ** samples.size; smaller than 4.5e-5
            assert samples[samples < -0.998].size > 0
            assert samples[samples > 0.998].size > 0
        else:
            npt.assert_allclose(mean.real, true_mean, atol=tol)
            npt.assert_allclose(mean.imag, true_mean, atol=tol)
            npt.assert_allclose(np.var(samples.real), true_var, atol=tol)
            npt.assert_allclose(np.var(samples.imag), true_var, atol=tol)
            assert samples[samples.real < -1].size == 0
            assert samples[samples.imag < -1].size == 0
            assert samples[samples.real > 1].size == 0
            assert samples[samples.imag > 1].size == 0
            assert samples[samples.real < -0.998].size > 0
            assert samples[samples.imag < -0.998].size > 0
            assert samples[samples.real > 0.998].size > 0
            assert samples[samples.imag > 0.998].size > 0


@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (3, 0), (0, 3)], ids=['1->1', '1->2', '0->3', '3->0'])
@pytest.mark.parametrize('dtype', [Dtype.float64, Dtype.complex128], ids=['real', 'complex'])
def test_SymmetricTensor_from_random_normal(leg_nums, dtype, make_compatible_tensor, np_random):
    # use larger block size to reliably check distributions
    # based on some test runs, this corresponds to up to 9e5 samples
    # but may also only be 5e2 samples for a single leg in codomain and domain
    T: SymmetricTensor = make_compatible_tensor(*leg_nums, max_block_size=80)
    # TODO do we want to test nontrivial means?
    sigma = np_random.uniform(high=3.0)

    if T.symmetry.has_complex_topological_data and dtype.is_real:
        with pytest.raises(ValueError, match='must have complex dtype'):
            _ = SymmetricTensor.from_random_normal(
                codomain=T.codomain, domain=T.domain, sigma=sigma, backend=T.backend, dtype=dtype
            )
        return

    rand_tens = SymmetricTensor.from_random_normal(
        codomain=T.codomain, domain=T.domain, sigma=sigma, backend=T.backend, dtype=dtype
    )
    rand_tens.test_sanity()

    if not T.symmetry.can_be_dropped:
        return  # TODO what can we even test?

    if isinstance(T.backend, backends.NoSymmetryBackend):
        samples = np.asarray(rand_tens.data).flatten()
    else:
        samples = np.concatenate([np.asarray(block).flatten() for block in rand_tens.data.blocks])
    # test distribution if there are enough samples
    if samples.size > 1e4:
        true_mean = 0
        true_var = sigma**2
        tol = 10 * samples.size**-0.5
        mean = np.mean(samples)
        if dtype.is_real:
            npt.assert_allclose(mean, true_mean, atol=tol)
            npt.assert_allclose(np.var(samples), true_var, rtol=tol)
        else:
            npt.assert_allclose(mean.real, true_mean, atol=tol)
            npt.assert_allclose(mean.imag, true_mean, atol=tol)
            npt.assert_allclose(np.var(samples.real), true_var / 2, rtol=tol)
            npt.assert_allclose(np.var(samples.imag), true_var / 2, rtol=tol)
            # absolute values are Rayleigh distributed
            npt.assert_allclose(np.mean(np.abs(samples)), np.sqrt(np.pi / 4) * sigma, atol=tol)
            npt.assert_allclose(np.var(np.abs(samples)), (4 - np.pi) / 4 * true_var, rtol=tol)


def test_from_random_normal_rejects_nonpositive_sigma(make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(1, 1)
    with pytest.raises(ValueError, match='sigma'):
        SymmetricTensor.from_random_normal(codomain=T.codomain, domain=T.domain, sigma=0.0, backend=T.backend)


@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (3, 0), (0, 3)], ids=['1->1', '1->2', '0->3', '3->0'])
def test_SymmetricTensor_from_tree_pairs(make_compatible_tensor, leg_nums, np_random):
    T: SymmetricTensor = make_compatible_tensor(*leg_nums, use_pipes=False)

    numpy_block_backend = NumpyBlockBackend.from_factory('cpu')
    # build two valid dicts
    trees1 = {}
    trees2 = {}
    for coupled in T.codomain.sector_decomposition:
        for X, _, mults1, _ in T.codomain.iter_tree_blocks([coupled]):
            for Y, _, mults2, _ in T.domain.iter_tree_blocks([coupled]):
                shape = [*mults1, *reversed(mults2)]
                if len(trees1) == 0 or np_random.choice([True, False]):
                    trees1[X, Y] = np_random.uniform(size=shape) + 1j * np_random.uniform(size=shape)
                if len(trees2) == 0 or np_random.choice([True, False]):
                    trees2[X, Y] = np_random.uniform(size=shape) + 1j * np_random.uniform(size=shape)
    # build the dict for the linear combination T3 = a * T1 + b * T2
    a = 1.3
    b = -3 + 4.7j
    trees3 = {}
    for pair, block in trees1.items():
        trees3[pair] = a * block
    for pair, block in trees2.items():
        trees3[pair] = trees3.get(pair, 0) + b * block

    T_from_trees1 = SymmetricTensor.from_tree_pairs(trees1, T.codomain, T.domain, backend=T.backend)
    T_from_trees1.test_sanity()
    T_from_trees2 = SymmetricTensor.from_tree_pairs(trees2, T.codomain, T.domain, backend=T.backend)
    T_from_trees2.test_sanity()
    T_from_trees3 = SymmetricTensor.from_tree_pairs(trees3, T.codomain, T.domain, backend=T.backend)
    T_from_trees3.test_sanity()

    assert tensors.almost_equal(T_from_trees3, a * T_from_trees1 + b * T_from_trees2)

    # compare to numpy
    if T.symmetry.can_be_dropped:
        for T_res, trees_dict in zip([T_from_trees1, T_from_trees2, T_from_trees3], [trees1, trees2, trees3]):
            T_np = T_res.to_numpy(understood_braiding=True)
            expect = np.zeros_like(T_np)
            for (X, Y), block in trees_dict.items():
                # [a1...aJ,b1...bK]
                symmetry_data = np.tensordot(
                    X.to_dense_block(understood_braiding=True).to_numpy().conj(),
                    Y.to_dense_block(understood_braiding=True).to_numpy(),
                    (-1, -1),
                )
                # [a1...aJ,b1...bK] & [a1...aJ,bK...b1]
                symmetry_data = np.transpose(
                    symmetry_data, [*range(T.num_codomain_legs), *reversed(range(T.num_codomain_legs, T.num_legs))]
                )
                contribution = np.kron(symmetry_data, block.to_numpy())
                codom_idcs = [
                    slice(*l.slices[l.sector_decomposition_where(a)]) for l, a in zip(T.codomain, X.uncoupled)
                ]
                dom_idcs = [slice(*l.slices[l.sector_decomposition_where(b)]) for l, b in zip(T.domain, Y.uncoupled)]
                expect[(*codom_idcs, *reversed(dom_idcs))] += contribution

            perms = [l.inverse_basis_perm for l in conventional_leg_order(T_res)]
            expect = expect[np.ix_(*perms)]
            npt.assert_almost_equal(T_np, expect)


def test_SymmetricTensor_tofrom_dense_block_trivial_sector(make_compatible_tensor):
    tens = make_compatible_tensor(codomain=['a'], use_pipes=False)
    (leg,) = tens.legs
    block_size = leg.sector_multiplicity(tens.symmetry.trivial_sector)

    if isinstance(tens.backend, backends.FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='to_dense_block_trivial_sector not implemented'):
            _ = tens.to_dense_block_trivial_sector()
        pytest.xfail('FTBackend does not support to_dense_block_trivial_sector yet')

    block = tens.to_dense_block_trivial_sector()
    assert tens.backend.block_backend.get_shape(block) == (block_size,)
    tens2 = SymmetricTensor.from_dense_block_trivial_sector(vector=block, space=leg, backend=tens.backend, label='a')
    tens2.test_sanity()
    assert tensors.almost_equal(tens, tens2)
    block2 = tens2.to_dense_block_trivial_sector()
    npt.assert_array_almost_equal_nulp(
        tens.backend.block_backend.to_numpy(block), tens.backend.block_backend.to_numpy(block2), 100
    )


def test_fixes_124(np_random):
    """Check if the bug discussed in PR #124 is fixed"""
    symm = _SU2().as_Symmetry()
    backend = get_backend(symm, 'numpy')
    a = ElementarySpace(symm, [[1]], [1])
    b = ElementarySpace(symm, [[1]], [1])
    c = ElementarySpace(symm, [[2]], [2])
    codomain = TensorProduct([a, b])
    domain = TensorProduct([c])

    trees = {}
    for coupled in codomain.sector_decomposition:
        for X, _, mults1, _ in codomain.iter_tree_blocks([coupled]):
            for Y, _, mults2, _ in domain.iter_tree_blocks([coupled]):
                shape = [*mults1, *reversed(mults2)]
                if len(trees) == 0 or np_random.choice([True, False]):
                    trees[X, Y] = np.ones(shape, float)

    T = SymmetricTensor.from_tree_pairs(trees, codomain, domain, backend=backend)
    T.test_sanity()

    T_np = T.to_numpy()
    expect = np.zeros_like(T_np)
    for (X, Y), block in trees.items():
        symmetry_data = np.tensordot(
            X.to_dense_block(understood_braiding=True).to_numpy().conj(),
            Y.to_dense_block(understood_braiding=True).to_numpy(),
            (-1, -1),
        )
        # [a1...aJ,b1...bK] & [a1...aJ,bK...b1]
        symmetry_data = np.transpose(
            symmetry_data, [*range(T.num_codomain_legs), *reversed(range(T.num_codomain_legs, T.num_legs))]
        )
        contribution = np.kron(symmetry_data, block.to_numpy())
        codom_idcs = [slice(*l.slices[l.sector_decomposition_where(a)]) for l, a in zip(T.codomain, X.uncoupled)]
        dom_idcs = [slice(*l.slices[l.sector_decomposition_where(b)]) for l, b in zip(T.domain, Y.uncoupled)]
        expect[(*codom_idcs, *reversed(dom_idcs))] += contribution
    perms = [l.inverse_basis_perm for l in conventional_leg_order(T)]
    expect = expect[np.ix_(*perms)]
    npt.assert_almost_equal(T_np, expect)


def test_fixes_23():
    # See PR #23
    sym = _SU2().as_Symmetry()
    site = ElementarySpace(sym, SectorArray.from_sector(_SU2.spin_half))
    block = np.zeros((2,) * 6, float)
    tens = SymmetricTensor.from_dense_block(block, codomain=[site] * 3, domain=[site] * 3)
    tens.test_sanity()
    npt.assert_almost_equal(tensors.norm(tens).to_numpy(), 0)


def test_DiagonalTensor(make_compatible_tensor):
    T: DiagonalTensor = make_compatible_tensor(cls=DiagonalTensor)
    T.test_sanity()

    print('checking sector_argmin')
    # --------------------------------------------------------------------------------------------
    sector_real_T = tensors.real(T)
    sector_real_T.labels = ['a', 'b']
    mask, minimum = sector_real_T.sector_argmin()

    mask.test_sanity()
    assert mask.labels == ['a', 'a*']
    assert mask.is_projection
    assert mask.small_leg.num_sectors == 1
    # mask must select the `minimum` value
    assert tensors.almost_equal(
        tensors.apply_mask_DiagonalTensor(sector_real_T, mask),
        minimum * DiagonalTensor.from_eye(mask.small_leg, backend=sector_real_T.backend),
    )
    # minimum must be the actual minimum
    npt.assert_almost_equal(minimum.as_float64(), sector_real_T.min().as_float64())

    for sector in sector_real_T.leg.sector_decomposition:
        sector_mask, sector_minimum = sector_real_T.sector_argmin(sector)
        sector_mask.test_sanity()
        assert sector_mask.small_leg.num_sectors == 1
        assert np.sum(sector_mask.small_leg.multiplicities) == 1
        # mask must select the `minimum` value
        assert tensors.almost_equal(
            tensors.apply_mask_DiagonalTensor(sector_real_T, sector_mask),
            sector_minimum * DiagonalTensor.from_eye(sector_mask.small_leg, backend=sector_real_T.backend),
        )
        # mask must select within the correct sector
        assert sector_mask.small_leg.sector_decomposition[0] == sector
        assert sector_minimum.dtype == sector_real_T.dtype
        # must the minimum within that sector
        sector_selector = DiagonalTensor.from_sector_block_func(
            lambda shape, coupled: np.full(shape, coupled == sector, dtype=bool),
            leg=sector_real_T.leg,
            backend=sector_real_T.backend,
            dtype=Dtype.bool,
        )
        sector_values = tensors.apply_mask_DiagonalTensor(sector_real_T, Mask.from_DiagonalTensor(sector_selector))
        npt.assert_almost_equal(sector_minimum.as_float64(), sector_values.min().as_float64())

    if sector_real_T.symmetry.can_be_dropped:
        sector_real_T_np = sector_real_T.diagonal_as_numpy()
        selected = np.flatnonzero(mask.as_numpy_mask())
        assert selected.size == mask.small_leg.dim
        npt.assert_almost_equal(sector_real_T_np[selected], minimum.as_float64())

    # --------------------------------------------------------------------------------------------
    if not T.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    print('checking diagonal_as_numpy')
    np_diag = T.diagonal_as_numpy()

    print('checking from_diag_block')
    tens = DiagonalTensor.from_diag_block(np_diag, leg=T.leg, backend=T.backend)
    tens.test_sanity()
    res = tens.diagonal_as_numpy()
    npt.assert_array_almost_equal_nulp(res, np_diag, 100)

    print('checking to_numpy')
    np_full = T.to_numpy(understood_braiding=True)
    npt.assert_array_almost_equal_nulp(np_full, np.diag(np_diag), 100)

    print('checking from zero')
    zero_tens = DiagonalTensor.from_zero(T.leg, backend=T.backend)
    zero_tens.test_sanity()
    npt.assert_array_almost_equal_nulp(zero_tens.diagonal_as_numpy(), np.zeros_like(np_diag), 100)

    print('checking from eye')
    tens = DiagonalTensor.from_eye(T.leg, backend=T.backend)
    tens.test_sanity()
    npt.assert_array_almost_equal_nulp(tens.diagonal_as_numpy(), np.ones_like(np_diag), 100)

    # TODO from_random_*
    # TODO from_tensor

    print('checking repr and str')
    _ = str(T)
    _ = repr(T)
    _ = str(zero_tens)
    _ = repr(zero_tens)

    # TODO elementwise dunder methods. loop over operator.XXX functions?
    # TODO float(), complex(), bool()

    print('checking min / max')
    real_T = tensors.real(T)
    real_np = real_T.diagonal_as_numpy()
    npt.assert_almost_equal(real_T.max().as_float64(), np.max(real_np))
    npt.assert_almost_equal(real_T.min().as_float64(), np.min(real_np))

    print('checking argmin')
    i0 = real_T.argmin()
    assert i0 == int(np.argmin(real_np))
    npt.assert_almost_equal(real_T[i0, i0].as_float64(), real_T.min().as_float64())
    with pytest.raises(ValueError, match='argmin is not defined'):
        T.as_dtype(Dtype.complex64).argmin()
    sob = np.asarray(real_T.leg.sectors_of_basis)
    for sec in real_T.leg.sector_decomposition:
        i_s = real_T.argmin(sec)
        assert real_T.leg.sector_decomposition_where(real_T.leg.idx_to_sector(i_s)) == (
            real_T.leg.sector_decomposition_where(sec)
        )
        mask = np.all(sob == np.asarray(sec), axis=-1)
        assert i_s == int(np.argmin(np.where(mask, real_np, np.inf)))
        npt.assert_almost_equal(real_T[i_s, i_s].as_float64(), np.min(real_np[mask]))

    print('checking as_dtype')
    # use as_dtype and compare to to_backend
    dtype = Dtype.complex64
    T2 = T.as_dtype(dtype)
    T2.test_sanity()
    T3 = T.to_backend(T.backend, dtype)
    T3.test_sanity()
    assert T.dtype != T2.dtype
    assert T2.dtype == dtype
    assert T3.dtype == dtype
    assert tensors.almost_equal(T, T2)
    assert tensors.almost_equal(T, T3)


def test_Identity(compatible_backend, make_compatible_space, make_compatible_tensor):
    leg = make_compatible_space()
    tens = tensors.Identity(leg=leg, backend=compatible_backend)
    tens.test_sanity()
    eye_diag = tensors.DiagonalTensor.from_eye(leg, backend=compatible_backend)
    eye_symm = tensors.SymmetricTensor.from_eye([leg], backend=compatible_backend)

    if tens.symmetry.can_be_dropped:
        t_np = tens.to_numpy(understood_braiding=True)
        npt.assert_almost_equal(t_np, np.eye(leg.dim))

    assert tensors.almost_equal(tens, eye_diag, allow_different_types=True)
    assert tensors.almost_equal(tens, eye_symm, allow_different_types=True)

    with pytest.raises(ValueError, match='argmin is not supported for Identity'):
        tens.argmin()

    assert tensors.norm(tens - eye_symm) < 1e-14
    assert tensors.norm(tens - eye_diag) < 1e-14

    # identity property in contraction
    A = make_compatible_tensor(codomain=[leg])
    B = make_compatible_tensor(domain=[leg])
    A_contr = tensors.compose(tens, A)
    assert tensors.norm(A_contr - A) < 1e-15

    B_contr = tensors.compose(B, tens)
    assert tensors.norm(B_contr - B) < 1e-14

    print('checking as_dtype')
    # use as_dtype and compare to to_backend
    dtype = Dtype.complex64
    tens2 = tens.as_dtype(dtype)
    tens2.test_sanity()
    tens3 = tens.to_backend(tens.backend, dtype)
    tens3.test_sanity()
    assert tens.dtype != tens2.dtype
    assert tens2.dtype == dtype
    assert tens3.dtype == tens2.dtype
    assert tensors.almost_equal(tens, tens2)
    assert tensors.almost_equal(tens, tens3)

    # TODO test partial_compose!


def test_Mask(make_compatible_tensor, compatible_symmetry_backend, np_random):
    M_projection: Mask = make_compatible_tensor(cls=Mask)
    backend = M_projection.backend
    symmetry = M_projection.symmetry
    large_leg = M_projection.domain[0]
    small_leg = M_projection.codomain[0]

    assert M_projection.is_projection is True
    M_projection.test_sanity()
    if symmetry.can_be_dropped:
        M_projection_np = M_projection.as_numpy_mask()

    print('checking inclusion Mask')
    M_inclusion: Mask = tensors.dagger(M_projection)
    assert M_inclusion.is_projection is False
    M_inclusion.test_sanity()

    print('checking properties')
    assert M_projection.large_leg == large_leg
    assert M_inclusion.large_leg == large_leg
    assert M_projection.small_leg == small_leg
    assert M_inclusion.small_leg == small_leg

    print('checking from_eye')
    for is_projection in [True, False]:
        M_eye = Mask.from_eye(large_leg, is_projection=is_projection, backend=backend)
        assert M_eye.is_projection is is_projection
        M_eye.test_sanity()

    if symmetry.can_be_dropped:
        # checks that rely on dense block representations
        print('checking from_block_mask / as_block_mask')
        block_mask = np_random.choice([True, False], large_leg.dim, replace=True)
        if not M_projection.symmetry.fusion_style.value == 0:
            # can only have True XOR False for all entries corresponding to the same multiplicity index
            for slc, sector in zip(large_leg.slices, large_leg.defining_sectors):
                dim = symmetry.sector_dim(sector)
                if dim > 1:
                    slc = slice(*slc)
                    stop = int(len(block_mask[slc]) // dim)
                    block_mask[slc] = np.tile(block_mask[slc][:stop], dim)
            block_mask = block_mask[large_leg.inverse_basis_perm]
        M = Mask.from_block_mask(block_mask, large_leg=large_leg, backend=backend)
        M.test_sanity()
        assert M.large_leg == large_leg
        assert M.small_leg.dim == np.sum(block_mask)
        npt.assert_array_equal(M.as_numpy_mask(), block_mask)

        print('checking from_indices')
        indices = np.where(block_mask)[0]
        M = Mask.from_indices(indices, large_leg=large_leg, backend=backend)
        M.test_sanity()
        assert M.large_leg == large_leg
        assert M.small_leg.dim == np.sum(block_mask)
        npt.assert_array_equal(M.as_numpy_mask(), block_mask)

    print('checking from_DiagonalTensor / as_DiagonalTensor')
    diag = M_projection.as_DiagonalTensor(dtype=Dtype.float32)
    assert diag.leg == large_leg
    assert diag.dtype == Dtype.float32
    diag.test_sanity()
    #
    diag = M_projection.as_DiagonalTensor(dtype=Dtype.bool)
    assert diag.leg == large_leg
    assert diag.dtype == Dtype.bool
    diag.test_sanity()
    M = Mask.from_DiagonalTensor(diag)
    if symmetry.can_be_dropped:
        npt.assert_array_equal(M_projection_np, M.as_numpy_mask())
    assert (M == M_projection).all()
    #
    diag = M_inclusion.as_DiagonalTensor(dtype=Dtype.bool)
    assert diag.leg == large_leg
    assert diag.dtype == Dtype.bool
    diag.test_sanity()
    M = Mask.from_DiagonalTensor(diag)  # should reproduce the *projection* Mask.
    if symmetry.can_be_dropped:
        npt.assert_array_equal(M_projection_np, M.as_numpy_mask())
    assert (M == M_projection).all()

    print('checking from_random')
    M = Mask.from_random(large_leg, small_leg=None, backend=backend)
    M.test_sanity()
    # specifying small_leg is currently only possible if the legs have no permutation
    large_leg_no_perm = ElementarySpace(M.symmetry, large_leg.defining_sectors, large_leg.multiplicities)
    M2 = Mask.from_random(large_leg_no_perm, small_leg=None, backend=backend)
    M2.test_sanity()
    M3 = Mask.from_random(large_leg_no_perm, small_leg=M2.small_leg, backend=backend)
    M3.test_sanity()
    assert M2.small_leg == M3.small_leg

    print('checking from_zero')
    M_zero = Mask.from_zero(large_leg, backend=backend)
    M_zero.test_sanity()
    assert M_zero.small_leg.dim == 0

    print('checking bool()')
    with pytest.raises(TypeError, match='The truth value of a Mask is ambiguous.'):
        _ = bool(M_projection)

    print('checking .any() and .all()')
    if symmetry.can_be_dropped:
        assert M_projection.all() == np.all(M_projection_np)
        assert M_inclusion.all() == np.all(M_projection_np)
        assert M_projection.any() == np.any(M_projection_np)
        assert M_inclusion.any() == np.any(M_projection_np)
    assert M_eye.all()
    assert M_eye.any()
    assert not M_zero.all()
    assert not M_zero.any()

    print('checking to_numpy vs as_SymmetricTensor')
    M_SymmetricTensor = M_projection.as_SymmetricTensor()
    assert M_SymmetricTensor.shape == M_projection.shape
    M_SymmetricTensor.test_sanity()
    if symmetry.can_be_dropped:
        res_via_Symmetric = M_SymmetricTensor.to_numpy(understood_braiding=True)
        res_direct = M_projection.to_numpy(understood_braiding=True)
        npt.assert_almost_equal(res_via_Symmetric, res_direct)
    print('   also for inclusion Mask')
    M_SymmetricTensor = M_inclusion.as_SymmetricTensor()
    assert M_SymmetricTensor.shape == M_inclusion.shape
    M_SymmetricTensor.test_sanity()
    if symmetry.can_be_dropped:
        res_via_Symmetric = M_SymmetricTensor.to_numpy(understood_braiding=True)
        res_direct = M_inclusion.to_numpy(understood_braiding=True)
        npt.assert_almost_equal(res_via_Symmetric, res_direct)

    # TODO check binary operands: &, ==, !=, &, |, ^ :
    #   left and right
    #   with bool and with other mask
    #   with projection Masks and with inclusion Masks

    # TODO check orthogonal complement, also for inclusion Mask!

    # TODO check exactly one True per column and at most one True per row?

    # TODO check to_numpy() vs as_numpy_mask() (if not already done...)

    print('checking repr and str')
    _ = str(M_projection)
    _ = repr(M_projection)
    _ = str(M_inclusion)
    _ = repr(M_inclusion)
    _ = str(M_zero)
    _ = repr(M_zero)

    print('checking as_dtype')
    # use as_dtype and compare to to_backend
    dtype = Dtype.bool  # only bool possible
    for T in [M_projection, M_inclusion]:
        T2 = T.as_dtype(dtype)
        T2.test_sanity()
        T3 = T.to_backend(T.backend, dtype)
        T3.test_sanity()
        assert T.dtype == T2.dtype
        assert T2.dtype == dtype
        assert T3.dtype == dtype
        assert tensors.almost_equal(T, T2)
        assert tensors.almost_equal(T, T3)


@pytest.mark.deselect_invalid_ChargedTensor_cases(get_cls=lambda kw: ChargedTensor)
@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (3, 0), (0, 3)], ids=['1->1', '1->2', '0->3', '3->0'])
def test_ChargedTensor(make_compatible_tensor, leg_nums):
    T: ChargedTensor = make_compatible_tensor(*leg_nums, cls=ChargedTensor)
    backend = T.backend

    T.test_sanity()
    assert T.num_codomain_legs == leg_nums[0]
    assert T.num_domain_legs == leg_nums[1]

    if not T.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    print('checking to_numpy')
    numpy_block = T.to_numpy(understood_braiding=True)
    assert T.shape == numpy_block.shape

    print('checking from_zero')
    zero_tens = ChargedTensor.from_zero(
        codomain=T.codomain, domain=T.domain, charge=T.charge_leg, charged_state=T.charged_state, backend=backend
    )
    zero_tens.test_sanity()
    zero_tens_np = zero_tens.to_numpy(understood_braiding=True)
    npt.assert_array_almost_equal_nulp(zero_tens_np, np.zeros(T.shape), 10)

    print('checking repr and str')
    _ = str(T)
    _ = repr(T)
    _ = str(zero_tens)
    _ = repr(zero_tens)

    print('checking as_dtype')
    # use as_dtype and compare to to_backend
    dtype = Dtype.complex64
    T2 = T.as_dtype(dtype)
    T2.test_sanity()
    T3 = T.to_backend(T.backend, dtype)
    T3.test_sanity()
    assert T.dtype != T2.dtype
    assert T2.dtype == dtype
    assert T3.dtype == dtype
    try:
        assert tensors.almost_equal(T, T2)
        assert tensors.almost_equal(T, T3)
    except NotImplementedError as e:
        pytest.xfail(str(e))


@pytest.mark.deselect_invalid_ChargedTensor_cases(get_cls=lambda kw: ChargedTensor)
def test_ChargedTensor_to_dense_block_single_sector(
    make_compatible_sectors, make_compatible_tensor, compatible_symmetry
):
    sector = make_compatible_sectors(1)[0]
    charge_leg = ElementarySpace(compatible_symmetry, SectorArray.from_sector(sector))
    inv_part = make_compatible_tensor(codomain=1, domain=[charge_leg], labels=[None, '!'])
    tens = ChargedTensor(inv_part, charged_state=[1])
    leg = tens.codomain[0]

    if tens.symmetry.qdim(sector) > 1:
        msg = 'to_dense_block_single_sector does not support higher-dim sectors'
        with pytest.raises(NotImplementedError, match=msg):
            _ = tens.to_dense_block_single_sector()
        pytest.xfail(msg)

    if isinstance(tens.backend, backends.FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='inv_part_to_dense_block_single_sector not implemented'):
            _ = tens.to_dense_block_single_sector()
        pytest.xfail('FTBackend does not support ChargedTensor.to_dense_block_single_sector yet')

    block = tens.to_dense_block_single_sector()
    if isinstance(leg, ElementarySpace):
        block_size = leg.sector_multiplicity(sector)
        assert tens.backend.block_backend.get_shape(block) == (block_size,)
    else:
        # TODO for LegPipe, cant do .sector_multiplicity to get the expected size
        pass

    tens2 = ChargedTensor.from_dense_block_single_sector(vector=block, space=leg, sector=sector, backend=tens.backend)
    tens2.test_sanity()
    assert tens2.charge_leg == tens.charge_leg
    assert tensors.almost_equal(tens, tens2)
    block2 = tens2.to_dense_block_single_sector()
    npt.assert_array_almost_equal_nulp(
        tens.backend.block_backend.to_numpy(block), tens.backend.block_backend.to_numpy(block2), 100
    )


# fmt: off
@pytest.mark.parametrize('symmetry_backend', ['abelian', pytest.param('fusion_tree', marks=pytest.mark.FusionTree)])
def test_explicit_blocks(symmetry_backend, block_backend):
    """Do detailed tests with concrete examples.

    Convert a small dense block to a Tensor.
    Construct the expected data (blocks) manually and compare.

    This is useful e.g. for debugging from_dense_block and to check that the data format is
    what we expect.
    """
    backend = get_backend(symmetry_backend, block_backend)
    all_qi = z4_symmetry.all_sectors()
    q0, q1, q2, q3 = all_qi
    basis1 = [q3, q3, q2, q0, q3, q2]  # basis_perm [3, 2, 5, 0, 1, 4]
    basis2 = [q2, q0, q1, q2, q3, q0, q1]  # basis_perm [1, 5, 2, 6, 0, 3, 4]
    s1 = ElementarySpace.from_basis(z4_symmetry, basis1)  # sectors = [0, 2, 3]
    s2 = ElementarySpace.from_basis(z4_symmetry, basis2)  # sectors = [0, 1, 2, 3]

    print(f'\n\nBOTH LEGS IN CODOMAIN:\n')

    #             s2 : 2,  0,  1,  2,  3,  0,  1     s1
    data = np.array([[ 0,  0,  1,  0,  0,  0,  2],  # 3
                     [ 0,  0,  3,  0,  0,  0,  4],  # 3
                     [ 5,  0,  0,  6,  0,  0,  0],  # 2
                     [ 0,  7,  0,  0,  0,  8,  0],  # 0
                     [ 0,  0,  9,  0,  0,  0, 10],  # 3
                     [11,  0,  0, 12,  0,  0,  0],],  # 2
                    dtype=float)

    print('after applying basis perm:')
    print(data[np.ix_(s1.basis_perm, s2.basis_perm)])
    # q: 0   0   1   1   2   2   3       q
    # [[ 7.  8.  0.  0.  0.  0.  0.]     0
    #  [ 0.  0.  0.  0.  5.  6.  0.]     2
    #  [ 0.  0.  0.  0. 11. 12.  0.]     2
    #  [ 0.  0.  1.  2.  0.  0.  0.]     3
    #  [ 0.  0.  3.  4.  0.  0.  0.]     3
    #  [ 0.  0.  9. 10.  0.  0.  0.]]    3
    block_00 = np.asarray([[7, 8]])
    block_31 = np.asarray([[1, 2], [3, 4], [9, 10]])
    block_22 = np.asarray([[5, 6], [11, 12]])

    # non-symmetric block:
    non_symmetric_data = data.copy()
    non_symmetric_data[0, 0] = 42
    with pytest.raises(ValueError, match='not symmetric'):
        t = SymmetricTensor.from_dense_block(non_symmetric_data, codomain=[s1, s2], backend=backend)
    # now continue with the symmetric block

    t = SymmetricTensor.from_dense_block(data, codomain=[s1, s2], backend=backend)
    t.test_sanity()

    # explicitly check the ``t.data`` vs what we expect
    if symmetry_backend == 'abelian':
        # listing this in an order such that the resulting block_inds are lexsorted:
        # blocks allowed for q:                [] -> [0, 0]  ;  [] -> [3, 1]  ;  [] -> [2, 2]
        # indices in .sector_decomposition:    [] -> [0, 0]  ;  [] -> [2, 1]  ;  [] -> [1, 2]
        expect_block_inds = np.array([[0, 0], [2, 1], [1, 2]])
        expect_blocks = [block_00, block_31, block_22]
        #
        valid_block_inds = backends.abelian._valid_block_inds(t.codomain, t.domain)
        npt.assert_array_equal(expect_block_inds, valid_block_inds)
        #
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            npt.assert_array_almost_equal_nulp(t.backend.block_backend.to_numpy(actual), expect, 100)

    elif symmetry_backend == 'fusion_tree':
        npt.assert_array_equal(t.data.block_inds, np.array([0, 0])[None, :])
        forest_block_00 = block_00.reshape((-1, 1))
        forest_block_22 = block_22.reshape((-1, 1))
        forest_block_31 = block_31.reshape((-1, 1))
        # forest blocks are sorted C-style, i.e. first by first row.
        expect_block = np.concatenate([forest_block_00, forest_block_22, forest_block_31], axis=0)
        assert len(t.data.blocks) == 1
        actual = t.backend.block_backend.to_numpy(t.data.blocks[0])
        npt.assert_array_almost_equal_nulp(actual, expect_block, 100)

    else:
        raise RuntimeError

    # check conversion back
    npt.assert_array_almost_equal_nulp(t.to_numpy(), data)

    # =======================================================
    # =======================================================
    # =======================================================
    print(f'\n\nONE LEG EACH IN DOMAIN AND CODOMAIN:\n')
    # note that this setup changes the charge rule! different entries are now allowed than before

    #             s2 : 2,  0,  1,  2,  3,  0,  1     s1
    data = np.array([[ 0,  0,  0,  0, -1,  0,  0],  # 3
                     [ 0,  0,  0,  0, -2,  0,  0],  # 3
                     [ 5,  0,  0,  6,  0,  0,  0],  # 2
                     [ 0,  7,  0,  0,  0,  8,  0],  # 0
                     [ 0,  0,  0,  0, -3,  0,  0],  # 3
                     [11,  0,  0, 12,  0,  0,  0],],  # 2
                    dtype=float,)

    print('after applying basis perm:')
    print(data[np.ix_(s1.basis_perm, s2.basis_perm)])
    # q: 0   0   1   1   2   2   3      q
    # [[ 7.  8.  0.  0.  0.  0.  0.]    0
    #  [ 0.  0.  0.  0.  5.  6.  0.]    2
    #  [ 0.  0.  0.  0. 11. 12.  0.]    2
    #  [ 0.  0.  0.  0.  0.  0. -1.]    3
    #  [ 0.  0.  0.  0.  0.  0. -2.]    3
    #  [ 0.  0.  0.  0.  0.  0. -3.]]   3
    block_00 = np.asarray([[7, 8]])
    block_22 = np.asarray([[5, 6], [11, 12]])
    block_33 = np.asarray([[-1], [-2], [-3]])

    # non-symmetric block:
    non_symmetric_data = data.copy()
    non_symmetric_data[0, 0] = 42
    with pytest.raises(ValueError, match='not symmetric'):
        t = SymmetricTensor.from_dense_block(non_symmetric_data, codomain=[s1], domain=[s2], backend=backend)
    # now continue with the symmetric block

    t = SymmetricTensor.from_dense_block(data, codomain=[s1], domain=[s2], backend=backend)
    t.test_sanity()

    # explicitly check the ``t.data`` vs what we expect
    if symmetry_backend == 'abelian':
        # listing this in an order such that the resulting block_inds are lexsorted:
        # blocks allowed for q:               [0] -> [0]  ;  [2] -> [2]  ;  [3] -> [3]
        # indices in .sector_decomposition:   [0] -> [0]  ;  [2] -> [1]  ;  [3] -> [2]
        # block_inds row:                     [0, 0]      ;  [1, 2]      ;  [2, 3]
        expect_block_inds = np.array([[0, 0], [1, 2], [2, 3]])
        expect_blocks = [block_00, block_22, block_33]
        #
        valid_block_inds = backends.abelian._valid_block_inds(t.codomain, t.domain)
        npt.assert_array_equal(expect_block_inds, valid_block_inds)
        #
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            npt.assert_array_almost_equal_nulp(t.backend.block_backend.to_numpy(actual), expect, 100)

    elif symmetry_backend == 'fusion_tree':
        # expect coupled sectors q0, q2, q3.
        # codomain.sector_decomposition == [q0, q2, q3]
        # domain.sector_decomposition == [q0, q1, q2, q3]
        npt.assert_array_equal(t.data.block_inds, np.array([[0, 0], [1, 2], [2, 3]]))
        expect_blocks = [block_00, block_22, block_33]
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            npt.assert_array_almost_equal_nulp(t.backend.block_backend.to_numpy(actual), expect, 100)

    else:
        raise RuntimeError

    # check conversion back
    npt.assert_array_almost_equal_nulp(t.to_numpy(), data)

    # =======================================================
    # =======================================================
    # =======================================================
    print(f'\n\nFOUR LEG EXAMPLE (2 -> 2):\n')
    s = ElementarySpace.from_basis(z4_symmetry, [q1, q0, q2])  # basis_perm [1, 0, 2]
    data = np.zeros((3, 3, 3, 3), float)
    # set the allowed elements manually
    # note the leg order for the dense array is [*codomain, *reversed(domain)] !!
    #                        SECTORS PER LEG  |  DOMAIN -> coupled -> CODOMAIN
    data[1, 1, 1, 1] = 1  # [0, 0, 0, 0]     |  [0, 0] -> 0 -> [0, 0]
    data[1, 1, 2, 2] = 2  # [0, 0, 2, 2]     |  [2, 2] -> 0 -> [0, 0]
    data[2, 2, 1, 1] = 3  # [2, 2, 0, 0]     |  [0, 0] -> 0 -> [2, 2]
    data[2, 2, 2, 2] = 4  # [2, 2, 2, 2]     |  [2, 2] -> 0 -> [2, 2]
    #
    data[0, 1, 0, 1] = 5  # [1, 0, 1, 0]     |  [0, 1] -> 1 -> [1, 0]
    data[0, 1, 1, 0] = 6  # [1, 0, 0, 1]     |  [1, 0] -> 1 -> [1, 0]
    #
    data[0, 0, 0, 0] = 7  # [1, 1, 1, 1]     |  [1, 1] -> 2 -> [1, 1]
    data[0, 0, 1, 2] = 8  # [1, 1, 0, 2]     |  [2, 0] -> 2 -> [1, 1]
    data[1, 2, 0, 0] = 9  # [0, 2, 1, 1]     |  [1, 1] -> 2 -> [0, 2]
    data[1, 2, 1, 2] = 10  # [0, 2, 0, 2]     |  [2, 0] -> 2 -> [0, 2]
    #
    data[0, 2, 0, 2] = 11  # [1, 2, 1, 2]     |  [2, 1] -> 3 -> [1, 2]
    data[0, 2, 2, 0] = 12  # [1, 2, 2, 1]     |  [1, 2] -> 3 -> [1, 2]
    data[2, 0, 0, 2] = 13  # [2, 1, 1, 2]     |  [2, 1] -> 3 -> [2, 1]
    data[2, 0, 2, 0] = 14  # [2, 1, 2, 1]     |  [1, 2] -> 3 -> [2, 1]

    # non-symmetric block:
    non_symmetric_data = data.copy()
    non_symmetric_data[0, 0, 1, 1] = 42
    with pytest.raises(ValueError, match='not symmetric'):
        t = SymmetricTensor.from_dense_block(non_symmetric_data, codomain=[s, s], domain=[s, s], backend=backend)
    # now continue with the symmetric block

    t = SymmetricTensor.from_dense_block(data, codomain=[s, s], domain=[s, s], backend=backend)
    t.test_sanity()

    # explicitly check the ``t.data`` vs what we expect
    if symmetry_backend == 'abelian':
        # all sectors appear only once, so each allowed entry is its own block.
        # In this case, the value of a sector is also its index in s.sector_decomposition
        # Thus the block inds are just the "SECTORS PER LEG" above.
        expect_block_inds = np.asarray(
            [
                [0, 0, 0, 0],
                [0, 0, 2, 2],
                [2, 2, 0, 0],
                [2, 2, 2, 2],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
                [1, 1, 0, 2],
                [0, 2, 1, 1],
                [0, 2, 0, 2],
                [1, 2, 1, 2],
                [1, 2, 2, 1],
                [2, 1, 1, 2],
                [2, 1, 2, 1],
            ],
            dtype=int,
        )
        expect_blocks = [np.asarray([[x]], dtype=float) for x in range(1, 15)]
        perm = np.lexsort(expect_block_inds.T)
        expect_block_inds = expect_block_inds[perm]
        expect_blocks = [expect_blocks[n] for n in perm]
        #
        # have not set all entries, so expect_block_inds should be a subset of _valid_block_inds
        valid_block_inds = backends.abelian._valid_block_inds(t.codomain, t.domain)
        for i, j in iter_common_noncommon_sorted_arrays(expect_block_inds, valid_block_inds):
            assert j is not None  # j=None would mean that the row of expect_block_inds is not in valid_block_inds
            actual_block = t.backend.block_backend.to_numpy(t.data.blocks[j])
            if i is None:
                expect_block = np.zeros_like(actual_block)
            else:
                expect_block = expect_blocks[i]
            npt.assert_array_almost_equal_nulp(actual_block, expect_block, 100)

    elif symmetry_backend == 'fusion_tree':
        # check block_inds. first make sure the (co)domain.sector_decomposition are what we expect
        npt.assert_array_equal(t.codomain.sector_decomposition, np.stack([q0, q1, q2, q3]))
        npt.assert_array_equal(t.domain.sector_decomposition, np.stack([q0, q1, q2, q3]))
        # expect coupled sectors [q0, q1, q2, q3]
        npt.assert_array_equal(t.data.block_inds, np.repeat(np.arange(4)[:, None], 2, axis=1))
        #
        # build the blocks for fixed coupled sectors
        # note: when setting the data we listed the uncoupled sectors of the domain

        #      dom uncoupled:  (0, 0)  ;  (2, 2)  |  codom uncoupled:
        block_0 = np.asarray([[    1,         2],  #   (0, 0)
                              [    3,         4],],  # (2, 2)
                             dtype=float,)
        #      dom uncoupled:  (0, 1)  ;  (1, 0)  |  codom uncoupled:
        block_1 = np.asarray([[    0,         0],  #  (0, 1)
                              [    5,         6]],  # (1, 0)
                             dtype=float,)
        #      dom uncoupled:  (0, 2)  ;  (1, 1)  ;  (2, 0)  |  codom uncoupled:
        block_2 = np.asarray([[    0,         9,        10],  #  (0, 2)
                              [    0,         7,         8],  #  (1, 1)
                              [    0,         0,         0]],  # (2, 0)
                             dtype=float)
        #      dom uncoupled:  (1, 2)  ;  (2, 1)  |  codom uncoupled:
        block_3 = np.asarray([[   12,        11],  # (1, 2)
                              [   14,        13]],  # (2, 1)
                             dtype=float)
        expect_blocks = [block_0, block_1, block_2, block_3]
        assert len(expect_blocks) == len(t.data.blocks)
        for i, (actual, expect) in enumerate(zip(t.data.blocks, expect_blocks)):
            print(f'checking blocks[{i}]')
            npt.assert_array_almost_equal_nulp(t.backend.block_backend.to_numpy(actual), expect, 100)

    else:
        raise RuntimeError
# fmt: on


@pytest.mark.parametrize('symmetry_backend', [pytest.param('fusion_tree', marks=pytest.mark.FusionTree)])
def test_from_block_su2_symm(symmetry_backend, block_backend):
    backend = get_backend(symmetry_backend, block_backend)
    sym = _SU2().as_Symmetry()
    spin_half = ElementarySpace(sym, [[1]])

    # basis order: [down, up]  ->  might look unusual
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, 1.0j], [-1.0j, 0]], dtype=complex)
    sz = 0.5 * np.array([[-1.0, 0.0], [0.0, +1.0]], dtype=complex)
    heisenberg_4 = sum(si[:, :, None, None] * si[None, None, :, :] for si in [sx, sy, sz])  # [p1, p1*, p2, p2*]
    print(heisenberg_4.transpose([0, 2, 1, 3]).reshape((4, 4)))
    heisenberg_4 = np.transpose(heisenberg_4, [0, 2, 3, 1])  # [p1, p2, p2*, p1*]

    tens_4 = SymmetricTensor.from_dense_block(
        heisenberg_4,
        codomain=[spin_half, spin_half],
        domain=[spin_half, spin_half],
        backend=backend,
        labels=[['p1', 'p2'], ['p1*', 'p2*']],
    )
    tens_4.test_sanity()
    npt.assert_array_equal(tens_4.data.block_inds, np.array([[0, 0], [1, 1]]))  # spin 0, spin 1
    # The blocks are the eigenvalue of the Heisenberg coupling in the fixed total spin sectors
    # For singlet states (coupled=spin-0), we have eigenvalue -3/4
    # For triplet states (coupled=spin-1), we have eigenvalue +1/4
    expect_spin_0 = backend.block_backend.as_scalar(-3 / 4)
    expect_spin_1 = backend.block_backend.as_scalar(1 / 4)
    assert backend.block_backend.allclose(tens_4.data.blocks[0], expect_spin_0._block)
    assert backend.block_backend.allclose(tens_4.data.blocks[1], expect_spin_1._block)

    recovered_block = tens_4.to_dense_block().to_numpy()
    print()
    print('got:')
    print(recovered_block.reshape((4, 4)))
    print()
    print('expect:')
    print(heisenberg_4.reshape((4, 4)))

    assert np.allclose(recovered_block, heisenberg_4)


@pytest.mark.parametrize(
    'codomain_dims, domain_dims, labels, abelian_pipe',
    [
        ([42, 42], [42, 42], ['a', 'b', 'c', 'd'], None),
        ([42, 42], [42, 42], ['a', 'b', 'c', 'd'], None),
        ([42], [42, 42], ['a', 'b', 'c'], None),
        ([42, 42, 12345, 42], [1], ['a', 'b', 'c', 'lorem', 'ipsum'], None),
        ([], [42, 42], ['a', 'b'], None),
        ([42, 42, 42], [], ['a', 'b', 'c'], None),
        ([[3, 2], 42], [42, 42], ['a', 'b', 'c', 'd'], True),
        ([42, 42], [[4, 5], 42], ['a', 'b', 'c', 'd'], False),
        ([[3, 3], 42], [42, [4, 5]], ['a', 'b', 'c', 'd'], True),
    ],
)
def test_Tensor_ascii_diagram(codomain_dims, domain_dims, labels, abelian_pipe, np_random):
    """
    You may find useful (see comments in :func:`test_Tensor_str_repr`)::

        pytest -rP -k test_Tensor_ascii_diagram | less

    """
    codomain = []
    domain = []
    for co_domain, dims in zip([codomain, domain], [codomain_dims, domain_dims]):
        for d in dims:
            if isinstance(d, int):
                co_domain.append(ElementarySpace.from_trivial_sector(dim=d, is_dual=np_random.choice([True, False])))
            else:
                pipe_legs = []
                for _d in d:
                    pipe_legs.append(
                        ElementarySpace.from_trivial_sector(dim=_d, is_dual=np_random.choice([True, False]))
                    )
                if abelian_pipe:
                    pipe = AbelianLegPipe(pipe_legs, is_dual=np_random.choice([True, False]))
                else:
                    pipe = LegPipe(pipe_legs, is_dual=np_random.choice([True, False]))
                co_domain.append(pipe)
    T = DummyTensor(codomain, domain, backend=get_backend(), labels=labels, dtype=Dtype.complex128)
    print(f'domain dualities: {", ".join(str(getattr(l, "is_dual", None)) for l in T.domain)}')
    print(f'codomain dualities: {", ".join(str(getattr(l, "is_dual", None)) for l in T.codomain)}')
    print(T.ascii_diagram)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, codomain, domain',
    [
        pytest.param(SymmetricTensor, ['vL'], ['vR', 'p'], id='Sym-vL-p-vR'),
        pytest.param(SymmetricTensor, ['p'], ['p*'], id='Sym-p-p*'),
        pytest.param(DiagonalTensor, ['p'], ['p*'], id='Diag-p-p*'),
        pytest.param(Mask, ['vL'], ['vL*'], id='Mask-vL-vL*'),
        pytest.param(ChargedTensor, ['vL'], ['vR', 'p'], id='Charged-vL-p-vR'),
    ],
)
def test_Tensor_str_repr(cls, codomain, domain, make_compatible_tensor, str_max_lines=30, repr_max_lines=30):
    """Check if str and repr work.

    Automatically, we can only check if they run at all.
    To check if the output is sensible and useful, a human should look at it.
    Run e.g.::

        pytest -rP -k test_Tensor_str_repr

    to select only this test (``-k`` flag) and see the output (``-rP``), even if it passes.
    Since the output is rather long, it is convenient to write the output to file.
    To do that, and directly open that file in your favorite editor, run e.g. for VS Code::

        pytest -rP -k test_Tensor_str_repr > playground/test_Tensor_str_repr.txt && code playground/test_Tensor_str_repr.txt

    or for vim::

        pytest -rP -k test_Tensor_str_repr > playground/test_Tensor_str_repr.txt && vim playground/test_Tensor_str_repr.txt

    Assumes your cwd is the repository root, such that the file is generated in playground and therefore gitignored.

    Or without creating a file::

        pytest -rP -k test_Tensor_str_repr | less

    """
    terminal_width = 80
    T = make_compatible_tensor(codomain=codomain, domain=domain, cls=cls)

    print('repr(T):')
    res = repr(T)
    print(res)
    lines = res.split('\n')
    assert all(len(line) <= terminal_width for line in lines)
    assert len(lines) <= repr_max_lines
    #
    print()
    print('str(T):')
    res = str(T)
    print(res)
    lines = res.split('\n')
    assert all(len(line) <= terminal_width for line in lines)
    assert len(lines) <= str_max_lines


# TENSOR FUNCTIONS


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, domain, codomain, is_dual',
    [
        pytest.param(SymmetricTensor, 2, 2, True),
        pytest.param(SymmetricTensor, 2, 0, False),
        pytest.param(SymmetricTensor, 0, 2, True),
        pytest.param(SymmetricTensor, 1, 3, False),
        pytest.param(DiagonalTensor, 1, 1, True),
        pytest.param(Mask, 1, 1, False),
        pytest.param(ChargedTensor, 2, 2, False),
        pytest.param(ChargedTensor, 3, 0, True),
    ],
)
def test_add_trivial_leg(cls, domain, codomain, is_dual, make_compatible_tensor, np_random):
    tens: Tensor = make_compatible_tensor(domain, codomain, cls=cls)

    compare_numpy = tens.symmetry.can_be_dropped

    if cls in [DiagonalTensor, Mask]:
        catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    if compare_numpy:
        tens_np = tens.to_numpy(understood_braiding=True)

    print('via positional arg')
    pos = np_random.choice(tens.num_legs + 1)
    with catch_warnings:
        res = tensors.add_trivial_leg(tens, pos, is_dual=is_dual)
    res.test_sanity()
    if compare_numpy:
        res_np = res.to_numpy(understood_braiding=True)
        expect = np.expand_dims(tens_np, pos)
        npt.assert_array_almost_equal_nulp(res_np, expect, 100)

    print('to_domain')
    pos = np_random.choice(tens.num_domain_legs + 1)
    with catch_warnings:
        res = tensors.add_trivial_leg(tens, domain_pos=pos, is_dual=is_dual)
    res.test_sanity()
    if compare_numpy:
        res_np = res.to_numpy(understood_braiding=True)
        expect = np.expand_dims(tens_np, -1 - pos)
        npt.assert_array_almost_equal_nulp(res_np, expect, 100)

    print('to_codomain')
    pos = np_random.choice(tens.num_codomain_legs + 1)
    with catch_warnings:
        res = tensors.add_trivial_leg(tens, codomain_pos=pos, is_dual=is_dual)
    res.test_sanity()
    if compare_numpy:
        res_np = res.to_numpy(understood_braiding=True)
        expect = np.expand_dims(tens_np, pos)
        npt.assert_array_almost_equal_nulp(res_np, expect, 100)


def test_add_trivial_leg_rejects_duplicate_label(make_compatible_tensor):
    tens = make_compatible_tensor(codomain=1, domain=1, labels=['existing', None])

    with pytest.raises(ValueError, match='already has label'):
        tensors.add_trivial_leg(tens, label='existing')

    res = tensors.add_trivial_leg(tens, label='new')
    assert res.labels == ['new', 'existing', None]


def test_tensor_constructor_rejects_duplicate_labels(make_compatible_tensor):
    with pytest.raises(ValueError, match='Duplicate leg labels'):
        make_compatible_tensor(codomain=1, domain=1, labels=['duplicate', 'duplicate'])


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize('cls', [DiagonalTensor, SymmetricTensor, ChargedTensor])
def test_almost_equal(cls, make_compatible_tensor):
    T: Tensor = make_compatible_tensor(cls=cls)

    if cls is ChargedTensor:
        T: ChargedTensor
        T_diff = make_compatible_tensor(like=T)
    else:
        T_diff: Tensor = make_compatible_tensor(domain=T.domain, codomain=T.codomain, cls=cls)

    T2 = T + 1e-7 * T_diff
    try:
        assert tensors.almost_equal(T, T2, rtol=1e-5, atol=1e-5)
        assert not tensors.almost_equal(T, T2, rtol=1e-10, atol=1e-10)
    except NotImplementedError as e:
        pytest.xfail(str(e))


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, codomain, domain, which_leg',
    [
        pytest.param(SymmetricTensor, 2, 2, 1, id='Symm-2-2-codom'),
        pytest.param(SymmetricTensor, 2, 2, -1, id='Symm-2-2-dom'),
        pytest.param(ChargedTensor, 2, 2, 1, id='Charged-2-2-dom'),
        pytest.param(ChargedTensor, 2, 2, -1, id='Charged-2-2-dom'),
        pytest.param(DiagonalTensor, 1, 1, -1, id='Diag-dom'),
        pytest.param(Mask, 1, 1, 0, id='Mask-codom'),
        pytest.param(Mask, 1, 1, -1, id='Mask-dom'),
    ],
)
def test_apply_mask(cls, codomain, domain, which_leg, make_compatible_tensor, compatible_backend):
    num_legs = codomain + domain
    labels = list('abcdefghijkl')[:num_legs]

    M: Mask = make_compatible_tensor(cls=Mask)
    num_legs = domain + codomain
    which_leg = to_valid_idx(which_leg, num_legs)
    if which_leg >= codomain:
        domain = [None] * domain
        domain[num_legs - which_leg - 1] = M.large_leg.dual
    else:
        codomain = [None] * codomain
        codomain[which_leg] = M.large_leg

    if cls is Mask and which_leg == 0:
        with pytest.raises(NotImplementedError, match='Mask generation broken'):
            _ = make_compatible_tensor(codomain=codomain, domain=domain, labels=labels, cls=cls)
        pytest.xfail(reason='Mask generation broken')
    T: tensors.Tensor = make_compatible_tensor(codomain=codomain, domain=domain, labels=labels, cls=cls)

    if cls is Mask:
        with pytest.raises(NotImplementedError, match='tensors._compose_with_Mask not implemented for Mask'):
            _ = tensors.apply_mask(T, M, which_leg)
        pytest.xfail('apply_mask(Mask, Mask) not yet supported')
    elif cls is DiagonalTensor:
        catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    with catch_warnings:
        res = tensors.apply_mask(T, M, which_leg)
    res.test_sanity()

    in_domain, co_domain_idx, leg_idx = T._parse_leg_idx(which_leg)
    expect_legs = T.legs
    expect_legs[leg_idx] = M.small_leg
    assert res.legs == expect_legs
    assert res.labels == T.labels

    compare_numpy = T.symmetry.can_be_dropped

    if compare_numpy:
        T_np = T.to_numpy(understood_braiding=True)
        mask_np = M.as_numpy_mask()
        expect = T_np.compress(mask_np, leg_idx)
        npt.assert_almost_equal(res.to_numpy(understood_braiding=True), expect)


def test_apply_mask_DiagonalTensor(make_compatible_tensor):
    tensor: DiagonalTensor = make_compatible_tensor(cls=DiagonalTensor, labels=['a', 'b'])
    mask: Mask = make_compatible_tensor(domain=[tensor.leg], cls=Mask)

    res = tensors.apply_mask_DiagonalTensor(tensor, mask)
    res.test_sanity()
    assert isinstance(res, DiagonalTensor)
    assert res.leg == mask.small_leg
    assert res.labels == tensor.labels

    if not tensor.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    diag = tensor.diagonal_as_numpy()
    mask_np = mask.as_numpy_mask()
    expect = diag[mask_np]
    npt.assert_almost_equal(res.diagonal_as_numpy(), expect)


@pytest.mark.parametrize('in_domain', [False, True], ids=['codom', 'dom'])
def test_mask_and_scale_axis_with_pipes_on_other_legs(in_domain, make_compatible_tensor):
    # acting on a leg ``x`` that has two pipes in front of it (in the flat leg order), e.g. like
    # masking the new leg in ``truncated_svd`` of a tensor with pipes
    M: Mask = make_compatible_tensor(cls=Mask)
    D: DiagonalTensor = make_compatible_tensor(codomain=[M.large_leg], cls=DiagonalTensor)
    if in_domain:
        codomain, domain = [None], [None, None, None, None, M.large_leg.dual]
        labels = ['y', 'x', 'd', 'c', 'b', 'a']
        groups = [['d', 'c'], ['b', 'a']]
    else:
        codomain, domain = [None, None, None, None, M.large_leg], [None]
        labels = ['a', 'b', 'c', 'd', 'x', 'y']
        groups = [['a', 'b'], ['c', 'd']]
    T = make_compatible_tensor(codomain, domain, labels=labels, max_block_size=2, use_pipes=False)
    T_piped = tensors.combine_legs(T, *groups)

    res = tensors.apply_mask(T_piped, M, 'x')
    res.test_sanity()
    assert_tensors_almost_equal(res, tensors.combine_legs(tensors.apply_mask(T, M, 'x'), *groups))

    res = tensors.scale_axis(T_piped, D, 'x')
    res.test_sanity()
    assert_tensors_almost_equal(res, tensors.combine_legs(tensors.scale_axis(T, D, 'x'), *groups))


@pytest.mark.parametrize(
    'cls, codomain, domain, num_codomain_legs',
    [
        pytest.param(SymmetricTensor, 2, 2, 2),
        pytest.param(SymmetricTensor, 2, 2, 1),
        pytest.param(SymmetricTensor, 2, 2, 4),
    ],
)
def test_bend_legs(cls, codomain, domain, num_codomain_legs, make_compatible_tensor):
    tensor: Tensor = make_compatible_tensor(codomain, domain, cls=cls)

    res = tensors.bend_legs(tensor, num_codomain_legs)
    res.test_sanity()
    assert res.legs == tensor.legs

    if tensor.symmetry.can_be_dropped:
        tensor_np = tensor.to_numpy(understood_braiding=True)
        res_np = res.to_numpy(understood_braiding=True)
        npt.assert_almost_equal(res_np, tensor_np)

    bent_back = tensors.bend_legs(tensor, num_codomain_legs=tensor.num_codomain_legs)
    bent_back.test_sanity()
    assert_tensors_almost_equal(bent_back, tensor)


@pytest.mark.parametrize(
    'use_pipes',
    [
        pytest.param(True, id='pipes'),
        pytest.param(False, id='no pipes'),
    ],
)
def test_combine_split(use_pipes, make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(
        ['a', 'b'], ['d', 'c'], use_pipes=use_pipes, max_blocks=3, max_block_size=3
    )
    assert T.labels == ['a', 'b', 'c', 'd']

    # 1) combine in codomain
    combined1 = tensors.combine_legs(T, [0, 1])
    combined1.test_sanity()
    assert combined1.labels == ['(a.b)', 'c', 'd']
    assert len(combined1.codomain) == 1
    assert combined1.codomain[0].legs == [T.codomain[0], T.codomain[1]]
    assert combined1.domain == T.domain
    #
    split1 = tensors.split_legs(combined1, 0)
    split1.test_sanity()
    assert split1.labels == ['a', 'b', 'c', 'd']
    assert split1.codomain == T.codomain
    assert split1.domain == T.domain
    assert tensors.almost_equal(split1, T)

    # 2) combine in domain
    combined2 = tensors.combine_legs(T, [2, 3])
    combined2.test_sanity()
    assert combined2.labels == ['a', 'b', '(c.d)']
    assert combined2.codomain == T.codomain
    assert len(combined2.domain) == 1
    assert combined2.domain[0].legs == [T.domain[0], T.domain[1]]
    #
    split2 = tensors.split_legs(combined2, '(c.d)')
    split2.test_sanity()
    assert split2.labels == ['a', 'b', 'c', 'd']
    assert split2.codomain == T.codomain
    assert split2.domain == T.domain
    assert tensors.almost_equal(split2, T)

    # 3) combine in codomain with braid
    combined3 = tensors.combine_legs(T, [1, 0], levels={'a': 0, 'b': 1, 'c': 2, 'd': 3})
    combined3.test_sanity()
    assert combined3.labels == ['(b.a)', 'c', 'd']
    assert len(combined3.codomain) == 1
    assert combined3.codomain[0].legs == [T.codomain[1], T.codomain[0]]
    assert combined3.domain == T.domain
    #
    if use_pipes:
        split3 = tensors.split_legs(combined3, '(b.a)')
    else:
        # check that it can be called without specifying the leg
        split3 = tensors.split_legs(combined3)
    split3.test_sanity()
    assert split3.labels == ['b', 'a', 'c', 'd']
    assert split3.codomain.factors == [T.codomain[1], T.codomain[0]]
    assert split3.domain == T.domain
    expect3 = tensors.permute_legs(T, [1, 0], levels={'a': 0, 'b': 1, 'c': 2, 'd': 3})
    assert tensors.almost_equal(split3, expect3)

    # 4) combine, one from codomain, one from domain
    if T.symmetry.has_trivial_braid:
        combined4 = tensors.combine_legs(T, [1, 2])
    else:
        # need to do explicit permute_legs since we need the bend_right
        combined4 = tensors.combine_legs(tensors.permute_legs(T, [0, 1, 2], [3], bend_right=True), [1, 2])
    combined4.test_sanity()
    assert combined4.labels == ['a', '(b.c)', 'd']
    assert len(combined4.codomain) == 2
    assert combined4.codomain[0] == T.codomain[0]
    assert combined4.codomain[1].legs == [T.codomain[1], T.domain[1].dual]
    assert combined4.domain.factors == [T.domain[0]]
    #
    split4 = tensors.split_legs(combined4, 1)
    split4.test_sanity()
    assert split4.labels == ['a', 'b', 'c', 'd']
    assert split4.codomain.factors == [T.codomain[0], T.codomain[1], T.domain[1].dual]
    assert split4.domain.factors == [T.domain[0]]
    assert tensors.almost_equal(split4, tensors.permute_legs(T, [0, 1, 2], bend_right=True))

    if use_pipes:
        # check splitting a non-combined leg (that is a pipe) raises
        with pytest.raises(ValueError, match='Invalid format for a combined label'):
            _ = tensors.split_legs(combined4)
        with pytest.raises(ValueError, match='Invalid format for a combined label'):
            _ = tensors.split_legs(T, 0)
    else:
        # check splitting a non-combined leg raises
        with pytest.raises(ValueError, match='Not a LegPipe.'):
            _ = tensors.split_legs(combined4, 0)

    # 5) check compatibility with bending legs
    #  5a) combine in domain
    combined5 = tensors.combine_legs(T, [2, 3])
    combined5.test_sanity()
    assert combined5.labels == ['a', 'b', '(c.d)']
    assert combined5.codomain.factors == T.codomain.factors
    assert combined5.domain[0].legs == T.domain.factors
    #
    combined_then_bent = tensors.bend_legs(combined5, num_domain_legs=0)
    combined_then_bent.test_sanity()
    bent_individually = tensors.bend_legs(T, num_domain_legs=0)
    bent_then_combined = tensors.combine_legs(bent_individually, [2, 3])
    bent_then_combined.test_sanity()
    assert combined_then_bent.legs == bent_then_combined.legs
    assert tensors.almost_equal(combined_then_bent, bent_then_combined)

    #  5b) check split * bend_pipe * combine == bend_legs
    split5 = tensors.split_legs(combined_then_bent, 2)
    split5.test_sanity()
    assert split5.labels == ['a', 'b', 'c', 'd']
    assert split5.codomain.factors == T.legs
    assert split5.domain.factors == []
    expect5 = tensors.bend_legs(T, num_domain_legs=0)
    assert tensors.almost_equal(split5, expect5), 'bending does not commute through combine!'

    # 5c) combine in codomain
    assert combined1.labels == ['(a.b)', 'c', 'd']
    combined_then_bent = tensors.bend_legs(combined1, num_codomain_legs=0)
    combined_then_bent.test_sanity()
    bent_individually = tensors.bend_legs(T, num_codomain_legs=0)
    bent_then_combined = tensors.combine_legs(bent_individually, [0, 1])
    bent_then_combined.test_sanity()
    assert combined_then_bent.legs == bent_then_combined.legs
    assert tensors.almost_equal(combined_then_bent, bent_then_combined)

    # 5d) combine across codomain
    assert combined4.labels == ['a', '(b.c)', 'd']
    assert combined4.codomain.num_factors == 2
    combined_then_bent = tensors.bend_legs(combined4, num_codomain_legs=0)
    combined_then_bent.test_sanity()
    bent_individually = tensors.bend_legs(T, num_codomain_legs=0)
    bent_then_combined = tensors.combine_legs(bent_individually, [1, 2])
    bent_then_combined.test_sanity()
    assert combined_then_bent.legs == bent_then_combined.legs
    assert tensors.almost_equal(combined_then_bent, bent_then_combined)

    # check that combine_legs().to_numpy() is the same as to_numpy().reshape()
    if T.symmetry.can_be_dropped:
        T_np = T.to_numpy(understood_braiding=True)
        a, b, c, d = T_np.shape

        expect1 = np.reshape(T_np, (a * b, c, d))
        combined1_np = combined1.to_numpy(understood_braiding=True)
        npt.assert_almost_equal(combined1_np, expect1)

        expect2 = np.reshape(T_np, (a, b, c * d))
        combined2_np = combined2.to_numpy(understood_braiding=True)
        npt.assert_almost_equal(combined2_np, expect2)

        combined3_np = combined3.to_numpy(understood_braiding=True)
        expect3 = np.reshape(swap_gate_numpy.transpose(T_np, T.legs, [1, 0, 2, 3]), (a * b, c, d))
        npt.assert_almost_equal(combined3_np, expect3)

        expect4 = np.reshape(T_np, (a, b * c, d))
        combined4_np = combined4.to_numpy(understood_braiding=True)
        npt.assert_almost_equal(combined4_np, expect4)

    # 6) check contracting combined leg versus contracting the individual legs
    if T.symmetry.has_trivial_braid:
        # this example requires implicit bends
        T2: SymmetricTensor = make_compatible_tensor(
            [None, T.codomain[1].dual, T.domain[1]], [None], labels=['x', 'b*', 'c*', 'y']
        )
        contracted_individual = tensors.tdot(T, T2, ['b', 'c'], ['b*', 'c*'])
        contracted_individual.test_sanity()
        T2_combined = tensors.combine_legs(T2, ['c*', 'b*'], pipe_dualities=[True])
        contracted_via_pipes = tensors.tdot(combined4, T2_combined, '(b.c)', '(c*.b*)')
        contracted_via_pipes.test_sanity()
        assert tensors.almost_equal(contracted_individual, contracted_via_pipes)

    # 7) check contracting combined leg versus contracting the individual legs
    #    without implicit bends
    T3: SymmetricTensor = make_compatible_tensor(
        [T._as_domain_leg('d'), T._as_domain_leg('c')], [None, None], labels=['d*', 'c*', 'x', 'y']
    )
    contracted_individual = tensors.compose(T, T3)
    contracted_individual.test_sanity()
    T3_combined = tensors.combine_legs(T3, ['d*', 'c*'], pipe_dualities=[True])
    contracted_via_pipes = tensors.compose(combined5, T3_combined)
    contracted_via_pipes.test_sanity()
    assert tensors.almost_equal(contracted_individual, contracted_via_pipes)


def test_swap_gate_numpy(np_random):
    from lasyten.testing import random_tensor

    symm = fermion_parity

    leg = ElementarySpace.from_basis(symm, [[0], [1]])
    T = random_tensor(symm, [leg, leg], [leg, leg], list('abcd'), max_multiplicity=1, np_random=np_random)
    T_np = T.to_numpy(understood_braiding=True)
    A = tensors.permute_legs(T, [1, 0], [3, 2])
    A_np = A.to_numpy(understood_braiding=True)
    expect = swap_gate_numpy.transpose(T_np, T.legs, [1, 0, 2, 3])
    npt.assert_almost_equal(A_np, expect)


@pytest.mark.parametrize(
    'use_pipes, in_domain',
    [
        pytest.param(True, False, id='codomain pipes'),
        pytest.param(False, False, id='codomain no pipes'),
        pytest.param(True, True, id='domain pipes'),
        pytest.param(False, True, id='domain no pipes'),
    ],
)
def test_combine_split_with_dualities(use_pipes, in_domain, make_compatible_tensor):
    # use dense blocks as check that pipes are consistent across the different backends
    labels = ['a', 'b', 'c', 'd']
    kwargs = dict(use_pipes=use_pipes, max_blocks=3, max_block_size=3)
    if in_domain:
        T: SymmetricTensor = make_compatible_tensor([], labels[::-1], **kwargs)
    else:
        T: SymmetricTensor = make_compatible_tensor(labels, [], **kwargs)
    assert T.labels == labels

    if T.symmetry.can_be_dropped:
        other_backends = [backends.NoSymmetryBackend, backends.AbelianBackend, backends.FusionTreeBackend]
        other_backends = [bak(T.backend.block_backend) for bak in other_backends if not isinstance(T.backend, bak)]
        other_backends = [bak for bak in other_backends if bak.supports_symmetry(T.symmetry)]
        T_np = T.to_dense_block(understood_braiding=True)
        T_bak = [
            SymmetricTensor.from_dense_block(
                block=T_np,
                codomain=[bak.make_pipe(leg, leg.is_dual) if isinstance(leg, LegPipe) else leg for leg in T.codomain],
                domain=[bak.make_pipe(leg, leg.is_dual) if isinstance(leg, LegPipe) else leg for leg in T.domain],
                backend=bak,
                labels=T.labels,
                dtype=T.dtype,
                understood_braiding=True,
            )
            for bak in other_backends
        ]
        T_np = T.backend.block_backend.to_numpy(T_np)
        for T2 in T_bak:
            npt.assert_almost_equal(T_np, T2.to_numpy(understood_braiding=True))

    for dual in [True, False]:
        combined = tensors.combine_legs(T, [0, 1], pipe_dualities=[dual])
        combined.test_sanity()
        split_idcs = [0] if use_pipes else None
        split = tensors.split_legs(combined, split_idcs)
        split.test_sanity()
        assert tensors.almost_equal(split, T)
        if T.symmetry.can_be_dropped:
            npt.assert_almost_equal(T_np, split.to_numpy(understood_braiding=True))
            for T2 in T_bak:
                combined2 = tensors.combine_legs(T2, [0, 1], pipe_dualities=[dual])
                combined2.test_sanity()
                npt.assert_almost_equal(
                    combined.to_numpy(understood_braiding=True), combined2.to_numpy(understood_braiding=True)
                )

        combined = tensors.combine_legs(T, [2, 3], pipe_dualities=[dual])
        combined.test_sanity()
        split_idcs = [2] if use_pipes else None
        split = tensors.split_legs(combined, split_idcs)
        split.test_sanity()
        assert tensors.almost_equal(split, T)
        if T.symmetry.can_be_dropped:
            npt.assert_almost_equal(T_np, split.to_numpy(understood_braiding=True))
            for T2 in T_bak:
                combined2 = tensors.combine_legs(T2, [2, 3], pipe_dualities=[dual])
                combined2.test_sanity()
                npt.assert_almost_equal(
                    combined.to_numpy(understood_braiding=True), combined2.to_numpy(understood_braiding=True)
                )

    for dual in [[True, True], [True, False], [False, True], [False, False]]:
        combined = tensors.combine_legs(T, [0, 1], [2, 3], pipe_dualities=dual)
        combined.test_sanity()
        split = tensors.split_legs(combined)
        split.test_sanity()
        assert tensors.almost_equal(split, T)
        if T.symmetry.can_be_dropped:
            npt.assert_almost_equal(T_np, split.to_numpy(understood_braiding=True))
            for T2 in T_bak:
                combined2 = tensors.combine_legs(T2, [0, 1], [2, 3], pipe_dualities=dual)
                combined2.test_sanity()
                npt.assert_almost_equal(
                    combined.to_numpy(understood_braiding=True), combined2.to_numpy(understood_braiding=True)
                )


def test_combine_split_pr_16():
    """Check if the bug addressed in PR :pull:`16` is fixed"""
    backend = get_backend('abelian', 'numpy')
    symmetry = u1_symmetry * z3_symmetry

    a = ElementarySpace(
        symmetry,
        defining_sectors=[[-2, 0], [-1, 0], [-2, 1], [-2, 2]],
        multiplicities=[1, 2, 4, 4],
        basis_perm=[8, 0, 7, 3, 6, 2, 4, 10, 1, 5, 9],
        is_dual=True,
    )
    b = ElementarySpace(
        symmetry,
        defining_sectors=[[-3, 0], [0, 0], [-3, 1], [-3, 2]],
        multiplicities=[1, 1, 1, 1],
        basis_perm=None,
        is_dual=False,
    )
    c = ElementarySpace(
        symmetry,
        defining_sectors=[[-4, 0], [-3, 0], [-7, 1], [-6, 1], [-3, 1]],
        multiplicities=[5, 5, 5, 5, 5],
        basis_perm=None,
        is_dual=False,
    )
    d = ElementarySpace(
        symmetry,
        defining_sectors=[[-2, 0], [1, 0], [2, 1]],
        multiplicities=[3, 3, 2],
        basis_perm=[6, 3, 4, 0, 7, 2, 5, 1],
        is_dual=True,
    )

    T = tensors.SymmetricTensor.from_random_normal([a, b], [d, c], backend=backend)
    combined5 = tensors.combine_legs(T, [2, 3])
    combined5.test_sanity()
    assert combined5.codomain.factors == T.codomain.factors
    assert combined5.domain[0].legs == T.domain.factors

    re_split = tensors.split_legs(combined5, 2)
    assert tensors.almost_equal(T, re_split)

    bent5 = tensors.bend_legs(combined5, num_domain_legs=0)
    split5 = tensors.split_legs(bent5, 2)
    split5.test_sanity()
    assert split5.codomain.factors == T.legs
    assert split5.domain.factors == []
    expect5 = tensors.bend_legs(T, num_domain_legs=0)

    assert tensors.almost_equal(split5, expect5)


@pytest.mark.parametrize(
    'cls_A, cls_B, cod_A, shared, dom_B',
    [
        pytest.param(SymmetricTensor, SymmetricTensor, 2, 2, 2, id='Sym@Sym-2-2-2'),
    ],
)
def test_compose(cls_A, cls_B, cod_A, shared, dom_B, make_compatible_tensor):
    labels_A = [list('abcd')[:cod_A], list('efgh')[:shared]]
    labels_B = [list('ijkl')[:shared], list('mnop')[:dom_B]]
    A: Tensor = make_compatible_tensor(codomain=cod_A, domain=shared, labels=labels_A, cls=cls_A)
    B: Tensor = make_compatible_tensor(codomain=A.domain, domain=dom_B, labels=labels_B, cls=cls_B)

    res = tensors.compose(A, B, relabel1={'a': 'x'}, relabel2={'m': 'y'})

    if not A.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    if cod_A == 0 == dom_B:  # scalar result
        assert isinstance(res, (float, complex))
        res_np = res
    else:
        res.test_sanity()
        assert res.codomain == A.codomain
        assert res.domain == B.domain
        expect_labels = []
        if A.num_codomain_legs > 0:
            expect_labels.append('x')
            expect_labels.extend(A.codomain_labels[1:])
        if B.num_domain_legs > 0:
            expect_labels.extend(reversed(B.domain_labels[1:]))
            expect_labels.append('y')
        assert res.labels == expect_labels

        res_np = res.to_numpy(understood_braiding=True)

    axes = [list(range(cod_A, cod_A + shared)), list(reversed(range(shared)))]
    expect = np.tensordot(A.to_numpy(understood_braiding=True), B.to_numpy(understood_braiding=True), axes)
    npt.assert_almost_equal(res_np, expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, cod, dom',
    [
        pytest.param(SymmetricTensor, 2, 2, id='Sym-2-2'),
        pytest.param(SymmetricTensor, 3, 0, id='Sym-3-0'),
        pytest.param(SymmetricTensor, 1, 1, id='Sym-1-1'),
        pytest.param(SymmetricTensor, 0, 3, id='Sym-3-0'),
        pytest.param(ChargedTensor, 2, 2, id='Charged-2-2'),
        pytest.param(ChargedTensor, 3, 0, id='Charged-3-0'),
        pytest.param(ChargedTensor, 1, 1, id='Charged-1-1'),
        pytest.param(ChargedTensor, 0, 3, id='Charged-3-0'),
        pytest.param(DiagonalTensor, 1, 1, id='Diag'),
        pytest.param(Mask, 1, 1, id='Mask'),
    ],
)
def test_dagger(cls, cod, dom, make_compatible_tensor, np_random):
    T_labels = list('abcdefghi')[: cod + dom]
    T: Tensor = make_compatible_tensor(cod, dom, cls=cls, labels=T_labels)

    if cls is ChargedTensor and not T.symmetry.has_symmetric_braid:
        # TODO : should be ok to just choose levels, as long as charge leg is very top or very bot
        with pytest.raises(BraidChiralityUnspecifiedError):
            _ = tensors.dagger(T)
        return

    how_to_call = np_random.choice(['dagger()', '.hc', '.dagger'])
    print(how_to_call)
    if how_to_call == 'dagger()':
        res = tensors.dagger(T)
    if how_to_call == '.hc':
        res = T.hc
    if how_to_call == '.dagger':
        res = T.dagger
    res.test_sanity()

    assert res.codomain == T.domain
    assert res.domain == T.codomain
    assert res.labels == [f'{l}*' for l in reversed(T_labels)]

    compare_numpy = T.symmetry.can_be_dropped

    if compare_numpy:
        T_np = T.to_numpy(understood_braiding=True)
        res_np = res.to_numpy(understood_braiding=True)
        expect = np.conj(np.transpose(T_np, list(reversed(range(cod + dom)))))

        if not T.symmetry.has_trivial_braid and cls is ChargedTensor:
            # TODO this should not be the case....?
            npt.assert_almost_equal(res_np, expect) or np.allclose(res_np, -expect)
        else:
            npt.assert_almost_equal(res_np, expect)

    double_dagger = res.dagger
    double_dagger.test_sanity()
    assert tensors.almost_equal(double_dagger, T)


@pytest.mark.parametrize(
    'lasyten_func, numpy_func, dtype, kwargs',
    # dtype=None indicates that we need to special case the tensor creations to fulfill constraints.
    [
        pytest.param(tensors.angle, np.angle, Dtype.complex128, {}, id='angle()-complex'),
        pytest.param(tensors.angle, np.angle, Dtype.float64, {}, id='angle()-real'),
        pytest.param(tensors.imag, np.imag, Dtype.complex128, {}, id='imag()-complex'),
        pytest.param(tensors.imag, np.imag, Dtype.float64, {}, id='imag()-real'),
        pytest.param(tensors.real, np.real, Dtype.complex128, {}, id='real()-complex'),
        pytest.param(tensors.real, np.real, Dtype.float64, {}, id='real()-real'),
        pytest.param(tensors.real_if_close, np.real_if_close, Dtype.complex128, {}, id='real_if_close()'),
        pytest.param(tensors.real_if_close, np.real_if_close, Dtype.float64, dict(tol=100), id='real_if_close()'),
        pytest.param(tensors.real_if_close, np.real_if_close, None, {}, id='real_if_close()'),
        pytest.param(tensors.sqrt, np.sqrt, None, {}, id='sqrt()'),
        pytest.param(DiagonalTensor.__abs__, np.abs, Dtype.float64, {}, id='abs()-real'),
        pytest.param(DiagonalTensor.__abs__, np.abs, Dtype.complex128, {}, id='abs()-complex'),
        pytest.param(tensors.real, np.real, Dtype.float64, {}, id='real()-real'),
        pytest.param(tensors.complex_conj, np.conj, Dtype.float64, {}, id='conj()-real'),
        pytest.param(tensors.complex_conj, np.conj, Dtype.complex128, {}, id='conj()-complex'),
        pytest.param(tensors.cutoff_inverse, None, Dtype.float64, dict(cutoff=1e-16), id='cutoff_inverse())-real'),
        pytest.param(tensors.cutoff_inverse, None, Dtype.complex128, dict(cutoff=0.3), id='cutoff_inverse())-complex'),
    ],
    # TODO more functions? exp, log
)
def test_DiagonalTensor_elementwise_unary(lasyten_func, numpy_func, dtype, kwargs, make_compatible_tensor):
    if dtype is not None:
        D: DiagonalTensor = make_compatible_tensor(cls=DiagonalTensor, dtype=dtype)
    elif lasyten_func is tensors.sqrt:
        # need positive
        D: DiagonalTensor = abs(make_compatible_tensor(cls=DiagonalTensor, dtype=Dtype.float64))
    elif lasyten_func is tensors.real_if_close:
        # want almost real
        rp = make_compatible_tensor(cls=DiagonalTensor, dtype=Dtype.float64)
        ip = make_compatible_tensor(domain=rp.domain, cls=DiagonalTensor, dtype=Dtype.float64)
        D = rp + 1e-12j * ip
    else:
        raise ValueError

    res = lasyten_func(D, **kwargs)
    res.test_sanity()

    if not D.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    if lasyten_func is tensors.cutoff_inverse:

        def numpy_func(a, cutoff=kwargs['cutoff']):
            return 1 / np.where(np.abs(a) < cutoff, np.inf, a)

        npt.assert_almost_equal(numpy_func(1), 1)
        npt.assert_almost_equal(numpy_func(0), 0)
        npt.assert_almost_equal(numpy_func(2 * kwargs['cutoff']), 0.5 / kwargs['cutoff'])
        npt.assert_almost_equal(numpy_func(0.5 * kwargs['cutoff']), 0)

    res_np = res.diagonal_as_numpy()
    expect = numpy_func(D.diagonal_as_numpy())
    npt.assert_almost_equal(res_np, expect)


@pytest.mark.parametrize(
    'cls, op, dtype',
    [
        pytest.param(DiagonalTensor, operator.add, Dtype.complex128, id='+'),
        pytest.param(DiagonalTensor, operator.ge, Dtype.float64, id='>='),
        pytest.param(DiagonalTensor, operator.gt, Dtype.float64, id='>'),
        pytest.param(DiagonalTensor, operator.le, Dtype.float64, id='<='),
        pytest.param(DiagonalTensor, operator.lt, Dtype.float64, id='<'),
        pytest.param(
            DiagonalTensor, operator.le, Dtype.bool, id='<='
        ),  # this implicitly checks diagonal_as_numpy() for bool
        pytest.param(DiagonalTensor, operator.mul, Dtype.complex128, id='*'),
        pytest.param(DiagonalTensor, operator.pow, Dtype.complex128, id='**'),
        pytest.param(DiagonalTensor, operator.sub, Dtype.complex128, id='-'),
    ],
)
def test_DiagonalTensor_elementwise_binary(cls, op, dtype, make_compatible_tensor, np_random):
    t1: DiagonalTensor = make_compatible_tensor(cls=cls, dtype=dtype)
    if op == operator.pow:
        dtype = Dtype.float64
        # complex exponentials t1**(r+i) = exp(np.log(t1)*(r+i)) have a high precision loss due to log,
        # making the "assert_almost_equal" fail in some cases
        # hence make second tensor real
    t2: DiagonalTensor = make_compatible_tensor(domain=t1.domain, cls=cls, dtype=dtype)
    if dtype == Dtype.bool:
        scalar = bool(np_random.choice([True, False]))
    elif dtype.is_real:
        scalar = np_random.uniform()
    else:
        scalar = np_random.uniform() + 1.0j * np_random.uniform()

    if not t1.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    t1_np = t1.diagonal_as_numpy()
    t2_np = t2.diagonal_as_numpy()
    print('With other tensor')
    res = op(t1, t2)
    res.test_sanity()
    res_np = res.diagonal_as_numpy()
    expect = op(t1_np, t2_np)
    npt.assert_almost_equal(res_np, expect)

    print('With scalar')
    res = op(t1, scalar)
    res.test_sanity()
    res_np = res.diagonal_as_numpy()
    expect = op(t1_np, scalar)
    npt.assert_almost_equal(res_np, expect)

    if op == operator.add:
        # had issues with implicit conversion of int to Scalar types,
        # explicitly check different versions of scalar types
        for scalar in [3, 3.0, np.float64(3), t1.backend.block_backend.as_scalar(3)]:
            res_int = t1 + scalar
            res_int.test_sanity()
            npt.assert_almost_equal(res_int.diagonal_as_numpy(), t1_np + 3)


@pytest.mark.parametrize(
    'cls, dom, new_leg_dual',
    [
        pytest.param(SymmetricTensor, 1, False, id='Sym-1-False'),
        pytest.param(SymmetricTensor, 1, True, id='Sym-1-True'),
        pytest.param(SymmetricTensor, 2, False, id='Sym-2-False'),
        pytest.param(DiagonalTensor, 1, False, id='Diag-False'),
        pytest.param(DiagonalTensor, 1, True, id='Diag-True'),
    ],
)
def test_eigh(cls, dom, new_leg_dual, make_compatible_tensor):
    # prepare hermitian tensor
    T: Tensor = make_compatible_tensor(dom, dom, cls=cls)
    T: Tensor = make_compatible_tensor(T.domain, T.domain, cls=cls)
    T = T + T.hc
    T.set_labels(list('efghijk')[: 2 * dom])
    T.test_sanity()

    new_leg_dual_arg = {'new_leg_dual': True} if new_leg_dual else {}
    W, V = tensors.eigh(T, new_labels=['a', 'b', 'c'], **new_leg_dual_arg)
    W.test_sanity()
    V.test_sanity()
    assert W.labels == ['b', 'c']
    assert V.codomain_labels == T.codomain_labels
    assert V.domain_labels == ['a']

    assert tensors.almost_equal(V @ W @ V.hc, T, allow_different_types=True)  # is decomposition
    assert tensors.almost_equal(V @ V.hc, SymmetricTensor.from_eye(V.codomain, T.backend))  # unitary 1)
    assert tensors.almost_equal(V.hc @ V, SymmetricTensor.from_eye(V.domain, T.backend))  # unitary 2)


@pytest.mark.parametrize(
    'cls, dom, new_leg_dual',
    [
        pytest.param(SymmetricTensor, 1, False, id='Sym-1-False'),
        pytest.param(SymmetricTensor, 1, True, id='Sym-1-True'),
        pytest.param(SymmetricTensor, 2, False, id='Sym-2-False'),
        pytest.param(DiagonalTensor, 1, False, id='Diag-False'),
        pytest.param(DiagonalTensor, 1, True, id='Diag-True'),
    ],
)
def test_eig(cls, dom, new_leg_dual, make_compatible_tensor):
    T: Tensor = make_compatible_tensor(dom, dom, cls=cls)
    T: Tensor = make_compatible_tensor(T.domain, T.domain, cls=cls)
    T.set_labels(list('efghijk')[: 2 * dom])
    T.test_sanity()

    new_leg_dual_arg = {'new_leg_dual': True} if new_leg_dual else {}
    W, V = tensors.eig(T, new_labels=['a', 'b', 'c'], **new_leg_dual_arg)
    W.test_sanity()
    V.test_sanity()
    assert W.labels == ['b', 'c']
    assert V.codomain_labels == T.codomain_labels
    assert V.domain_labels == ['a']

    assert tensors.almost_equal(T @ V, V @ W, allow_different_types=True)


@pytest.mark.parametrize(
    'cls, dom, new_leg_dual',
    [
        pytest.param(SymmetricTensor, 1, False, id='Sym-1-False'),
        pytest.param(SymmetricTensor, 1, True, id='Sym-1-True'),
        pytest.param(SymmetricTensor, 2, False, id='Sym-2-False'),
        pytest.param(DiagonalTensor, 1, False, id='Diag-False'),
        pytest.param(DiagonalTensor, 1, True, id='Diag-True'),
    ],
)
def test_eigvalsh(cls, dom, new_leg_dual, make_compatible_tensor):
    T: Tensor = make_compatible_tensor(dom, dom, cls=cls)
    T: Tensor = make_compatible_tensor(T.domain, T.domain, cls=cls)
    T = T + T.hc
    T.set_labels(list('efghijk')[: 2 * dom])
    T.test_sanity()

    new_leg_dual_arg = {'new_leg_dual': True} if new_leg_dual else {}
    W = tensors.eigvalsh(T, new_labels=['b', 'c'], **new_leg_dual_arg)
    W.test_sanity()
    assert W.labels == ['b', 'c']
    W2, _V = tensors.eigh(T, new_labels=['a', 'b', 'c'], new_leg_dual=new_leg_dual)
    assert tensors.almost_equal(W, W2, allow_different_types=True)


@pytest.mark.parametrize(
    'cls, dom, new_leg_dual',
    [
        pytest.param(SymmetricTensor, 1, False, id='Sym-1-False'),
        pytest.param(SymmetricTensor, 1, True, id='Sym-1-True'),
        pytest.param(SymmetricTensor, 2, False, id='Sym-2-False'),
        pytest.param(DiagonalTensor, 1, False, id='Diag-False'),
        pytest.param(DiagonalTensor, 1, True, id='Diag-True'),
    ],
)
def test_eigvals(cls, dom, new_leg_dual, make_compatible_tensor):
    T: Tensor = make_compatible_tensor(dom, dom, cls=cls)
    T: Tensor = make_compatible_tensor(T.domain, T.domain, cls=cls)
    T.set_labels(list('efghijk')[: 2 * dom])
    T.test_sanity()

    new_leg_dual_arg = {'new_leg_dual': True} if new_leg_dual else {}
    W = tensors.eigvals(T, new_labels=['b', 'c'], **new_leg_dual_arg)
    W.test_sanity()
    assert W.labels == ['b', 'c']
    W2, _V = tensors.eig(T, new_labels=['a', 'b', 'c'], new_leg_dual=new_leg_dual)
    assert tensors.almost_equal(W, W2, allow_different_types=True)


@pytest.mark.parametrize(
    'cls, dom, new_leg_dual',
    [
        pytest.param(SymmetricTensor, 1, False, id='Sym-1-False'),
        pytest.param(SymmetricTensor, 1, True, id='Sym-1-True'),
        pytest.param(SymmetricTensor, 2, False, id='Sym-2-False'),
        pytest.param(DiagonalTensor, 1, False, id='Diag-False'),
    ],
)
def test_slice_leg(cls, dom, new_leg_dual, make_compatible_tensor, compatible_backend):
    kwargs = {}
    if isinstance(compatible_backend, backends.FusionTreeBackend):
        kwargs['use_pipes'] = False
    T: Tensor = make_compatible_tensor(dom, dom, cls=cls, **kwargs)
    T: Tensor = make_compatible_tensor(T.domain, T.domain, cls=cls, **kwargs)
    T = T + T.hc
    T.set_labels(list('efghijk')[: 2 * dom])

    W, V = tensors.eigh(T, new_labels=['a', 'b', 'c'], new_leg_dual=new_leg_dual)
    eval_leg = V.domain_labels[0]
    space = V.domain[0]

    def check_psi(psi, sector):
        psi.test_sanity()
        assert isinstance(psi, HiddenLegTensor)
        assert psi.labels[-1] == '!slice'
        assert HiddenLegTensor.is_hidden_leg_label(psi.labels[-1])
        T_psi = T @ psi
        return T_psi

    if T.symmetry.can_be_dropped:
        W_real = W if W.dtype.is_real else tensors.real(W)
        i0 = W_real.argmin()
        sector = space.idx_to_sector(i0)
        psi = V.slice_leg(eval_leg, i0)
        T_psi = check_psi(psi, sector)
        assert space.idx_to_sector(i0) == W.leg.idx_to_sector(i0)
        assert tensors.almost_equal(T_psi, W[i0, i0] * psi, allow_different_types=True)

        sec_idx, mult_idx = space.parse_index(i0)
        sec = space.sector_decomposition[sec_idx]
        m = mult_idx % space.sector_multiplicity(sec)
        psi_sec = V.slice_leg(eval_leg, sec, m)
        assert tensors.almost_equal(psi, psi_sec, allow_different_types=True)
        assert tensors.almost_equal(psi, tensors.slice_leg(V, eval_leg, i0), allow_different_types=True)

        dim = T.symmetry.sector_dim(sec)
        if dim > 1:
            for j in range(int(space.dim)):
                if j == i0:
                    continue
                s_j, m_j = space.parse_index(j)
                if space.sector_decomposition[s_j] == sec and (m_j % space.sector_multiplicity(sec)) == m:
                    psi_same = V.slice_leg(eval_leg, j)
                    assert tensors.almost_equal(psi, psi_same, allow_different_types=True)
                    break
    else:
        with pytest.raises(SymmetryError):
            V.slice_leg(eval_leg, 0)
        sec = space.sector_decomposition[0]
        psi = V.slice_leg(eval_leg, sec, 0)
        check_psi(psi, sec)

    if ChargedTensor.supports_symmetry(T.symmetry):
        C: ChargedTensor = make_compatible_tensor(1, 1, cls=ChargedTensor, **kwargs)
        with pytest.raises(ValueError, match='ChargedTensor'):
            C.slice_leg(0, 0)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, codomain, domain, which_leg',
    [
        pytest.param(SymmetricTensor, 2, 2, 1, id='Symm-2-2-codom'),
        pytest.param(SymmetricTensor, 2, 2, 3, id='Symm-2-2-dom'),
        pytest.param(ChargedTensor, 2, 2, 1, id='Charged-2-2-codom'),
        pytest.param(ChargedTensor, 2, 2, 3, id='Charged-2-2-dom'),
        pytest.param(DiagonalTensor, 1, 1, 1, id='Diag-dom'),
        pytest.param(Mask, 1, 1, 0, id='Mask-codom'),
        pytest.param(Mask, 1, 1, 1, id='Mask-dom'),
    ],
)
def test_enlarge_leg(cls, codomain, domain, which_leg, make_compatible_tensor, make_compatible_space):
    num_legs = codomain + domain
    labels = list('abcdefghijkl')[:num_legs]

    large_leg = make_compatible_space()
    projection: Mask = make_compatible_tensor(domain=[large_leg], cls=Mask)
    M: Mask = tensors.dagger(projection)
    #
    T_codomain = [None] * codomain
    T_domain = [None] * domain
    if which_leg >= codomain:
        domain_idx = codomain + domain - 1 - which_leg
        T_domain[domain_idx] = M.small_leg.dual
    else:
        T_codomain[which_leg] = M.small_leg

    if cls is Mask and which_leg == 0:
        with pytest.raises(NotImplementedError, match='Mask generation broken'):
            _ = make_compatible_tensor(codomain=T_codomain, domain=T_domain, labels=labels, cls=cls)
        pytest.xfail('Mask generation broken')

    T: Tensor = make_compatible_tensor(codomain=T_codomain, domain=T_domain, labels=labels, cls=cls)

    if cls is Mask:
        with pytest.raises(NotImplementedError, match='tensors._compose_with_Mask not implemented for Mask'):
            _ = tensors.enlarge_leg(T, M, which_leg)
        pytest.xfail('apply_mask(Mask, Mask) not yet supported')
    if cls is DiagonalTensor:
        catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    with catch_warnings:
        res = tensors.enlarge_leg(T, M, which_leg)
    res.test_sanity()

    _, _, leg_idx = T._parse_leg_idx(which_leg)
    expect_legs = T.legs[:]
    expect_legs[leg_idx] = M.large_leg
    assert res.legs == expect_legs
    assert res.labels == T.labels

    compare_numpy = T.symmetry.can_be_dropped

    if compare_numpy:
        T_np = T.to_numpy(understood_braiding=True)
        mask_np = M.as_numpy_mask()
        idcs = (slice(None, None, None),) * leg_idx + (mask_np,)
        expect_shape = list(T.shape)
        expect_shape[leg_idx] = M.shape[0]
        expect = np.zeros(expect_shape, dtype=T_np.dtype)
        expect[idcs] = T_np
        npt.assert_almost_equal(res.to_numpy(understood_braiding=True), expect)

    # TODO what can we check without resorting to numpy?


@pytest.mark.parametrize('n', [1, np.inf, 2, 3])
def test_entropy(n, make_compatible_tensor):
    D: DiagonalTensor = make_compatible_tensor(cls=DiagonalTensor)
    D = abs(D)  # non-negative
    D = D / tensors.norm(D)  # normalized
    p = D**2

    if not D.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    # make sure we have created a valid probability distribution
    D_np = D.diagonal_as_numpy()
    p_np = p.diagonal_as_numpy()
    assert np.all(D_np >= 0)
    npt.assert_almost_equal(np.sum(p_np), 1)

    ent = tensors.entropy(p, n)
    assert isinstance(ent, float)
    expect = tensors.entropy(p_np, n)
    npt.assert_almost_equal(ent, expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, cod, dom',
    [
        pytest.param(SymmetricTensor, 2, 2, id='Sym-2-2'),
        pytest.param(SymmetricTensor, 1, 3, id='Sym-1-3'),
        pytest.param(SymmetricTensor, 3, 0, id='Sym-3-0'),
        pytest.param(DiagonalTensor, 1, 1, id='Diag'),
        pytest.param(Mask, 1, 1, id='Mask-1-1'),
        pytest.param(ChargedTensor, 2, 2, id='Charged-2-2'),
        pytest.param(ChargedTensor, 0, 3, id='Charged-0-3'),
        pytest.param(ChargedTensor, 3, 1, id='Charged-3-1'),
    ],
)
def test_getitem(cls, cod, dom, make_compatible_tensor, np_random):
    T: Tensor = make_compatible_tensor(cod, dom, cls=cls)

    if not T.symmetry.can_be_dropped:
        with pytest.raises(SymmetryError, match='Can not access elements'):
            _ = T[(0,) * (cod + dom)]
        # error is expected behavior
        return

    if (cls is SymmetricTensor or cls is ChargedTensor) and isinstance(T.backend, backends.FusionTreeBackend):
        catch_warnings = pytest.warns(UserWarning, match='Accessing individual entries')
    else:
        catch_warnings = nullcontext()

    T_np = T.to_numpy(understood_braiding=True)
    random_idx = tuple(np_random.choice(d) for d in T.shape)

    def assert_same(a, b):
        if T.dtype == Dtype.bool:
            assert bool(a) == bool(b)
        else:
            npt.assert_almost_equal(a, b)

    with catch_warnings:
        entry = T[random_idx]
    assert isinstance(entry, Scalar)
    assert_same(entry.to_numpy(), T_np[random_idx])

    # trying to set items raises
    with pytest.raises(TypeError, match='.* do.* not support item assignment.'):
        T[random_idx] = T.dtype.zero_scalar

    non_zero_idcs = np.where(np.abs(T_np) > 1e-2)
    which = np_random.choice(len(non_zero_idcs[0]))
    non_zero_idx = tuple(ax[which] for ax in non_zero_idcs)
    assert len(non_zero_idx) > 0
    with catch_warnings:
        entry = T[non_zero_idx]
    assert_same(entry.to_numpy(), T_np[non_zero_idx])

    zero_idcs = np.where(np.abs(T_np) < 1e-8)
    if len(zero_idcs[0]) > 0:
        which = np_random.choice(len(zero_idcs[0]))
        zero_idx = tuple(ax[which] for ax in zero_idcs)
        with catch_warnings:
            entry = T[zero_idx]
        assert_same(entry.to_numpy(), T_np[zero_idx])


@pytest.mark.parametrize('trunc', [None, 1e-10])
def test_horizontal_factorization(trunc, make_compatible_tensor, compatible_symmetry):
    cod = 4
    cod_cut = 2
    dom = 3
    dom_cut = 1
    if compatible_symmetry.is_equivalent_to(_SU2()):
        # use fewer legs, to make it not super slow
        cod = 3
        dom = 2

    T_labels = list('efghijklmn')[: dom + cod]
    T: Tensor = make_compatible_tensor(
        cod, dom, labels=T_labels, cls=SymmetricTensor, use_pipes=False, max_blocks=3, max_block_size=3
    )

    A, B = tensors.horizontal_factorization(
        T, codomain_cut=cod_cut, domain_cut=dom_cut, new_labels=['a', 'b'], cutoff_singular_values=trunc
    )
    A.test_sanity()
    B.test_sanity()
    assert A.num_codomain_legs == cod_cut
    assert A.num_domain_legs == dom_cut + 1
    assert A.labels == T.labels[:cod_cut] + ['a'] + T.labels[-dom_cut:]
    assert B.num_codomain_legs == 1 + (cod - cod_cut)
    assert B.num_domain_legs == dom - dom_cut
    assert B.labels == ['b'] + T.labels[cod_cut:-dom_cut]

    if T.symmetry.can_be_dropped:
        A_np = A.to_numpy(understood_braiding=True)  # [*J1, a, *K1_rev]
        B_np = B.to_numpy(understood_braiding=True)  # [b, *J2, *K2_rev]
        T2_np = np.tensordot(A_np, B_np, (cod_cut, 0))  # [*J1, *K1_rev, *J2, *K2_rev]
        T2_np = np.transpose(
            T2_np, [*range(cod_cut), *range(cod_cut + dom_cut, cod + dom), *range(cod_cut, cod_cut + dom_cut)]
        )
        # [*J1, *J2, *J2_rev, *K1_rev] = [*J, *J_rev]
        T_np = T.to_numpy(understood_braiding=True)
        npt.assert_almost_equal(T2_np, T_np)

    A_bent = tensors.permute_legs(A, [*range(cod_cut + 1, A.num_legs), *range(cod_cut)], [cod_cut], bend_right=False)
    B_bent = tensors.permute_legs(B, [0], [*reversed(range(1, B.num_legs))], bend_right=True)
    T2 = tensors.compose(A_bent, B_bent)
    T2 = tensors.permute_legs(
        T2, T.codomain_labels, T.domain_labels, bend_right=[False] * T2.num_codomain_legs + [True] * T2.num_domain_legs
    )
    assert tensors.almost_equal(T2, T), 'input tensor not reproduced'


@pytest.mark.parametrize('trunc', [False, None, 1e-100, 1e-10])
def test_fixes_scale_axis_bug(trunc):
    # there was a bug in scale_axis that this test isolated
    s = u1_symmetry * z3_symmetry
    backend = backends.get_backend('fusion_tree')
    codomain = TensorProduct(
        [
            ElementarySpace.from_defining_sectors(s, [[-1, 0], [-2, 2]], [2, 3], is_dual=True),
            ElementarySpace.from_defining_sectors(s, [[1, 1], [-3, 2], [0, 2]], [3, 1, 1], is_dual=True),
            ElementarySpace.from_defining_sectors(s, [[-2, 0], [0, 0], [123, 1]], [3, 1, 2], is_dual=True),
        ]
    )
    domain = TensorProduct(
        [
            ElementarySpace.from_defining_sectors(s, [[-2, 0]], [3]),
            ElementarySpace.from_defining_sectors(s, [[-124, 0], [-1, 1], [1, 1]], [3, 3, 3]),
            ElementarySpace.from_defining_sectors(s, [[2, 0]], [3]),
            ElementarySpace.from_defining_sectors(s, [[-2, 0], [1, 1], [123, 1]], [1, 1, 3]),
        ]
    )
    T = SymmetricTensor.from_random_normal(codomain, domain, backend=backend)
    T.test_sanity()

    if trunc is False:
        U, S, Vh = tensors.svd(T)
    else:
        U, S, Vh, _, _ = tensors.truncated_svd(T, svd_min=trunc)
    U.test_sanity()
    S.test_sanity()
    Vh.test_sanity()

    S_Vh = tensors.compose(S, Vh)
    S_Vh.test_sanity()

    U_S = tensors.compose(U, S)
    U_S.test_sanity()

    T2 = tensors.compose(U, S_Vh)
    T2.test_sanity()
    assert tensors.almost_equal(T, T2)

    T3 = tensors.compose(U_S, Vh)
    T3.test_sanity()
    assert tensors.almost_equal(T, T3)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, cod, dom, do_dagger, allow_basis_perm',
    [
        pytest.param(SymmetricTensor, 2, 2, True, True, id='Sym-2-2-True'),
        pytest.param(SymmetricTensor, 2, 2, False, True, id='Sym-2-2-False'),
        pytest.param(SymmetricTensor, 3, 0, True, True, id='Sym-3-0-True'),
        pytest.param(SymmetricTensor, 0, 2, False, True, id='Sym-0-2-False'),
        pytest.param(ChargedTensor, 2, 2, True, True, id='Charged-2-2-True'),
        pytest.param(ChargedTensor, 2, 2, False, True, id='Charged-2-2-False'),
        pytest.param(ChargedTensor, 3, 0, True, True, id='Charged-3-0-True'),
        pytest.param(ChargedTensor, 0, 2, False, True, id='Charged-0-2-False'),
        pytest.param(DiagonalTensor, 1, 1, True, True, id='Diag-1-1-True'),
        pytest.param(DiagonalTensor, 1, 1, False, True, id='Diag-1-1-False'),
        pytest.param(Mask, 1, 1, True, False, id='Mask-1-1-True'),
        pytest.param(Mask, 1, 1, False, False, id='Mask-1-1-False'),
    ],
    # TODO also test mixed types
)
def test_inner(cls, cod, dom, do_dagger, allow_basis_perm, make_compatible_tensor):
    """
    Parameters
    ----------
    cod, dom, do_dagger : from parametrize
    allow_basis_perm : bool from parametrize
        Mask random generation does not allow basis perms right now. This flag allows generating
        only legs with trivial basis perm.
    """
    A: Tensor = make_compatible_tensor(cod, dom, cls=cls, allow_basis_perm=allow_basis_perm)
    if do_dagger:
        B: Tensor = make_compatible_tensor(codomain=A.codomain, domain=A.domain, cls=cls)
    else:
        B: Tensor = make_compatible_tensor(codomain=A.domain, domain=A.codomain, cls=cls)

    if cls is Mask:
        with pytest.raises(NotImplementedError, match='tensors._compose_with_Mask not implemented for Mask'):
            _ = tensors.inner(A, B, do_dagger=do_dagger)
        pytest.xfail('apply_mask(Mask, Mask) not yet supported')

    res = tensors.inner(A, B, do_dagger=do_dagger)
    assert isinstance(res, A.backend.block_backend.Scalar)
    res = res.to_numpy()
    assert isinstance(res, (bool, float, complex, np.bool_, np.float32, np.float64, np.complex64, np.complex128))

    if A.symmetry.has_trivial_braid:
        if do_dagger:
            expect = np.sum(np.conj(A.to_numpy()) * B.to_numpy())
        else:
            expect = np.sum(np.transpose(A.to_numpy(), [*reversed(range(A.num_legs))]) * B.to_numpy())
        npt.assert_almost_equal(res, expect)
    else:
        pass  # TODO need to check some other way


def test_is_scalar():
    # python scalars
    assert tensors.is_scalar(42.0)
    assert tensors.is_scalar(1.0 + 3.0j)
    assert tensors.is_scalar(4e543j)
    assert tensors.is_scalar(3)
    assert tensors.is_scalar(True)
    # objects which are not scalar, just bc of their type
    assert not tensors.is_scalar('42')
    assert not tensors.is_scalar([3, 3, 1])
    assert not tensors.is_scalar([3])
    assert not tensors.is_scalar({'a': 3})
    assert not tensors.is_scalar(np.array(1))
    assert not tensors.is_scalar(np.array([True]))
    assert not tensors.is_scalar(np.array([1.0, 2]))
    # Tensors
    leg1 = ElementarySpace(z4_symmetry, [[2]])
    leg2 = ElementarySpace.from_defining_sectors(z4_symmetry, [[2], [3], [1]])
    backend = get_backend()
    scalar_tens1 = DummyTensor([leg1, leg1], [leg1, leg1], backend=backend, labels=None, dtype=Dtype.float64)
    scalar_tens2 = DummyTensor([], [leg1, leg1], backend=backend, labels=None, dtype=Dtype.float64)
    scalar_tens3 = DummyTensor([leg1], [leg1], backend=backend, labels=None, dtype=Dtype.float64)
    non_scalar_tens = DummyTensor([leg1, leg1], [leg2], backend=backend, labels=None, dtype=Dtype.float64)
    assert tensors.is_scalar(scalar_tens1)
    assert tensors.is_scalar(scalar_tens2)
    assert tensors.is_scalar(scalar_tens3)
    assert not tensors.is_scalar(non_scalar_tens)


def test_item(make_compatible_tensor, make_compatible_sectors, compatible_symmetry, np_random):
    sector_candidates = make_compatible_sectors(10)
    dims = compatible_symmetry.batch_qdim(sector_candidates)
    one_dim_sectors = sector_candidates[dims == 1, :]
    if len(one_dim_sectors) == 0:
        sector = compatible_symmetry.trivial_sector
    else:
        sector = np_random.choice(one_dim_sectors)
    leg = ElementarySpace(compatible_symmetry, [sector])
    T: SymmetricTensor = make_compatible_tensor([leg, leg], [leg, leg], cls=SymmetricTensor)

    res = tensors.item(T)

    if not T.symmetry.can_be_dropped:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc

    expect = T.to_numpy(understood_braiding=True).item()
    npt.assert_almost_equal(res.to_numpy(), expect)

    leg2 = ElementarySpace(compatible_symmetry, [sector], [3])
    non_scalar_T = make_compatible_tensor([leg, leg2], [leg, leg], cls=SymmetricTensor)
    with pytest.raises(ValueError, match='Not a scalar'):
        _ = tensors.item(non_scalar_T)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize('cls', [SymmetricTensor, DiagonalTensor, Mask, ChargedTensor])
def test_linear_combination(cls, make_compatible_tensor):
    if cls in [SymmetricTensor, ChargedTensor]:
        v = make_compatible_tensor(cls=cls, codomain=2, domain=2, max_block_size=3, max_blocks=3)
    elif cls is Mask:
        # Generating the second mask with the same legs can only be done right now if
        # permutation is trivial
        v = make_compatible_tensor(cls=cls, allow_basis_perm=False)
    else:
        v = make_compatible_tensor(cls=cls)
    w = make_compatible_tensor(like=v)

    compare_numpy = w.symmetry.can_be_dropped

    if compare_numpy:
        v_np = v.to_numpy(understood_braiding=True)
        w_np = w.to_numpy(understood_braiding=True)

    for valid_scalar in [0, 1.0, 2.0 + 3.0j, -42]:
        scalar_dtype = Dtype.float64 if v.dtype == Dtype.bool else v.dtype
        if isinstance(valid_scalar, complex) and valid_scalar.imag != 0 and scalar_dtype.is_real:
            scalar_dtype = scalar_dtype.to_complex
        scalar = v.backend.block_backend.as_scalar(valid_scalar, scalar_dtype)
        try:
            res = tensors.linear_combination(valid_scalar, v, 2 * valid_scalar, w)
        except NotImplementedError as e:
            pytest.xfail(str(e))
        res.test_sanity()
        if compare_numpy:
            expect = valid_scalar * v_np + 2 * valid_scalar * w_np
            npt.assert_almost_equal(res.to_numpy(understood_braiding=True), expect)
        if valid_scalar == 0:
            res_scalar = tensors.linear_combination(scalar, v, 2 * scalar, w)
            res_mixed = tensors.linear_combination(valid_scalar, v, 2 * scalar, w)
            res_scalar.test_sanity()
            res_mixed.test_sanity()
            if compare_numpy:
                npt.assert_allclose(res_scalar.to_numpy(understood_braiding=True), expect)
                npt.assert_allclose(res_mixed.to_numpy(understood_braiding=True), expect)
            continue
        # res = a * v + 2 * a * w  =>  v = res / a - 2 * w
        z = tensors.linear_combination(1.0 / valid_scalar, res, -2, w)
        z.test_sanity()
        assert tensors.almost_equal(v, z, allow_different_types=True)
        res_scalar = tensors.linear_combination(scalar, v, 2 * scalar, w)
        res_mixed = tensors.linear_combination(valid_scalar, v, 2 * scalar, w)
        res_scalar.test_sanity()
        res_mixed.test_sanity()
        if compare_numpy:
            npt.assert_allclose(res_scalar.to_numpy(understood_braiding=True), expect)
            npt.assert_allclose(res_mixed.to_numpy(understood_braiding=True), expect)
    for invalid_scalar in [None, (1, 2), v, 'abc']:
        with pytest.raises(TypeError, match='unsupported scalar types'):
            _ = tensors.linear_combination(invalid_scalar, v, invalid_scalar, w)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, cod, dom, leg, codomain_pos, domain_pos, levels',
    [
        pytest.param(SymmetricTensor, 2, 2, 0, 1, None, None, id='Sym-a'),
        pytest.param(SymmetricTensor, 2, 2, 0, 0, None, None, id='Sym-b'),
        pytest.param(SymmetricTensor, 2, 2, 0, None, 0, None, id='Sym-c'),
        pytest.param(SymmetricTensor, 2, 2, 3, 1, None, None, id='Sym-d'),
        pytest.param(SymmetricTensor, 2, 2, 3, 0, None, None, id='Sym-e'),
        pytest.param(SymmetricTensor, 2, 2, 3, None, 0, None, id='Sym-f'),
        pytest.param(DiagonalTensor, 1, 1, 0, None, 1, None, id='Diag-a'),
        pytest.param(DiagonalTensor, 1, 1, 0, 0, None, None, id='Diag-b'),
        pytest.param(ChargedTensor, 2, 2, 0, 1, None, None, id='Charged-a'),
        pytest.param(ChargedTensor, 2, 2, 0, None, 1, None, id='Charged-b'),
        pytest.param(ChargedTensor, 2, 2, 3, 0, None, None, id='Charged-c'),
    ],
)
def test_move_leg(cls, cod, dom, leg, codomain_pos, domain_pos, levels, make_compatible_tensor, np_random):
    """Test `move_leg` for tensors without specifying any levels. In the case where the symmetry
    is anyonic, that is, `move_leg` needs levels as input, random levels are chosen and the
    resulting numerical values are not checked against another tensor.
    Such checks are however done within the tests provided in `test_backend_nonabelian`.
    """
    assert sum(x is None for x in [domain_pos, codomain_pos]) == 1

    T_labels = list('abcdefghi')[: cod + dom]
    T: Tensor = make_compatible_tensor(cod, dom, labels=T_labels, cls=cls)

    if cls in [DiagonalTensor, Mask]:
        if (leg, codomain_pos, domain_pos) == (0, 0, None):
            # that set of inputs means no actual legs move and no conversion happens
            catch_warnings = nullcontext()
        else:
            catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    if not T.symmetry.has_symmetric_braid:
        # need to specify levels for moving the leg
        levels = list(np_random.permutation(T.num_legs))

    codomain_perm = [n for n in range(cod) if n != leg]
    domain_perm = [n for n in reversed(range(cod, cod + dom)) if n != leg]
    if codomain_pos is not None:
        codomain_perm[codomain_pos:codomain_pos] = [leg]
    if domain_pos is not None:
        domain_perm[domain_pos:domain_pos] = [leg]
    perm = [*codomain_perm, *reversed(domain_perm)]
    bend_right = np_random.choice([True, False])

    with catch_warnings:
        res = tensors.move_leg(
            T, leg, codomain_pos=codomain_pos, domain_pos=domain_pos, levels=levels, bend_right=bend_right
        )
    res.test_sanity()

    assert res.labels == [T_labels[n] for n in perm]
    assert res.legs == [T.get_leg(n) for n in perm]
    assert res.num_codomain_legs == cod + int(codomain_pos is not None) - int(leg < cod)

    compare_numpy = T.symmetry.can_be_dropped

    if compare_numpy:
        T_np = T.to_numpy(understood_braiding=True)
        res_np = res.to_numpy(understood_braiding=True)
        expect = swap_gate_numpy.permute_legs(
            T_np, T.num_codomain_legs, T.legs, codomain=codomain_perm, domain=domain_perm, bend_right=bend_right
        )
        npt.assert_almost_equal(res_np, expect, decimal=14)


def test_left_bends_fermions(block_backend, np_random):
    #
    #    |  |  |
    #    |  tens
    #    |  |  |
    #    .--.  |
    #          |
    backend = get_backend(fermion_parity, block_backend)
    tens = random_tensor(fermion_parity, 2, 2, backend=backend, np_random=np_random)
    res = tensors.permute_legs(tens, [1], [0, 3, 2], bend_right=False)
    tens_np = tens.to_numpy(understood_braiding=True)
    expect = np.transpose(tens_np, [1, 2, 3, 0])
    res_np = res.to_numpy(understood_braiding=True)
    npt.assert_almost_equal(np.abs(res_np), np.abs(expect))
    npt.assert_almost_equal(res_np, expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, cod, dom',
    [
        pytest.param(SymmetricTensor, 2, 2, id='Sym-2-2'),
        pytest.param(SymmetricTensor, 3, 1, id='Sym-3-1'),
        pytest.param(SymmetricTensor, 3, 0, id='Sym-2-0'),
        pytest.param(SymmetricTensor, 0, 2, id='Sym-0-2'),
        pytest.param(ChargedTensor, 0, 2, id='Charged-0-2'),
        pytest.param(ChargedTensor, 2, 2, id='Charged-2-2'),
        pytest.param(DiagonalTensor, 1, 1, id='Diag'),
        pytest.param(Mask, 1, 1, id='Mask'),
    ],
)
def test_norm(cls, cod, dom, make_compatible_tensor):
    T: Tensor = make_compatible_tensor(cod, dom, cls=cls)

    res = tensors.norm(T)
    assert isinstance(res, Scalar)

    if T.symmetry.can_be_dropped:
        expect = np.linalg.norm(T.to_numpy(understood_braiding=True))
        npt.assert_almost_equal(res.to_numpy(), expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases(
    get_cls=lambda kw: ChargedTensor if ChargedTensor in [kw['cls_A'], kw['cls_B']] else None
)
@pytest.mark.parametrize(
    'cls_A, cls_B, cA, dA, cB, dB',
    [
        pytest.param(SymmetricTensor, SymmetricTensor, 1, 2, 2, 1, id='Sym@Sym-1-2-2-1'),
        pytest.param(SymmetricTensor, SymmetricTensor, 2, 1, 1, 2, id='Sym@Sym-2-1-1-2'),
        pytest.param(SymmetricTensor, SymmetricTensor, 0, 3, 2, 0, id='Sym@Sym-0-3-2-0'),
        pytest.param(ChargedTensor, ChargedTensor, 1, 2, 2, 1, id='Charged@Charged-1-2-2-1'),
        pytest.param(ChargedTensor, ChargedTensor, 0, 3, 2, 0, id='Charged@Charged-0-3-2-0'),
        pytest.param(ChargedTensor, SymmetricTensor, 1, 2, 2, 1, id='Charged@Sym-1-2-2-1'),
        pytest.param(SymmetricTensor, ChargedTensor, 0, 3, 2, 0, id='Sym@Charged-0-3-2-0'),
    ],
)
def test_outer(cls_A, cls_B, cA, dA, cB, dB, make_compatible_tensor, compatible_symmetry):
    if compatible_symmetry.has_factor(_SU2):
        # need to make the test case smaller, so this does not need to many resources
        kwargs = dict(use_pipes=False, max_blocks=3)
        cA = dA = cB = dB = 1
    else:
        kwargs = {}
    A_labels = list('abcdefg')[: cA + dA]
    B_labels = list('hijklmn')[: cB + dB]
    A: Tensor = make_compatible_tensor(cA, dA, cls=cls_A, labels=A_labels, **kwargs)
    B: Tensor = make_compatible_tensor(cB, dB, cls=cls_B, labels=B_labels, **kwargs)

    if cls_A is ChargedTensor and cls_B is ChargedTensor:
        msg = reason = 'state_tensor_product not implemented'
        with pytest.raises(NotImplementedError, match=msg):
            _ = tensors.outer(A, B, relabel1={'a': 'x'}, relabel2={'h': 'y'})
        pytest.xfail(reason)

    res = tensors.outer(A, B, relabel1={'a': 'x'}, relabel2={'h': 'y'})

    res.test_sanity()
    A_relabelled = ['x', *A_labels[1:]]
    B_relabelled = ['y', *B_labels[1:]]
    assert res.labels == [*A_relabelled[:cA], *B_relabelled, *A_relabelled[cA:]]

    compare_numpy = A.symmetry.can_be_dropped

    if compare_numpy:
        perm = [*range(cA), *range(cA + dA, cA + cB + dB + dA), *range(cA, cA + dA)]
        A_np = A.to_numpy(understood_braiding=True)
        B_np = B.to_numpy(understood_braiding=True)
        res_np = res.to_numpy(understood_braiding=True)
        expect = np.transpose(np.tensordot(A_np, B_np, [(), ()]), perm)
        npt.assert_almost_equal(res_np, expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases(
    get_cls=lambda kw: ChargedTensor if ChargedTensor in [kw['cls_A'], kw['cls_B']] else None
)
@pytest.mark.parametrize(
    'cls_A, cls_B, legs_A, legs_B, A_contr_leg',
    [
        pytest.param(SymmetricTensor, SymmetricTensor, [2, 2], [1, 1], 1, id='Sym-Sym-2-2-1-1-codom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [3, 1], [2, 2], 1, id='Sym-Sym-3-1-2-2-codom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [3, 1], [1, 2], 1, id='Sym-Sym-3-1-1-2-codom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [3, 0], [2, 2], 1, id='Sym-Sym-3-0-2-2-codom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [3, 1], [0, 2], 1, id='Sym-Sym-3-1-0-2-codom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [2, 2], [1, 1], 2, id='Sym-Sym-2-2-1-1-dom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [1, 3], [2, 2], 2, id='Sym-Sym-1-3-2-2-dom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [1, 3], [2, 1], 2, id='Sym-Sym-1-3-2-1-dom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [0, 3], [2, 2], 0, id='Sym-Sym-0-3-2-2-dom'),
        pytest.param(SymmetricTensor, SymmetricTensor, [0, 3], [2, 0], 1, id='Sym-Sym-0-3-2-0-dom'),
        pytest.param(SymmetricTensor, Mask, [2, 2], [1, 1], 1, id='Sym-Mask-2-2-1-1-codom'),
        pytest.param(SymmetricTensor, Mask, [2, 2], [1, 1], 2, id='Sym-Mask-2-2-1-1-dom'),
        pytest.param(SymmetricTensor, DiagonalTensor, [2, 2], [1, 1], 1, id='Sym-Diag-2-2-1-1-codom'),
        pytest.param(SymmetricTensor, DiagonalTensor, [2, 2], [1, 1], 2, id='Sym-Diag-2-2-1-1-dom'),
        pytest.param(SymmetricTensor, ChargedTensor, [3, 1], [1, 2], 1, id='Sym-Charged-3-1-1-2-codom'),
        pytest.param(SymmetricTensor, ChargedTensor, [1, 3], [2, 1], 2, id='Sym-Charged-1-3-2-1-dom'),
        pytest.param(ChargedTensor, SymmetricTensor, [3, 1], [1, 2], 1, id='Charged-Sym-3-1-1-2-codom'),
        pytest.param(ChargedTensor, SymmetricTensor, [1, 3], [2, 1], 2, id='Charged-Sym-1-3-2-1-dom'),
        pytest.param(ChargedTensor, Mask, [2, 2], [1, 1], 1, id='Charged-Mask-2-2-1-1-codom'),
        pytest.param(ChargedTensor, Mask, [2, 2], [1, 1], 2, id='Charged-Mask-2-2-1-1-dom'),
        pytest.param(ChargedTensor, DiagonalTensor, [2, 2], [1, 1], 1, id='Charged-Diag-2-2-1-1-codom'),
        pytest.param(ChargedTensor, DiagonalTensor, [2, 2], [1, 1], 2, id='Charged-Diag-2-2-1-1-dom'),
        pytest.param(ChargedTensor, ChargedTensor, [3, 1], [1, 2], 1, id='Charged-Charged-3-1-1-2-codom'),
        pytest.param(ChargedTensor, ChargedTensor, [1, 3], [2, 1], 2, id='Charged-Charged-1-3-2-1-dom'),
    ],
)
def test_partial_compose(cls_A, cls_B, legs_A, legs_B, A_contr_leg, make_compatible_tensor, np_random):
    labels_A = [*list('abcde')[: legs_A[0]], *list('fghij')[: legs_A[1]][::-1]]
    labels_B = [*list('klmno')[: legs_B[0]], *list('pqrst')[: legs_B[1]][::-1]]
    A: Tensor = make_compatible_tensor(codomain=legs_A[0], domain=legs_A[1], labels=labels_A, cls=cls_A)

    if (A.has_pipes and cls_B is Mask) or (isinstance(A.get_leg(A_contr_leg), LegPipe) and cls_B is DiagonalTensor):
        # need to design such that the diagonal one doesnt need pipes
        pytest.xfail()

    if A_contr_leg < A.num_codomain_legs:
        codom_B = legs_B[0]
        dom_B = A.codomain[A_contr_leg : A_contr_leg + legs_B[1]]
        relabel1 = {'e': 'x'}
        relabel2 = {'i': 'y'}
        num_contr_legs = legs_B[1]
    else:
        domain_idx = A.num_legs - A_contr_leg - legs_B[0]
        codom_B = A.domain[domain_idx : domain_idx + legs_B[0]]
        dom_B = legs_B[1]
        relabel1 = {'a': 'x'}
        relabel2 = {'p': 'y'}
        num_contr_legs = legs_B[0]

    if cls_B is Mask and A_contr_leg >= A.num_codomain_legs:
        with pytest.raises(NotImplementedError, match='Mask generation broken'):
            _ = make_compatible_tensor(codomain=codom_B, domain=dom_B, labels=labels_B, cls=cls_B)
        pytest.xfail(reason='Mask generation broken')

    B: Tensor = make_compatible_tensor(codomain=codom_B, domain=dom_B, labels=labels_B, cls=cls_B)

    if isinstance(A, ChargedTensor) and isinstance(B, ChargedTensor):
        with pytest.raises(NotImplementedError, match='state_tensor_product not implemented'):
            _ = tensors.partial_compose(A, B, A_contr_leg, relabel1=relabel1, relabel2=relabel2)
        pytest.xfail(reason='state_tensor_product not implemented')

    res = tensors.partial_compose(A, B, A_contr_leg, relabel1=relabel1, relabel2=relabel2)
    res.test_sanity()

    if isinstance(B, Mask):
        expect = tensors.apply_mask(A, B, A_contr_leg)
        expect.relabel({A.labels[A_contr_leg]: B.labels[int(A_contr_leg >= A.num_codomain_legs)]})
        expect.relabel(relabel1)
        expect.relabel(relabel2)
    elif isinstance(B, DiagonalTensor):
        expect = tensors.scale_axis(A, B, A_contr_leg)
        expect.relabel({A.labels[A_contr_leg]: B.labels[int(A_contr_leg >= A.num_codomain_legs)]})
        expect.relabel(relabel1)
        expect.relabel(relabel2)
    else:
        # we can treat the invariant part of ChargedTensor in the same way, but need to take some extra care for the charge leg
        A_ = A.invariant_part if isinstance(A, ChargedTensor) else A
        B_ = B.invariant_part if isinstance(B, ChargedTensor) else B
        c = ChargedTensor._CHARGE_LEG_LABEL
        if isinstance(B, ChargedTensor):
            relabel2 = {**relabel2, c: f'{c}B'}

        # compare to compose (bend legs accordingly before and after composing)
        A_last_leg = A_contr_leg + num_contr_legs - 1
        if A_contr_leg < A_.num_codomain_legs:
            if isinstance(B, ChargedTensor):
                B_ = tensors.move_leg(B_, c, codomain_pos=B_.num_codomain_legs - 1, bend_right=True)
            inter_codom = list(range(A_contr_leg, A_last_leg + 1))
            inter_dom = list(reversed(range(A_contr_leg))) + list(reversed(range(A_last_leg + 1, A_.num_legs)))
            num_bend_right = A_.num_codomain_legs - 1 - A_last_leg
            inter_bend_right = [False] * A_contr_leg + [True] * (A_.num_legs - A_contr_leg)
            expect = tensors.compose(
                B_,
                tensors.permute_legs(A_, inter_codom, inter_dom, bend_right=inter_bend_right),
                relabel1=relabel2,
                relabel2=relabel1,
            )
            final_codom = list(range(expect.num_legs - A_contr_leg, expect.num_legs)) + list(
                range(expect.num_codomain_legs + num_bend_right)
            )
            final_dom = list(reversed(range(expect.num_codomain_legs + num_bend_right, expect.num_legs - A_contr_leg)))
            final_bend_right = [True] * (expect.num_legs - A_contr_leg) + [False] * A_contr_leg
            expect = tensors.permute_legs(expect, final_codom, final_dom, bend_right=final_bend_right)
        else:
            inter_codom = list(range(A_last_leg + 1, A_.num_legs)) + list(range(A_contr_leg))
            inter_dom = list(reversed(range(A_contr_leg, A_last_leg + 1)))
            num_bend_left = A_.num_legs - 1 - A_last_leg
            num_bend_right = A_contr_leg - A.num_codomain_legs
            inter_bend_right = [True] * A_contr_leg + [False] * (A_.num_legs - A_contr_leg)
            expect = tensors.compose(
                tensors.permute_legs(A_, inter_codom, inter_dom, bend_right=inter_bend_right),
                B_,
                relabel1=relabel1,
                relabel2=relabel2,
            )
            final_codom = list(range(num_bend_left, expect.num_codomain_legs - num_bend_right))
            final_dom = list(reversed(range(num_bend_left))) + list(
                reversed(range(expect.num_codomain_legs - num_bend_right, expect.num_legs))
            )
            final_bend_right = [False] * num_bend_left + [True] * (expect.num_legs - num_bend_left)
            expect = tensors.permute_legs(expect, final_codom, final_dom, bend_right=final_bend_right)

        if isinstance(A, ChargedTensor) and isinstance(B, ChargedTensor):
            expect = tensors.move_leg(expect, f'{c}B', domain_pos=1, bend_right=True)
            expect = ChargedTensor.from_two_charge_legs(expect, state1=A.charged_state, state2=B.charged_state)
        elif isinstance(A, ChargedTensor):
            expect = ChargedTensor.from_invariant_part(expect, A.charged_state)
        elif isinstance(B, ChargedTensor):
            expect = tensors.move_leg(expect, f'{c}B', domain_pos=0, bend_right=True).relabel({f'{c}B': c})
            expect = ChargedTensor.from_invariant_part(expect, B.charged_state)

    assert res.labels == expect.labels
    assert tensors.almost_equal(res, expect)


@pytest.mark.parametrize('in_domain', [False, True])
def test_partial_compose_after_pipe(in_domain, make_compatible_tensor, make_compatible_space):
    # A pipe *before* the contracted leg: leg indices count the pipe once, but the FusionTreeBackend
    # has trees over its components. Compare with contracting before combining the legs.
    X0, X1, Y, D, W = [make_compatible_space(max_sectors=3, max_mult=2) for _ in range(5)]
    if in_domain:
        A0 = make_compatible_tensor([D], [X0, X1, Y], labels=['d', 'y', 'x1', 'x0'], use_pipes=False)
        B = make_compatible_tensor([Y], [W], labels=['y', 'z'], use_pipes=False)
        pipe_legs = ['x1', 'x0']
    else:
        A0 = make_compatible_tensor([X0, X1, Y], [D], labels=['x0', 'x1', 'y', 'd'], use_pipes=False)
        B = make_compatible_tensor([W], [Y], labels=['z', 'y'], use_pipes=False)
        pipe_legs = ['x0', 'x1']
    A = tensors.combine_legs(A0, pipe_legs)
    expect = tensors.combine_legs(tensors.partial_compose(A0, B, 'y'), pipe_legs)
    pipe_factors = A.domain.factors if in_domain else A.codomain.factors
    assert isinstance(pipe_factors[0], LegPipe)  # the pipe comes before 'y'
    res = tensors.partial_compose(A, B, 'y')
    res.test_sanity()
    assert res.labels == expect.labels
    assert tensors.almost_equal(res, expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, codom, dom',
    [
        pytest.param(SymmetricTensor, ['a', 'b', 'a'], ['c', 'd'], id='Sym-aba-cd'),
        pytest.param(SymmetricTensor, ['a', 'b'], ['b', 'a'], id='Sym-ab-ba'),
        pytest.param(SymmetricTensor, ['a', 'c'], ['b', 'a'], id='Sym-ac-ba'),
        pytest.param(SymmetricTensor, ['a', 'b'], ['c', 'd'], id='Sym-ab-cd'),
        pytest.param(ChargedTensor, ['a', 'b'], ['b', 'a'], id='Charged-ab-ba'),
        pytest.param(ChargedTensor, ['a', 'b', 'a'], ['c', 'd'], id='Charged-aba-cd'),
        pytest.param(DiagonalTensor, ['a'], ['a'], id='Diag-a-a'),
    ],
)
def test_partial_trace(cls, codom, dom, make_compatible_space, make_compatible_tensor, np_random):
    #
    # 1) Prepare inputs
    #
    trace_legs = {l: make_compatible_space() for l in duplicate_entries([*codom, *dom])}
    # build compatible legs.
    # If we see a label for the second time, use opposite duality than the first time, and different label.
    # In the domain, use opposite duality than in the codomain.
    seen_labels = []
    codomain_spaces = []
    codomain_labels = []
    for l in codom:
        if l in seen_labels:
            codomain_spaces.append(trace_legs[l].dual)
            codomain_labels.append(f'{l}*')
        elif l in trace_legs:
            codomain_spaces.append(trace_legs[l])
            seen_labels.append(l)
            codomain_labels.append(l)
        else:
            codomain_spaces.append(make_compatible_space())
            codomain_labels.append(l)
    domain_spaces = []
    domain_labels = []
    for l in dom:
        if l in seen_labels:
            domain_spaces.append(trace_legs[l])
            domain_labels.append(f'{l}*')
        elif l in trace_legs:
            domain_spaces.append(trace_legs[l].dual)
            domain_labels.append(l)
            seen_labels.append(l)
        else:
            domain_spaces.append(make_compatible_space())
            domain_labels.append(l)
    #
    T: Tensor = make_compatible_tensor(
        codomain_spaces, domain_spaces, cls=cls, labels=[*codomain_labels, *reversed(domain_labels)]
    )
    #
    how_to_call = np_random.choice(['positions', 'labels'])
    labels = T.labels
    pairs_positions = [(labels.index(l), labels.index(f'{l}*')) for l in trace_legs]
    if how_to_call == 'positions':
        pairs = pairs_positions
    if how_to_call == 'labels':
        pairs = [(l, f'{l}*') for l in trace_legs]
    #
    # 2) Call the actual function
    #
    levels = None
    if isinstance(T.backend, backends.FusionTreeBackend) and T.symmetry.braiding_style.value >= 20:
        levels = list(np_random.permutation(T.num_legs - len(pairs)))
        idcs1 = [p[0] for p in pairs_positions]
        idcs2 = [p[1] for p in pairs_positions]
        for idx in np.argsort(idcs2):
            level = levels[idcs1[idx]]
            levels = [l + 1 if l > level else l for l in levels]
            levels.insert(idcs2[idx], level + 1)

    if cls is ChargedTensor and len(pairs) * 2 == T.num_legs:
        pass  # full trace of ChargedTensor with specified charged_state is allowed
    #
    res = tensors.partial_trace(T, *pairs, levels=levels)
    #
    # 3) Test the result
    compare_numpy = T.symmetry.can_be_dropped

    if compare_numpy:
        num_open = T.num_legs - 2 * len(pairs)
        if num_open == 0:
            assert isinstance(res, Scalar)
            res_np = res.to_numpy()
        else:
            assert isinstance(res, cls)
            res.test_sanity()
            assert res.labels == [l for l in T.labels if l[0] not in trace_legs]
            assert res.codomain.factors == [
                sp for sp, l in zip(T.codomain, T.codomain_labels) if l[0] not in trace_legs
            ]
            assert res.domain.factors == [sp for sp, l in zip(T.domain, T.domain_labels) if l[0] not in trace_legs]
            res_np = res.to_numpy(understood_braiding=True)
        #
        idcs1 = [p[0] for p in pairs_positions]
        idcs2 = [p[1] for p in pairs_positions]
        remaining = [n for n in range(T.num_legs) if n not in idcs1 and n not in idcs2]
        T_block = T.to_dense_block(understood_braiding=True)
        expect = T.backend.block_backend.trace_partial(T_block, idcs1, idcs2, remaining)
        expect = T.backend.block_backend.to_numpy(expect)

        if T.symmetry.has_trivial_braid:
            npt.assert_almost_equal(res_np, expect)
        else:
            # TODO need to carefully build the expected result.
            pass

    # TODO what else can we check?


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, num_cod, num_dom, codomain, domain, levels, bend_right',
    [
        pytest.param(SymmetricTensor, 2, 2, [0, 1], [3, 2], None, True, id='Symmetric-2-2-trivial'),
        pytest.param(SymmetricTensor, 2, 2, [1, 0], [3, 2], [0, 1, 2, 3], True, id='Symmetric-2-2-braid_cod'),
        pytest.param(SymmetricTensor, 2, 2, [0, 1], [2, 3], [0, 1, 2, 3], True, id='Symmetric-2-2-braid_dom'),
        pytest.param(SymmetricTensor, 2, 2, [1, 0], [2, 3], [0, 1, 2, 3], True, id='Symmetric-2-2-braid_both'),
        pytest.param(SymmetricTensor, 2, 2, [0, 1, 2], [3], None, True, id='Symmetric-2-2-bend_right'),
        pytest.param(SymmetricTensor, 2, 2, [3, 0, 1], [2], None, False, id='Symmetric-2-2-bend_left'),
        pytest.param(SymmetricTensor, 2, 2, [0, 3], [1, 2], [0, 1, 2, 3], True, id='Symmetric-2-2-general'),
        pytest.param(SymmetricTensor, 2, 2, [0, 3], [1, 2], [0, 1, 2, 3], False, id='Symmetric-2-2-general2'),
        pytest.param(SymmetricTensor, 3, 3, [1, 4, 0], [2, 3, 5], [4, 1, 3, 5, 0, 2], True, id='Symmetric-3-3-general'),
        pytest.param(DiagonalTensor, 1, 1, [0], [1], [0, 1], True, id='Diagonal-trivial'),
        pytest.param(DiagonalTensor, 1, 1, [1], [0], [0, 1], True, id='Diagonal-swap'),
        pytest.param(DiagonalTensor, 1, 1, [1], [0], [0, 1], [True, False], id='Diagonal-transpose'),
        pytest.param(DiagonalTensor, 1, 1, [1, 0], [], [0, 1], True, id='Diagonal-general'),
        pytest.param(Mask, 1, 1, [0], [1], [0, 1], True, id='Mask-trivial'),
        pytest.param(Mask, 1, 1, [1], [0], [0, 1], True, id='Mask-swap'),
        pytest.param(Mask, 1, 1, [1], [0], [0, 1], [True, False], id='Mask-transpose'),
        pytest.param(Mask, 1, 1, [1, 0], [], [0, 1], True, id='Mask-general'),
        pytest.param(ChargedTensor, 2, 2, [0, 1], [3, 2], None, True, id='Charged-2-2-trivial'),
        pytest.param(ChargedTensor, 2, 2, [1, 0], [2, 3], [0, 1, 2, 3], True, id='Charged-2-2-braid'),
        pytest.param(ChargedTensor, 2, 2, [0, 1, 2], [3], None, True, id='Charged-2-2-bend'),
        pytest.param(ChargedTensor, 2, 2, [0, 3], [1, 2], [0, 1, 2, 3], True, id='Charged-2-2-general'),
    ],
)
def test_permute_legs(
    cls, num_cod, num_dom, codomain, domain, levels, bend_right, make_compatible_tensor, compatible_symmetry, np_random
):
    if compatible_symmetry.has_factor(_SU2) and (num_cod + num_dom) > 4:
        # make sure we dont need symmetry data for too large sectors
        sectors = [[0], [1], [2]]
        legs = []
        for _ in range(num_cod + num_dom):
            mults = np_random.integers(1, 5, size=3)
            is_dual = np_random.choice([True, False])
            legs.append(ElementarySpace(compatible_symmetry, sectors, mults, is_dual))
        T_codomain = legs[:num_cod]
        T_domain = legs[num_cod:]
    else:
        T_codomain = num_cod
        T_domain = num_dom
    T = make_compatible_tensor(T_codomain, T_domain, max_block_size=3, cls=cls)

    if cls in [DiagonalTensor, Mask]:
        if (codomain, domain) == ([0], [1]):
            # trivial permutation -> no warning expected
            catch_warnings = nullcontext()
        elif (codomain, domain) == ([1], [0]) and (bend_right == [True, False] or T.symmetry.has_trivial_braid):
            # special case transpose -> no warning expected
            catch_warnings = nullcontext()
        else:
            catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    with catch_warnings:
        res = tensors.permute_legs(T, codomain, domain, levels, bend_right=bend_right)
    res.test_sanity()

    if issubclass(cls, (DiagonalTensor, Mask)):
        res2 = tensors.permute_legs(T.as_SymmetricTensor(), codomain, domain, levels, bend_right=bend_right)
        assert tensors.almost_equal(res2, res, allow_different_types=True)

    for n, i in enumerate(codomain):
        assert res.codomain[n] == T._as_codomain_leg(i)
    for n, i in enumerate(domain):
        assert res.domain[n] == T._as_domain_leg(i)
    assert res.codomain_labels == [T.labels[n] for n in codomain]
    assert res.domain_labels == [T.labels[n] for n in domain]

    compare_numpy = T.symmetry.can_be_dropped

    if compare_numpy:
        T_np = T.to_numpy(understood_braiding=True)
        actual = res.to_numpy(understood_braiding=True)
        perm = [*codomain, *reversed(domain)]
        perm_is_trivial = perm == list(range(len(perm)))
        expect = swap_gate_numpy.permute_legs(
            T_np,
            num_codomain_legs=T.num_codomain_legs,
            legs=T.legs,
            codomain=codomain,
            domain=domain,
            bend_right=bend_right,
        )
        npt.assert_almost_equal(actual, expect, decimal=14)

    # construct the instructions needed to undo the original instructions
    leg_perm = [*codomain, *reversed(domain)]
    inv_leg_perm = inverse_permutation(leg_perm)
    codomain2 = inv_leg_perm[:num_cod]
    domain2 = inv_leg_perm[num_cod:][::-1]
    levels2 = None if levels is None else [levels[i] for i in leg_perm]

    res2 = tensors.permute_legs(res, codomain2, domain2, levels2, bend_right=bend_right)
    res2.test_sanity()
    assert tensors.almost_equal(res2, T, allow_different_types=True)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, dom, cod, new_leg_dual',
    [
        pytest.param(SymmetricTensor, 1, 1, False, id='Sym-1-1-False'),
        pytest.param(SymmetricTensor, 1, 1, True, id='Sym-1-1-True'),
        pytest.param(SymmetricTensor, 3, 1, False, id='Sym-3-1-False'),
        pytest.param(SymmetricTensor, 2, 2, False, id='Sym-2-2-False'),
        pytest.param(SymmetricTensor, 2, 2, True, id='Sym-2-2-True'),
        pytest.param(DiagonalTensor, 1, 1, False, id='Diag-False'),
        pytest.param(DiagonalTensor, 1, 1, True, id='Diag-True'),
        pytest.param(Mask, 1, 1, False, id='Mask-False'),
        pytest.param(Mask, 1, 1, True, id='Mask-True'),
        pytest.param(ChargedTensor, 1, 1, False, id='Charged-1-1-False'),
        pytest.param(ChargedTensor, 1, 1, True, id='Charged-1-1-True'),
        pytest.param(ChargedTensor, 2, 1, False, id='Charged-2-1-False'),
        pytest.param(ChargedTensor, 1, 2, False, id='Charged-1-2-True'),
    ],
)
def test_qr_lq(cls, dom, cod, new_leg_dual, make_compatible_tensor):
    T_labels = list('efghijk')[: dom + cod]
    T: Tensor = make_compatible_tensor(dom, cod, cls=cls, labels=T_labels)

    Q, R = tensors.qr(T, new_leg_dual=new_leg_dual, charge_leg_top=True)
    Q.test_sanity()
    R.test_sanity()
    assert tensors.almost_equal(Q @ R, T, allow_different_types=True)
    eye = tensors.SymmetricTensor.from_eye(Q.domain, backend=T.backend)
    assert tensors.almost_equal(Q.hc @ Q, eye, allow_different_types=True)

    if isinstance(T, ChargedTensor):
        assert isinstance(R, ChargedTensor)
        assert isinstance(Q, SymmetricTensor)
        # for charge_leg_top = True above; for charge_leg_top = False below

        Q, R = tensors.qr(T, new_leg_dual=new_leg_dual, charge_leg_top=False)
        Q.test_sanity()
        R.test_sanity()
        assert isinstance(Q, ChargedTensor)
        assert isinstance(R, SymmetricTensor)
        assert tensors.almost_equal(Q @ R, T, allow_different_types=True)
        Q_iso = tensors.move_leg(Q.invariant_part, Q._CHARGE_LEG_LABEL, codomain_pos=0, bend_right=False)
        # different position of charge_leg -> different total charges -> different eye
        eye = tensors.SymmetricTensor.from_eye(Q_iso.domain, backend=T.backend)
        assert tensors.almost_equal((Q_iso.hc @ Q_iso), eye, allow_different_types=True)

    L, Q2 = tensors.lq(T, new_leg_dual=new_leg_dual, charge_leg_top=False)
    L.test_sanity()
    Q2.test_sanity()
    assert tensors.almost_equal(L @ Q2, T, allow_different_types=True)
    eye = tensors.SymmetricTensor.from_eye(Q2.codomain, backend=T.backend)
    assert tensors.almost_equal(Q2 @ Q2.hc, eye, allow_different_types=True)

    if isinstance(T, ChargedTensor):
        assert isinstance(L, ChargedTensor)
        assert isinstance(Q2, SymmetricTensor)
        # for charge_leg_top = False above; for charge_leg_top = True below

        L, Q2 = tensors.lq(T, new_leg_dual=new_leg_dual, charge_leg_top=True)
        L.test_sanity()
        Q2.test_sanity()
        assert isinstance(Q2, ChargedTensor)
        assert isinstance(L, SymmetricTensor)
        assert tensors.almost_equal(L @ Q2, T, allow_different_types=True)
        Q2_iso = Q2.invariant_part
        # compose does not contract the charge legs, which we do contract for the isometry
        eye = tensors.SymmetricTensor.from_eye(Q2_iso.codomain, backend=T.backend)
        assert tensors.almost_equal(Q2_iso @ Q2_iso.hc, eye, allow_different_types=True)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize('cls', [SymmetricTensor, DiagonalTensor, Mask, ChargedTensor])
def test_scalar_multiply(cls, make_compatible_tensor):
    if cls in [SymmetricTensor, ChargedTensor]:
        T = make_compatible_tensor(cls=cls, codomain=2, domain=2, max_block_size=3, max_blocks=3)
    else:
        T = make_compatible_tensor(cls=cls)

    if isinstance(T, Mask):
        catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    compare_numpy = T.symmetry.can_be_dropped
    if compare_numpy:
        T_np = T.to_numpy(understood_braiding=True)

    for valid_scalar in [0, 1.0, 2.0 + 3.0j, -42]:
        scalar_dtype = Dtype.float64 if T.dtype == Dtype.bool else T.dtype
        if isinstance(valid_scalar, complex) and valid_scalar.imag != 0 and scalar_dtype.is_real:
            scalar_dtype = scalar_dtype.to_complex
        scalar = T.backend.block_backend.as_scalar(valid_scalar, scalar_dtype)
        with catch_warnings:
            res = tensors.scalar_multiply(valid_scalar, T)
        res.test_sanity()
        if compare_numpy:
            npt.assert_almost_equal(res.to_numpy(understood_braiding=True), valid_scalar * T_np)
        with catch_warnings:
            res_scalar = tensors.scalar_multiply(scalar, T)
        res_scalar.test_sanity()
        if compare_numpy:
            npt.assert_almost_equal(res_scalar.to_numpy(understood_braiding=True), valid_scalar * T_np)
        with catch_warnings:
            left_mul = scalar * T
            right_mul = T * scalar
        left_mul.test_sanity()
        right_mul.test_sanity()
        if compare_numpy:
            npt.assert_almost_equal(left_mul.to_numpy(understood_braiding=True), valid_scalar * T_np)
            npt.assert_almost_equal(right_mul.to_numpy(understood_braiding=True), valid_scalar * T_np)
    for invalid_scalar in [None, (1, 2), T, 'abc']:
        with pytest.raises(TypeError, match='unsupported scalar type'):
            _ = tensors.scalar_multiply(invalid_scalar, T)


def test_block_backend_as_scalar_passthrough(make_compatible_tensor):
    T = make_compatible_tensor(cls=SymmetricTensor, codomain=1, domain=1, max_block_size=2, max_blocks=2)
    bb = T.backend.block_backend
    scalar = bb.as_scalar(1.5)
    same = bb.as_scalar(scalar)
    assert same.dtype == scalar.dtype
    npt.assert_equal(same.to_numpy(), scalar.to_numpy())

    int_scalar = bb.as_scalar(2)
    assert int_scalar.dtype == Dtype.int64
    assert int_scalar.as_int64() == 2
    assert bb.as_scalar(True).dtype == Dtype.bool
    assert bb.as_scalar(True).as_bool() is True


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, codom, dom, which_leg',
    [
        pytest.param(SymmetricTensor, 2, 2, 1, id='Sym-2-2-1'),
        pytest.param(SymmetricTensor, 2, 2, 3, id='Sym-2-2-3'),
        pytest.param(SymmetricTensor, 0, 2, 1, id='Sym-0-2-1'),
        pytest.param(SymmetricTensor, 3, 0, 1, id='Sym-3-0-1'),
        pytest.param(ChargedTensor, 2, 2, 1, id='Charged-2-2-1'),
        pytest.param(ChargedTensor, 2, 2, 3, id='Charged-2-2-3'),
        pytest.param(DiagonalTensor, 1, 1, 0, id='Diag-0'),
        pytest.param(DiagonalTensor, 1, 1, 1, id='Diag-1'),
        pytest.param(Mask, 1, 1, 0, id='Mask-0'),
        pytest.param(Mask, 1, 1, 1, id='Mask-1'),
    ],
)
def test_scale_axis(cls, codom, dom, which_leg, make_compatible_tensor, np_random):
    # 1) Prepare
    D = make_compatible_tensor(cls=DiagonalTensor, labels=['x', 'y'])
    T_labels = list('abcdefghi')[: codom + dom]
    if which_leg >= codom:
        num_legs = codom + dom
        dom = [None] * dom
        dom[num_legs - which_leg - 1] = D.leg if np_random.choice([True, False]) else D.leg.dual
    else:
        codom = [None] * codom
        codom[which_leg] = D.leg if np_random.choice([True, False]) else D.leg.dual
    if cls is Mask and which_leg == 0:
        with pytest.raises(NotImplementedError, match='Mask generation broken'):
            _ = make_compatible_tensor(codom, dom, cls=cls, labels=T_labels)
        pytest.xfail(reason='Mask generation broken')
    T: tensors.Tensor = make_compatible_tensor(codom, dom, cls=cls, labels=T_labels)

    if cls is Mask:
        catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
    else:
        catch_warnings = nullcontext()

    # 2) Call functions
    how_to_call = np_random.choice(['by_idx', 'by_label'])
    if how_to_call == 'by_idx':
        with catch_warnings:
            res = tensors.scale_axis(T, D, which_leg)
    if how_to_call == 'by_label':
        with catch_warnings:
            res = tensors.scale_axis(T, D, T_labels[which_leg])

    # 3) check tensor properties
    res.test_sanity()
    assert isinstance(res, SymmetricTensor if cls is Mask else cls)
    assert res.codomain == T.codomain
    assert res.domain == T.domain
    assert res.labels == T_labels

    # 4) compare to numpy
    compare_numpy = T.symmetry.can_be_dropped
    if compare_numpy:
        T_np = T.to_numpy(understood_braiding=True)
        expect = np.swapaxes(T_np, which_leg, -1)  # swap axis to be scaled to the back
        expect = expect * D.diagonal_as_numpy()  # broadcasts to last axis of expect
        expect = np.swapaxes(expect, which_leg, -1)  # swap back
        npt.assert_almost_equal(res.to_numpy(understood_braiding=True), expect, decimal=14)


def test_squeeze_legs(make_compatible_tensor, compatible_symmetry):
    trivial_leg = ElementarySpace.from_trivial_sector(1, symmetry=compatible_symmetry)
    T = make_compatible_tensor(
        [None, trivial_leg, trivial_leg, None], [None, None, trivial_leg], labels=list('abcdefg')
    )

    res_all = tensors.squeeze_legs(T)
    res_all.test_sanity()
    res_1 = tensors.squeeze_legs(T, 1)
    res_1.test_sanity()
    res_2 = tensors.squeeze_legs(T, [1, 4])
    res_2.test_sanity()

    assert res_all.labels == ['a', 'd', 'f', 'g']
    assert res_1.labels == ['a', 'c', 'd', 'e', 'f', 'g']
    assert res_2.labels == ['a', 'c', 'd', 'f', 'g']

    if T.symmetry.can_be_dropped:
        T_np = T.to_numpy(understood_braiding=True)
        expect_all = T_np[:, 0, 0, :, 0, :, :]
        expect_1 = T_np[:, 0]
        expect_2 = T_np[:, 0, :, :, 0]
        npt.assert_almost_equal(res_all.to_numpy(understood_braiding=True), expect_all)
        npt.assert_almost_equal(res_1.to_numpy(understood_braiding=True), expect_1)
        npt.assert_almost_equal(res_2.to_numpy(understood_braiding=True), expect_2)
    else:
        return  # TODO  Need to re-design checks, cant use .to_numpy() etc


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, dom, cod, new_leg_dual',
    [
        pytest.param(SymmetricTensor, 1, 1, False, id='Sym-1-1-False'),
        pytest.param(SymmetricTensor, 1, 3, False, id='Sym-1-3-False'),
        pytest.param(SymmetricTensor, 3, 1, False, id='Sym-3-1-False'),
        pytest.param(SymmetricTensor, 2, 2, False, id='Sym-2-2-False'),
        pytest.param(SymmetricTensor, 2, 2, True, id='Sym-2-2-True'),
        pytest.param(DiagonalTensor, 1, 1, False, id='Diag-False'),
        pytest.param(DiagonalTensor, 1, 1, True, id='Diag-True'),
        pytest.param(Mask, 1, 1, False, id='Mask-False'),
        pytest.param(Mask, 1, 1, True, id='Mask-True'),
        pytest.param(ChargedTensor, 1, 1, False, id='Charged-1-1-False'),
        pytest.param(ChargedTensor, 1, 1, True, id='Charged-1-1-True'),
        pytest.param(ChargedTensor, 2, 1, False, id='Charged-2-1-False'),
        pytest.param(ChargedTensor, 1, 2, False, id='Charged-1-2-True'),
    ],
)
def test_svd(cls, dom, cod, new_leg_dual, make_compatible_tensor):
    """Test svd and various related functions. Covers

    - svd
    - truncated_svd
        - truncate_singular_values
        - svd_apply_mask
    """
    T_labels = list('efghijklmn')[: dom + cod]
    T: Tensor = make_compatible_tensor(dom, cod, labels=T_labels, cls=cls)

    print('Normal (non-truncated) SVD')

    U, S, Vh = tensors.svd(T, new_labels=['a', 'b', 'c', 'd'], new_leg_dual=new_leg_dual)
    U.test_sanity()
    S.test_sanity()
    Vh.test_sanity()
    assert U.labels == [*T.codomain_labels, 'a']
    assert S.labels == ['b', 'c']
    assert Vh.labels == ['d', *reversed(T.domain_labels)]

    assert isinstance(S, DiagonalTensor)
    assert (S >= 0).all()
    npt.assert_almost_equal(tensors.norm(S).to_numpy(), tensors.norm(T).to_numpy())

    assert tensors.almost_equal(U @ S @ Vh, T, allow_different_types=True)
    eye = tensors.SymmetricTensor.from_eye(S.domain, backend=T.backend)
    assert tensors.almost_equal(U.hc @ U, eye, allow_different_types=True)
    if isinstance(Vh, ChargedTensor):
        # need to contract charge_leg
        assert tensors.almost_equal(Vh.invariant_part @ Vh.invariant_part.hc, eye, allow_different_types=True)
    else:
        assert tensors.almost_equal(Vh @ Vh.hc, eye, allow_different_types=True)

    if isinstance(T, ChargedTensor):
        assert isinstance(Vh, ChargedTensor)
        assert isinstance(U, SymmetricTensor)

        # for charge_leg_top = True above; for charge_leg_top = False below
        U, S, Vh = tensors.svd(T, new_labels=['a', 'b', 'c', 'd'], new_leg_dual=new_leg_dual, charge_leg_top=False)
        U.test_sanity()
        S.test_sanity()
        Vh.test_sanity()
        assert U.labels == [*T.codomain_labels, 'a']
        assert S.labels == ['b', 'c']
        assert Vh.labels == ['d', *reversed(T.domain_labels)]

        assert isinstance(S, DiagonalTensor)
        assert (S >= 0).all()
        npt.assert_almost_equal(tensors.norm(S).to_numpy(), tensors.norm(T).to_numpy())

        assert isinstance(U, ChargedTensor)
        assert isinstance(Vh, SymmetricTensor)

        assert tensors.almost_equal(U @ S @ Vh, T, allow_different_types=True)
        eye = tensors.SymmetricTensor.from_eye(S.domain, backend=T.backend)
        U_iso = tensors.move_leg(U.invariant_part, U._CHARGE_LEG_LABEL, codomain_pos=0, bend_right=False)
        assert tensors.almost_equal(U_iso.hc @ U_iso, eye, allow_different_types=True)
        assert tensors.almost_equal(Vh @ Vh.hc, eye, allow_different_types=True)

    S_full = S
    did_any_nontrivial_trunction = False
    print('Truncated SVD')
    for svd_min, normalize_to, chi_max in [
        (1e-14, None, None),
        (0.1, None, None),
        (0.3, None, None),
        (0.999, None, None),
        (0.4, 2.7, None),
        (0.0, None, 1),
        (0.0, 2.7, 1),
    ]:
        norm_T = tensors.norm(T).to_numpy()
        if svd_min >= S_full.max() / norm_T:
            with pytest.warns(UserWarning, match="truncation: can't satisfy constraint for svd_min"):
                U, S, Vh, err, renormalize = tensors.truncated_svd(T, svd_min=svd_min, chi_max=chi_max)
            return

        U, S, Vh, err, renormalize = tensors.truncated_svd(
            T, new_leg_dual=new_leg_dual, normalize_to=normalize_to, svd_min=svd_min, chi_max=chi_max
        )
        min_chi_needed = int(tensors.trace(S_full > 1e-14).to_numpy())
        if S.leg.dim < min_chi_needed:
            assert err > 0  # make sure we did a non-trivial truncation
            did_any_nontrivial_trunction = True
        U.test_sanity()
        S.test_sanity()
        Vh.test_sanity()
        lambda_kept = S / norm_T if normalize_to is None else (renormalize / norm_T) * S

        # 1) Check that T = T_approx + T_ortho = renormalize * U @ S @ Vh + T_ortho
        #   -> Uh @ T @ V = renormalize * S
        if not isinstance(T, ChargedTensor):
            assert tensors.almost_equal(U.hc @ T @ Vh.hc, (1.0 + 0.0j) * renormalize * S, allow_different_types=True)
        else:
            pass  # projection is not defined here for charged tensors or non-trivial braidings

        # 2) Check isometric property of U and Vh
        eye = tensors.SymmetricTensor.from_eye(S.domain, backend=T.backend)
        assert tensors.almost_equal(U.hc @ U, eye, allow_different_types=True)
        if isinstance(Vh, ChargedTensor):
            assert tensors.almost_equal(Vh.invariant_part @ Vh.invariant_part.hc, eye, allow_different_types=True)
        else:
            assert tensors.almost_equal(Vh @ Vh.hc, eye, allow_different_types=True)

        # 3) Check singular value properties of S
        if not T.symmetry.is_abelian:
            # TODO this is probably not the intended definition of svd_min!
            qdims = tensors.DiagonalTensor.from_sector_block_func(
                lambda shape, c: T.symmetry.qdim(c) * np.ones(shape), leg=lambda_kept.leg, backend=T.backend
            )
            assert (lambda_kept * qdims >= svd_min).all()
        else:
            assert (lambda_kept >= svd_min).all()

        # 4) Check properties of err / renormalize
        assert err**2 == pytest.approx(1 - tensors.norm(lambda_kept).to_numpy() ** 2)
        if normalize_to is not None:
            assert err**2 == pytest.approx(1 - (normalize_to * renormalize / norm_T) ** 2)
        T_approx = renormalize * (U @ S @ Vh)
        if isinstance(T, ChargedTensor):
            distance = tensors.norm(T - T_approx).to_numpy()
        else:
            distance = tensors.norm(T.as_SymmetricTensor() - T_approx).to_numpy()
        assert err == pytest.approx(distance / norm_T)

        # 5) Check definition of renor

        # check that U @ S @ Vd recovers the original tensor up to the error incurred
        T_approx = U @ S @ Vh / renormalize
        # check isometric properties

        if isinstance(T, ChargedTensor):
            assert isinstance(Vh, ChargedTensor)
            assert isinstance(U, SymmetricTensor)

            # for charge_leg_top = True above; for charge_leg_top = False below
            U, S, Vh, err, renormalize = tensors.truncated_svd(
                T,
                new_leg_dual=new_leg_dual,
                normalize_to=normalize_to,
                svd_min=svd_min,
                chi_max=chi_max,
                charge_leg_top=False,
            )
            U.test_sanity()
            S.test_sanity()
            Vh.test_sanity()
            assert isinstance(U, ChargedTensor)
            assert isinstance(Vh, SymmetricTensor)
            T_approx = renormalize * (U @ S @ Vh)
            distance = tensors.norm(T - T_approx).to_numpy()
            assert err == pytest.approx(distance / norm_T)
            eye = tensors.SymmetricTensor.from_eye(S.domain, backend=T.backend)
            U_iso = tensors.move_leg(U.invariant_part, U._CHARGE_LEG_LABEL, codomain_pos=0, bend_right=False)
            assert tensors.almost_equal(U_iso.hc @ U_iso, eye, allow_different_types=True)
            assert tensors.almost_equal(Vh @ Vh.hc, eye, allow_different_types=True)

    assert did_any_nontrivial_trunction or isinstance(T, tensors.Mask)


@pytest.mark.deselect_invalid_ChargedTensor_cases(
    get_cls=lambda kw: ChargedTensor if ChargedTensor in [kw['cls_A'], kw['cls_B']] else None
    # the fixture only listens to ``get_cls(kwargs) is ChargedTensor``, so we need to return
    # ChargedTensor if *either* of the two classes if ChargedTensor.
)
@pytest.mark.parametrize(
    'cls_A, cls_B, labels_A, labels_B, contr_A, contr_B',
    [
        pytest.param(
            SymmetricTensor,
            SymmetricTensor,
            [['a', 'b'], ['c', 'd']],
            [['c', 'e'], ['a', 'f']],
            [0, 3],
            [3, 0],
            id='Sym-Sym-4-2-4',
        ),
        pytest.param(
            SymmetricTensor, SymmetricTensor, [['a', 'b'], ['c']], [['d', 'e'], ['f']], [], [], id='Sym-Sym-3-0-3'
        ),
        pytest.param(
            SymmetricTensor,
            SymmetricTensor,
            [['a', 'b'], ['c', 'd']],
            [['c', 'a'], ['d', 'b']],
            [0, 1, 3, 2],
            [1, 2, 0, 3],
            id='Sym-Sym-4-4-4',
        ),
        pytest.param(
            SymmetricTensor, SymmetricTensor, [[], ['a', 'b']], [['b', 'c'], ['d']], [0], [0], id='Sym-Sym-2-1-3'
        ),
        #
        pytest.param(
            SymmetricTensor,
            ChargedTensor,
            [['a', 'b'], ['c', 'd']],
            [['c', 'e'], ['a', 'f']],
            [0, 3],
            [3, 0],
            id='Sym-Charged-4-2-4',
        ),
        pytest.param(
            ChargedTensor, SymmetricTensor, [['a', 'b'], ['c']], [['d', 'e'], ['f']], [], [], id='Charged-Sym-3-0-3'
        ),
        pytest.param(
            SymmetricTensor,
            ChargedTensor,
            [['a', 'b'], ['c', 'd']],
            [['c', 'a'], ['d', 'b']],
            [0, 1, 3, 2],
            [1, 2, 0, 3],
            id='Sym-Charged-4-4-4',
        ),
        pytest.param(
            SymmetricTensor, ChargedTensor, [[], ['a', 'b']], [['b', 'c'], ['d']], [0], [0], id='Sym-Charged-2-1-3'
        ),
        #
        # Note: need to put DiagonalTensor first to get correct legs. If SymmetricTensor is first,
        # it generates independent legs, which can not both be on a diagonalTensor.
        pytest.param(
            DiagonalTensor,
            SymmetricTensor,
            [['c'], ['b']],
            [['a', 'b'], ['c', 'd']],
            [1, 0],
            [1, 3],
            id='Diag-Sym-4-2-2',
        ),
        pytest.param(
            SymmetricTensor, DiagonalTensor, [['a', 'b'], ['c', 'd']], [['e'], ['b']], [1], [1], id='Sym-Diag-4-1-2'
        ),
        pytest.param(
            SymmetricTensor, DiagonalTensor, [['a', 'b'], ['c', 'd']], [['e'], ['f']], [], [], id='Sym-Diag-4-0-2'
        ),
        #
        # Note: If both legs of a mask are contracted, we should generate the mask first. otherwise its legs may be invalid.
        pytest.param(
            Mask, SymmetricTensor, [['c'], ['b']], [['a', 'b'], ['c', 'd']], [1, 0], [1, 3], id='Sym-Mask-4-2-2'
        ),
        pytest.param(SymmetricTensor, Mask, [['a', 'b'], ['c', 'd']], [['e'], ['b']], [1], [1], id='Sym-Mask-4-1-2'),
        pytest.param(SymmetricTensor, Mask, [['a', 'b'], ['c', 'd']], [['e'], ['f']], [], [], id='Sym-Mask-4-0-2'),
        #
        pytest.param(
            ChargedTensor, DiagonalTensor, [['a', 'b'], ['c', 'd']], [['e'], ['b']], [1], [1], id='Charged-Diag-4-1-2'
        ),
        pytest.param(ChargedTensor, Mask, [['a', 'b'], ['c', 'd']], [['e'], ['b']], [1], [1], id='Charged@-ask-4-1-2'),
        #
        pytest.param(DiagonalTensor, DiagonalTensor, [['a'], ['b']], [['c'], ['b']], [1], [1], id='Diag-Diag-2-1-2'),
        pytest.param(
            DiagonalTensor, DiagonalTensor, [['a'], ['b']], [['b'], ['a']], [1, 0], [0, 1], id='Diag-Diag-2-2-2'
        ),
        pytest.param(DiagonalTensor, DiagonalTensor, [['a'], ['b']], [['c'], ['d']], [], [], id='Diag-Diag-2-0-2'),
        #
        pytest.param(Mask, Mask, [['a'], ['b']], [['c'], ['b']], [1], [1], id='Mask-Mask-2-1-2'),
        #
        # TODO: having issues randomly generating the masks in this case...
        # pytest.param(Mask, Mask, [['a'], ['b']], [['a'], ['b']], [0, 1], [0, 1], id='Mask-Mask-2-2-2'),
        #
        pytest.param(Mask, Mask, [['a'], ['b']], [['c'], ['d']], [], [], id='Mask-Mask-2-0-2'),
    ],
)
@pytest.mark.parametrize('use_pipes', [False, 0.3])
def test_tdot(
    cls_A: type[tensors.Tensor],
    cls_B: type[tensors.Tensor],
    labels_A: list[list[str]],
    labels_B: list[list[str]],
    contr_A: list[int],
    contr_B: list[int],
    use_pipes: bool,
    make_compatible_tensor,
    np_random,
):
    if use_pipes and cls_A in [Mask, DiagonalTensor] or cls_B in [Mask, DiagonalTensor]:
        # need to design such that the diagonal one doesnt need pipes
        pytest.xfail()

    A: Tensor = make_compatible_tensor(
        codomain=len(labels_A[0]),
        domain=len(labels_A[1]),
        labels=[*labels_A[0], *reversed(labels_A[1])],
        max_block_size=3,
        max_blocks=3,
        cls=cls_A,
        use_pipes=use_pipes,
    )

    # create B such that legs with the same label can be contracted
    B: Tensor = make_compatible_tensor(
        codomain=[A._as_domain_leg(l) if A.has_label(l) else None for l in labels_B[0]],
        domain=[A._as_codomain_leg(l) if A.has_label(l) else None for l in labels_B[1]],
        labels=[*labels_B[0], *reversed(labels_B[1])],
        max_block_size=2,
        max_blocks=3,
        cls=cls_B,
        use_pipes=use_pipes,
    )

    num_contr = len(contr_A)
    num_open_A = A.num_legs - num_contr
    num_open_B = B.num_legs - num_contr
    num_open = num_open_A + num_open_B
    # make sure we defined compatible legs
    for ia, ib in zip(contr_A, contr_B):
        assert A._as_domain_leg(ia) == B._as_codomain_leg(ib), (
            f'{ia} / {A.labels[ia]} incompatible with {ib} / {B.labels[ib]}'
        )

    expect_codomain = [A._as_codomain_leg(n) for n in range(A.num_legs) if n not in contr_A]
    expect_domain = [B._as_domain_leg(n) for n in range(B.num_legs) if n not in contr_B][::-1]
    expect_legs = [A.get_leg(n) for n in range(A.num_legs) if n not in contr_A] + [
        B.get_leg(n) for n in range(B.num_legs) if n not in contr_B
    ]
    expect_labels = [A._labels[n] for n in range(A.num_legs) if n not in contr_A] + [
        B._labels[n] for n in range(B.num_legs) if n not in contr_B
    ]

    if (cls_A is Mask and cls_B is Mask) and num_contr > 0:
        with pytest.raises(NotImplementedError, match='tensors._compose_with_Mask not implemented for Mask'):
            _ = tensors.tdot(A, B, contr_A, contr_B)
        pytest.xfail('apply_mask(Mask, Mask) not yet supported')

    # expected errors:
    if not A.symmetry.has_symmetric_braid and (contr_A, contr_B) != ([1, 0], [0, 1]):
        # would need to specify levels and/or bend_right
        msg = 'Legs can not be permuted automatically. Explicitly use permute_legs()'
        with pytest.raises(SymmetryError, match=msg):
            _ = tensors.tdot(A, B, contr_A, contr_B)
        return
    if not A.symmetry.has_trivial_braid and (contr_A, contr_B) not in [([1, 0], [1, 3]), ([1, 0], [0, 1])]:
        # would need to specify bend_right
        msg = 'Legs can not be permuted automatically. Explicitly use permute_legs()'
        with pytest.raises(SymmetryError, match=msg):
            _ = tensors.tdot(A, B, contr_A, contr_B)
        return

    if not A.symmetry.has_symmetric_braid:
        if cls_A is not DiagonalTensor:
            levels_A = list(np_random.permutation(A.num_legs))
            codomain_A = [i for i in range(A.num_legs) if i not in contr_A]
            if any([isinstance(leg, LegPipe) for leg in A.legs]):
                with pytest.raises(RuntimeError, match='iter_tree_blocks can not deal with pipes'):
                    _ = tensors.permute_legs(A, codomain=codomain_A, domain=contr_A, levels=levels_A)
                pytest.xfail(reason='FTbackend cant deal with pipes yet')
            if cls_A is Mask and len(codomain_A) != 1:
                # otherwise, mask transpose is used
                catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
            else:
                catch_warnings = nullcontext()
            with catch_warnings:
                A = tensors.permute_legs(A, codomain=codomain_A, domain=contr_A, levels=levels_A)
            contr_A = [A.num_legs - 1 - i for i in range(num_contr)]
        if cls_B is not DiagonalTensor:
            levels_B = list(np_random.permutation(B.num_legs))
            domain_B = [i for i in range(B.num_legs) if i not in contr_B][::-1]
            if cls_B is Mask and len(domain_B) != 1:
                catch_warnings = pytest.warns(UserWarning, match='Converting to SymmetricTensor *')
            else:
                catch_warnings = nullcontext()
            with catch_warnings:
                B = tensors.permute_legs(B, codomain=contr_B, domain=domain_B, levels=levels_B)
            contr_B = list(range(num_contr))

    compare_numpy = A.symmetry.has_symmetric_braid
    # if the braid is trivial, we can compare to braiding the to_numpy() representations
    # for a symmetric braid, we can do to_numpy(), but the numpy rep loses the braiding information
    # for general braids, we cant even do to_numpy()

    res = tensors.tdot(A, B, contr_A, contr_B)

    if num_open == 0:
        # scalar result
        assert isinstance(res, Scalar)
        res_np = res.to_numpy()
    else:
        # tensor result
        res.test_sanity()
        if compare_numpy:
            res_np = res.to_numpy(understood_braiding=True)
        assert res.codomain.factors == expect_codomain
        assert res.domain.factors == expect_domain
        assert res.legs == expect_legs
        assert res.labels == expect_labels

    if compare_numpy:
        A_np = A.to_numpy(understood_braiding=True)
        B_np = B.to_numpy(understood_braiding=True)
        if A.symmetry.has_trivial_braid:
            expect = np.tensordot(A_np, B_np, [contr_A, contr_B])
        else:
            # need to be careful about the braiding
            A2 = swap_gate_numpy.permute_legs(A_np, A.num_codomain_legs, A.legs, codomain=contr_A, bend_right=True)
            B2 = swap_gate_numpy.permute_legs(B_np, B.num_codomain_legs, B.legs, domain=contr_B, bend_right=True)
            expect = np.tensordot(A2, B2, ([*range(num_contr)], [*reversed(range(B.num_legs - num_contr, B.num_legs))]))
        npt.assert_almost_equal(res_np, expect, decimal=12)


@pytest.mark.parametrize(
    'cod, dom, row, col',
    [
        pytest.param(1, 1, 3, 3, id='Tens-1-1-Grid-3-3'),
        pytest.param(2, 1, 2, 3, id='Tens-2-1-Grid-2-3'),
        pytest.param(2, 2, 2, 2, id='Tens-2-2-Grid-2-2'),
        pytest.param(3, 1, 2, 2, id='Tens-3-1-Grid-2-2'),
        pytest.param(1, 3, 2, 2, id='Tens-1-3-Grid-2-2'),
    ],
)
def test_tensor_from_grid(cod, dom, row, col, make_compatible_tensor, make_compatible_space, np_random):
    codomain = [make_compatible_space()] + [None] * (cod - 1)
    domain = [None] * (dom - 1) + [make_compatible_space()]
    T: SymmetricTensor = make_compatible_tensor(codomain, domain, cls=SymmetricTensor)
    dual_codom = T.codomain[0].is_dual
    dual_dom = T.domain[-1].is_dual

    # build grid -> first finish first row and column, then fill in the rest
    grid = [[T]]
    for _ in range(col - 1):
        space = make_compatible_space(is_dual=dual_dom)
        grid[0].append(make_compatible_tensor(T.codomain, [*T.domain[:-1], space], cls=SymmetricTensor))
    for _ in range(row - 1):
        space = make_compatible_space(is_dual=dual_codom)
        grid.append([make_compatible_tensor([space, *T.codomain[1:]], T.domain, cls=SymmetricTensor)])
    for i in range(1, row):
        for j in range(1, col):
            grid[i].append(make_compatible_tensor(grid[i][0].codomain, grid[0][j].domain, cls=SymmetricTensor))

    # permuting legs should commute with building the full tensor
    # make sure codomain[0] and domain[-1] stay in their positions
    perm = np_random.permutation(T.num_legs)
    idx = np.where(perm == 0)[0][0]
    perm[[0, idx]] = perm[[idx, 0]]
    idx = np.where(perm == T.num_codomain_legs)[0][0]
    perm_codom = perm[:idx]
    perm_dom = perm[idx:][::-1]
    levels = np_random.permutation(T.num_legs)

    res1 = tensors.tensor_from_grid(grid)
    assert isinstance(res1, SymmetricTensor)
    assert not isinstance(res1, HiddenLegTensor)
    if row > 1:
        assert isinstance(res1.codomain[0], DirectSumSpace)
        assert len(res1.codomain[0].spaces) == row
    if col > 1:
        assert isinstance(res1.domain[-1], DirectSumSpace)
        assert len(res1.domain[-1].spaces) == col
    # check to_numpy
    if T.symmetry.can_be_dropped:
        res_np = [
            np.concatenate([op.to_numpy(understood_braiding=True) for op in row], axis=T.num_codomain_legs)
            for row in grid
        ]
        res_np = np.concatenate(res_np, axis=0)
        npt.assert_almost_equal(res1.to_numpy(understood_braiding=True), res_np)

    # check permute_legs commutes with tensor_from_grid
    bend_right = np_random.choice([True, False])
    res1 = tensors.permute_legs(res1, perm_codom, perm_dom, levels, bend_right=bend_right)
    res2 = [
        [tensors.permute_legs(op, perm_codom, perm_dom, levels, bend_right=bend_right) for op in row] for row in grid
    ]
    res2 = tensors.tensor_from_grid(res2)

    assert res1.codomain == res2.codomain
    assert res1.domain == res2.domain
    assert res1.backend.almost_equal(res1, res2, rtol=1e-12, atol=1e-12)
    if T.symmetry.can_be_dropped:
        npt.assert_almost_equal(res1.to_numpy(understood_braiding=True), res2.to_numpy(understood_braiding=True))


def test_tensor_from_grid_empty_rows_and_columns(make_compatible_tensor, make_compatible_space):
    T: SymmetricTensor = make_compatible_tensor([None], [None], cls=SymmetricTensor, use_pipes=False)
    row_summands = [T.codomain[0], make_compatible_space(is_dual=T.codomain[0].is_dual)]
    col_summands = [T.domain[-1], make_compatible_space(is_dual=T.domain[-1].is_dual)]
    grid = [[T, None], [None, None]]

    with pytest.raises(ValueError, match='row_summands'):
        tensors.tensor_from_grid(grid, col_summands=col_summands)
    with pytest.raises(ValueError, match='col_summands'):
        tensors.tensor_from_grid(grid, row_summands=row_summands)

    result = tensors.tensor_from_grid(
        grid,
        row_labels=['row0', 'row1'],
        col_labels=['col0', 'col1'],
        row_summands=row_summands,
        col_summands=col_summands,
    )
    assert result.codomain[0].spaces == row_summands
    assert result.domain[-1].spaces == col_summands
    assert result.codomain[0].summand_labels == ['row0', 'row1']
    assert result.domain[-1].summand_labels == ['col0', 'col1']


def test_tensor_grid_cell_roundtrip(make_compatible_tensor, make_compatible_space):
    """Recover grid cells via DirectSumSpace projections (works without dense bases)."""
    T: SymmetricTensor = make_compatible_tensor([None], [None], cls=SymmetricTensor, use_pipes=False)
    dual_codom = T.codomain[0].is_dual
    dual_dom = T.domain[-1].is_dual

    row_spaces = [T.codomain[0], make_compatible_space(is_dual=dual_codom)]
    col_spaces = [T.domain[-1], make_compatible_space(is_dual=dual_dom)]
    grid = [
        [make_compatible_tensor([row_spaces[i]], [col_spaces[j]], cls=SymmetricTensor) for j in range(2)]
        for i in range(2)
    ]
    # zero cell
    grid[1][0] = None

    res = tensors.tensor_from_grid(grid, labels=['wL', 'wR'], row_labels=['IdL', 'IdR'], col_labels=['IdL', 'IdR'])
    assert isinstance(res.codomain[0], DirectSumSpace)
    assert isinstance(res.domain[-1], DirectSumSpace)
    assert res.codomain[0].summand_labels == ['IdL', 'IdR']
    assert res.domain[-1].summand_labels == ['IdL', 'IdR']

    recovered = []
    row_refs = [0, 'IdR']
    col_refs = ['IdL', -1]
    for i in range(2):
        row = []
        for j in range(2):
            cell = tensors.tensor_grid_cell(res, row_refs[i], col_refs[j], row_leg='wL', col_leg='wR')
            assert cell.labels == ['wL', 'wR']
            if grid[i][j] is None:
                assert tensors.almost_equal(cell, tensors.zero_like(cell))
                row.append(None)
            else:
                row.append(cell)
        recovered.append(row)

    # Restacking recovered cells rebuilds the original stacked tensor.
    res2 = tensors.tensor_from_grid(recovered, labels=['wL', 'wR'])
    assert tensors.almost_equal(res, res2)

    # The operation only depends on the current legs, not tensor_from_grid provenance.
    direct = make_compatible_tensor([res.codomain[0]], [res.domain[-1]], cls=SymmetricTensor, use_pipes=False)
    direct.set_labels(['wL', 'wR'])
    expected = tensors.apply_mask(
        direct,
        direct.get_leg('wL').projection_onto_summand(1, backend=direct.backend),
        'wL',
    )
    expected = tensors.apply_mask(
        expected,
        expected.get_leg('wR').projection_onto_summand(0, backend=direct.backend),
        'wR',
    )
    projected = tensors.grid_project(direct, ['wL', 'wR'], ['IdR', 'IdL'])
    assert tensors.almost_equal(projected, expected)


def test_tensor_subgrid(make_compatible_tensor, make_compatible_space):
    """tensor_subgrid returns the subgrid, consistent with tensor_grid_cell."""
    T: SymmetricTensor = make_compatible_tensor([None], [None], cls=SymmetricTensor, use_pipes=False)
    dual_codom = T.codomain[0].is_dual
    dual_dom = T.domain[-1].is_dual

    row_spaces = [T.codomain[0]] + [make_compatible_space(is_dual=dual_codom) for _ in range(2)]
    col_spaces = [T.domain[-1]] + [make_compatible_space(is_dual=dual_dom) for _ in range(2)]
    grid = [
        [make_compatible_tensor([row_spaces[i]], [col_spaces[j]], cls=SymmetricTensor) for j in range(3)]
        for i in range(3)
    ]
    labels = ['A', 'B', 'C']
    res = tensors.tensor_from_grid(grid, labels=['wL', 'wR'], row_labels=labels, col_labels=labels)

    rows = ['C', 0]
    cols = [1, 'A']
    sub = tensors.tensor_subgrid(res, rows, cols, row_leg='wL', col_leg='wR')
    assert sub.labels == ['wL', 'wR']
    assert isinstance(sub.codomain[0], DirectSumSpace)
    assert isinstance(sub.domain[-1], DirectSumSpace)
    assert sub.codomain[0].summand_labels == ['C', 'A']
    assert sub.domain[-1].summand_labels == ['B', 'A']

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            expected = tensors.tensor_grid_cell(res, r, c, row_leg='wL', col_leg='wR')
            actual = tensors.tensor_grid_cell(sub, i, j, row_leg='wL', col_leg='wR')
            assert tensors.almost_equal(actual, expected)

    # Selecting everything in order reproduces the tensor.
    full = tensors.tensor_subgrid(res, [0, 1, 2], [0, 1, 2], row_leg='wL', col_leg='wR')
    assert tensors.almost_equal(full, res)

    with pytest.raises(ValueError):
        tensors.tensor_subgrid(res, [], [0], row_leg='wL', col_leg='wR')

    # Subgrids are recombined by tensor_from_grid (without row/col labels): nested summand
    # labels are flattened, so the original labels are recovered.
    row_blocks = [['A', 'B'], ['C']]
    col_blocks = [['A'], ['B', 'C']]
    subs = [[tensors.tensor_subgrid(res, rb, cb, row_leg='wL', col_leg='wR') for cb in col_blocks] for rb in row_blocks]
    recombined = tensors.tensor_from_grid(subs, labels=['wL', 'wR'])
    assert recombined.codomain[0].summand_labels == labels
    assert recombined.domain[-1].summand_labels == labels
    assert tensors.almost_equal(recombined, res)


def test_mpo_partition_and_make_U_I_pattern(compatible_symmetry, compatible_backend):
    """DSS Mask patterns used by TeNPy make_U_I / make_U_II (IdL ⊕ other ⊕ IdR)."""
    sym = compatible_symmetry
    backend = compatible_backend
    if not sym.can_be_dropped:
        pytest.skip('dense reference needs can_be_dropped')

    # All-trivial virtual sectors so public-basis order matches summand order.
    Id = ElementarySpace.from_trivial_sector(1, symmetry=sym, is_dual=False)
    other = ElementarySpace.from_trivial_sector(2, symmetry=sym, is_dual=False)
    p = ElementarySpace.from_trivial_sector(2, symmetry=sym, is_dual=False)

    left = DirectSumSpace([Id, other, Id], is_dual=False)  # IdL=0, other=1, IdR=2
    Id_d = Id.dual
    other_d = other.dual
    right = DirectSumSpace([Id_d, other_d, Id_d], is_dual=True)
    p_dual = p.dual

    rng = np.random.default_rng(0)

    def dense_cell(wL, wR):
        # shape (wL, p, wR, p*) with domain factors [p*, wR] → numpy (wL, p, wR, p*)
        block = rng.normal(size=(wL.dim, p.dim, wR.dim, p.dim)) + 1j * rng.normal(size=(wL.dim, p.dim, wR.dim, p.dim))
        return SymmetricTensor.from_dense_block(
            block,
            [wL, p],
            [p_dual, wR],
            backend=backend,
            labels=['wL', 'p', 'wR', 'p*'],
            understood_braiding=True,
        )

    grid = [
        [dense_cell(Id, Id_d), dense_cell(Id, other_d), dense_cell(Id, Id_d)],
        [dense_cell(other, Id_d), dense_cell(other, other_d), dense_cell(other, Id_d)],
        [None, dense_cell(Id, other_d), dense_cell(Id, Id_d)],
    ]
    W = tensors.tensor_from_grid(grid, labels=['wL', 'p', 'wR', 'p*'])
    Wflat = W.to_numpy(leg_order=['wL', 'wR', 'p', 'p*'], understood_braiding=True)

    projected_IdL = tensors.grid_project(W, ['wL'], [0], squeeze=True)
    expected_IdL = tensors.squeeze_legs(
        tensors.apply_mask(W, W.get_leg('wL').projection_onto_summand(0, backend=backend), 'wL'),
        ['wL'],
    )
    assert tensors.almost_equal(projected_IdL, expected_IdL)

    IdL, IdR = 0, 2
    other_idx = [1]
    sl_IdL = slice(0, 1)
    sl_other = slice(1, 3)
    sl_IdR = slice(3, 4)
    assert Wflat.shape[:2] == (4, 4)

    left_leg = W.get_leg('wL')
    right_leg = W.get_leg('wR')

    D = tensors.apply_mask(
        tensors.apply_mask(W, left_leg.projection_onto_summand(IdL, backend=backend), 'wL'),
        right_leg.projection_onto_summand(IdR, backend=backend),
        'wR',
    )
    C = tensors.apply_mask(
        tensors.apply_mask(W, left_leg.projection_onto_summand(IdL, backend=backend), 'wL'),
        right_leg.projection_onto_summands(keep_indices=other_idx, backend=backend),
        'wR',
    )
    B = tensors.apply_mask(
        tensors.apply_mask(W, left_leg.projection_onto_summands(keep_indices=other_idx, backend=backend), 'wL'),
        right_leg.projection_onto_summand(IdR, backend=backend),
        'wR',
    )
    A = tensors.apply_mask(
        tensors.apply_mask(W, left_leg.projection_onto_summands(keep_indices=other_idx, backend=backend), 'wL'),
        right_leg.projection_onto_summands(keep_indices=other_idx, backend=backend),
        'wR',
    )

    npt.assert_almost_equal(
        D.to_numpy(leg_order=['wL', 'wR', 'p', 'p*'], understood_braiding=True),
        Wflat[sl_IdL, sl_IdR],
    )
    npt.assert_almost_equal(
        C.to_numpy(leg_order=['wL', 'wR', 'p', 'p*'], understood_braiding=True),
        Wflat[sl_IdL, sl_other],
    )
    npt.assert_almost_equal(
        B.to_numpy(leg_order=['wL', 'wR', 'p', 'p*'], understood_braiding=True),
        Wflat[sl_other, sl_IdR],
    )
    npt.assert_almost_equal(
        A.to_numpy(leg_order=['wL', 'wR', 'p', 'p*'], understood_braiding=True),
        Wflat[sl_other, sl_other],
    )

    dt = 0.25
    col_IdR = tensors.apply_mask(W, right_leg.projection_onto_summand(IdR, backend=backend), 'wR')
    col_on_IdL = tensors.enlarge_leg(col_IdR, right_leg.inclusion_of_summand(IdL, backend=backend), 'wR')
    U = tensors.linear_combination(1.0, W, dt, col_on_IdL)
    keep = right_leg.projection_onto_summands(keep_indices=[0, 1], backend=backend)
    U_kept = tensors.apply_mask(U, keep, 'wR')

    expect = Wflat.copy()
    expect[:, sl_IdL] = expect[:, sl_IdL] + dt * expect[:, sl_IdR]
    expect = np.concatenate([expect[:, sl_IdL], expect[:, sl_other]], axis=1)
    npt.assert_almost_equal(
        U_kept.to_numpy(leg_order=['wL', 'wR', 'p', 'p*'], understood_braiding=True),
        expect,
    )


def test_tensor_from_grid_hidden_labels(make_compatible_tensor, np_random):
    T = make_compatible_tensor([None], [None], cls=SymmetricTensor, use_pipes=False)
    labels = np_random.choice([['row', '!column'], ['!row', 'column'], ['!row', '!column']])

    result = tensors.tensor_from_grid([[T]], labels=labels)

    assert isinstance(result, HiddenLegTensor)
    assert all(result.labels == labels)
    assert result.hidden_leg_idcs() == [i for i, l in enumerate(labels) if l.startswith('!')]


@pytest.mark.parametrize(
    'cls, legs',
    [
        pytest.param(SymmetricTensor, 2, id='Sym-2'),
        pytest.param(SymmetricTensor, 1, id='Sym-1'),
        pytest.param(ChargedTensor, 2, id='Charged-2'),
        pytest.param(ChargedTensor, 1, id='Charged-1'),
        pytest.param(DiagonalTensor, 1, id='Diag'),
    ],
)
def test_trace(cls, legs, make_compatible_tensor, compatible_symmetry, make_compatible_sectors, make_compatible_space):
    co_domain_spaces = [make_compatible_space() for _ in range(legs)]
    if cls is ChargedTensor:
        if not compatible_symmetry.can_be_dropped:
            return  # ChargedTensor not supported for this symmetry
        tensor: ChargedTensor = make_compatible_tensor(co_domain_spaces, co_domain_spaces, cls=ChargedTensor)
    else:
        tensor: Tensor = make_compatible_tensor(co_domain_spaces, co_domain_spaces, cls=cls)

    res = tensors.trace(tensor)
    assert isinstance(res, Scalar)
    res_np = res.to_numpy()

    if tensor.symmetry.can_be_dropped:
        expect = tensor.to_numpy(understood_braiding=True)
        while expect.ndim > 0:
            expect = np.trace(expect, axis1=0, axis2=-1)
        npt.assert_almost_equal(res_np, expect)


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize(
    'cls, cod, dom',
    [
        pytest.param(SymmetricTensor, 2, 2, id='Sym-2-2'),
        pytest.param(SymmetricTensor, 3, 0, id='Sym-3-0'),
        pytest.param(SymmetricTensor, 1, 1, id='Sym-1-1'),
        pytest.param(SymmetricTensor, 0, 3, id='Sym-0-3'),
        pytest.param(ChargedTensor, 2, 2, id='Charged-2-2'),
        pytest.param(ChargedTensor, 3, 0, id='Charged-3-0'),
        pytest.param(ChargedTensor, 1, 1, id='Charged-1-1'),
        pytest.param(ChargedTensor, 0, 3, id='Charged-3-0'),
        pytest.param(DiagonalTensor, 1, 1, id='Diag'),
        pytest.param(Mask, 1, 1, id='Mask'),
    ],
)
@pytest.mark.parametrize('use_pipes', [False, 0.3])
def test_transpose(cls, cod, dom, make_compatible_tensor, np_random, use_pipes):
    labels = list('abcdefghi')[: cod + dom]
    tensor: Tensor = make_compatible_tensor(cod, dom, cls=cls, labels=labels, use_pipes=use_pipes)

    if isinstance(tensor, ChargedTensor) and not tensor.symmetry.has_trivial_braid:
        with pytest.raises(SymmetryError, match='not defined'):
            _ = tensor.T
        return

    how_to_call = np_random.choice(['transpose()', '.T'])
    print(how_to_call)
    if how_to_call == 'transpose()':
        res = tensors.transpose(tensor)
    if how_to_call == '.T':
        res = tensor.T
    res.test_sanity()

    assert res.codomain == tensor.domain.dual
    assert res.domain == tensor.codomain.dual
    assert res.labels == [*labels[cod:], *labels[:cod]]

    double_transpose = tensors.transpose(res)
    double_transpose.test_sanity()
    assert_tensors_almost_equal(double_transpose, tensor)

    # make sure the res agrees with both equivalent definitions of the transpose
    # note that for SymmetricTensors, the left_transpose is a copy of the implementation,
    #      but for DiagonalTensor and Mask checking vs left_transpose is not trivial.
    left_transpose = tensors.permute_legs(
        tensor, [*range(cod, cod + dom)], [*reversed(range(cod))], bend_right=[False] * cod + [True] * dom
    )
    right_transpose = tensors.permute_legs(
        tensor, [*range(cod, cod + dom)], [*reversed(range(cod))], bend_right=[True] * cod + [False] * dom
    )
    assert_tensors_almost_equal(res, left_transpose)
    assert_tensors_almost_equal(res, right_transpose)

    if cls in [DiagonalTensor, Mask]:
        res2 = tensors.transpose(tensor.as_SymmetricTensor())
        assert_tensors_almost_equal(res.as_SymmetricTensor(), res2)

    compare_numpy = tensor.symmetry.can_be_dropped

    if compare_numpy:
        res_np = res.to_numpy(understood_braiding=True)
        npt.assert_almost_equal(res_np, left_transpose.to_numpy(understood_braiding=True))
        npt.assert_almost_equal(res_np, right_transpose.to_numpy(understood_braiding=True))

        tensor_np = tensor.to_numpy(understood_braiding=True)
        expect_right_transpose = swap_gate_numpy.permute_legs(
            tensor_np,
            num_codomain_legs=cod,
            legs=tensor.legs,
            codomain=[*range(cod, cod + dom)],
            domain=[*reversed(range(cod))],
            bend_right=[True] * cod + [False] * dom,
        )
        expect_left_transpose = swap_gate_numpy.permute_legs(
            tensor_np,
            num_codomain_legs=cod,
            legs=tensor.legs,
            codomain=[*range(cod, cod + dom)],
            domain=[*reversed(range(cod))],
            bend_right=[False] * cod + [True] * dom,
        )

        npt.assert_almost_equal(res_np, expect_right_transpose, err_msg='right')
        npt.assert_almost_equal(res_np, expect_left_transpose, err_msg='left')


def test_bug_linear_combinations(make_compatible_tensor):
    #
    A = make_compatible_tensor(cls=tensors.DiagonalTensor)
    B = make_compatible_tensor(cls=tensors.SymmetricTensor, codomain=A.codomain, domain=A.domain)

    (B - A).test_sanity()
    (B + A).test_sanity()
    (A - B).test_sanity()
    (A + B).test_sanity()


@pytest.mark.deselect_invalid_ChargedTensor_cases(get_sym=lambda kw: kw['symm'])
@pytest.mark.parametrize(
    'cls, cod, dom, combine',
    [
        pytest.param(SymmetricTensor, 1, 1, None, id='Sym-1-1-nopipes'),
        pytest.param(SymmetricTensor, 2, 1, [0, 1], id='Sym-1-1-pipes'),
        pytest.param(SymmetricTensor, 2, 2, None, id='Sym-2-2'),
        pytest.param(SymmetricTensor, 3, 0, None, id='Sym-3-0'),
        pytest.param(SymmetricTensor, 0, 3, None, id='Sym-0-3'),
        pytest.param(ChargedTensor, 2, 2, None, id='Charged'),
        pytest.param(DiagonalTensor, 1, 1, None, id='Diag'),
        pytest.param(Mask, 1, 1, None, id='Mask'),
    ],
)
@pytest.mark.parametrize(
    'symm, original_backend',
    [
        pytest.param(no_symmetry, 'no_symmetry', id='NoSymmetry-NS'),
        pytest.param(no_symmetry, 'abelian', id='NoSymmetry-AB'),
        pytest.param(no_symmetry, 'fusion_tree', id='NoSymmetry-FT'),
        pytest.param(u1_symmetry, 'abelian', id='U1-AB'),
        pytest.param(u1_symmetry, 'fusion_tree', id='U1-FT'),
        pytest.param(su2_symmetry, 'fusion_tree', id='_SU2-FT'),
        pytest.param(fibonacci_anyon_category, 'fusion_tree', id='Fib-FT'),
    ],
)
@pytest.mark.parametrize('target_backend', ['no_symmetry', 'abelian', 'fusion_tree'], ids=['NS', 'AB', 'FT'])
@pytest.mark.parametrize('original_block_backend', ['numpy'])
@pytest.mark.parametrize('target_block_backend', ['numpy', pytest.param('torch', marks=pytest.mark.torch)])
def test_to_backend(
    cls,
    cod,
    dom,
    combine,
    symm,
    original_backend,
    original_block_backend,
    target_backend,
    target_block_backend,
    np_random,
):
    if target_block_backend == 'torch':
        _ = pytest.importorskip('torch')
    b1 = get_backend(original_backend, original_block_backend)
    b2 = get_backend(target_backend, target_block_backend)

    tens = random_tensor(symm, cod, dom, backend=b1, np_random=np_random, cls=cls)
    if combine is not None:
        tens = tensors.combine_legs(tens, combine)
        cod = tens.num_codomain_legs
        dom = tens.num_domain_legs

    if not b2.supports_symmetry(symm):
        with pytest.raises(SymmetryError, match='backend does not support symmetry'):
            res = tens.to_backend(b2)
        return

    res = tens.to_backend(b2)
    res.test_sanity()
    assert res.backend == b2
    assert type(res) is type(tens)
    assert_equivalent_legs(res, tens)
    assert res.labels == tens.labels

    recovered = res.to_backend(b1)
    recovered.test_sanity()
    assert recovered.backend == b1
    assert tensors.almost_equal(recovered, tens)

    if symm.can_be_dropped:
        tens_np = tens.to_numpy(understood_braiding=True)
        res_np = res.to_numpy(understood_braiding=True)
        npt.assert_almost_equal(res_np, tens_np)


@pytest.mark.parametrize('cls', [SymmetricTensor, DiagonalTensor, Mask])
def test_reject_exclamation_in_labels(cls, make_compatible_tensor):
    with pytest.raises(ValueError, match="must not contain '!'"):
        make_compatible_tensor(cls=cls, codomain=1, domain=1, labels=['!a', 'b'])


def test_HiddenLegTensor_basic(make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(codomain=2, domain=1, labels=['a', 'b', 'c'])
    H = HiddenLegTensor(T, ['c'])
    H.test_sanity()
    assert isinstance(H, HiddenLegTensor)
    assert H.labels[-1] == '!c'
    assert H.num_legs == T.num_legs
    assert H.hidden_leg_idcs() == [2]
    assert H.public_leg_idcs() == [0, 1]
    unhidden = H.unhide_legs()
    assert unhidden.labels == T.labels
    assert tensors.almost_equal(unhidden, T)


def test_HiddenLegTensor_implicit_dual_contraction(make_compatible_tensor):
    A_sym: SymmetricTensor = make_compatible_tensor(codomain=2, domain=1, labels=['a', 'b', 'h'], use_pipes=False)
    # Contract public 'b' and implicitly '!h'↔'!h*': B's domain must carry duals of A's b and h.
    B_sym: SymmetricTensor = make_compatible_tensor(
        codomain=1,
        domain=[A_sym.get_leg('b'), A_sym.get_leg('h')],
        labels=['c', 'h*', 'b'],
        use_pipes=False,
    )
    A = HiddenLegTensor(A_sym, ['h'])
    B = HiddenLegTensor(B_sym, ['h*'])
    try:
        res = tensors.tdot(A, B, ['b'], ['b'])
    except (NotImplementedError, SymmetryError, BraidChiralityUnspecifiedError) as e:
        pytest.xfail(str(e))
    res.test_sanity()
    # Hidden duals contract implicitly; only public open legs remain.
    assert '!h' not in (res.labels or []) and '!h*' not in (res.labels or [])
    assert set(lab for lab in res.labels if lab) == {'a', 'c'}
    if isinstance(res, HiddenLegTensor):
        assert not res.hidden_leg_idcs()


@pytest.mark.parametrize('which_leg', [0, 1, 3])
def test_HiddenLegTensor_scale_axis_keeps_hidden(which_leg, make_compatible_tensor):
    # labels a, b | (domain) c, h -> leg 2 is the hidden leg, we scale the public legs
    D = make_compatible_tensor(cls=DiagonalTensor, labels=['x', 'y'])
    codom, dom = [None, None], [None, None]
    if which_leg < 2:
        codom[which_leg] = D.leg
    else:
        dom[3 - which_leg] = D.leg
    T: SymmetricTensor = make_compatible_tensor(codom, dom, labels=['a', 'b', 'h', 'c'])
    H = HiddenLegTensor(T, ['h'])
    res = tensors.scale_axis(H, D, which_leg)
    res.test_sanity()
    assert isinstance(res, HiddenLegTensor)
    assert res.labels == H.labels
    assert res.hidden_leg_idcs() == H.hidden_leg_idcs()
    assert tensors.almost_equal(res.unhide_legs(), tensors.scale_axis(T, D, which_leg))


def test_HiddenLegTensor_zero_like_keeps_hidden(make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(codomain=2, domain=2, labels=['a', 'b', 'h', 'c'])
    H = HiddenLegTensor(T, ['h'])
    res = tensors.zero_like(H)
    res.test_sanity()
    assert isinstance(res, HiddenLegTensor)
    assert res.labels == H.labels
    assert res.hidden_leg_idcs() == H.hidden_leg_idcs()
    assert tensors.norm(res).to_numpy() == 0


@pytest.mark.parametrize('pos', [dict(codomain_pos=0), dict(codomain_pos=2), dict(domain_pos=0), dict(domain_pos=2)])
def test_HiddenLegTensor_add_squeeze_trivial_leg_keeps_hidden(pos, make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(codomain=2, domain=2, labels=['a', 'b', 'h', 'c'])
    H = HiddenLegTensor(T, ['h'])
    res = tensors.add_trivial_leg(H, label='t', **pos)
    res.test_sanity()
    assert isinstance(res, HiddenLegTensor)
    assert [l for l in res.labels if l != 't'] == H.labels
    assert [res.labels[i] for i in res.hidden_leg_idcs()] == ['!h']
    assert tensors.almost_equal(res.unhide_legs(), tensors.add_trivial_leg(T, label='t', **pos))

    squeezed = tensors.squeeze_legs(res, 't')
    squeezed.test_sanity()
    assert isinstance(squeezed, HiddenLegTensor)
    assert squeezed.labels == H.labels
    assert squeezed.hidden_leg_idcs() == H.hidden_leg_idcs()
    assert tensors.almost_equal(squeezed.unhide_legs(), T)


def test_HiddenLegTensor_equal_hidden_label_error(make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(codomain=3, labels=['a', 'b', 'h'])
    A = HiddenLegTensor(T, ['h'])
    B = HiddenLegTensor(T, ['h'])
    with pytest.raises(ValueError, match='hidden'):
        _ = tensors.tdot(A, B, [], [])


def test_HiddenLegTensor_partial_trace_leaves_hidden(make_compatible_tensor, make_compatible_space):
    a = make_compatible_space()
    b = make_compatible_space()
    h = make_compatible_space()
    # Domain factors are dualized when read as legs, so pass `a` (not a.dual) so legs
    # expose a dual pair (a, a*).
    T: SymmetricTensor = make_compatible_tensor(
        codomain=[a, b],
        domain=[a, h],
        labels=['a', 'b', 'h', 'a*'],
    )
    H = HiddenLegTensor(T, ['h'])
    try:
        res = tensors.partial_trace(H, ('a', 'a*'))
    except (NotImplementedError, SymmetryError, BraidChiralityUnspecifiedError) as e:
        pytest.xfail(str(e))
    res.test_sanity()
    assert isinstance(res, HiddenLegTensor)
    assert '!h' in res.labels


def test_HiddenLegTensor_scalar_ops_error_with_hidden(make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(codomain=1, domain=1, labels=['a', 'h'])
    H = HiddenLegTensor(T, ['h'])
    with pytest.raises(ValueError, match='hidden'):
        _ = tensors.trace(H)
    # Equal (non-dual) hidden labels must raise on contraction.
    H2 = HiddenLegTensor(
        make_compatible_tensor(codomain=T.codomain, domain=T.domain, labels=['a', 'h']),
        ['h'],
    )
    with pytest.raises(ValueError, match='equal hidden|Cannot contract HiddenLegTensors'):
        _ = tensors.tdot(H, H2, [], [])


def test_HiddenLegTensor_split_legs_skips_hidden(make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(codomain=3, domain=1, labels=['a', 'b', 'c', 'h'], use_pipes=False)
    try:
        combined = tensors.combine_legs(T, [0, 1])
    except (NotImplementedError, SymmetryError, BraidChiralityUnspecifiedError) as e:
        pytest.xfail(str(e))
    H = HiddenLegTensor(combined, ['h'])
    hidden_idx = H.hidden_leg_idcs()[0]
    with pytest.raises(ValueError, match='hidden'):
        _ = tensors.split_legs(H, hidden_idx)
    # split_legs() without args only splits public pipes; hidden legs are untouched.
    try:
        split = tensors.split_legs(H)
    except (NotImplementedError, SymmetryError, BraidChiralityUnspecifiedError) as e:
        pytest.xfail(str(e))
    split.test_sanity()
    assert isinstance(split, HiddenLegTensor)
    assert '!h' in split.labels


def test_HiddenLegTensor_combine_legs_rejects_hidden(make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(codomain=3, labels=['a', 'b', 'h'])
    H = HiddenLegTensor(T, ['h'])
    with pytest.raises(ValueError, match='hidden'):
        _ = tensors.combine_legs(H, [0, 2])


def test_HiddenLegTensor_dagger_stars_hidden_labels(make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(codomain=1, domain=1, labels=['a', 'h'])
    H = HiddenLegTensor(T, ['h'])
    Hd = H.hc
    Hd.test_sanity()
    assert isinstance(Hd, HiddenLegTensor)
    assert '!h*' in Hd.labels


def test_HiddenLegTensor_permute_leaves_hidden_in_place(make_compatible_tensor):
    T: SymmetricTensor = make_compatible_tensor(codomain=2, domain=1, labels=['a', 'b', 'h'], use_pipes=False)
    H = HiddenLegTensor(T, ['h'])
    try:
        res = tensors.permute_legs(H, [1, 0], [2])
    except (NotImplementedError, SymmetryError, BraidChiralityUnspecifiedError) as e:
        pytest.xfail(str(e))
    res.test_sanity()
    assert isinstance(res, HiddenLegTensor)
    assert res.labels[2] == '!h'  # hidden stayed in domain
    assert '!h' in res.labels
    assert res.labels[2] == '!h'


@pytest.mark.parametrize('hidden_in_domain', [True, False])
def test_HiddenLegTensor_decompositions_keep_hidden(hidden_in_domain, make_compatible_tensor):
    if hidden_in_domain:
        T = make_compatible_tensor(codomain=2, domain=2, labels=['vL', 'p', 'vR', 'h'], use_pipes=False)
    else:
        T = make_compatible_tensor(codomain=3, domain=1, labels=['h', 'vL', 'p', 'vR'], use_pipes=False)
    H = HiddenLegTensor(T, ['h'])

    def check(left, right):
        hidden, public = (right, left) if hidden_in_domain else (left, right)
        for X in [left, right]:
            X.test_sanity()
        assert isinstance(hidden, HiddenLegTensor) and '!h' in hidden.labels
        assert not isinstance(public, HiddenLegTensor) and '!h' not in public.labels
        res = tensors.compose(left, right)
        assert isinstance(res, HiddenLegTensor)
        assert res.labels == H.labels
        assert tensors.almost_equal(res, H)

    check(*tensors.qr(H, new_labels=['vR', 'vL']))
    check(*tensors.lq(H, new_labels=['vR', 'vL']))
    U, S, Vh = tensors.svd(H, new_labels=['vR', 'vL'])
    check(tensors.compose(U, S), Vh)
    U, S, Vh, _, _ = tensors.truncated_svd(H, new_labels=['vR', 'vL'])
    check(U, tensors.compose(S, Vh))
    # partial_compose keeps the hidden leg of tensor1
    vL = H.codomain.factors[1] if not hidden_in_domain else H.codomain.factors[0]
    X = make_compatible_tensor(codomain=[vL], domain=[vL], labels=['vL', 'x'], use_pipes=False)
    res = tensors.partial_compose(H, X, 'vL')
    res.test_sanity()
    assert isinstance(res, HiddenLegTensor)
    assert res.labels == H.labels
    expect = tensors.partial_compose(H.unhide_legs(), X, 'vL')
    assert tensors.almost_equal(res.unhide_legs(), expect)


def test_HiddenLegTensor_to_numpy_public_leg_order(make_compatible_tensor):
    T = make_compatible_tensor(codomain=2, domain=2, labels=['a', 'h', 'b', 'c'], use_pipes=False)
    if not T.symmetry.can_be_dropped:
        pytest.skip('no dense representation')
    H = HiddenLegTensor(T, ['h'])
    full = H.to_numpy(understood_braiding=True)
    # hidden legs need not be named; they are the last legs of the result
    res = H.to_numpy(['c', 'a', 'b'], understood_braiding=True)
    npt.assert_array_equal(res, np.transpose(full, [3, 0, 2, 1]))
    res = H.to_numpy(['c', 'a', 'b', '!h'], understood_braiding=True)
    npt.assert_array_equal(res, np.transpose(full, [3, 0, 2, 1]))


@pytest.mark.parametrize('do_dagger', [True, False])
def test_HiddenLegTensor_inner(do_dagger, make_compatible_tensor):
    labels = ['a', 'b', 'c', 'h']
    T: SymmetricTensor = make_compatible_tensor(codomain=2, domain=2, labels=labels, use_pipes=False)
    T = HiddenLegTensor(T, ['h'])
    if do_dagger:
        T2: SymmetricTensor = make_compatible_tensor(codomain=T.codomain, domain=T.domain, labels=labels)
        T2 = HiddenLegTensor(T2, ['h'])
    else:
        T2: SymmetricTensor = make_compatible_tensor(
            codomain=T.domain, domain=T.codomain, labels=[l + '*' for l in labels[::-1]]
        )
        T2 = HiddenLegTensor(T2, ['h*'])

    res = tensors.inner(T, T2, do_dagger=do_dagger)
    assert isinstance(res, T.backend.block_backend.Scalar)
    res = res.to_numpy()
    assert isinstance(res, (bool, float, complex, np.float32, np.float64, np.complex64, np.complex128))

    if T.symmetry.has_trivial_braid:
        if do_dagger:
            expect = np.sum(np.conj(T.to_numpy()) * T2.to_numpy())
        else:
            expect = np.sum(np.transpose(T.to_numpy(), [*reversed(range(T.num_legs))]) * T2.to_numpy())
        npt.assert_almost_equal(res, expect)
    else:
        pass  # TODO need to check some other way

    # norm(T) == sqrt(inner(T, T, do_dagger=True)); hidden legs are contracted implicitly.
    npt.assert_almost_equal((tensors.norm(T) ** 2).to_numpy(), tensors.inner(T, T, do_dagger=True).to_numpy())
    npt.assert_almost_equal((tensors.norm(T2) ** 2).to_numpy(), tensors.inner(T2, T2, do_dagger=True).to_numpy())


def _move_hidden_leg_pair(make_compatible_tensor, hidden_space, extra_hidden=False, hide_on_B=False, **tensor_kwargs):
    """Build contractible A (HiddenLegTensor) and B with A.vR dual to B.vL."""
    kwargs = {'use_pipes': False, 'max_blocks': 3, 'max_block_size': 3, **tensor_kwargs}
    A_codomain = [hidden_space, None, None, None]
    A_labels = ['charge', 'pA', 'vL', 'vR']
    if extra_hidden:
        A_codomain.append(None)
        A_labels.append('h2')
    A_sym: SymmetricTensor = make_compatible_tensor(codomain=A_codomain, labels=A_labels, **kwargs)
    which = ['charge', 'h2'] if extra_hidden else ['charge']
    A = HiddenLegTensor(A_sym, which)
    B_codomain = [None, A.get_leg('vR').dual, None]
    B_labels = ['pB', 'vL', 'vR']
    if hide_on_B:
        B_codomain.append(None)
        B_labels.append('k')
    B_sym: SymmetricTensor = make_compatible_tensor(codomain=B_codomain, labels=B_labels, **kwargs)
    B = HiddenLegTensor(B_sym, ['k']) if hide_on_B else B_sym
    return A, B


def _assert_tdot_equal(expect, got):
    """``tdot`` leftover hidden sits on A vs B; align then compare."""
    try:
        got2 = tensors.permute_legs(got, list(expect.codomain_labels), list(expect.domain_labels), bend_right=True)
    except (NotImplementedError, SymmetryError, BraidChiralityUnspecifiedError) as e:
        pytest.xfail(str(e))
    assert tensors.almost_equal(expect, got2, allow_different_types=True)


def test_move_hidden_leg_example_and_tdot(make_compatible_tensor, compatible_symmetry):
    h = ElementarySpace.from_trivial_sector(1, symmetry=compatible_symmetry)
    A, B = _move_hidden_leg_pair(make_compatible_tensor, h)
    try:
        expect = tensors.tdot(A, B, 'vR', 'vL')
        A2, B2 = tensors.move_hidden_leg(A, B, 'vR', 'vL', '!charge', target_codomain_pos=2)
        got = tensors.tdot(A2, B2, 'vR', 'vL')
    except (NotImplementedError, SymmetryError, BraidChiralityUnspecifiedError) as e:
        pytest.xfail(str(e))
    A2.test_sanity()
    B2.test_sanity()
    assert isinstance(A2, SymmetricTensor) and not isinstance(A2, HiddenLegTensor)
    assert isinstance(B2, HiddenLegTensor)
    assert A2.labels == ['pA', 'vL', 'vR']
    assert B2.labels == ['pB', 'vL', '!charge', 'vR']
    assert not isinstance(A2.get_leg('vR'), LegPipe)
    assert not isinstance(B2.get_leg('vL'), LegPipe)
    _assert_tdot_equal(expect, got)


def test_move_hidden_leg_domain_pos(make_compatible_tensor, compatible_symmetry):
    h = ElementarySpace.from_trivial_sector(1, symmetry=compatible_symmetry)
    A, B = _move_hidden_leg_pair(make_compatible_tensor, h)
    try:
        expect = tensors.tdot(A, B, 'vR', 'vL')
        A2, B2 = tensors.move_hidden_leg(A, B, 'vR', 'vL', '!charge', target_domain_pos=0)
        got = tensors.tdot(A2, B2, 'vR', 'vL')
    except (NotImplementedError, SymmetryError, BraidChiralityUnspecifiedError) as e:
        pytest.xfail(str(e))
    A2.test_sanity()
    B2.test_sanity()
    assert isinstance(B2, HiddenLegTensor)
    assert B2.labels[-1] == '!charge'
    _assert_tdot_equal(expect, got)


def test_move_hidden_leg_dim_gt_one_keeps_pipes(make_compatible_tensor, compatible_symmetry):
    # Keep the hidden space tiny: default make_compatible_space() is far too large for _SU2/FusionTree.
    h = ElementarySpace.from_trivial_sector(2, symmetry=compatible_symmetry)
    A, B = _move_hidden_leg_pair(make_compatible_tensor, h, max_blocks=2, max_block_size=2)
    try:
        expect = tensors.tdot(A, B, 'vR', 'vL')
        A2, B2 = tensors.move_hidden_leg(A, B, 'vR', 'vL', '!charge', target_codomain_pos=2)
        got = tensors.tdot(A2, B2, 'vR', 'vL')
    except (NotImplementedError, SymmetryError, BraidChiralityUnspecifiedError) as e:
        pytest.xfail(str(e))
    A2.test_sanity()
    B2.test_sanity()
    assert isinstance(A2.get_leg('vR'), LegPipe)
    assert isinstance(B2.get_leg('vL'), LegPipe)
    assert A2.get_leg('vR') == B2.get_leg('vL').dual
    try:
        _assert_tdot_equal(expect, got)
    except AssertionError:
        if isinstance(A.backend, backends.FusionTreeBackend) and compatible_symmetry.braiding_style.value >= 10:
            pytest.xfail('FusionTree dim>1 hidden with non-trivial braiding')
        raise


def test_move_hidden_leg_leftover_and_existing_hidden(make_compatible_tensor, compatible_symmetry):
    h = ElementarySpace.from_trivial_sector(1, symmetry=compatible_symmetry)
    A, B = _move_hidden_leg_pair(make_compatible_tensor, h, extra_hidden=True, hide_on_B=True)
    try:
        expect = tensors.tdot(A, B, 'vR', 'vL')
        A2, B2 = tensors.move_hidden_leg(A, B, 'vR', 'vL', '!charge', target_codomain_pos=2)
        got = tensors.tdot(A2, B2, 'vR', 'vL')
    except (NotImplementedError, SymmetryError, BraidChiralityUnspecifiedError) as e:
        pytest.xfail(str(e))
    A2.test_sanity()
    B2.test_sanity()
    assert isinstance(A2, HiddenLegTensor)
    assert '!h2' in A2.labels
    assert '!charge' not in A2.labels
    assert isinstance(B2, HiddenLegTensor)
    assert '!charge' in B2.labels
    assert '!k' in B2.labels
    # Sliding ``!charge`` onto B changes its order relative to leftover ``!h2``.
    # Aligning the two ``tdot`` results would braid those hidden wires; check contraction only.


def test_move_hidden_leg_errors(make_compatible_tensor, compatible_symmetry):
    h = ElementarySpace.from_trivial_sector(1, symmetry=compatible_symmetry)
    A, B = _move_hidden_leg_pair(make_compatible_tensor, h)
    with pytest.raises(ValueError, match='exactly one'):
        tensors.move_hidden_leg(A, B, 'vR', 'vL', '!charge')
    with pytest.raises(ValueError, match='exactly one'):
        tensors.move_hidden_leg(A, B, 'vR', 'vL', '!charge', 2, target_domain_pos=0)
    with pytest.raises(ValueError, match='hidden label'):
        tensors.move_hidden_leg(A, B, 'vR', 'vL', 'charge', 2)
    with pytest.raises(ValueError, match='public'):
        tensors.move_hidden_leg(A, B, '!charge', 'vL', '!charge', 2)
    with pytest.raises(ValueError, match='not contractible'):
        tensors.move_hidden_leg(A, B, 'vL', 'vL', '!charge', 2)
    if ChargedTensor.supports_symmetry(compatible_symmetry):
        C = make_compatible_tensor(codomain=1, domain=1, cls=ChargedTensor, use_pipes=False)
        with pytest.raises((TypeError, ValueError)):
            tensors.move_hidden_leg(A, C, 'vR', 0, '!charge', 0)
    D = make_compatible_tensor(codomain=1, domain=1, cls=DiagonalTensor, use_pipes=False)
    with pytest.raises((TypeError, ValueError)):
        tensors.move_hidden_leg(A, D, 'vR', 0, '!charge', 0)
    B_same = HiddenLegTensor(
        make_compatible_tensor(
            codomain=[h, A.get_leg('vR').dual, None], labels=['charge', 'vL', 'vR'], use_pipes=False
        ),
        ['charge'],
    )
    with pytest.raises(ValueError, match='already has label'):
        tensors.move_hidden_leg(A, B_same, 'vR', 'vL', '!charge', 1)
    B_dual = HiddenLegTensor(
        make_compatible_tensor(
            codomain=[h.dual, A.get_leg('vR').dual, None], labels=['charge*', 'vL', 'vR'], use_pipes=False
        ),
        ['charge*'],
    )
    with pytest.raises(ValueError, match='dual hidden'):
        tensors.move_hidden_leg(A, B_dual, 'vR', 'vL', '!charge', 1)


@pytest.mark.parametrize('dual_pipes', [False, True])
def test_flatten_pipe_leg(dual_pipes, make_compatible_tensor, make_compatible_space):
    # combine two bonds like for an MPS: vL pipe in the codomain, the dual vR pipe in the domain.
    # For `dual_pipes`, the pipe *factors* in the codomain and domain are dual pipes instead.
    V0 = make_compatible_space()
    V1 = make_compatible_space()
    labels = ['vL0', 'vL1', 'vR1', 'vR0']
    A0 = make_compatible_tensor([V0, V1], [V0, V1], labels=labels, use_pipes=False)
    B0 = make_compatible_tensor([V0, V1], [V0, V1], labels=labels, use_pipes=False)

    def combine(T):
        dualities = [True, False] if dual_pipes else [False, True]
        T = tensors.combine_legs(T, ['vL0', 'vL1'], ['vR1', 'vR0'], pipe_dualities=dualities)
        return T.relabel({'(vL0.vL1)': 'vL', '(vR1.vR0)': 'vR'})

    A, B = [combine(T) for T in [A0, B0]]
    assert A.codomain == A.domain
    assert A.codomain.factors[0].is_dual == dual_pipes
    expect = tensors.compose(A, B)

    A2, B2, expect2 = [tensors.flatten_pipe_leg(tensors.flatten_pipe_leg(T, 'vL'), 'vR') for T in [A, B, expect]]
    for T, T2 in [(A, A2), (B, B2), (expect, expect2)]:
        T2.test_sanity()
        assert T2.labels == T.labels
        for label in ['vL', 'vR']:
            leg, leg2 = T.get_leg(label), T2.get_leg(label)
            assert isinstance(leg, LegPipe)
            assert isinstance(leg2, ElementarySpace) and not isinstance(leg2, LegPipe)
            assert leg2.is_dual == leg.is_dual
            leg_flat = leg.as_ElementarySpace(leg.is_dual)
            npt.assert_array_equal(leg2.sector_decomposition, leg_flat.sector_decomposition)
            npt.assert_array_equal(leg2.multiplicities, leg_flat.multiplicities)
        npt.assert_almost_equal(tensors.norm(T2).to_numpy(), tensors.norm(T).to_numpy())
    assert A2.codomain == A2.domain
    # the flattened legs are consistent, such that contractions give the same data
    assert tensors.almost_equal(tensors.compose(A2, B2), expect2)
    # ... also the same as contracting the separate legs, i.e. the flattened legs keep their meaning
    expect0 = combine(tensors.compose(A0, B0))
    expect0 = tensors.flatten_pipe_leg(tensors.flatten_pipe_leg(expect0, 'vL'), 'vR')
    assert tensors.almost_equal(tensors.compose(A2, B2), expect0)
    # flattening commutes with (planar) permutations of the other legs, here the pipe is not the
    # first factor of the domain
    W = make_compatible_space()
    C = make_compatible_tensor([V0, V1], [W, V0, V1], labels=labels + ['w'], use_pipes=False)
    C = combine(C)
    assert C.domain_labels == ['w', 'vR']
    C2 = tensors.flatten_pipe_leg(C, 'vR')
    C3 = tensors.planar_permute_legs(C, domain=['vR'])
    C3 = tensors.flatten_pipe_leg(C3, 'vR')
    C3 = tensors.planar_permute_legs(C3, codomain=C.codomain_labels, domain=C.domain_labels)
    assert C2.labels == C.labels
    assert tensors.almost_equal(C2, C3)
    # flattening a leg bent to the other side (i.e. the dual pipe factor) is consistent with
    # flattening it in place
    B3 = tensors.planar_permute_legs(B, codomain=['vR', 'vL'], domain=[])
    assert B3.codomain.factors[0].is_dual != B.domain.factors[0].is_dual
    B3 = tensors.flatten_pipe_leg(tensors.flatten_pipe_leg(B3, 'vL'), 'vR')
    B3 = tensors.planar_permute_legs(B3, codomain=B.codomain_labels, domain=B.domain_labels)
    assert tensors.almost_equal(B3, B2)
    # two pipes on the same side: flattening commutes, also if another pipe precedes the flattened one
    D = make_compatible_tensor([V0, V1, V0, V1], [W], labels=['a0', 'a1', 'b0', 'b1', 'w'], use_pipes=False)
    D = tensors.combine_legs(D, ['a0', 'a1'], ['b0', 'b1'])
    D_ab = tensors.flatten_pipe_leg(tensors.flatten_pipe_leg(D, '(a0.a1)'), '(b0.b1)')
    D_ba = tensors.flatten_pipe_leg(tensors.flatten_pipe_leg(D, '(b0.b1)'), '(a0.a1)')
    assert tensors.almost_equal(D_ab, D_ba)
    # flattening a leg which is not a pipe does nothing
    assert tensors.almost_equal(tensors.flatten_pipe_leg(A2, 0), A2)
    # also works with a pipe in the codomain of a HiddenLegTensor
    H = HiddenLegTensor(
        tensors.add_trivial_leg(A, codomain_pos=0, label='h'),
        ['h'],
    )
    H2 = tensors.flatten_pipe_leg(H, 'vL')
    assert isinstance(H2, HiddenLegTensor)
    assert H2.labels == H.labels
    assert not isinstance(H2.get_leg('vL'), LegPipe)
