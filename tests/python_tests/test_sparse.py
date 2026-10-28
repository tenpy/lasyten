"""A collection of tests for lasyten.sparse"""

# Copyright (C) TeNPy Developers, Apache license
import numpy as np
import numpy.testing as npt
import pytest

from lasyten import Dtype, SymmetryError, backends, sparse, tensors
from lasyten.symmetries import TensorProduct
from lasyten.tensors import (
    ChargedTensor,
    SymmetricTensor,
    Tensor,
    almost_equal,
    inner,
    norm,
    tdot,
)

# define a few simple operators to test the wrappers:


class ScalingDummyOperator(sparse.LinearOperator):
    def __init__(self, factor, vector_legs, vector_labels=None):
        super().__init__(vector_legs=vector_legs, dtype=Dtype.complex128, vector_labels=vector_labels)
        self.factor = factor

    def matvec(self, vec: Tensor) -> Tensor:
        return self.factor * vec

    def to_tensor(self, backend=None) -> Tensor:
        assert backend is not None, 'backend kwarg is needed for ScalingDummyOperator.to_tensor'
        return self.factor * SymmetricTensor.from_eye(self.vector_legs, backend=backend, labels=self.vector_labels)

    def adjoint(self):
        return ScalingDummyOperator(np.conj(self.factor), self.vector_legs, self.vector_labels)


class TensorDummyOperator(sparse.LinearOperator):
    def __init__(self, tensor: SymmetricTensor):
        assert tensor.labels == ['a', 'b', 'b*', 'a*']
        acts_on = ['a', 'b']
        super().__init__(
            vector_legs=tensor.get_leg(acts_on),
            dtype=tensor.dtype,
            vector_labels=acts_on,
        )
        self.tensor = tensor

    def matvec(self, vec: Tensor) -> Tensor:
        return tdot(self.tensor, vec, ['b*', 'a*'], ['b', 'a'])

    def to_tensor(self, backend=None, **kw) -> Tensor:
        return self.tensor

    def adjoint(self):
        return TensorDummyOperator(self.tensor.dagger)


def check_to_tensor(op: sparse.LinearOperator, vec: Tensor):
    """perform common checks of the LinearOperator.to_tensor method"""
    res_matvec = op.matvec(vec)
    if not vec.symmetry.has_trivial_braid:
        return
    tensor = op.to_tensor(backend=vec.backend)
    _ = op.to_matrix(backend=vec.backend)  # just check if it runs...
    N = vec.num_legs
    res_tensor = tdot(tensor, vec, list(range(N, 2 * N)), list(reversed(range(N))))
    assert almost_equal(res_matvec, res_tensor)


def test_TensorLinearOperator(make_compatible_tensor):
    vec = make_compatible_tensor(codomain=1, labels=['v'], use_pipes=False)
    leg = vec.legs[0]
    H = make_compatible_tensor(codomain=[leg], domain=[leg], labels=['w', 'v'], use_pipes=False)

    op = sparse.TensorLinearOperator(H)
    assert almost_equal(op.matvec(vec), tdot(H, vec, [1], [0]))
    check_to_tensor(op, vec)

    adj = op.adjoint()
    assert almost_equal(adj.matvec(vec), tdot(H.dagger, vec, [1], [0]))

    H_h = H + H.dagger
    hop = sparse.TensorLinearOperator(H_h)
    assert almost_equal(hop.matvec(vec), hop.adjoint().matvec(vec))
    mat = hop.to_matrix(backend=vec.backend)
    assert mat.num_legs == 2


def test_SumLinearOperator(make_compatible_tensor):
    vec = make_compatible_tensor(codomain=['a', 'b'], use_pipes=False)
    a, b = vec.legs
    T = make_compatible_tensor(codomain=[a, b], domain=[a, b], labels=['a', 'b', 'b*', 'a*'], use_pipes=False)

    factor1 = 2.4
    factor3 = 3.1 - 42.0j
    op1 = ScalingDummyOperator(factor1, vec.legs, vec.labels)
    op2 = TensorDummyOperator(T)
    op3 = ScalingDummyOperator(factor3, vec.legs, vec.labels)

    print('single operator')
    op = sparse.SumLinearOperator(op1)
    assert almost_equal(op.matvec(vec), factor1 * vec)
    check_to_tensor(op, vec)

    print('two operators')
    if not vec.symmetry.has_trivial_braid:
        return
    op = sparse.SumLinearOperator(op2, op1)
    assert almost_equal(op.matvec(vec), factor1 * vec + tdot(T, vec, ['b*', 'a*'], ['b', 'a']))
    check_to_tensor(op, vec)

    print('three operators')
    op = sparse.SumLinearOperator(op1, op2, op3)
    assert almost_equal(op.matvec(vec), (factor1 + factor3) * vec + tdot(T, vec, ['b*', 'a*'], ['b', 'a']))
    check_to_tensor(op, vec)


def test_ShiftedLinearOperator(make_compatible_tensor):
    vec = make_compatible_tensor(codomain=['a', 'b'], use_pipes=False)
    factor = 3.2
    op1 = ScalingDummyOperator(factor=factor, vector_legs=vec.legs, vector_labels=vec.labels)
    shift = 5.0j

    op = sparse.ShiftedLinearOperator(op1, shift)
    assert almost_equal(op.matvec(vec), (factor + shift) * vec)
    check_to_tensor(op, vec)


@pytest.mark.parametrize(['penalty', 'project_operator'], [(None, True), (2.0 - 0.3j, True), (-4, False)])
@pytest.mark.skip(reason='ProjectedLinearOperator.to_tensor not implemented yet')
def test_ProjectedLinearOperator(make_compatible_tensor, penalty, project_operator):
    vec = make_compatible_tensor(codomain=['a', 'b'], use_pipes=False)
    o1 = make_compatible_tensor(like=vec, use_pipes=False)
    assert (o1_norm := norm(o1)) > 0
    o1 = o1 / o1_norm
    o2 = make_compatible_tensor(like=vec, use_pipes=False)

    o2 = o2 - inner(o1, o2) * o1
    assert (o2_norm := norm(o2)) > 0
    o2 = o2 / o2_norm
    factor = 3.2
    original_op = ScalingDummyOperator(factor=factor, vector_legs=o1.legs, vector_labels=o1.labels)

    projected_op = sparse.ProjectedLinearOperator(
        original_op, [o1, o2], project_operator=project_operator, penalty=penalty
    )

    print('check vector in ortho_vecs subspace')
    if project_operator:
        expect = 0.0 * o1
    else:
        expect = original_op.matvec(o1)
    if penalty is not None:
        expect += penalty * o1
    assert almost_equal(projected_op.matvec(o1), expect)

    print('check vector orthogonal to ortho_vecs')
    vec1 = vec - inner(o1, vec) * o1 - inner(o2, vec) * o2
    expect = original_op.matvec(vec1)
    res = projected_op.matvec(vec1)
    assert almost_equal(res, expect)

    check_to_tensor(projected_op, vec)


def _xfail_ft_dense_block_sector(backend, fn):
    if not isinstance(backend, backends.FusionTreeBackend):
        return fn()
    with pytest.raises(
        (NotImplementedError, SymmetryError),
        match='from_dense_block_trivial_sector|to_dense_block_trivial_sector|'
        'from_dense_block_single_sector|inv_part_from_dense_block_single_sector|'
        'inv_part_to_dense_block_single_sector|Dense block representation is not supported|'
        'non-trivial braids|sector_dim',
    ):
        fn()
    pytest.xfail('FTBackend does not support dense-block sector conversions yet')


@pytest.mark.parametrize('use_hermitian', [True, False])
def test_NumpyArrayLinearOperator_sector(make_compatible_tensor, use_hermitian, tol=1e-12):
    vec = make_compatible_tensor(codomain=['a', 'b'], use_pipes=False)
    a, b = vec.legs
    H = make_compatible_tensor(codomain=[a, b], domain=[a, b], labels=['a', 'b', 'b*', 'a*'], use_pipes=False)

    def lasyten_matvec(v):
        return tdot(H, v, ['b*', 'a*'], ['b', 'a'])

    cls = sparse.HermitianNumpyArrayLinearOperator if use_hermitian else sparse.NumpyArrayLinearOperator
    H_op = cls(
        lasyten_matvec,
        legs=vec.legs,
        backend=vec.backend,
        dtype=vec.dtype.to_numpy_dtype(),
        labels=vec.labels,
        charge_sector='trivial',
    )

    def roundtrip():
        vec_flat = H_op.tensor_to_flat_array(vec)
        vec2 = H_op.flat_array_to_tensor(vec_flat)
        assert almost_equal(vec, vec2)
        res_flat = H_op.matvec(vec_flat)
        npt.assert_allclose(res_flat, H_op.tensor_to_flat_array(lasyten_matvec(vec)), atol=tol, rtol=tol)
        # from_Tensor on a two-leg endomorphism, if the space contains the trivial sector
        try:
            H2 = make_compatible_tensor(codomain=[a], domain=[a], labels=['w', 'v'], use_pipes=False)
            op2 = cls.from_Tensor(H2, legs1=['v'], legs2=['w'], charge_sector='trivial')
        except ValueError:
            return
        v1 = make_compatible_tensor(codomain=[a], labels=['w'], use_pipes=False)
        npt.assert_allclose(
            op2.matvec(op2.tensor_to_flat_array(v1)),
            op2.tensor_to_flat_array(tdot(H2, v1, ['v'], ['w'])),
            atol=tol,
            rtol=tol,
        )

    _xfail_ft_dense_block_sector(H.backend, roundtrip)


def test_NumpyArrayLinearOperator_from_matvec_and_vector(make_compatible_tensor, np_random):
    vec = make_compatible_tensor(codomain=['a', 'b'], use_pipes=False)
    a, b = vec.legs
    H = make_compatible_tensor(codomain=[a, b], domain=[a, b], labels=['a', 'b', 'b*', 'a*'], use_pipes=False)

    def lasyten_matvec(v):
        return tdot(H, v, ['b*', 'a*'], ['b', 'a'])

    def run():
        op, vec_flat = sparse.NumpyArrayLinearOperator.from_matvec_and_vector(lasyten_matvec, vec)
        vec2 = op.flat_array_to_tensor(vec_flat)
        assert almost_equal(vec, vec2)
        npt.assert_allclose(op.matvec(vec_flat), op.tensor_to_flat_array(lasyten_matvec(vec)))

        if not ChargedTensor.supports_symmetry(vec.symmetry):
            return
        domain = TensorProduct(list(vec.legs))
        nontrivial = [s for s in domain.sector_decomposition if not np.array_equal(s, vec.symmetry.trivial_sector)]
        if not nontrivial:
            return
        sector = nontrivial[0]
        if vec.symmetry.qdim(sector) > 1:
            return
        size = int(domain.block_size(sector))
        block = np_random.normal(size=size) + 1j * np_random.normal(size=size)
        charged = ChargedTensor.from_dense_block_single_sector(
            vector=block, space=op.pipe, sector=sector, backend=vec.backend
        )
        if charged.num_legs == 1 and len(vec.legs) > 1:
            charged = tensors.split_legs(charged, 0)
        if any(vec.labels.as_list()):
            charged.labels.set(vec.labels.as_list())
        op2, cflat = sparse.NumpyArrayLinearOperator.from_matvec_and_vector(lasyten_matvec, charged)
        npt.assert_allclose(op2.matvec(cflat), op2.tensor_to_flat_array(lasyten_matvec(charged)))

    _xfail_ft_dense_block_sector(vec.backend, run)


def test_NumpyArrayLinearOperator_all_sectors(make_compatible_tensor, np_random, tol=1e-12):
    vec = make_compatible_tensor(codomain=['a', 'b'], use_pipes=False)
    a, b = vec.legs
    H = make_compatible_tensor(codomain=[a, b], domain=[a, b], labels=['a', 'b', 'b*', 'a*'], use_pipes=False)

    def lasyten_matvec(v):
        return tdot(H, v, ['b*', 'a*'], ['b', 'a'])

    kwargs = dict(
        lasyten_matvec=lasyten_matvec,
        legs=vec.legs,
        backend=vec.backend,
        dtype=vec.dtype.to_numpy_dtype(),
        labels=vec.labels,
        charge_sector=None,
    )
    if not vec.symmetry.can_be_dropped:
        with pytest.raises(SymmetryError, match='charge_sector=None'):
            sparse.NumpyArrayLinearOperator(**kwargs)
        return

    H_op = sparse.NumpyArrayLinearOperator(**kwargs)
    assert H_op.shape[0] == int(H_op.domain.dim)

    flat = np_random.normal(size=H_op.shape[0]) + 1j * np_random.normal(size=H_op.shape[0])
    tens = H_op.flat_array_to_tensor(flat)
    assert isinstance(tens, ChargedTensor)
    assert tens.charged_state is not None
    npt.assert_allclose(H_op.tensor_to_flat_array(tens), flat, atol=tol, rtol=tol)
    npt.assert_allclose(H_op.matvec(flat), H_op.tensor_to_flat_array(lasyten_matvec(tens)), atol=tol, rtol=tol)

    # a SymmetricTensor (trivial total charge) also converts through the dense all-sector map
    triv_flat = H_op.tensor_to_flat_array(vec)
    npt.assert_allclose(triv_flat, H_op.tensor_to_flat_array(H_op.flat_array_to_tensor(triv_flat)), atol=tol, rtol=tol)

    op2, _ = sparse.NumpyArrayLinearOperator.from_matvec_and_vector(lasyten_matvec, tens)
    assert op2.charge_sector is None


@pytest.mark.parametrize('num_legs', [1, 2])
def test_gram_schmidt(make_compatible_tensor, num_legs, num_vecs=5, tol=1e-15):
    first = make_compatible_tensor(codomain=num_legs, use_pipes=False)
    vecs_old = [first] + [make_compatible_tensor(like=first, use_pipes=False) for _ in range(num_vecs - 1)]
    # note: depending on the dimension of `legs` (which is random),
    # some of those can be linearly dependent!

    vecs_new = sparse.gram_schmidt(vecs_old)  # rtol=tol is too small for some random spaces
    assert len(vecs_new) <= len(vecs_old)
    ovs = np.zeros((len(vecs_new), len(vecs_new)), dtype=np.complex128)
    for i, v in enumerate(vecs_new):
        for j, w in enumerate(vecs_new):
            ovs[i, j] = inner(v, w).to_numpy()
    atol = 2 * first.num_parameters * (num_vecs) ** 2 * tol
    npt.assert_allclose(ovs, np.eye(len(vecs_new)), atol=atol)
