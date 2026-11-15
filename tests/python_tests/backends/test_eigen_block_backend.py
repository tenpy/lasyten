"""Tests for the native Eigen/LAPACK block backend, compared against numpy."""

import numpy as np
import numpy.testing as npt
import pytest

from lasyten.block_backends import EigenBlockBackend
from lasyten.block_backends.dtypes import Dtype


@pytest.fixture(scope='module')
def bb():
    return EigenBlockBackend.from_factory('eigen')


def _rand(shape, complex_=False, seed=0):
    rng = np.random.default_rng(seed)
    res = rng.normal(size=shape)
    if complex_:
        res = res + 1j * rng.normal(size=shape)
    return res


def test_factory(bb):
    assert bb.get_backend_name() == 'EigenBlockBackend'
    assert EigenBlockBackend.from_factory('cpu') is bb
    with pytest.raises(Exception):
        EigenBlockBackend.from_factory('cuda')


def test_zeros_ones_eye(bb):
    z = bb.zeros([2, 3], Dtype.float64)
    assert bb.get_shape(z) == (2, 3)
    assert bb.get_dtype(z) == Dtype.float64
    npt.assert_array_equal(bb.to_numpy(z), np.zeros((2, 3)))
    o = bb.ones_block([3], Dtype.complex128)
    npt.assert_array_equal(bb.to_numpy(o), np.ones(3, dtype=complex))
    e = bb.eye_matrix(4, Dtype.float64)
    npt.assert_array_equal(bb.to_numpy(e), np.eye(4))


def test_to_numpy_roundtrip(bb):
    for cplx in [False, True]:
        arr = _rand((2, 3, 4), cplx)
        block = bb.block_from_numpy(arr)
        npt.assert_array_equal(bb.to_numpy(block), arr)
        # non-contiguous input
        block_t = bb.block_from_numpy(arr.transpose(2, 0, 1))
        npt.assert_array_equal(bb.to_numpy(block_t), arr.transpose(2, 0, 1))


def test_reshape_permute(bb):
    arr = _rand((2, 3, 4))
    block = bb.block_from_numpy(arr)
    npt.assert_array_equal(bb.to_numpy(bb.reshape(block, [6, 4])), arr.reshape(6, 4))
    npt.assert_array_equal(bb.to_numpy(bb.permute_axes(block, [2, 0, 1])), arr.transpose(2, 0, 1))


def test_elementwise(bb):
    a = _rand((3, 4), True, seed=1)
    b = _rand((3, 4), True, seed=2)
    ba, bbl = bb.block_from_numpy(a), bb.block_from_numpy(b)
    npt.assert_allclose(bb.to_numpy(bb.abs(ba)), np.abs(a))
    npt.assert_allclose(bb.to_numpy(ba + bbl), a + b)
    npt.assert_allclose(bb.to_numpy(ba * bbl), a * b)
    npt.assert_allclose(bb.to_numpy(bb.conj(ba)), a.conj())


@pytest.mark.parametrize('cplx', [False, True])
def test_matrix_dot_tdot(bb, cplx):
    a = _rand((3, 4, 5), cplx, seed=3)
    b = _rand((5, 4, 2), cplx, seed=4)
    ba, bbl = bb.block_from_numpy(a), bb.block_from_numpy(b)
    res = bb.tdot(ba, bbl, [1, 2], [1, 0])
    npt.assert_allclose(bb.to_numpy(res), np.tensordot(a, b, [[1, 2], [1, 0]]), atol=1e-12)
    m1, m2 = _rand((3, 4), cplx, seed=5), _rand((4, 6), cplx, seed=6)
    res = bb.matrix_dot(bb.block_from_numpy(m1), bb.block_from_numpy(m2))
    npt.assert_allclose(bb.to_numpy(res), m1 @ m2, atol=1e-12)


@pytest.mark.parametrize('cplx', [False, True])
def test_matrix_svd(bb, cplx):
    a = _rand((5, 3), cplx, seed=7)
    for algorithm in bb.possible_svd_algorithms():
        u, s, vh = bb.matrix_svd(bb.block_from_numpy(a), algorithm)
        u, s, vh = bb.to_numpy(u), bb.to_numpy(s), bb.to_numpy(vh)
        npt.assert_allclose(s, np.linalg.svd(a, compute_uv=False), atol=1e-12)
        npt.assert_allclose((u * s) @ vh, a, atol=1e-12)
        npt.assert_allclose(u.conj().T @ u, np.eye(3), atol=1e-12)


@pytest.mark.parametrize('cplx', [False, True])
@pytest.mark.parametrize('full', [False, True])
def test_matrix_qr(bb, cplx, full):
    a = _rand((5, 3), cplx, seed=8)
    q, r = bb.matrix_qr(bb.block_from_numpy(a), full)
    q, r = bb.to_numpy(q), bb.to_numpy(r)
    k = 5 if full else 3
    assert q.shape == (5, k) and r.shape == (k, 3)
    npt.assert_allclose(q @ r, a, atol=1e-12)
    npt.assert_allclose(q.conj().T @ q, np.eye(k), atol=1e-12)


@pytest.mark.parametrize('cplx', [False, True])
def test_eigh(bb, cplx):
    a = _rand((4, 4), cplx, seed=9)
    h = a + a.conj().T
    w, v = bb.eigh(bb.block_from_numpy(h))
    w, v = bb.to_numpy(w), bb.to_numpy(v)
    npt.assert_allclose(w, np.linalg.eigvalsh(h), atol=1e-12)
    npt.assert_allclose((v * w) @ v.conj().T, h, atol=1e-12)


@pytest.mark.parametrize('np_dtype', [np.bool_, np.int64, np.float32, np.float64, np.complex64, np.complex128])
def test_ndarray_roundtrip(bb, np_dtype):
    """nb::ndarray boundary: copying import, zero-copy export."""
    if np_dtype is np.bool_:
        arr = _rand((2, 3, 4)) > 0
    else:
        arr = (_rand((2, 3, 4), complex_=np.issubdtype(np_dtype, np.complexfloating)) * 3).astype(np_dtype)
    block = bb.block_from_ndarray(arr)
    out = bb.to_ndarray(block)
    assert out.dtype == arr.dtype
    npt.assert_array_equal(out, arr)
    # import copies: modifying the source afterwards does not affect the block
    arr_copy = arr.copy()
    arr[...] = 0
    npt.assert_array_equal(bb.to_ndarray(block), arr_copy)
    # export is a view onto the block's memory, which keeps the block alive
    view = block.to_ndarray()
    npt.assert_array_equal(view, arr_copy)
    assert view.base is not None or not view.flags.owndata
    npt.assert_array_equal(bb.to_numpy(block), arr_copy)


def test_ndarray_noncontiguous_and_dtype(bb):
    arr = _rand((2, 3, 4))
    block = bb.block_from_ndarray(arr.transpose(2, 0, 1))
    npt.assert_array_equal(bb.to_ndarray(block), arr.transpose(2, 0, 1))
    block32 = bb.block_from_ndarray(arr, Dtype.float32)
    assert bb.get_dtype(block32) == Dtype.float32
    npt.assert_allclose(bb.to_ndarray(block32), arr.astype(np.float32))
    # 0-d and empty arrays
    assert bb.get_shape(bb.block_from_ndarray(np.array(2.5))) == ()
    assert bb.get_shape(bb.block_from_ndarray(np.zeros((0, 3)))) == (0, 3)


def test_ndarray_view_aliases_block(bb):
    block = bb.zeros([2, 2], Dtype.float64)
    view = block.to_ndarray()
    view[0, 1] = 5.0
    npt.assert_array_equal(bb.to_numpy(block), np.array([[0.0, 5.0], [0.0, 0.0]]))
    del block  # the view keeps the buffer alive
    assert view[0, 1] == 5.0


def test_ndarray_unsupported_dtype(bb):
    with pytest.raises(TypeError):
        bb.block_from_ndarray(np.zeros(3, dtype=np.int32))
