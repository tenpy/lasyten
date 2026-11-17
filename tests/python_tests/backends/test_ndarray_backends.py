"""Native nb::ndarray I/O for NumpyBlockBackend and TorchBlockBackend (leftover #10)."""

from __future__ import annotations

import importlib.util

import numpy as np
import numpy.testing as npt
import pytest


@pytest.fixture
def numpy_bb():
    from lasyten.block_backends import NumpyBlockBackend

    return NumpyBlockBackend.from_factory('cpu')


def test_numpy_ndarray_roundtrip(numpy_bb):
    arr = np.arange(6.0).reshape(2, 3)
    block = numpy_bb.block_from_ndarray(arr)
    out = numpy_bb.to_ndarray(block)
    npt.assert_array_equal(out, arr)
    # Import is an owning copy (source may be GC'd); export is still zero-copy.
    arr[0, 0] = -1.0
    npt.assert_array_equal(numpy_bb.to_numpy(block)[0, 0], 0.0)
    out[0, 0] = 99.0
    npt.assert_array_equal(numpy_bb.to_numpy(block), out)
    ro = block.to_ndarray(writable=False)
    assert not ro.flags.writeable


def test_numpy_block_from_ndarray_owns_storage_after_gc(numpy_bb):
    """MPS from_Bflat / product-state regression: Block must outlive the source array.

    ``nb::cast(nb::ndarray)`` + ``py::array::ensure`` used to leave the Block aliasing a
    buffer that died with the caller's ndarray → segfault on ``Block.dtype`` after GC.
    """
    import gc

    from lasyten import Dtype

    blocks = []
    for i in range(4):
        # MPS product-state Bflat shape (p, vL, vR)
        B = np.zeros((2, 1, 1), dtype=float)
        B[0 if i % 2 == 0 else 1, 0, 0] = 1.0
        blocks.append(numpy_bb.block_from_ndarray(B, dtype=Dtype.float64, device='cpu'))
        del B
    gc.collect()
    for i, block in enumerate(blocks):
        assert block.dtype == Dtype.float64
        assert numpy_bb.get_shape(block) == (2, 1, 1)
        out = numpy_bb.to_numpy(block)
        assert out[0 if i % 2 == 0 else 1, 0, 0] == 1.0


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='torch not installed')
def test_torch_ndarray_cpu_roundtrip():
    import torch

    from lasyten.block_backends import TorchBlockBackend

    bb = TorchBlockBackend.from_factory('cpu:0')
    arr = np.random.default_rng(0).standard_normal((2, 3))
    block = bb.block_from_ndarray(arr)
    view = bb.to_ndarray(block)
    assert isinstance(view, torch.Tensor)
    assert view.device.type == 'cpu'
    npt.assert_allclose(view.detach().cpu().numpy(), arr)
    # zero-copy: mutate through the view
    view[0, 0] = 1.25
    npt.assert_allclose(bb.to_numpy(block)[0, 0], 1.25)


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='torch not installed')
def test_torch_ndarray_cuda_export_if_available():
    import torch

    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from lasyten.block_backends import TorchBlockBackend

    bb = TorchBlockBackend.from_factory('cuda:0')
    arr = np.arange(4.0).reshape(2, 2)
    block = bb.block_from_ndarray(arr, device='cuda:0')
    view = bb.to_ndarray(block)
    assert isinstance(view, torch.Tensor)
    assert view.device.type == 'cuda'
    npt.assert_allclose(view.detach().cpu().numpy(), arr)


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='torch not installed')
def test_torch_ndarray_cuda_import_keeps_device():
    """CUDA torch tensors import without bouncing through host memory."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from lasyten.block_backends import TorchBlockBackend

    bb = TorchBlockBackend.from_factory('cuda:0')
    src = torch.arange(6.0, device='cuda:0').reshape(2, 3)
    block = bb.block_from_ndarray(src)
    view = bb.to_ndarray(block)
    assert view.device.type == 'cuda'
    npt.assert_allclose(view.detach().cpu().numpy(), src.detach().cpu().numpy())


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='torch not installed')
def test_eigen_rejects_cuda_ndarray():
    """Eigen/Numpy are host-only; CUDA inputs raise TypeError."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from lasyten.block_backends import EigenBlockBackend

    bb = EigenBlockBackend.from_factory('cpu')
    src = torch.zeros(2, 2, dtype=torch.float64, device='cuda:0')
    with pytest.raises(TypeError, match='CPU array'):
        bb.block_from_ndarray(src)


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='torch not installed')
def test_eigen_to_ndarray_framework_torch():
    """framework='torch' retags Eigen storage as a torch.Tensor view."""
    import torch

    from lasyten.block_backends import EigenBlockBackend

    bb = EigenBlockBackend.from_factory('cpu')
    block = bb.block_from_ndarray(np.arange(4.0).reshape(2, 2))
    view = bb.to_ndarray(block, framework='torch')
    assert isinstance(view, torch.Tensor)
    npt.assert_allclose(view.detach().cpu().numpy(), np.arange(4.0).reshape(2, 2))
    # default remains NumPy
    assert isinstance(block.to_ndarray(), np.ndarray)


@pytest.mark.skipif(importlib.util.find_spec('jax') is None, reason='jax not installed')
def test_eigen_to_ndarray_framework_jax():
    import jax.numpy as jnp

    from lasyten.block_backends import EigenBlockBackend

    bb = EigenBlockBackend.from_factory('cpu')
    block = bb.block_from_ndarray(np.arange(3.0))
    view = bb.to_ndarray(block, framework='jax')
    assert isinstance(view, jnp.ndarray) or type(view).__module__.startswith('jax')
    npt.assert_allclose(np.asarray(view), np.arange(3.0))


@pytest.mark.skipif(importlib.util.find_spec('tensorflow') is None, reason='tensorflow not installed')
def test_eigen_to_ndarray_framework_tensorflow():
    import tensorflow as tf

    from lasyten.block_backends import EigenBlockBackend

    bb = EigenBlockBackend.from_factory('cpu')
    block = bb.block_from_ndarray(np.arange(3.0))
    view = bb.to_ndarray(block, framework='tensorflow')
    assert isinstance(view, tf.Tensor)
    npt.assert_allclose(view.numpy(), np.arange(3.0))


def test_to_ndarray_framework_invalid():
    from lasyten.block_backends import EigenBlockBackend

    bb = EigenBlockBackend.from_factory('cpu')
    block = bb.block_from_ndarray(np.zeros(2))
    with pytest.raises(TypeError, match='framework'):
        bb.to_ndarray(block, framework='cupy')
