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
    # zero-copy: mutating the view mutates the block
    out[0, 0] = 99.0
    npt.assert_array_equal(numpy_bb.to_numpy(block), out)
    ro = block.to_ndarray(writable=False)
    assert not ro.flags.writeable


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
