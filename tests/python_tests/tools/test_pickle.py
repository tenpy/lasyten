"""Test pickle support of the nanobind-bound lasyten types.

The pickled state mirrors the logical fields that ``save_hdf5`` / ``from_hdf5`` persist (see
``test_hdf5.py``), but no HDF5 file is involved; pickling is intended for in-Python workflows
(``multiprocessing``, ``copy.deepcopy``, ...).
"""

# Copyright (C) TeNPy Developers, Apache license

import copy
import pickle

import numpy as np
import pytest

import lasyten
from lasyten import U1, Dtype, NoSymmetry, Sector, SectorArray

PROTOCOLS = list(range(2, pickle.HIGHEST_PROTOCOL + 1))


def roundtrip(obj, protocol=pickle.HIGHEST_PROTOCOL):
    return pickle.loads(pickle.dumps(obj, protocol=protocol))


# --- pilot types ---------------------------------------------------------------------------------


@pytest.mark.parametrize('protocol', PROTOCOLS)
def test_pickle_dtype(protocol):
    for dtype in Dtype.__members__.values():
        assert roundtrip(dtype, protocol) == dtype


@pytest.mark.parametrize('protocol', PROTOCOLS)
def test_pickle_sector(protocol):
    sector = Sector([1, -2, 3])
    loaded = roundtrip(sector, protocol)
    assert isinstance(loaded, Sector)
    assert loaded == sector
    assert loaded.to_numpy().dtype == np.int64
    np.testing.assert_array_equal(loaded.to_numpy(), [1, -2, 3])
    assert roundtrip(NoSymmetry().trivial_sector) == NoSymmetry().trivial_sector


@pytest.mark.parametrize('protocol', PROTOCOLS)
def test_pickle_sector_array(protocol):
    sectors = SectorArray(np.array([[0, 1], [2, 3], [-4, 5]], dtype=np.int64))
    loaded = roundtrip(sectors, protocol)
    assert isinstance(loaded, SectorArray)
    assert loaded == sectors
    assert loaded.shape == (3, 2)
    # an empty array keeps its ``sector_ind_len``
    empty = SectorArray.empty(3)
    loaded_empty = roundtrip(empty, protocol)
    assert loaded_empty.shape == (0, 3)
    assert loaded_empty == empty


@pytest.mark.parametrize('protocol', PROTOCOLS)
def test_pickle_no_symmetry_and_u1(protocol):
    no_symm = roundtrip(NoSymmetry(), protocol)
    assert isinstance(no_symm, NoSymmetry)
    assert no_symm.is_equivalent_to(NoSymmetry())
    assert no_symm == NoSymmetry()

    u1 = roundtrip(U1(), protocol)
    assert isinstance(u1, U1)
    assert u1.is_equivalent_to(U1())
    assert u1.descriptive_name is None
    assert u1.trivial_shift is True

    # constructor args (``descriptive_name``, ``trivial_shift``) survive
    named = roundtrip(U1('charge', trivial_shift=False), protocol)
    assert named.descriptive_name == 'charge'
    assert named.trivial_shift is False
    assert named == U1('charge', trivial_shift=False)


def test_pickle_block_and_scalar(block_backend):
    be = lasyten.get_backend('no_symmetry', block_backend).block_backend
    block = be.as_block(np.arange(6, dtype=np.float64).reshape(2, 3))
    loaded = roundtrip(block)
    assert type(loaded) is type(block)
    assert loaded.shape == block.shape
    assert loaded.dtype == block.dtype
    np.testing.assert_array_equal(loaded.to_numpy(), block.to_numpy())
    assert loaded.get_backend() is be

    for value, dtype in [(3.5, Dtype.float64), (1 + 2j, Dtype.complex128)]:
        scalar = be.as_scalar(value)
        loaded_scalar = roundtrip(scalar)
        assert loaded_scalar.dtype == scalar.dtype == dtype
        assert loaded_scalar.as_complex128() == pytest.approx(value)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128, np.int64, np.bool_])
def test_pickle_block_dtypes(block_backend, dtype):
    be = lasyten.get_backend('no_symmetry', block_backend).block_backend
    arr = (np.arange(24).reshape(2, 3, 4) % 3 + 1).astype(dtype)
    block = be.as_block(arr)
    loaded = roundtrip(block)
    assert loaded.dtype == block.dtype
    np.testing.assert_array_equal(loaded.to_numpy(), arr)


def test_pickle_block_backend(block_backend):
    be = lasyten.get_backend('no_symmetry', block_backend).block_backend
    assert roundtrip(be) is be


def test_pickle_copy_deepcopy():
    sectors = SectorArray([[0], [1]])
    assert copy.deepcopy(sectors) == sectors
    assert copy.copy(U1('x')) == U1('x')


# --- nested containers ---------------------------------------------------------------------------


def test_pickle_nested_containers(block_backend):
    be = lasyten.get_backend('no_symmetry', block_backend).block_backend
    block = be.as_block(np.arange(4, dtype=np.float64).reshape(2, 2))
    data = {
        'dtype': Dtype.float64,
        'symmetries': [NoSymmetry(), U1('a'), (U1(), NoSymmetry())],
        'sectors': {'one': Sector([1]), 'many': SectorArray([[0, 1], [1, 0]])},
        'block': block,
        'scalar': be.as_scalar(2.0),
        'plain': [1, 2.5, 'three', None],
    }
    loaded = roundtrip(data)
    assert loaded['dtype'] == Dtype.float64
    assert loaded['symmetries'][1] == U1('a')
    assert loaded['symmetries'][2][0] == U1()
    assert loaded['sectors']['one'] == Sector([1])
    assert loaded['sectors']['many'] == SectorArray([[0, 1], [1, 0]])
    np.testing.assert_array_equal(loaded['block'].to_numpy(), block.to_numpy())
    assert loaded['scalar'].as_float64() == 2.0
    assert loaded['plain'] == data['plain']

    # shared objects stay shared (pickle memo)
    shared = U1('shared')
    loaded_pair = roundtrip([shared, shared])
    assert loaded_pair[0] is loaded_pair[1]
