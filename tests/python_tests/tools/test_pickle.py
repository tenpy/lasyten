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


# --- symmetries, spaces, backends ----------------------------------------------------------------


@pytest.mark.parametrize('protocol', PROTOCOLS)
def test_pickle_symmetry(any_symmetry, protocol):
    loaded = roundtrip(any_symmetry, protocol)
    assert type(loaded) is type(any_symmetry)
    assert loaded == any_symmetry
    assert loaded.is_equivalent_to(any_symmetry)
    assert loaded.num_sectors == any_symmetry.num_sectors
    assert loaded.trivial_sector == any_symmetry.trivial_sector
    for factor in getattr(any_symmetry, 'factors', []):
        assert roundtrip(factor, protocol) == factor


def test_pickle_symmetry_factors():
    factors = [
        lasyten.ZN(3),
        lasyten.FermionParity(),
        lasyten.FermionNumber(),
        lasyten.FibonacciAnyonCategory(),
        lasyten.IsingAnyonCategory(),
        lasyten.ZNAnyonCategory(3, 1),
        lasyten.SU2Symmetry() if hasattr(lasyten, 'SU2Symmetry') else lasyten.NoSymmetry(),
    ]
    for factor in factors:
        loaded = roundtrip(factor)
        assert type(loaded) is type(factor)
        assert loaded == factor


def test_pickle_space(make_any_space):
    for is_dual in [False, True]:
        space = make_any_space(is_dual=is_dual)
        loaded = roundtrip(space)
        assert type(loaded) is type(space)
        assert loaded == space
        assert loaded.is_dual == is_dual
        assert loaded.symmetry == space.symmetry
        assert loaded.dim == space.dim
        if space.symmetry.can_be_dropped:
            np.testing.assert_array_equal(loaded.basis_perm, space.basis_perm)
        loaded.test_sanity()


def test_pickle_space_with_basis_perm():
    symmetry = U1().as_Symmetry()
    space = lasyten.ElementarySpace(
        symmetry, SectorArray([[0], [1], [-1]]), [2, 1, 3], is_dual=True, basis_perm=np.array([2, 0, 1, 3, 5, 4])
    )
    loaded = roundtrip(space)
    assert loaded == space
    assert loaded.is_dual
    np.testing.assert_array_equal(loaded.basis_perm, space.basis_perm)


def test_pickle_tensor_product_and_pipes(make_any_space):
    spaces = [make_any_space(max_sectors=3, max_mult=2) for _ in range(3)]
    product = lasyten.TensorProduct(spaces)
    loaded = roundtrip(product)
    assert loaded == product
    assert loaded.num_factors == product.num_factors

    ds = lasyten.DirectSumSpace(spaces[:1] * 2, is_dual=spaces[0].is_dual)
    loaded_ds = roundtrip(ds)
    assert type(loaded_ds) is type(ds)
    assert loaded_ds == ds


def test_pickle_tensor_backend(any_backend):
    loaded = roundtrip(any_backend)
    assert type(loaded) is type(any_backend)
    assert loaded == any_backend
    assert loaded.symmetry_backend if hasattr(any_backend, 'symmetry_backend') else True


# --- tensors -------------------------------------------------------------------------------------


def assert_tensor_roundtrip(tensor, protocol=pickle.HIGHEST_PROTOCOL):
    loaded = roundtrip(tensor, protocol)
    assert type(loaded) is type(tensor)
    loaded.test_sanity()
    assert loaded.dtype == tensor.dtype
    assert loaded.device == tensor.device
    assert loaded.shape == tensor.shape
    assert loaded.labels == tensor.labels
    assert loaded.codomain == tensor.codomain
    assert loaded.domain == tensor.domain
    assert loaded.backend == tensor.backend
    assert lasyten.almost_equal(loaded, tensor, rtol=0, atol=0)
    return loaded


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('leg_nums', [(1, 1), (2, 1), (0, 2)], ids=['1->1', '1->2', '0->2'])
def test_pickle_symmetric_tensor(make_compatible_tensor, leg_nums, protocol):
    tensor = make_compatible_tensor(*leg_nums)
    assert_tensor_roundtrip(tensor, protocol)


@pytest.mark.parametrize('dtype', [Dtype.float64, Dtype.complex128, Dtype.complex64])
def test_pickle_symmetric_tensor_dtype(make_compatible_tensor, compatible_symmetry, dtype):
    if not dtype.is_complex and not compatible_symmetry.can_be_dropped:
        pytest.skip('symmetry requires complex dtype')
    tensor = make_compatible_tensor(2, 1, dtype=dtype, labels=['a', 'b', 'c'])
    loaded = assert_tensor_roundtrip(tensor)
    assert loaded.dtype == dtype
    assert loaded.labels == ['a', 'b', 'c']


def test_pickle_diagonal_tensor_and_identity(make_compatible_space, make_compatible_tensor):
    leg = make_compatible_space()
    diag = make_compatible_tensor(codomain=[leg], domain=[leg], cls=lasyten.DiagonalTensor, labels=['a', 'b'])
    assert_tensor_roundtrip(diag)
    dtype = Dtype.float64 if diag.symmetry.can_be_dropped else Dtype.complex128
    eye = lasyten.Identity.from_eye(leg, backend=diag.backend, labels=['a', 'b'], dtype=dtype)
    assert_tensor_roundtrip(eye)


def test_pickle_mask(make_compatible_space, make_compatible_tensor):
    leg = make_compatible_space(allow_basis_perm=False)
    mask = make_compatible_tensor(codomain=[leg], domain=[leg], cls=lasyten.Mask)
    loaded = assert_tensor_roundtrip(mask)
    assert loaded.is_projection == mask.is_projection


@pytest.mark.deselect_invalid_ChargedTensor_cases(get_cls=lambda kw: lasyten.ChargedTensor)
def test_pickle_charged_tensor(make_compatible_tensor):
    tensor = make_compatible_tensor(2, 1, cls=lasyten.ChargedTensor)
    loaded = roundtrip(tensor)
    assert type(loaded) is type(tensor)
    loaded.test_sanity()
    assert loaded.dtype == tensor.dtype
    assert loaded.shape == tensor.shape
    assert loaded.labels == tensor.labels
    assert loaded.charge_leg == tensor.charge_leg
    assert lasyten.almost_equal(loaded, tensor, rtol=0, atol=0)


def test_pickle_tensor_in_container(make_compatible_tensor, make_compatible_space):
    tensor = make_compatible_tensor(1, 1)
    data = {'tensors': [tensor, tensor], 'space': tensor.codomain[0], 'meta': {'n': 1}}
    loaded = roundtrip(data)
    # the pickle memo preserves shared references
    assert loaded['tensors'][0] is loaded['tensors'][1]
    assert lasyten.almost_equal(loaded['tensors'][0], tensor, rtol=0, atol=0)
    assert loaded['space'] == tensor.codomain[0]
    assert copy.deepcopy(tensor).labels == tensor.labels


def test_pickle_leg_labels():
    labels = lasyten.tensors.LegLabels(['a', 'b', 'c'])
    loaded = roundtrip(labels)
    assert loaded == labels


# --- config and sites ----------------------------------------------------------------------------


def test_pickle_config():
    with lasyten.temporary_options(print_linewidth=77, check_fusion=False):
        modified = lasyten.get_config()
        loaded = roundtrip(modified)
        assert type(loaded) is type(modified)
        assert loaded.print_linewidth == 77
        assert loaded.check_fusion is False
        assert str(loaded) == str(modified)


def test_pickle_site():
    site = lasyten.models.SpinlessFermionSite(num_species=1, conserve='N')
    loaded = roundtrip(site)
    assert type(loaded) is type(site)
    loaded.test_sanity()
    assert loaded.leg == site.leg
    assert loaded.state_labels == site.state_labels
    assert sorted(loaded.onsite_operators) == sorted(site.onsite_operators)
    assert lasyten.almost_equal(loaded.get_op('N'), site.get_op('N'))
