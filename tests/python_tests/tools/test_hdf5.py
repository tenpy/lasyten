"""Test output to and import from hdf5."""

import os
import warnings

import io_test
import numpy as np
import pytest

import lasyten
from lasyten.tools import hdf5_io

h5py = pytest.importorskip('h5py')

datadir_hdf5 = [f for f in io_test.datadir_files if f.endswith('.hdf5')]


def export_to_datadir():
    filename = io_test.get_datadir_filename('exported_from_tenpy_{0}.hdf5')
    data = io_test.gen_example_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings('ignore', category=UserWarning)
        with h5py.File(filename, 'w') as f:
            hdf5_io.save_to_hdf5(f, data)
    for w in caught:
        msg = str(w.message)
        expected = 'without explicit HDF5 format' in msg
        if expected:
            expected = any(
                t in msg
                for t in [
                    'io_test.DummyClass',
                    'tenpy.tools.events.EventHandler',
                    'tenpy.tools.events.Listener',
                    'method',
                ]
            )
        if not expected:
            warnings.showwarning(w.message, w.category, w.filename, w.lineno, w.file, w.line)


@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_hdf5_block_and_scalar_io(block_backend, tmp_path):
    """Roundtrip C++ NumpyBlockBackend Block and Scalar via HDF5."""

    be = lasyten.get_backend('no_symmetry', block_backend).block_backend
    block = be.as_block(np.arange(6, dtype=np.float64).reshape(2, 3))
    scalar = be.as_scalar(3.5)

    filename = tmp_path / 'block_scalar.hdf5'
    with h5py.File(str(filename), 'w') as f:
        hdf5_io.save_to_hdf5(f, {'block': block, 'scalar': scalar})
    with h5py.File(str(filename), 'r') as f:
        loaded = hdf5_io.load_from_hdf5(f)

    block_loaded = loaded['block']
    scalar_loaded = loaded['scalar']
    np.testing.assert_array_equal(block_loaded.to_numpy(), block.to_numpy())
    assert block_loaded.shape == block.shape
    assert scalar_loaded.as_float64() == pytest.approx(3.5)
    assert scalar_loaded.dtype == scalar.dtype


@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_hdf5_export_import(tmp_path):
    """Try subsequent export and import to pickle."""
    data = io_test.gen_example_data()
    filename = tmp_path / 'test.hdf5'
    with h5py.File(str(filename), 'w') as f:
        hdf5_io.save_to_hdf5(f, data)
    with h5py.File(str(filename), 'r') as f:
        data_imported = hdf5_io.load_from_hdf5(f)
    io_test.assert_equal_data(data_imported, data)


@pytest.mark.parametrize('fn', datadir_hdf5)
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_import_from_datadir(fn):
    print('import ', fn)
    filename = os.path.join(io_test.datadir, fn)
    with h5py.File(filename, 'r') as f:
        data = hdf5_io.load_from_hdf5(f)
    if 'version' not in data:
        raise ValueError(f'Version not found in data: {data.keys()}')
    data_expected = io_test.gen_example_data(data['version'])
    io_test.assert_equal_data(data, data_expected)


def _hdf5_roundtrip(obj, tmp_path, filename='sym.hdf5'):
    path = tmp_path / filename
    with h5py.File(str(path), 'w') as f:
        hdf5_io.save_to_hdf5(f, obj)
    with h5py.File(str(path), 'r') as f:
        return hdf5_io.load_from_hdf5(f)


def _with_descriptive_name(obj, name):
    obj.descriptive_name = name
    return obj


def _symmetry_hdf5_cases():
    S = lasyten.symmetries
    cases = [
        ('NoSymmetry', S.NoSymmetry(), {}),
        ('NoSymmetry named', _with_descriptive_name(S.NoSymmetry(), 'trivial'), {'descriptive_name': 'trivial'}),
        ('U1', S.U1(), {}),
        ('U1 named', S.U1('Sz'), {'descriptive_name': 'Sz'}),
        ('U1 no trivial_shift', S.U1('N', False), {'descriptive_name': 'N', 'trivial_shift': False}),
        ('ZN(2)', S.ZN(2), {'N': 2}),
        ('ZN(3)', S.ZN(3), {'N': 3}),
        ('ZN(5) named', S.ZN(5, 'mod5'), {'N': 5, 'descriptive_name': 'mod5'}),
        ('ZN(7) no shift', S.ZN(7, 'bar', False), {'N': 7, 'descriptive_name': 'bar', 'trivial_shift': False}),
        ('_SU2', S._SU2(), {}),
        ('_SU2 named', S._SU2('spin'), {'descriptive_name': 'spin'}),
        ('FermionNumber', S.FermionNumber(), {}),
        ('FermionNumber named', S.FermionNumber('N_f', False), {'descriptive_name': 'N_f', 'trivial_shift': False}),
        ('FermionParity', S.FermionParity(), {}),
        ('FermionParity named', S.FermionParity('P', False), {'descriptive_name': 'P', 'trivial_shift': False}),
        ('Fib left', S.FibonacciAnyonCategory('left'), {'handedness': 'left'}),
        ('Fib right', S.FibonacciAnyonCategory('right'), {'handedness': 'right'}),
        (
            'Fib named',
            _with_descriptive_name(S.FibonacciAnyonCategory('left'), 'fib'),
            {'handedness': 'left', 'descriptive_name': 'fib'},
        ),
        ('Ising nu=1', S.IsingAnyonCategory(1), {'nu': 1}),
        ('Ising nu=5', S.IsingAnyonCategory(5), {'nu': 5}),
        # Constructor canonicalizes nu modulo 16 into [0, 16).
        ('Ising nu=-3', S.IsingAnyonCategory(-3), {'nu': 13}),
        (
            'Ising named',
            _with_descriptive_name(S.IsingAnyonCategory(3), 'ising'),
            {'nu': 3, 'descriptive_name': 'ising'},
        ),
        ('SU2_k 4 left', S.SU2_kAnyonCategory(4, 'left'), {'k': 4, 'handedness': 'left'}),
        ('SU2_k 3 right', S.SU2_kAnyonCategory(3, 'right'), {'k': 3, 'handedness': 'right'}),
        (
            'SU2_k named',
            _with_descriptive_name(S.SU2_kAnyonCategory(2, 'left'), 'su2k'),
            {'k': 2, 'handedness': 'left', 'descriptive_name': 'su2k'},
        ),
        ('SU3_3', S.SU3_3AnyonCategory(), {}),
        ('SU3_3 named', _with_descriptive_name(S.SU3_3AnyonCategory(), 'su3_3'), {'descriptive_name': 'su3_3'}),
        ('Toric', S.ToricCodeCategory(), {}),
        ('Toric named', S.ToricCodeCategory('tc'), {'descriptive_name': 'tc'}),
        ('ZNAnyon 3,1', S.ZNAnyonCategory(3, 1), {'N': 3, 'n': 1}),
        ('ZNAnyon 5,2 named', S.ZNAnyonCategory(5, 2, 'any'), {'N': 5, 'n': 2, 'descriptive_name': 'any'}),
        ('ZNAnyon2 4,1', S.ZNAnyonCategory2(4, 1), {'N': 4, 'n': 1}),
        ('ZNAnyon2 6,5 named', S.ZNAnyonCategory2(6, 5, 'half'), {'N': 6, 'n': 5, 'descriptive_name': 'half'}),
        ('QDZN 3', S.QuantumDoubleZNAnyonCategory(3), {'N': 3}),
        ('QDZN 4 named', S.QuantumDoubleZNAnyonCategory(4, 'qd'), {'N': 4, 'descriptive_name': 'qd'}),
        ('ZN.as_Symmetry', S.ZN(6).as_Symmetry(), {}),
        ('U1.as_Symmetry named', S.U1('charge').as_Symmetry(), {}),
        ('product ZN x U1', S.ZN(3, 'parity') * S.U1('Sz'), {}),
        ('product Fib x U1', S.FibonacciAnyonCategory('right') * S.U1(), {}),
        (
            'product extra params',
            S.ZN(5, 'mod5', False) * S.FibonacciAnyonCategory('left') * S.U1('Sz'),
            {},
        ),
    ]
    return [pytest.param(obj, attrs, id=case_id) for case_id, obj, attrs in cases]


@pytest.mark.parametrize('obj,attrs', _symmetry_hdf5_cases())
@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_hdf5_symmetry_roundtrip(obj, attrs, tmp_path):
    loaded = _hdf5_roundtrip(obj, tmp_path)
    assert type(loaded) is type(obj)
    assert loaded == obj
    assert loaded.is_equivalent_to(obj)
    for name, expected in attrs.items():
        assert getattr(loaded, name) == expected
    if isinstance(obj, lasyten.symmetries.Symmetry):
        assert loaded.num_factors == obj.num_factors
        for f_loaded, f_orig in zip(loaded.factors, obj.factors, strict=True):
            assert type(f_loaded) is type(f_orig)
            assert f_loaded == f_orig
            if hasattr(f_orig, 'N'):
                assert f_loaded.N == f_orig.N
            if hasattr(f_orig, 'n'):
                assert f_loaded.n == f_orig.n
            if hasattr(f_orig, 'handedness'):
                assert f_loaded.handedness == f_orig.handedness
            if hasattr(f_orig, 'nu'):
                assert f_loaded.nu == f_orig.nu
            if hasattr(f_orig, 'k'):
                assert f_loaded.k == f_orig.k
            assert f_loaded.descriptive_name == f_orig.descriptive_name
            assert f_loaded.trivial_shift == f_orig.trivial_shift


@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_hdf5_symmetry_extra_params_are_distinguished(tmp_path):
    S = lasyten.symmetries
    pairs = [
        (S.ZN(3), S.ZN(4)),
        (S.ZN(3, 'a'), S.ZN(3, 'b')),
        (S.U1('Sz'), S.U1('N')),
        (S.FibonacciAnyonCategory('left'), S.FibonacciAnyonCategory('right')),
        (S.IsingAnyonCategory(1), S.IsingAnyonCategory(3)),
        (S.SU2_kAnyonCategory(3, 'left'), S.SU2_kAnyonCategory(4, 'left')),
        (S.SU2_kAnyonCategory(3, 'left'), S.SU2_kAnyonCategory(3, 'right')),
        (S.ZNAnyonCategory(3, 1), S.ZNAnyonCategory(3, 2)),
        (S.ZNAnyonCategory(4, 1), S.ZNAnyonCategory(5, 1)),
        (S.ZNAnyonCategory2(4, 1), S.ZNAnyonCategory2(6, 1)),
        (S.QuantumDoubleZNAnyonCategory(3), S.QuantumDoubleZNAnyonCategory(4)),
        (S.ZN(2).as_Symmetry(), S.ZN(3).as_Symmetry()),
        (S.ZN(3) * S.U1(), S.ZN(4) * S.U1()),
        (S.FibonacciAnyonCategory('left') * S.U1(), S.FibonacciAnyonCategory('right') * S.U1()),
    ]
    data = {f'a{i}': a for i, (a, _) in enumerate(pairs)}
    data.update({f'b{i}': b for i, (_, b) in enumerate(pairs)})
    loaded = _hdf5_roundtrip(data, tmp_path, 'pairs.hdf5')
    for i, (a, b) in enumerate(pairs):
        assert a != b
        la = loaded[f'a{i}']
        lb = loaded[f'b{i}']
        assert la == a
        assert lb == b
        assert la != lb
        assert type(la) is type(a)
        assert type(lb) is type(b)


@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_hdf5_leg_pipe_roundtrip(tmp_path):
    """Plain LegPipe (and nested pipes / TensorProduct factors) roundtrip via HDF5."""
    S = lasyten.symmetries
    symmetry = S.NoSymmetry().as_Symmetry()
    a = lasyten.ElementarySpace.from_trivial_sector(2, symmetry=symmetry)
    b = lasyten.ElementarySpace.from_trivial_sector(3, symmetry=symmetry, is_dual=True)
    pipe = lasyten.LegPipe([a, b], is_dual=True, combine_cstyle=False)
    nested = lasyten.LegPipe([pipe, a], is_dual=False, combine_cstyle=True)
    product = lasyten.TensorProduct([nested, b])

    loaded = _hdf5_roundtrip(
        {'pipe': pipe, 'nested': nested, 'product': product},
        tmp_path,
        'leg_pipe.hdf5',
    )
    for key, orig in [('pipe', pipe), ('nested', nested)]:
        got = loaded[key]
        assert type(got) is type(orig)
        assert got == orig
        assert got.is_dual == orig.is_dual
        assert got.combine_cstyle == orig.combine_cstyle
        assert got.num_legs == orig.num_legs
        assert got.dim == orig.dim
        got.test_sanity()

    got_prod = loaded['product']
    assert type(got_prod) is type(product)
    assert got_prod == product
    assert got_prod.num_factors == product.num_factors
    assert isinstance(got_prod.factors[0], lasyten.LegPipe)
