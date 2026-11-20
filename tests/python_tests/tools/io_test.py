"""Provide functionality for import (input) and export (output) tests.

The subfolder "data/" (given by `datadir`) in the directory of this file is used for storage of
test files to be imported. Files in this subfolder are detected by tests and checked for correct
import. In that way, we can test the import of data files exported by old TeNPy versions. To
generate the files, you can call the ``test_*.py`` files in this folder manually, e.g., ``python
test_hdf5.py``. This will generate the files with pre-defined data (see :func:`gen_example_data`)
and the tenpy version in the filename.
"""
# Copyright (C) TeNPy Developers, Apache license

import os
import types

import numpy as np

import lasyten

try:
    from packaging.version import parse as parse_version  # part of setuptools
except ImportError:

    def parse_version(version_str):
        """Hack to allow very basic version comparison.

        Does not handle cases of alpha, beta, pre and dev releases correctly.
        """
        return version_str.split('.')


__all__ = ['datadir', 'datadir_files', 'gen_example_data', 'assert_equal_data', 'get_datadir_filename']

datadir = os.path.join(os.path.dirname(__file__), 'data')
datadir_files = []
if os.path.isdir(datadir):
    datadir_files = os.listdir(datadir)


def _clean_version(version):
    if '+' in version:
        version = version.split('+')[0]  # discard '+GITHASH' / local date suffix
    return version


def gen_example_data(version=lasyten.__version__):
    version = _clean_version(version)

    SU2_SpinSite = lasyten.models.SpinSite(conserve='SU2', use_test_su2=True)
    FermionSite = lasyten.models.SpinlessFermionSite(num_species=1, conserve='N')
    leg = FermionSite.leg
    n_op = FermionSite.get_op('N')  # DiagonalTensor
    identity = FermionSite.get_op('Id')  # Identity
    # Two-leg identity map stored as a generic SymmetricTensor (not DiagonalTensor).
    symmetric = lasyten.tensors.SymmetricTensor.from_eye([leg], labels=['p'])
    mask = lasyten.tensors.Mask.from_block_mask(np.array([True, False]), large_leg=leg, labels=['p'])
    # ChargedTensor: hide a trivial charge leg on a non-zero operator.
    inv = lasyten.add_trivial_leg(n_op.as_SymmetricTensor(), domain_pos=0, label='#')
    charged = lasyten.tensors.ChargedTensor(inv, charged_state=[1.0])

    # HiddenLegTensor: SymmetricTensor with a ``!``-prefixed hidden leg.
    eye2 = lasyten.tensors.SymmetricTensor.from_eye([leg, leg], labels=['a', 'b'])
    hidden = lasyten.tensors.HiddenLegTensor(eye2, ['b'])

    # Spaces / pipes (abelian + plain FusionTree nesting).
    u1 = lasyten.symmetries.U1().as_Symmetry()
    es_a = lasyten.ElementarySpace.from_defining_sectors(u1, [[0], [1]], [1, 1])
    es_b = lasyten.ElementarySpace.from_defining_sectors(u1, [[0], [-1]], [1, 1], is_dual=True)
    abelian_pipe = lasyten.AbelianLegPipe([es_a, es_b], is_dual=False, combine_cstyle=True)
    direct_sum = lasyten.DirectSumSpace([es_a, es_a], is_dual=False)

    no_sym = lasyten.symmetries.NoSymmetry().as_Symmetry()
    ns_a = lasyten.ElementarySpace.from_trivial_sector(2, symmetry=no_sym)
    ns_b = lasyten.ElementarySpace.from_trivial_sector(3, symmetry=no_sym, is_dual=True)
    leg_pipe = lasyten.LegPipe([ns_a, ns_b], is_dual=True, combine_cstyle=False)
    nested_pipe = lasyten.LegPipe([leg_pipe, ns_a], is_dual=False, combine_cstyle=True)
    tensor_product = lasyten.TensorProduct([nested_pipe, ns_b])

    data = {
        'version': version,
        'None': None,
        'scalars': [0, np.int64(1), 2.0, np.float64(3.0), 4.0j, 'five'],
        'arrays': [np.array([6, 66]), np.array([]), np.zeros([])],
        'iterables': [[], [11, 12], tuple([]), tuple([1, 2, 3]), set([]), set([1, 2, 3])],
        'recursive': [0, None, 2, [3, None, 5]],
        'dict_complicated': {0: 1, 'asdf': 2, (1, 2): '3'},
        'exportable': lasyten.tools.hdf5_io.Hdf5Exportable(),
        'range': range(2, 8, 3),
        'dtypes': [np.dtype('int64'), np.dtype([('a', np.int32, 8), ('b', np.float64, 5)])],
        'symmetry': FermionSite.symmetry,
        'tensor': n_op,
        'tensors': {
            'SymmetricTensor': symmetric,
            'DiagonalTensor': n_op,
            'Identity': identity,
            'Mask': mask,
            'ChargedTensor': charged,
            'HiddenLegTensor': hidden,
        },
        'spaces': {
            'ElementarySpace': es_a,
            'AbelianLegPipe': abelian_pipe,
            'DirectSumSpace': direct_sum,
            'LegPipe': leg_pipe,
            'nested_LegPipe': nested_pipe,
            'TensorProduct': tensor_product,
        },
        'site': SU2_SpinSite,
    }
    data['recursive'][3][1] = data['recursive'][1] = data['recursive']
    data['exportable'].some_attr = 'something'

    return data


def assert_equal_data(data_imported, data_expected, max_recursion_depth=10):
    """Check that the imported data is as expected."""
    assert isinstance(data_imported, type(data_expected))
    if hasattr(data_expected, 'test_sanity'):
        data_imported.test_sanity()
    if isinstance(data_expected, dict):
        assert set(data_imported.keys()) == set(data_expected.keys())
        if max_recursion_depth > 0:
            for ki in data_imported.keys():
                assert_equal_data(data_imported[ki], data_expected[ki], max_recursion_depth - 1)
    elif isinstance(data_expected, (list, tuple)):
        if max_recursion_depth > 0:
            for vi, ve in zip(data_imported, data_expected):
                assert_equal_data(vi, ve, max_recursion_depth - 1)
    elif isinstance(data_expected, np.ndarray):
        assert np.linalg.norm(data_imported - data_expected) == 0.0  # should be exactly equal!
    elif isinstance(data_expected, np.dtype):
        assert data_imported == data_expected
    elif isinstance(data_expected, (int, float, np.int64, np.float64, complex, str, bytes, bool)):
        assert data_imported == data_expected
    elif isinstance(data_expected, range):
        assert tuple(data_imported) == tuple(data_expected)
    elif isinstance(data_expected, (lasyten.symmetries.Symmetry, lasyten.symmetries.SymmetryFactor)):
        # Extra parameters (N, handedness, nu, ...) are part of equality.
        assert data_imported == data_expected
    elif isinstance(
        data_expected,
        (
            lasyten.ElementarySpace,
            lasyten.AbelianLegPipe,
            lasyten.DirectSumSpace,
            lasyten.LegPipe,
            lasyten.TensorProduct,
        ),
    ):
        assert data_imported == data_expected
    elif isinstance(data_expected, lasyten.tensors.Tensor):
        assert type(data_imported) is type(data_expected)
        assert data_imported.dtype == data_expected.dtype
        assert data_imported.labels == data_expected.labels
        assert lasyten.almost_equal(data_imported, data_expected)
    elif isinstance(data_expected, (types.FunctionType, type)):
        # global variables where no copy should be made
        assert data_imported is data_expected
    elif isinstance(data_expected, lasyten.tools.hdf5_io.Hdf5Exportable):
        assert type(data_imported) is type(data_expected)
        assert getattr(data_imported, 'some_attr', None) == getattr(data_expected, 'some_attr', None)


def get_datadir_filename(template='pickled_from_tenpy_{0}.pkl'):
    """Determine filename for export to `datadir`."""
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    version = _clean_version(lasyten.__version__)
    fn = template.format(version)
    filename = os.path.join(datadir, fn)
    if os.path.exists(filename):
        raise ValueError('File already exists: ' + filename)
    print('export to datadir: ', fn)
    return filename
