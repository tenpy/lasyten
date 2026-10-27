"""Tests for reading F / R symbol files in the packed layout (``Layout = "packed_v1"``).

The packed fixtures are made from the weight-6 SU(2) files of the ``external/SUN_symbols`` submodule
with a small converter that follows the format of the SU(N) data generator.
"""

# Copyright (C) TeNPy Developers, Apache license
import gc
import os
import re
from pathlib import Path

import h5py
import numpy as np
import numpy.testing as npt
import pytest

from lasyten import _core, cache, symmetries, temporary_options
from lasyten.symmetries import SymmetryError

_SUN_SYMBOLS_SUBMODULE_DIR = Path(__file__).resolve().parents[2] / 'external' / 'SUN_symbols'

# F[2, 0][4, 0][6, 0][4, 0][4, 0][4, 0]: allowed channels, but the F symbol vanishes. Files from the
# old generator (no 'Gauge_Version') omit such blocks.
_ZERO_F_KEY = 'F[2, 0][4, 0][6, 0][4, 0][4, 0][4, 0]'


def _sec(w: int) -> np.ndarray:
    return np.array([w, 0])


_ZERO_F = tuple(_sec(w) for w in (2, 4, 6, 4, 4, 4))


@pytest.fixture(scope='module')
def su2_dir():
    """Directory with the SU(2) data files (CG 20, F 6, R 6): the configured path (``None``) or the git submodule."""
    for directory in (None, str(_SUN_SYMBOLS_SUBMODULE_DIR)):
        files = [
            symmetries.su_n_data_file_path(2, kind, h, path=directory) for kind, h in (('CG', 20), ('F', 6), ('R', 6))
        ]
        if all(os.path.exists(f) for f in files):
            return directory
    pytest.skip('Need the SU(2) data files (CG 20, F 6, R 6), e.g. the external/SUN_symbols submodule.')


def _parse_key(name: str) -> list[list[int]]:
    return [[int(x) for x in group.split(',')] for group in re.findall(r'\[([^\]]*)\]', name)]


def _to_packed(src: str, dst: str, kind: str, drop=(), layout: str = 'packed_v1', sort: bool = True) -> None:
    """Convert a folder-layout F or R file to the packed layout."""
    with h5py.File(src, 'r') as f:
        group = f[f'{kind}_sym']
        names = [n for n in group.keys() if n not in drop]
        labels = np.array([_parse_key(n) for n in names], dtype=np.int32)
        order = np.lexsort(labels.reshape(len(names), -1).T[::-1])
        if not sort:
            order = order[::-1]
        blocks = [np.asarray(group[names[i]][()], dtype=np.complex128) for i in order]
        offsets = np.concatenate([[0], np.cumsum([b.size for b in blocks])]).astype(np.int64)
        with h5py.File(dst, 'w') as g:
            for key, value in f.attrs.items():
                g.attrs[key] = value
            packed = g.create_group(f'{kind}_packed')
            packed.create_dataset('labels', data=labels[order])
            packed.create_dataset('shapes', data=np.array([list(b.shape) for b in blocks], dtype=np.int64))
            packed.create_dataset('offsets', data=offsets)
            packed.create_dataset('data', data=np.concatenate([b.ravel() for b in blocks]))
            g.attrs['Layout'] = layout


def _make_dir(su2_dir, dst: Path, f: str = 'folders', r: str = 'folders', **convert_kwargs) -> str:
    """A data directory with symlinked CG file and F / R files in the given layouts ('folders' or 'packed')."""
    dst.mkdir()
    for kind, layout in (('CG', 'folders'), ('F', f), ('R', r)):
        src = symmetries.su_n_data_file_path(2, kind, 20 if kind == 'CG' else 6, path=su2_dir)
        target = dst / os.path.basename(src)
        if layout == 'packed':
            _to_packed(src, str(target), kind, **convert_kwargs.get(kind, {}))
        else:
            os.symlink(src, target)
    return str(dst)


def _sun(path: str) -> symmetries.SUN:
    return symmetries.SUN(2, 20, f_hweight=6, r_hweight=6, path=path)


def _allowed_f_sextuples(sym, wmax: int):
    secs = [_sec(w) for w in range(wmax + 1)]
    for a in secs:
        for b in secs:
            fs = [f for f in secs if sym.can_fuse_to(a, b, f)]
            for c in secs:
                for e in (e for e in secs if sym.can_fuse_to(b, c, e)):
                    for f in fs:
                        for d in secs:
                            if sym.can_fuse_to(f, c, d) and sym.can_fuse_to(a, e, d):
                                yield a, b, c, d, e, f


def _allowed_r_triples(sym, wmax: int):
    secs = [_sec(w) for w in range(wmax + 1)]
    for a in secs:
        for b in secs:
            for c in secs:
                if sym.can_fuse_to(a, b, c):
                    yield a, b, c


def _assert_same_symbols(reference, other, wmax: int = 6) -> None:
    """Every F and R symbol up to weight `wmax` is identical (shape, dtype and all entries)."""
    sextuples = list(_allowed_f_sextuples(reference, wmax))
    triples = list(_allowed_r_triples(reference, wmax))
    n_f = 0
    with temporary_options(su_n_symbol_cache_mb=0):  # every call reads the file
        for s in sextuples:
            expect, actual = reference.f_symbol(*s), other.f_symbol(*s)
            assert actual.shape == expect.shape
            assert actual.dtype == expect.dtype
            npt.assert_array_equal(actual, expect)
            n_f += 1
        n_r = 0
        for s in triples:
            expect, actual = reference.r_symbol(*s), other.r_symbol(*s)
            assert actual.shape == expect.shape
            npt.assert_array_equal(actual, expect)
            n_r += 1
    assert n_f > 1000
    assert n_r > 10


def test_packed_equals_folders(su2_dir, tmp_path):
    folders = _sun(_make_dir(su2_dir, tmp_path / 'folders'))
    packed = _sun(_make_dir(su2_dir, tmp_path / 'packed', 'packed', 'packed'))
    _assert_same_symbols(folders, packed)


@pytest.mark.parametrize('f, r', [('packed', 'folders'), ('folders', 'packed')])
def test_mixed_layouts(su2_dir, tmp_path, f, r):
    folders = _sun(_make_dir(su2_dir, tmp_path / 'folders'))
    mixed = _sun(_make_dir(su2_dir, tmp_path / 'mixed', f, r))
    _assert_same_symbols(folders, mixed)


def test_data_read_per_lookup_equals_data_in_ram(su2_dir, tmp_path):
    """Blocks larger than the RAM limit are read one hyperslab at a time; the result does not change."""
    folders = _sun(_make_dir(su2_dir, tmp_path / 'folders'))
    d = _make_dir(su2_dir, tmp_path / 'packed', 'packed', 'packed')
    old = _core._set_packed_data_in_ram_max_bytes(0)
    try:
        before = cache.cache_info()['packed_files']
        packed = _sun(d)
        after = cache.cache_info()['packed_files']
        assert after['index_bytes'] > before['index_bytes']
        assert after['data_bytes_in_ram'] == before['data_bytes_in_ram']  # nothing was loaded
        _assert_same_symbols(folders, packed)
    finally:
        _core._set_packed_data_in_ram_max_bytes(old)


def test_legacy_zero_block_is_zero_in_packed_files(su2_dir, tmp_path):
    d = _make_dir(su2_dir, tmp_path / 'packed', 'packed', 'packed')
    with h5py.File(symmetries.su_n_data_file_path(2, 'F', 6, path=d), 'r') as f:
        if 'Gauge_Version' in f.attrs:
            pytest.skip('this data set is not a legacy file')
        assert 'F_packed' in f
    res = _sun(d).f_symbol(*_ZERO_F)
    assert res.shape == (1, 1, 1, 1)
    npt.assert_array_equal(res, 0)


@pytest.mark.parametrize('kind', ['F', 'R'])
def test_missing_key_of_new_gauge_file_is_an_error(su2_dir, tmp_path, kind):
    """With a 'Gauge_Version' attribute every allowed key must be present, in packed files as well."""
    drop = {'F': {'F': {'drop': [_ZERO_F_KEY]}}, 'R': {'R': {'drop': ['R[1, 0][1, 0][0, 0]']}}}[kind]
    d = _make_dir(su2_dir, tmp_path / 'data', 'packed', 'packed', **drop)
    with h5py.File(symmetries.su_n_data_file_path(2, kind, 6, path=d), 'r+') as f:
        f.attrs['Gauge_Version'] = 'diag_r_swap_v1'
    sym = _sun(d)
    with pytest.raises(RuntimeError, match='incomplete'):
        if kind == 'F':
            sym.f_symbol(*_ZERO_F)
        else:
            sym.r_symbol(_sec(1), _sec(1), _sec(0))


def test_not_allowed_channels_and_weights(su2_dir, tmp_path):
    sym = _sun(_make_dir(su2_dir, tmp_path / 'packed', 'packed', 'packed'))
    with temporary_options(check_fusion=False):
        with pytest.raises(SymmetryError):
            sym.f_symbol(_sec(1), _sec(1), _sec(1), _sec(1), _sec(1), _sec(1))
        with pytest.raises(SymmetryError):
            sym.r_symbol(_sec(1), _sec(1), _sec(1))
    with pytest.raises(ValueError, match='higher weight'):
        sym.f_symbol(_sec(7), _sec(0), _sec(0), _sec(7), _sec(0), _sec(7))


def test_unsupported_layout_is_rejected(su2_dir, tmp_path):
    d = _make_dir(su2_dir, tmp_path / 'data', 'packed', 'folders', F={'layout': 'packed_v2'})
    with pytest.raises(ValueError, match='unsupported Layout'):
        _sun(d)


def test_unsorted_labels_are_rejected(su2_dir, tmp_path):
    d = _make_dir(su2_dir, tmp_path / 'data', 'packed', 'folders', F={'sort': False})
    with pytest.raises(RuntimeError, match='strictly'):
        _sun(d)


def test_inconsistent_offsets_are_rejected(su2_dir, tmp_path):
    d = _make_dir(su2_dir, tmp_path / 'data', 'packed', 'folders')
    with h5py.File(symmetries.su_n_data_file_path(2, 'F', 6, path=d), 'r+') as f:
        f['F_packed/offsets'][-1] += 1
    with pytest.raises(RuntimeError, match='offsets'):
        _sun(d)


def test_wrong_group_is_rejected(su2_dir, tmp_path):
    """An R file used as the F file has no 'F_packed' group."""
    d = _make_dir(su2_dir, tmp_path / 'data', 'folders', 'packed')
    f_path = symmetries.su_n_data_file_path(2, 'F', 6, path=d)
    os.remove(f_path)
    os.symlink(symmetries.su_n_data_file_path(2, 'R', 6, path=d), f_path)
    with pytest.raises(ValueError, match=r"no group '/F_packed/'"):
        _sun(d)


def test_sanity_check_of_packed_files(su2_dir, tmp_path):
    d = _make_dir(su2_dir, tmp_path / 'good', 'packed', 'packed')
    sym = _sun(d)
    sym.sanity_check_hdf5(symmetries.su_n_data_file_path(2, 'F', 6, path=d))
    sym.sanity_check_hdf5(symmetries.su_n_data_file_path(2, 'R', 6, path=d))
    # without the all-trivial key the file is rejected when it is opened and by the sanity check
    bad = _make_dir(
        su2_dir, tmp_path / 'bad', 'packed', 'folders', F={'drop': ['F[0, 0][0, 0][0, 0][0, 0][0, 0][0, 0]']}
    )
    with pytest.raises(ValueError, match='all-trivial'):
        _sun(bad)
    with pytest.raises(ValueError, match='all-trivial'):
        sym.sanity_check_hdf5(symmetries.su_n_data_file_path(2, 'F', 6, path=bad))


def test_packed_memory_is_reported(su2_dir, tmp_path):
    d = _make_dir(su2_dir, tmp_path / 'packed', 'packed', 'packed')
    gc.collect()
    before = cache.cache_info()['packed_files']
    sym = _sun(d)
    during = cache.cache_info()['packed_files']
    assert during['files'] == before['files'] + 2
    assert during['index_bytes'] > before['index_bytes']
    assert during['data_bytes_in_ram'] > before['data_bytes_in_ram']
    del sym
    gc.collect()
    assert cache.cache_info()['packed_files'] == before
