"""Tests for the RAM caches (SU(N) symbols, tree plans, tree-size memo) and for how the SUN symmetry
treats missing F / R keys, and for the data-weight auto-detection of the production SU(2) symmetry."""

# Copyright (C) TeNPy Developers, Apache license
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import numpy.testing as npt
import pytest

from lasyten import cache, symmetries, temporary_options, tensors
from lasyten.symmetries import SymmetryError, trees

_SUN_SYMBOLS_SUBMODULE_DIR = Path(__file__).resolve().parents[2] / 'external' / 'SUN_symbols'


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


@pytest.fixture
def su2(su2_dir) -> symmetries.SUN:
    cache.clear_caches()
    return symmetries.SUN(2, 20, f_hweight=6, r_hweight=6, path=su2_dir)


def _sec(w: int) -> np.ndarray:
    return np.array([w, 0])


def _allowed_f_sextuples(sym, wmax: int):
    """All (a, b, c, d, e, f) up to weight `wmax` with all four fusion channels allowed."""
    secs = [_sec(w) for w in range(wmax + 1)]
    for a in secs:
        for b in secs:
            fs = [f for f in secs if sym.can_fuse_to(a, b, f)]
            for c in secs:
                es = [e for e in secs if sym.can_fuse_to(b, c, e)]
                for e in es:
                    for f in fs:
                        for d in secs:
                            if sym.can_fuse_to(f, c, d) and sym.can_fuse_to(a, e, d):
                                yield a, b, c, d, e, f


def test_cached_equals_uncached(su2, np_random):
    sextuples = list(_allowed_f_sextuples(su2, 4))
    picks = [sextuples[i] for i in np_random.choice(len(sextuples), size=min(150, len(sextuples)), replace=False)]
    with temporary_options(su_n_symbol_cache_mb=0):
        ref_f = [su2.f_symbol(*s) for s in picks]
        ref_r = [su2.r_symbol(s[0], s[1], s[5]) for s in picks]
        ref_n = [su2.n_symbol(s[0], s[1], s[5]) for s in picks]
        ref_out = [su2.fusion_outcomes(s[0], s[1]) for s in picks]
    cache.clear_caches()
    for _ in range(2):  # second round is served from the cache
        for s, f, r, n, out in zip(picks, ref_f, ref_r, ref_n, ref_out):
            npt.assert_array_equal(su2.f_symbol(*s), f)
            npt.assert_array_equal(su2.r_symbol(s[0], s[1], s[5]), r)
            assert su2.n_symbol(s[0], s[1], s[5]) == n
            npt.assert_array_equal(su2.fusion_outcomes(s[0], s[1]), out)
    info = cache.cache_info()['symbols']
    assert info['hits'] > 0
    assert info['entries'] > 0
    assert 0 < info['bytes'] <= info['capacity_bytes']


def test_disabled_cache_stores_nothing(su2):
    sextuples = list(_allowed_f_sextuples(su2, 2))
    with temporary_options(su_n_symbol_cache_mb=0):
        for s in sextuples[:20]:
            su2.f_symbol(*s)
        info = cache.cache_info()['symbols']
    assert info['entries'] == 0
    assert info['bytes'] == 0
    assert info['capacity_bytes'] == 0
    assert info['rejected'] > 0


def test_lru_respects_byte_cap(su2):
    """Reading every F symbol of the weight-6 file does not fit into 1 MB: least recently used entries go."""
    sextuples = list(_allowed_f_sextuples(su2, 6))
    with temporary_options(su_n_symbol_cache_mb=1):
        for s in sextuples:
            su2.f_symbol(*s)
            info = cache.cache_info()['symbols']
            assert info['bytes'] <= info['capacity_bytes'] == 2**20
        assert cache.cache_info()['symbols']['evictions'] > 0
        # without the fusion-rule check a call is a single lookup of the F symbol: the symbol used
        # last is still cached (a hit), the one used first is gone (a miss)
        with temporary_options(check_fusion=False):
            before = cache.cache_info()['symbols']
            su2.f_symbol(*sextuples[-1])
            after = cache.cache_info()['symbols']
            assert (after['hits'], after['misses']) == (before['hits'] + 1, before['misses'])
            su2.f_symbol(*sextuples[0])
            assert cache.cache_info()['symbols']['misses'] == after['misses'] + 1


def test_lowering_the_cap_evicts(su2):
    for s in _allowed_f_sextuples(su2, 6):
        su2.f_symbol(*s)
    big = cache.cache_info()['symbols']
    assert big['bytes'] > 2**20  # more than the new cap
    with temporary_options(su_n_symbol_cache_mb=1):
        small = cache.cache_info()['symbols']  # reading the info applies the new cap
        assert small['bytes'] <= 2**20
        assert small['entries'] < big['entries']
        assert small['evictions'] > big['evictions']


def test_clear_caches(su2):
    su2.f_symbol(*next(_allowed_f_sextuples(su2, 2)))
    assert cache.cache_info()['symbols']['entries'] > 0
    cache.clear_caches()
    assert cache.cache_info()['symbols']['entries'] == 0
    assert cache.cache_info()['tree_plans']['entries'] == 0


# --------------------------------------------------------------------------------------------
# missing F / R keys


def _copy_su2_dir(su2_dir, dst: Path) -> Path:
    dst.mkdir()
    for kind, h in (('CG', 20), ('F', 6), ('R', 6)):
        src = symmetries.su_n_data_file_path(2, kind, h, path=su2_dir)
        target = dst / os.path.basename(src)
        if kind == 'CG':
            os.symlink(src, target)  # the CG file is large and never modified
        else:
            shutil.copy(src, target)
    return dst


# F[2, 0][4, 0][6, 0][4, 0][4, 0][4, 0]: allowed channels, but the F symbol vanishes. Files from the
# old generator (no 'Gauge_Version') omit such blocks; the current generator writes them.
_ZERO_F = (_sec(2), _sec(4), _sec(6), _sec(4), _sec(4), _sec(4))


def test_legacy_file_omitted_zero_block_is_zero(su2_dir):
    sym = symmetries.SUN(2, 20, f_hweight=6, r_hweight=6, path=su2_dir)
    with h5py.File(symmetries.su_n_data_file_path(2, 'F', 6, path=su2_dir), 'r') as f:
        if 'Gauge_Version' in f.attrs:
            pytest.skip('this data set is not a legacy file')
        assert 'F[2, 0][4, 0][6, 0][4, 0][4, 0][4, 0]' not in f['F_sym']
    res = sym.f_symbol(*_ZERO_F)
    assert res.shape == (1, 1, 1, 1)
    npt.assert_array_equal(res, 0)
    # a stored symbol is still read from the file
    assert np.any(sym.f_symbol(_sec(1), _sec(1), _sec(1), _sec(1), _sec(0), _sec(0)) != 0)


def test_new_gauge_file_missing_key_is_an_error(su2_dir, tmp_path):
    d = _copy_su2_dir(su2_dir, tmp_path / 'data')
    with h5py.File(symmetries.su_n_data_file_path(2, 'F', 6, path=str(d)), 'r+') as f:
        f.attrs['Gauge_Version'] = 'diag_r_swap_v1'
    sym = symmetries.SUN(2, 20, f_hweight=6, r_hweight=6, path=str(d))
    with pytest.raises(RuntimeError, match='incomplete'):
        sym.f_symbol(*_ZERO_F)


def test_missing_r_key_is_an_error(su2_dir, tmp_path):
    d = _copy_su2_dir(su2_dir, tmp_path / 'data')
    with h5py.File(symmetries.su_n_data_file_path(2, 'R', 6, path=str(d)), 'r+') as f:
        del f['R_sym']['R[1, 0][1, 0][0, 0]']
    sym = symmetries.SUN(2, 20, f_hweight=6, r_hweight=6, path=str(d))
    with pytest.raises(RuntimeError, match='incomplete'):
        sym.r_symbol(_sec(1), _sec(1), _sec(0))


def test_not_allowed_channels_raise(su2):
    # spin 1/2 x spin 1/2 -> spin 1/2 is not allowed
    with temporary_options(check_fusion=False):
        with pytest.raises(SymmetryError):
            su2.f_symbol(_sec(1), _sec(1), _sec(1), _sec(1), _sec(1), _sec(1))
        with pytest.raises(SymmetryError):
            su2.r_symbol(_sec(1), _sec(1), _sec(1))
    with pytest.raises(SymmetryError):  # with the check enabled, the fusion rules are checked first
        su2.f_symbol(_sec(1), _sec(1), _sec(1), _sec(1), _sec(1), _sec(1))


@pytest.mark.parametrize('N, cg_h, f_h, r_h', [(2, 20, 6, 6), (3, 7, 4, 4)])
def test_sector_dim_matches_data_file(N, cg_h, f_h, r_h):
    for directory in (None, str(_SUN_SYMBOLS_SUBMODULE_DIR)):
        paths = [
            symmetries.su_n_data_file_path(N, k, h, path=directory) for k, h in (('CG', cg_h), ('F', f_h), ('R', r_h))
        ]
        if all(os.path.exists(p) for p in paths):
            break
    else:
        pytest.skip('Need SU(N) data files')
    sym = symmetries.SUN(N, cg_h, f_hweight=f_h, r_hweight=r_h, path=directory)
    with h5py.File(paths[0], 'r') as f:
        found = 0
        group = f[f'N_{N}']

        def visit(name, obj):
            nonlocal found
            if isinstance(obj, h5py.Dataset) and 'Irreplabel' in obj.attrs and 'Dimension' in obj.attrs:
                label = np.array(obj.attrs['Irreplabel'])
                if label[0] <= cg_h:
                    assert sym.sector_dim(label) == int(obj.attrs['Dimension'])
                    found += 1

        group.visititems(visit)
        assert found > 0


# --------------------------------------------------------------------------------------------
# SU(2) data weights of the production SU(2) symmetry


def test_su2_weights_are_autodetected(su2_dir, tmp_path):
    from lasyten.models.sites import SpinSite

    d = tmp_path / 'data'
    d.mkdir()
    for kind, h in (('CG', 20), ('F', 6), ('R', 6)):
        src = symmetries.su_n_data_file_path(2, kind, h, path=su2_dir)
        os.symlink(src, d / os.path.basename(src))
    # lower-weight decoy files (never opened, since the larger weight wins)
    os.symlink(
        symmetries.su_n_data_file_path(2, 'F', 6, path=su2_dir), d / 'su_n_clebsch_gordan_data_N2_F_hweight5.hdf5'
    )
    os.symlink(
        symmetries.su_n_data_file_path(2, 'R', 6, path=su2_dir), d / 'su_n_clebsch_gordan_data_N2_R_hweight5.hdf5'
    )

    with temporary_options(su_n_data_path=str(d)):
        factor = SpinSite(S=0.5, conserve='SU2').symmetry.factors[0]
        assert (factor.hweight_from_CG_hdf5(), factor.hweight_from_F_hdf5(), factor.hweight_from_R_hdf5()) == (20, 6, 6)
    with temporary_options(su_n_data_path=str(d), su2_f_hweight=6, su2_r_hweight=6, su2_cg_hweight=20):
        factor = SpinSite(S=0.5, conserve='SU2').symmetry.factors[0]
        assert factor.hweight_from_F_hdf5() == 6
    with temporary_options(su_n_data_path=str(d), su2_f_hweight=7):
        with pytest.raises(FileNotFoundError, match='hweight7'):
            SpinSite(S=0.5, conserve='SU2')


# --------------------------------------------------------------------------------------------
# tree caches


def test_tree_size_memo_matches_enumeration(any_symmetry, make_any_sectors):
    cache.clear_caches()
    sectors = make_any_sectors(4)
    if len(sectors) < 3:
        pytest.skip('symmetry has too few sectors')
    for coupled in sectors:
        for n in (3, min(4, len(sectors))):
            ft = trees.fusion_trees(any_symmetry, sectors[:n], coupled)
            assert len(ft) == len(list(ft))
            assert len(ft) == len(list(ft))  # second time from the memo


def test_plan_cache_does_not_change_results(make_compatible_tensor, compatible_symmetry, compatible_symmetry_backend):
    if compatible_symmetry_backend != 'fusion_tree':
        pytest.skip('the plan cache belongs to the fusion tree backend')
    T = make_compatible_tensor(2, 2, max_block_size=3)
    args = ([0, 3], [1, 2], [0, 1, 2, 3])
    cache.clear_caches()
    with temporary_options(tree_plan_cache_mb=0):
        ref = tensors.permute_legs(T, *args, bend_right=True)
        assert cache.cache_info()['tree_plans']['entries'] == 0
    cache.clear_caches()
    first = tensors.permute_legs(T, *args, bend_right=True)
    info1 = cache.cache_info()['tree_plans']
    second = tensors.permute_legs(T, *args, bend_right=True)
    info2 = cache.cache_info()['tree_plans']
    assert tensors.almost_equal(first, ref)
    assert tensors.almost_equal(second, ref)
    if info1['entries'] > 0:
        assert info2['hits'] > info1['hits']  # the second call reuses the cached mapping
        assert info2['bytes'] <= info2['capacity_bytes']


def test_plan_cache_byte_cap(make_compatible_tensor, compatible_symmetry, compatible_symmetry_backend):
    if compatible_symmetry_backend != 'fusion_tree':
        pytest.skip('the plan cache belongs to the fusion tree backend')
    T = make_compatible_tensor(2, 2, max_block_size=3)
    cache.clear_caches()
    # a budget of 1 MB is respected however many different permutations are applied
    with temporary_options(tree_plan_cache_mb=1):
        for codomain, domain in [
            ([0, 3], [1, 2]),
            ([0, 1], [3, 2]),
            ([2, 3], [0, 1]),
            ([1, 0], [2, 3]),
            ([3, 2], [1, 0]),
        ]:
            tensors.permute_legs(T, codomain, domain, [0, 1, 2, 3], bend_right=True)
            info = cache.cache_info()['tree_plans']
            assert info['bytes'] <= info['capacity_bytes'] == 2**20


# --------------------------------------------------------------------------------------------
# constructor checks


@pytest.mark.parametrize('kind, key', [('F', 'F[0, 0][0, 0][0, 0][0, 0][0, 0][0, 0]'), ('R', 'R[0, 0][0, 0][0, 0]')])
def test_incomplete_file_is_rejected_when_opening(su2_dir, tmp_path, kind, key):
    """The check of a symbol file (all-trivial key present) passes for the real data and fails without it."""
    d = _copy_su2_dir(su2_dir, tmp_path / 'data')
    symmetries.SUN(2, 20, f_hweight=6, r_hweight=6, path=str(d))  # complete files are accepted
    with h5py.File(symmetries.su_n_data_file_path(2, kind, 6, path=str(d)), 'r+') as f:
        del f[f'{kind}_sym'][key]
    with pytest.raises(ValueError, match='all-trivial'):
        symmetries.SUN(2, 20, f_hweight=6, r_hweight=6, path=str(d))
