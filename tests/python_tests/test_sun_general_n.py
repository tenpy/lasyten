"""Tests of the SU(N) symbol handling for N > 2 and of the automatic detection of the data weights.

The SU(3) / SU(4) data sets are too large for the repository. They are generated with the
``clebsch_gordan_coefficients`` package (``generate_symmetry_data.py -N 3 --hweight 4`` etc.) and found via the
environment variable ``LASYTEN_TEST_SUN_DIR``, a directory with the sub-directories

* ``N3_H4``, ``N3_H4_packed`` (SU(3), weight 4; folder and packed layout of the F / R files),
* ``N3_H5`` (SU(3), only its CG file is used) and
* ``N4_H3``, ``N4_H3_packed`` (SU(4), weight 3).

Tests that need data which is not available are skipped.
"""

# Copyright (C) TeNPy Developers, Apache license
import gc
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import numpy.testing as npt
import pytest

from lasyten import cache, symmetries, temporary_options
from lasyten.symmetries import SymmetryError

_SUN_SYMBOLS_SUBMODULE_DIR = Path(__file__).resolve().parents[2] / 'external' / 'SUN_symbols'
_CASES = {3: ('N3_H4', 4), 4: ('N4_H3', 3)}  # N -> (directory, highest weight)


def _base_dir() -> Path:
    base = os.environ.get('LASYTEN_TEST_SUN_DIR')
    if not base or not Path(base).is_dir():
        pytest.skip(
            'Set LASYTEN_TEST_SUN_DIR to a directory with generated SU(3) / SU(4) data (see the module docstring).'
        )
    return Path(base)


def _data_dir(name: str) -> str:
    path = _base_dir() / name
    if not path.is_dir():
        pytest.skip(f'LASYTEN_TEST_SUN_DIR has no sub-directory {name}.')
    return str(path)


@pytest.fixture(params=[3, 4])
def N(request) -> int:
    return request.param


@pytest.fixture
def folder_dir(N) -> str:
    return _data_dir(_CASES[N][0])


@pytest.fixture
def packed_dir(N) -> str:
    return _data_dir(_CASES[N][0] + '_packed')


def _sun(N: int, path: str) -> symmetries.SUN:
    h = _CASES[N][1]
    return symmetries.SUN(N, h, path=path)


def _f_path(N: int, path: str) -> str:
    return symmetries.su_n_data_file_path(N, 'F', _CASES[N][1], path=path)


def _r_path(N: int, path: str) -> str:
    return symmetries.su_n_data_file_path(N, 'R', _CASES[N][1], path=path)


def _sectors(row) -> list[np.ndarray]:
    return [np.array([int(x) for x in s]) for s in row]


def _sample_f_keys(N: int, path: str, n: int = 2000) -> np.ndarray:
    with h5py.File(_f_path(N, path), 'r') as f:
        labels = f['F_packed/labels'] if 'F_packed' in f else None
        if labels is None:
            names = list(f['F_sym'].keys())
            idx = np.random.default_rng(1).choice(len(names), min(n, len(names)), replace=False)
            return np.array(
                [
                    [[int(x) for x in grp.split(',')] for grp in name[2:-1].split('][')]
                    for name in (names[i] for i in idx)
                ]
            )
        idx = np.sort(np.random.default_rng(1).choice(labels.shape[0], min(n, labels.shape[0]), replace=False))
        return labels[idx.tolist()]


# --------------------------------------------------------------------------------------------
# automatic data weights


def test_auto_weights_su2_submodule():
    files = [
        symmetries.su_n_data_file_path(2, k, h, path=str(_SUN_SYMBOLS_SUBMODULE_DIR))
        for k, h in (('CG', 20), ('F', 6), ('R', 6))
    ]
    if not all(os.path.exists(f) for f in files):
        pytest.skip('Need the SU(2) data files of the external/SUN_symbols submodule.')
    sym = symmetries.SUN(2, path=str(_SUN_SYMBOLS_SUBMODULE_DIR))
    assert (sym.hweight_from_CG_hdf5(), sym.hweight_from_F_hdf5(), sym.hweight_from_R_hdf5()) == (20, 6, 6)


def test_auto_weights_general_n(N, folder_dir):
    sym = symmetries.SUN(N, path=folder_dir)
    h = _CASES[N][1]
    assert (sym.hweight_from_CG_hdf5(), sym.hweight_from_F_hdf5(), sym.hweight_from_R_hdf5()) == (h, h, h)


def test_auto_weights_pinned_by_option(tmp_path):
    """Two complete data sets (weights 4 and 5, different generator runs) in one directory."""
    d = tmp_path / 'data'
    d.mkdir()
    base = _base_dir()
    for src, h in (('N3_H4', 4), ('N3_H5', 5)):
        for kind in ('CG', 'F', 'R'):
            name = os.path.basename(symmetries.su_n_data_file_path(3, kind, h, path='.'))
            if not (base / src / name).exists():
                pytest.skip(f'missing {src}/{name}')
            os.symlink(base / src / name, d / name)

    def weights(sym):
        return (sym.hweight_from_CG_hdf5(), sym.hweight_from_F_hdf5(), sym.hweight_from_R_hdf5())

    assert weights(symmetries.SUN(3, path=str(d))) == (5, 5, 5)  # the largest data set
    with temporary_options(su_n_hweights='3:4'):  # f and r default to cg
        assert weights(symmetries.SUN(3, path=str(d))) == (4, 4, 4)
    with temporary_options(su_n_hweights='3:5;4:2'):  # entries of other N are ignored
        assert weights(symmetries.SUN(3, path=str(d))) == (5, 5, 5)
    with temporary_options(su_n_hweights='3:9'):  # pinned to a file that does not exist
        with pytest.raises(FileNotFoundError, match='hweight9'):
            symmetries.SUN(3, path=str(d))
    with temporary_options(su_n_hweights='3:4'):  # explicit arguments win over the option
        assert weights(symmetries.SUN(3, 5, path=str(d))) == (5, 5, 5)


def test_auto_weights_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match=r'No SU\(3\) CG data file found'):
        symmetries.SUN(3, path=str(tmp_path))


# --------------------------------------------------------------------------------------------
# the packed layout and the symbol cache for N > 2


def test_packed_equals_folders(N, folder_dir, packed_dir):
    folders, packed = _sun(N, folder_dir), _sun(N, packed_dir)
    keys = _sample_f_keys(N, packed_dir)
    n_multi = 0
    with temporary_options(su_n_symbol_cache_mb=0, check_fusion=False):  # every call reads the file
        for row in keys:
            expect, actual = folders.f_symbol(*_sectors(row)), packed.f_symbol(*_sectors(row))
            assert actual.shape == expect.shape
            npt.assert_array_equal(actual, expect)
            n_multi += expect.size > 1
        with h5py.File(_r_path(N, packed_dir), 'r') as f:
            r_keys = f['R_packed/labels'][()]
        for row in r_keys:
            expect, actual = folders.r_symbol(*_sectors(row)), packed.r_symbol(*_sectors(row))
            npt.assert_array_equal(actual, expect)
    assert n_multi > 10  # the sample contains blocks with outer multiplicities


def test_cache_does_not_change_symbols(N, folder_dir):
    sym = _sun(N, folder_dir)
    keys = _sample_f_keys(N, folder_dir, n=500)
    with temporary_options(su_n_symbol_cache_mb=0, check_fusion=False):
        reference = [sym.f_symbol(*_sectors(row)) for row in keys]
    cache.clear_caches()
    with temporary_options(su_n_symbol_cache_mb=256, check_fusion=False):
        for _ in range(2):  # the second round is served from the cache
            for row, expect in zip(keys, reference):
                npt.assert_array_equal(sym.f_symbol(*_sectors(row)), expect)
        assert cache.cache_info()['symbols']['hits'] > 0


def _copy_n3(tmp_path: Path, drop_gauge_version: bool) -> Path:
    """Copy of the N3_H4 set; with ``drop_gauge_version`` a fully legacy set (no gauge record in any file)."""
    src = Path(_data_dir('N3_H4'))
    dst = tmp_path / 'data'
    dst.mkdir()
    for kind in ('CG', 'F', 'R'):
        name = os.path.basename(symmetries.su_n_data_file_path(3, kind, 4, path='.'))
        if kind == 'CG' and not drop_gauge_version:
            os.symlink(src / name, dst / name)
        else:
            shutil.copy(src / name, dst / name)
    if drop_gauge_version:
        for kind in ('CG', 'F', 'R'):
            with h5py.File(dst / os.path.basename(symmetries.su_n_data_file_path(3, kind, 4, path='.')), 'r+') as f:
                del f.attrs['Gauge_Version']
                if kind == 'CG':
                    del f.attrs['CG_Run_Id']
    return dst


def _drop_f_key(dst: Path, shape: tuple) -> list[np.ndarray]:
    """Remove one F key with the given block shape (not the all-trivial key, which every file must have)."""
    with h5py.File(dst / os.path.basename(symmetries.su_n_data_file_path(3, 'F', 4, path='.')), 'r+') as f:
        group = f['F_sym']
        for name in group.keys():
            sectors = [np.array([int(x) for x in grp.split(',')]) for grp in name[2:-1].split('][')]
            if group[name].shape == shape and any(sec.any() for sec in sectors):
                del group[name]
                return sectors
    raise AssertionError(f'no F key with block shape {shape}')


# the shape is (N(bce), N(aed), N(abf), N(fcd)); non-palindromic shapes catch a wrong axis order
@pytest.mark.parametrize(
    'shape',
    [(1, 1, 1, 1), (2, 1, 1, 2), (1, 1, 1, 2), (2, 1, 1, 1), (1, 2, 1, 1), (1, 1, 2, 1), (2, 2, 2, 3)],
)
def test_legacy_file_missing_f_key_is_a_zero_block_of_true_shape(tmp_path, shape):
    dst = _copy_n3(tmp_path, drop_gauge_version=True)
    sectors = _drop_f_key(dst, shape)
    res = symmetries.SUN(3, 4, path=str(dst)).f_symbol(*sectors)
    assert res.shape == shape
    npt.assert_array_equal(res, 0)


def test_new_gauge_file_missing_keys_are_errors(tmp_path):
    dst = _copy_n3(tmp_path, drop_gauge_version=False)
    sectors = _drop_f_key(dst, (2, 1, 1, 2))
    r_name = None
    with h5py.File(dst / os.path.basename(symmetries.su_n_data_file_path(3, 'R', 4, path='.')), 'r+') as f:
        r_name = [n for n in f['R_sym'].keys() if any(int(x) for x in n.replace('][', ',').strip('R[]').split(','))][0]
        r_sectors = [np.array([int(x) for x in grp.split(',')]) for grp in r_name[2:-1].split('][')]
        del f['R_sym'][r_name]
    sym = symmetries.SUN(3, 4, path=str(dst))
    with pytest.raises(RuntimeError, match='incomplete'):
        sym.f_symbol(*sectors)
    with pytest.raises(RuntimeError):
        sym.r_symbol(*r_sectors)


def test_not_allowed_channels_and_weights():
    sym = symmetries.SUN(3, 4, path=_data_dir('N3_H4'))
    fund, triv = np.array([1, 0, 0]), np.array([0, 0, 0])
    with temporary_options(check_fusion=False):
        with pytest.raises(SymmetryError):
            sym.f_symbol(fund, fund, fund, triv, triv, triv)
    with pytest.raises(ValueError, match='highest weight 4') as exc:
        sym.f_symbol(np.array([5, 0, 0]), triv, triv, np.array([5, 0, 0]), triv, np.array([5, 0, 0]))
    msg = str(exc.value)
    assert os.path.basename(sym.Fpath) in msg
    assert 'irrep of weight 5' in msg
    assert 'Recalculate' in msg and 'larger highest weight' in msg


def test_cg_run_id_mismatch_is_refused(tmp_path):
    dst = _copy_n3(tmp_path, drop_gauge_version=False)
    with h5py.File(dst / os.path.basename(symmetries.su_n_data_file_path(3, 'R', 4, path='.')), 'r+') as f:
        f.attrs['CG_Run_Id'] = 'someone-elses-run'
    with pytest.raises(ValueError, match='different generator runs'):
        symmetries.SUN(3, 4, path=str(dst))


@pytest.mark.parametrize(
    'N, label, dim',
    [
        (3, [0, 0, 0], 1),
        (3, [1, 0, 0], 3),
        (3, [1, 1, 0], 3),
        (3, [2, 0, 0], 6),
        (3, [2, 1, 0], 8),
        (3, [3, 0, 0], 10),
        (3, [2, 2, 0], 6),
        (4, [1, 0, 0, 0], 4),
        (4, [1, 1, 0, 0], 6),
        (4, [2, 0, 0, 0], 10),
        (4, [2, 1, 1, 0], 15),
        # from weight 4 on, a float computation of the Weyl dimension formula is off by one (34, 83, 69)
        (4, [4, 0, 0, 0], 35),
        (4, [4, 1, 0, 0], 84),
        (4, [4, 1, 1, 0], 70),
    ],
)
def test_sector_dim_is_the_exact_integer(N, label, dim):
    sym = symmetries.SUN(N, path=_data_dir(_CASES[N][0]))
    assert sym.sector_dim(np.array(label)) == dim


# --------------------------------------------------------------------------------------------
# the contraction sequence of a DMRG update does not depend on the caches


def _update_workload(sym, N: int, bond_weight: int, mult: int):
    """Random tensors with the sector structure of a two-site DMRG update, and the function applying it."""
    import lasyten as lasy

    fund, triv = np.array([1] + [0] * (N - 1)), np.zeros(N, dtype=int)
    adj = np.array([2] + [1] * (N - 2) + [0])
    sectors, frontier = {tuple(triv)}, [triv]
    while frontier:
        new = []
        for s in frontier:
            for o in np.atleast_2d(np.asarray(sym.fusion_outcomes(s, fund))):
                t = tuple(int(x) for x in o)
                if t[0] <= bond_weight and t not in sectors:
                    sectors.add(t)
                    new.append(np.array(t))
        frontier = new
    secs = [list(s) for s in sorted(sectors)]
    V = lasy.ElementarySpace.from_defining_sectors(sym, secs, [mult] * len(secs))
    P = lasy.ElementarySpace.from_defining_sectors(sym, [list(fund)], [1])
    B = lasy.ElementarySpace.from_defining_sectors(sym, [list(triv), list(adj)], [1, 1])
    rnd = lasy.SymmetricTensor.from_random_normal
    theta = rnd([V, P, P], [V], labels=['vL', 'p0', 'p1', 'vR'])
    LP, RP = rnd([V, B], [V], labels=['a', 'b', 'c']), rnd([V], [B, V], labels=['r_out', 'r_v', 'r_w'])
    W1 = rnd([B, P], [B, P], labels=['w1_out', 'w1_p_out', 'w1_p_in', 'w1_in'])
    W2 = rnd([B, P], [B, P], labels=['w2_out', 'w2_p_out', 'w2_p_in', 'w2_in'])

    def apply(th):
        t1 = lasy.tdot(LP, th, ['c'], ['vL'])
        t2 = lasy.tdot(W1, t1, ['w1_in', 'w1_p_in'], ['b', 'p0'])
        t3 = lasy.tdot(W2, t2, ['w2_in', 'w2_p_in'], ['w1_out', 'p1'])
        t4 = lasy.tdot(RP, t3, ['r_w', 'r_out'], ['w2_out', 'vR'])
        out = lasy.permute_legs(t4, codomain=['a', 'w1_p_out', 'w2_p_out'], domain=['r_v'])
        out.set_labels(['vL', 'p0', 'p1', 'vR'])
        return out

    return theta, apply


@pytest.mark.parametrize('N, bond_weight', [(2, 2), (3, 1), (4, 1)])
def test_dmrg_update_is_independent_of_the_caches(N, bond_weight):
    if N == 2:
        if not (_SUN_SYMBOLS_SUBMODULE_DIR / 'su_n_clebsch_gordan_data_N2_F_hweight6.hdf5').exists():
            pytest.skip('Need the external/SUN_symbols submodule.')
        sym = symmetries.SUN(2, path=str(_SUN_SYMBOLS_SUBMODULE_DIR))
    else:
        sym = symmetries.SUN(N, path=_data_dir(_CASES[N][0]))
    theta, apply = _update_workload(sym, N, bond_weight, mult=2)

    def run(symbol_mb, plan_mb):
        cache.clear_caches()
        before = cache.cache_info()
        with temporary_options(su_n_symbol_cache_mb=symbol_mb, tree_plan_cache_mb=plan_mb):
            cur, outs = theta, []
            for _ in range(3):
                cur = apply(cur)
                outs.append(np.asarray(cur.to_dense_block()))
            info = cache.cache_info()
        for name in ('symbols', 'tree_plans'):  # counters since the start of this run
            for counter in ('hits', 'rejected'):
                info[name][counter] -= before[name][counter]
        return outs, info

    cached, info = run(256, 256)
    assert info['symbols']['hits'] > 0 and info['tree_plans']['entries'] > 0
    uncached, info = run(0, 0)
    # reading the statistics trims a cache to its capacity, so `entries == 0` alone proves nothing:
    # the insertions must have been refused
    assert info['symbols']['entries'] == 0 and info['symbols']['rejected'] > 0
    assert info['tree_plans']['entries'] == 0 and info['tree_plans']['rejected'] > 0
    assert info['symbols']['hits'] == 0 and info['tree_plans']['hits'] == 0
    small, info = run(1, 1)
    for a, b, c in zip(cached, uncached, small):
        assert a.shape == b.shape == c.shape
        npt.assert_array_equal(a, b)  # bit-identical, not just close
        npt.assert_array_equal(a, c)


# --------------------------------------------------------------------------------------------
# audit fixes: finished files only, gauge / run-id consistency, shared state, plan cache ownership


def _weights(sym) -> tuple[int, int, int]:
    return (sym.hweight_from_CG_hdf5(), sym.hweight_from_F_hdf5(), sym.hweight_from_R_hdf5())


def _complete_n3_dir(tmp_path: Path) -> Path:
    """A directory with the complete N3_H4 set (symbolic links, so the files are never modified)."""
    d = tmp_path / 'data'
    d.mkdir()
    src = Path(_data_dir('N3_H4'))
    for kind in ('CG', 'F', 'R'):
        name = os.path.basename(symmetries.su_n_data_file_path(3, kind, 4, path='.'))
        os.symlink(src / name, d / name)
    return d


def test_auto_detection_ignores_unfinished_unreadable_and_mislabelled_files(tmp_path):
    """The generator writes Highest_Weight last: a file with a standard name that is not finished (or not
    readable, or named for another weight) must neither be picked nor break the default symmetry."""
    d = _complete_n3_dir(tmp_path)
    cg4 = d / os.path.basename(symmetries.su_n_data_file_path(3, 'CG', 4, path='.'))
    unfinished = d / os.path.basename(symmetries.su_n_data_file_path(3, 'CG', 9, path='.'))
    shutil.copy(cg4.resolve(), unfinished)
    with h5py.File(unfinished, 'r+') as f:
        del f.attrs['Highest_Weight']
    unreadable = d / os.path.basename(symmetries.su_n_data_file_path(3, 'CG', 8, path='.'))
    unreadable.write_bytes(b'not an hdf5 file ' * 100)
    mislabelled = d / os.path.basename(symmetries.su_n_data_file_path(3, 'CG', 7, path='.'))
    shutil.copy(cg4.resolve(), mislabelled)  # says Highest_Weight 4 but is named for weight 7

    with pytest.warns(UserWarning) as record:
        sym = symmetries.SUN(3, path=str(d))
    assert _weights(sym) == (4, 4, 4)
    messages = ' '.join(str(w.message) for w in record)
    assert 'CG_hweight9' in messages and "no attribute 'Highest_Weight'" in messages
    assert 'CG_hweight8' in messages and 'cannot be opened' in messages
    assert 'CG_hweight7' in messages and 'is 4, not 7' in messages


def test_missing_highest_weight_error_names_the_file(tmp_path):
    d = _complete_n3_dir(tmp_path)
    f_name = os.path.basename(symmetries.su_n_data_file_path(3, 'F', 4, path='.'))
    (d / f_name).unlink()
    shutil.copy(Path(_data_dir('N3_H4')) / f_name, d / f_name)
    with h5py.File(d / f_name, 'r+') as f:
        del f.attrs['Highest_Weight']
    with pytest.raises(RuntimeError, match=f_name):
        symmetries.SUN(3, 4, path=str(d))


def _set_attrs(dst: Path, kind: str, **attrs) -> None:
    name = os.path.basename(symmetries.su_n_data_file_path(3, kind, 4, path='.'))
    path = dst / name
    if path.is_symlink():  # never modify the shared data
        target = path.resolve()
        path.unlink()
        shutil.copy(target, path)
    with h5py.File(path, 'r+') as f:
        for key, value in attrs.items():
            if value is None:
                del f.attrs[key]
            else:
                f.attrs[key] = value


def test_unrecorded_run_id_is_not_a_mismatch(tmp_path):
    """The generator writes 'unknown' for what a legacy CG file did not record: that must not be refused."""
    dst = _copy_n3(tmp_path, drop_gauge_version=False)
    _set_attrs(dst, 'F', CG_Run_Id='unknown')
    _set_attrs(dst, 'R', CG_Run_Id='')
    assert _weights(symmetries.SUN(3, 4, path=str(dst))) == (4, 4, 4)


def test_gauge_mix_is_refused_and_names_the_files(tmp_path):
    dst = _copy_n3(tmp_path, drop_gauge_version=False)
    _set_attrs(dst, 'F', Gauge_Version=None)  # legacy F file next to a CG file that records its gauge
    with pytest.raises(ValueError, match='not in the same gauge') as exc:
        symmetries.SUN(3, 4, path=str(dst))
    assert 'F_hweight4' in str(exc.value) and 'CG_hweight4' in str(exc.value)
    assert 'not recorded' in str(exc.value)
    (tmp_path / 'other').mkdir()
    dst = _copy_n3(tmp_path / 'other', drop_gauge_version=False)
    _set_attrs(dst, 'R', Gauge_Version='some_other_gauge')
    with pytest.raises(ValueError, match="'some_other_gauge'"):
        symmetries.SUN(3, 4, path=str(dst))


def test_fully_legacy_set_is_accepted(tmp_path):
    dst = _copy_n3(tmp_path, drop_gauge_version=True)  # no file records a gauge or a run
    assert _weights(symmetries.SUN(3, 4, path=str(dst))) == (4, 4, 4)


def test_symbol_files_with_a_gauge_next_to_a_legacy_cg_file_are_accepted(tmp_path):
    """Only a CG file that records its gauge makes the F / R files' gauge matter."""
    dst = _copy_n3(tmp_path, drop_gauge_version=False)
    _set_attrs(dst, 'CG', Gauge_Version=None, CG_Run_Id=None)
    assert _weights(symmetries.SUN(3, 4, path=str(dst))) == (4, 4, 4)


def test_sun_objects_on_the_same_files_share_their_state(N, packed_dir):
    cache.clear_caches()
    gc.collect()
    base = cache.cache_info()['packed_files']['files']
    first = _sun(N, packed_dir)
    one = cache.cache_info()['packed_files']['files']
    second = _sun(N, packed_dir)
    two = cache.cache_info()['packed_files']['files']
    assert one - base == 2  # the packed indices of the F and R file
    assert two == one  # the second object reuses them
    rows = _sample_f_keys(N, packed_dir, n=20)
    for row in rows:
        first.f_symbol(*_sectors(row))
    before = cache.cache_info()['symbols']
    for row in rows:
        second.f_symbol(*_sectors(row))
    after = cache.cache_info()['symbols']
    assert after['misses'] == before['misses'] and after['hits'] > before['hits']  # shared cache entries
    del first, second
    gc.collect()
    assert cache.cache_info()['packed_files']['files'] == base  # released with the last object


def test_dropped_symmetry_is_released_from_the_plan_cache(N, packed_dir):
    """Cached tree plans own their symmetry; once nothing else does, the plans (and with them the open
    files and the packed index) must not outlive it."""
    cache.clear_caches()
    gc.collect()
    base = cache.cache_info()['packed_files']['files']
    sym = _sun(N, packed_dir)
    theta, apply = _update_workload(sym, N, bond_weight=1, mult=1)
    apply(theta)
    info = cache.cache_info()
    assert info['tree_plans']['entries'] > 0
    assert info['packed_files']['files'] == base + 2
    del sym, theta, apply
    gc.collect()
    info = cache.cache_info()  # reading the statistics purges the plans of unused symmetries
    assert info['tree_plans']['entries'] == 0
    assert info['packed_files']['files'] == base


def test_plans_of_a_symmetry_in_use_are_kept(N, packed_dir):
    cache.clear_caches()
    sym = _sun(N, packed_dir)
    theta, apply = _update_workload(sym, N, bond_weight=1, mult=1)
    apply(theta)
    entries = cache.cache_info()['tree_plans']['entries']
    assert entries > 0
    assert cache.cache_info()['tree_plans']['entries'] == entries  # the symmetry is still used
    before = cache.cache_info()['tree_plans']['hits']
    apply(theta)
    assert cache.cache_info()['tree_plans']['hits'] > before
