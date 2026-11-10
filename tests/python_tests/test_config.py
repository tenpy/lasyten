import getpass

import numpy as np
import pytest

import lasyten as lasy
from lasyten.config import LasytenConfig, get_config, get_option, restore_defaults, set_options, temporary_options

u1_symmetry = lasy.U1().as_Symmetry()

FAKE_USER_CONFIG = """
print_linewidth: 93
maxlines_spaces: 11
maxlines_tensors: 22
default_block_backend: gpu
"""

FAKE_LOCAL_CONFIG = """
print_linewidth: 92
print_indent: 3
maxlines_spaces: 10
check_fusion: False
"""


@pytest.fixture
def setup_fake_config(tmp_path_factory, monkeypatch):
    with monkeypatch.context() as m:
        user_config_path = tmp_path_factory.mktemp('home').joinpath('.lasytenconfig.yaml')
        user_config_path.write_text(FAKE_USER_CONFIG)
        m.setenv('LASYTEN_CONFIG_FILE', str(user_config_path))

        cwd = tmp_path_factory.mktemp('wd')
        cwd.joinpath('.lasytenconfig.yaml').write_text(FAKE_LOCAL_CONFIG)
        m.chdir(cwd)

        m.setenv('LASYTEN_PRINT_LINEWIDTH', '91')
        m.setenv('LASYTEN_PRINT_INDENT', '2')
        m.setenv('LASYTEN_MAXLINES_TENSORS', '21')

        restore_defaults()

        yield  # <- this runs the actual test
    # exiting the context restores the old cwd and env

    # cleanup
    restore_defaults()


def test_config_precedence(setup_fake_config):
    """

    ========================  ======  ======  ======  ======  ======  ======
    option                    temp    rntm    env     local   user    deflt
    ========================  ======  ======  ======  ======  ======  ======
    print_linewidth           80      90      91      92      93      100
    ------------------------  ------  ------  ------  ------  ------  ------
    print_indent                      1       2       3               4
    ------------------------  ------  ------  ------  ------  ------  ------
    maxlines_spaces                                   10      11      15
    ------------------------  ------  ------  ------  ------  ------  ------
    maxlines_tensors                          21              22      30
    ------------------------  ------  ------  ------  ------  ------  ------
    check_fusion              True                    False           True
    ------------------------  ------  ------  ------  ------  ------  ------
    default_tensor_backend                                            abel
    ------------------------  ------  ------  ------  ------  ------  ------
    default_block_backend                                     gpu     numpy
    ------------------------  ------  ------  ------  ------  ------  ------
    coupling_cutoff                                                   1e-13
    ========================  ======  ======  ======  ======  ======  ======

    """

    assert get_option('print_linewidth') == 91
    assert get_option('print_indent') == 2
    assert get_option('maxlines_spaces') == 10
    assert get_option('maxlines_tensors') == 21
    assert get_option('check_fusion') is False
    assert get_option('default_tensor_backend') == 'abelian'
    assert get_option('default_block_backend') == 'gpu'
    assert get_option('coupling_cutoff') == 1e-13
    # now, doing an invalid fusion should not raise
    invalid_fusion = u1_symmetry.r_symbol(np.array([1]), np.array([1]), np.array([-1]))

    set_options(print_linewidth=90, print_indent=1)

    assert get_option('print_linewidth') == 90
    assert get_option('print_indent') == 1
    assert get_option('maxlines_spaces') == 10
    assert get_option('maxlines_tensors') == 21
    assert get_option('check_fusion') is False
    assert get_option('default_tensor_backend') == 'abelian'
    assert get_option('default_block_backend') == 'gpu'
    assert get_option('coupling_cutoff') == 1e-13
    # now, doing an invalid fusion should not raise
    invalid_fusion = u1_symmetry.r_symbol(np.array([1]), np.array([1]), np.array([-1]))

    with temporary_options(print_linewidth=80, check_fusion=True):
        assert get_option('print_linewidth') == 80
        assert get_option('print_indent') == 1
        assert get_option('maxlines_spaces') == 10
        assert get_option('maxlines_tensors') == 21
        assert get_option('check_fusion') is True
        assert get_option('default_tensor_backend') == 'abelian'
        assert get_option('default_block_backend') == 'gpu'
        assert get_option('coupling_cutoff') == 1e-13
        # now, doing an invalid fusion should raise
        with pytest.raises(lasy.SymmetryError, match='not consistent with fusion rules'):
            invalid_fusion = u1_symmetry.r_symbol(np.array([1]), np.array([1]), np.array([-1]))

    assert get_option('print_linewidth') == 90
    assert get_option('print_indent') == 1
    assert get_option('maxlines_spaces') == 10
    assert get_option('maxlines_tensors') == 21
    assert get_option('check_fusion') is False
    assert get_option('default_tensor_backend') == 'abelian'
    assert get_option('default_block_backend') == 'gpu'
    assert get_option('coupling_cutoff') == 1e-13
    # now, doing an invalid fusion should not raise
    invalid_fusion = u1_symmetry.r_symbol(np.array([1]), np.array([1]), np.array([-1]))


def test_options_consistency():
    config = LasytenConfig()  # default config
    assert config.get_option('implicit_scalar_conversion') is True
    for key in get_config().all_option_keys():
        default_val = get_option(key)
        config.set_option(key, default_val)  # make sure that the default is valid


def test_su_n_data_defaults():
    """Default SU(N) data location follows the external generator's convention."""
    assert get_option('su_n_data_path') == f'/home/{getpass.getuser()}/.tenpy/su_n_symmetry_data'
    assert get_option('su_n_data_filename_base') == 'su_n_clebsch_gordan_data'


def test_su_n_data_env_override(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv('LASYTEN_SU_N_DATA_PATH', '/some/where/else')
        m.setenv('LASYTEN_SU_N_DATA_FILENAME_BASE', 'my_base')
        restore_defaults()
        assert get_option('su_n_data_path') == '/some/where/else'
        assert get_option('su_n_data_filename_base') == 'my_base'
    restore_defaults()


def test_su_n_data_set_option_types():
    config = LasytenConfig()
    config.set_option('su_n_data_path', r'C:\Users\x\su_n')  # free-form, no allow-list
    assert config.get_option('su_n_data_path') == r'C:\Users\x\su_n'
    with pytest.raises(TypeError):
        config.set_option('su_n_data_path', 5)
    with pytest.raises(ValueError):
        config.set_option('su_n_data_filename_base', '')


def test_su_n_data_file_path():
    from lasyten.symmetries import su_n_data_file_path, su_n_data_filename

    assert su_n_data_filename(3, 'CG', 7) == 'su_n_clebsch_gordan_data_N3_CG_hweight7.hdf5'
    assert su_n_data_filename(3, 'F', 4, filename_base='foo') == 'foo_N3_F_hweight4.hdf5'
    assert su_n_data_filename(3, 'r', 4) == 'su_n_clebsch_gordan_data_N3_R_hweight4.hdf5'
    with pytest.raises(ValueError):
        su_n_data_filename(3, 'X', 4)

    assert su_n_data_file_path(3, 'CG', 7, path='/a/b') == '/a/b/su_n_clebsch_gordan_data_N3_CG_hweight7.hdf5'
    assert su_n_data_file_path(3, 'CG', 7, path='/a/b/') == '/a/b/su_n_clebsch_gordan_data_N3_CG_hweight7.hdf5'
    with temporary_options(su_n_data_path='/x', su_n_data_filename_base='b'):
        assert su_n_data_file_path(3, 'R', 4) == '/x/b_N3_R_hweight4.hdf5'


def test_cache_option_defaults():
    config = LasytenConfig()
    assert config.get_option('su_n_symbol_cache_mb') == 256
    assert config.get_option('tree_plan_cache_mb') == 256
    assert config.get_option('tree_size_memo_mb') == 32  # its own budget, not a share of the plan budget
    assert config.get_option('su_n_packed_ram_mb') == 256
    for key in ('su2_cg_hweight', 'su2_f_hweight', 'su2_r_hweight'):
        assert config.get_option(key) == 0  # 0 = detect from the data directory


@pytest.mark.parametrize(
    'key',
    [
        'su_n_symbol_cache_mb',
        'tree_plan_cache_mb',
        'tree_size_memo_mb',
        'su_n_packed_ram_mb',
        'su2_cg_hweight',
        'su2_f_hweight',
        'su2_r_hweight',
    ],
)
def test_cache_option_validation(key):
    config = LasytenConfig()
    config.set_option(key, 0)
    config.set_option(key, 64)
    assert config.get_option(key) == 64
    config.set_option(key, '128')  # envvars and YAML files give strings
    assert config.get_option(key) == 128
    with pytest.raises(ValueError):
        config.set_option(key, -1)


def test_cache_option_env_override(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv('LASYTEN_SU_N_SYMBOL_CACHE_MB', '17')
        m.setenv('LASYTEN_TREE_PLAN_CACHE_MB', '0')
        m.setenv('LASYTEN_TREE_SIZE_MEMO_MB', '5')
        m.setenv('LASYTEN_SU_N_PACKED_RAM_MB', '0')
        restore_defaults()
        assert get_option('su_n_symbol_cache_mb') == 17
        assert get_option('tree_plan_cache_mb') == 0
        assert get_option('tree_size_memo_mb') == 5
        assert get_option('su_n_packed_ram_mb') == 0
    restore_defaults()
    assert get_option('su_n_symbol_cache_mb') == 256
    assert get_option('tree_size_memo_mb') == 32


def test_cache_option_temporary():
    with temporary_options(su_n_symbol_cache_mb=3):
        assert get_option('su_n_symbol_cache_mb') == 3
    assert get_option('su_n_symbol_cache_mb') == 256


def test_su_n_hweights_option():
    config = LasytenConfig()
    assert config.get_option('su_n_hweights') == ''
    for spec in ('3:5', '3:5,4,4', '3:5;4:3,3,3', ' 3 : 5 , 4 ; 4:3 ', '3:0,0,0', ''):
        config.set_option('su_n_hweights', spec)
        assert config.get_option('su_n_hweights') == spec
    for bad in ('3', '3:', '3:a', '1:5', '3:5;3:4', '3:1,2,3,4', ':5', '3:-1', 'x:5'):
        with pytest.raises(ValueError, match='su_n_hweights'):
            config.set_option('su_n_hweights', bad)
    assert config.get_option('su_n_hweights') == ''  # a rejected value does not change the option


def test_su_n_hweights_env_override(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv('LASYTEN_SU_N_HWEIGHTS', '3:5;4:3')
        restore_defaults()
        assert get_option('su_n_hweights') == '3:5;4:3'
    restore_defaults()
    assert get_option('su_n_hweights') == ''


def test_unquoted_yaml_su_n_hweights_gets_a_hint():
    config = LasytenConfig()
    with pytest.raises(TypeError, match='quote'):
        config.update_from_yaml('su_n_hweights: 2:24')  # YAML reads the unquoted 2:24 as the number 144
    config.update_from_yaml("su_n_hweights: '2:24'")
    assert config.get_option('su_n_hweights') == '2:24'
