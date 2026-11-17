"""DOF public surfaces use nb::ndarray interchange (issue #326 item 2)."""

import numpy as np
import numpy.testing as npt

from lasyten.models import degrees_of_freedom, sites


def test_spin_dof_ndarray_dual_path():
    site = sites.SpinSite(0.5, 'None')
    spin = site.spin
    view = spin.spin_vector
    assert isinstance(view, np.ndarray)
    assert not view.flags.writeable
    owning = spin.to_numpy()
    assert isinstance(owning, np.ndarray)
    assert owning.flags.writeable
    npt.assert_array_equal(owning, view)
    # Owning copy is independent of the stored buffer.
    owning[0, 0, 0] = 123.0
    assert spin.spin_vector[0, 0, 0] != 123.0


def test_spin_vector_from_sp_accepts_ndarray():
    site = sites.SpinSite(0.5, 'None')
    sz = np.array(site.spin.spin_vector[:, :, 2], copy=True)
    sx = site.spin.spin_vector[:, :, 0]
    sy = site.spin.spin_vector[:, :, 1]
    sp = np.array(sx + 1j * sy, copy=True)
    rebuilt = degrees_of_freedom.SpinDOF.spin_vector_from_Sp(sz, sp)
    assert isinstance(rebuilt, np.ndarray)
    npt.assert_allclose(rebuilt, site.spin.spin_vector, atol=1e-12)


def test_occupation_get_creator_numpy_owning():
    site = sites.SpinlessFermionSite(conserve='parity')
    creator = site.occupation.get_creator_numpy(species=0)
    assert isinstance(creator, np.ndarray)
    assert creator.flags.writeable
    creator[0, 0] = 99.0
    # Stored creators must remain unchanged (owning copy).
    npt.assert_array_equal(
        site.occupation.creators[:, :, 0],
        site.occupation.get_creator_numpy(species=0),
    )


def test_bosonic_creation_ops_ndarray():
    c, a = degrees_of_freedom.BosonicDOF.creation_annihilation_ops_from_Nmax(np.asarray([2], dtype=int), dim=3)
    assert isinstance(c, np.ndarray) and isinstance(a, np.ndarray)
    assert c.shape == (3, 3, 1)
