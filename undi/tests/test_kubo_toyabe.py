import numpy as np
import pytest

pytest.importorskip('ase')

from ase import Atoms
from undi import MuonNuclearInteraction
from undi.kubo_toyabe import KT

angtom = 1.0e-10  # m


def test_isotopes_from_isotopes_module():
    # Cu: 63Cu and 65Cu, columns are abundance, spin, g-factor
    isotopes = KT.get_isotopes(29)
    assert isotopes.shape == (2, 3)
    assert np.isclose(isotopes[:, 0].sum(), 100., atol=0.1)
    assert np.allclose(isotopes[:, 1], 1.5)


def test_kubo_toyabe_limits():
    t = np.array([0., 1e-3, 1e3])
    kt = KT.kubo_toyabe(t, 1.)
    assert np.isclose(kt[0], 1.)  # full polarization at t=0
    assert np.isclose(kt[-1], 1. / 3.)  # 1/3 tail


def test_second_moments_single_species():
    # a muon (H) and one Cu atom 2 angstrom away
    atms = Atoms('HCu', positions=[[0, 0, 0], [2., 0, 0]], pbc=False)
    moments = KT.compute_second_moments(atms, cutoff_distances={29: 5.})
    assert list(moments.keys()) == [29]
    assert moments[29] > 0.


def test_second_moments_match_analytical_formula():
    # a muon surrounded by three 19F (100% abundant, spin 1/2) at different distances
    pos = np.array([[1.2, 0, 0], [0, 1.9, 0], [0, 0, -2.5]])

    # gamma_mu^2 * Delta^2, secular dipolar interaction, powder average, zero field
    hbar = 1.0545718e-34  # J s
    mu_0 = 1.2566371e-6  # N A^-2
    gamma_mu = MuonNuclearInteraction.gammas['mu']
    gamma_F = MuonNuclearInteraction.gammas['F']
    spin = 0.5
    r3 = (np.linalg.norm(pos, axis=1) * angtom) ** 3
    expected = gamma_mu ** 2 * np.sum(
        2. * (mu_0 / (4 * np.pi)) ** 2 * ((gamma_F * hbar) / r3) ** 2
        * (1. / 3.) * spin * (spin + 1))

    atms = Atoms('H' + 'F' * 3, positions=np.vstack([[0, 0, 0], pos]))
    moments = KT.compute_second_moments(atms, cutoff_distances={9: 10.})
    assert np.isclose(sum(moments.values()), expected, rtol=1e-3)
