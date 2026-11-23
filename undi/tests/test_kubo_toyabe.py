import numpy as np
import pytest

pytest.importorskip('pandas')
pytest.importorskip('ase')

from ase import Atoms
from undi.kubo_toyabe import KT


def test_isotope_data_is_packaged():
    # Cu: 63Cu and 65Cu
    isotopes = KT.get_isotopes(29)
    assert len(isotopes) >= 2
    assert np.isclose(isotopes[:, 0].sum(), 100., atol=0.1)


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
