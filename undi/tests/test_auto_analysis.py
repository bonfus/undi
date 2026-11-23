import numpy as np
import pytest
from ase import Atoms

import undi.auto_analysis as aa
from undi.auto_analysis import (
    check_enough_isotopes,
    execute_auto_analysis,
    gen_neighbouring_atomic_structure,
)


def cu_mu():
    """Cu and a muon (last atom, labelled H). Cu has 63Cu and 65Cu, both spin 3/2."""
    return Atoms('CuH', positions=[[0, 0, 0], [1.5, 0, 0]])


def test_check_enough_isotopes_finds_spins():
    info, candidates = check_enough_isotopes(cu_mu())
    assert list(info.keys()) == ['Cu']  # the muon is not considered
    assert {i.mass_number for i in candidates} == {63, 65}


def test_check_enough_isotopes_respects_thresholds():
    # 13C is only ~1.1% abundant
    structure = Atoms('CH', positions=[[0, 0, 0], [1.1, 0, 0]])
    _, candidates = check_enough_isotopes(structure)
    assert candidates == []
    _, candidates = check_enough_isotopes(structure, abundance_threshold=1.)
    assert [i.mass_number for i in candidates] == [13]
    # a spin_threshold above 3/2 excludes Cu
    _, candidates = check_enough_isotopes(cu_mu(), spin_threshold=2.)
    assert candidates == []


def test_check_enough_isotopes_without_muon_flag():
    info, _ = check_enough_isotopes(cu_mu(), without_muon=True)
    assert list(info.keys()) == ['Cu']
    info, _ = check_enough_isotopes(cu_mu(), without_muon=False)
    assert sorted(info.keys()) == ['Cu', 'H']


def test_hydrogen_is_never_a_candidate():
    # other H atoms (spin 1/2) are skipped; the last H is the muon
    structure = Atoms('HHH', positions=[[0, 0, 0], [3, 0, 0], [1.1, 0, 0]])
    info, candidates = check_enough_isotopes(structure)
    assert 'H' in info
    assert candidates == []


def test_execute_raises_without_isotopes():
    structure = Atoms('CH', positions=[[0, 0, 0], [1.1, 0, 0]])
    with pytest.raises(RuntimeError, match='No isotopes with spin'):
        execute_auto_analysis(structure)


def test_execute_raises_if_muon_not_last():
    structure = Atoms('HCu', positions=[[0, 0, 0], [1.5, 0, 0]])
    with pytest.raises(RuntimeError, match='last atom'):
        execute_auto_analysis(structure)


def test_other_hydrogens_are_not_neighbours():
    structure = Atoms('HCuH', positions=[[0, 0, 2.0], [1.5, 0, 0], [0, 0, 0]])
    data, hdim = gen_neighbouring_atomic_structure(
        structure, {'Cu': 63}, {'Cu': 1.5}, 10000)
    assert [d['Label'] for d in data] == ['mu', '63Cu']
    assert hdim == 8


class FakeInteraction:
    """Stands in for MuonNuclearInteraction: no C++ extension needed."""
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def translate_rotate_sample_vec(self, direction):
        pass

    def celio_on_steroids(self, t, *args, **kwargs):
        FakeInteraction.calls += 1
        return np.ones_like(t)


def test_convergence_check_returns_all_clusters(monkeypatch):
    monkeypatch.setattr(aa, 'MuonNuclearInteraction', FakeInteraction)
    FakeInteraction.calls = 0
    results = execute_auto_analysis(cu_mu(), convergence_check=True)
    # one result per isotope cluster: 63Cu and 65Cu
    assert sorted(r['cluster_isotopes']['Cu'] for r in results) == [63, 65]
    assert np.isclose(sum(r['probability'] for r in results), 1., atol=0.01)
    # the average over the iterations is normalised
    for r in results:
        assert np.allclose(r['signal_z_lf'], 1.)
    # only the Z LF signal is computed, no powder average etc.
    assert 'signal_powder_lf' not in results[0]
    assert FakeInteraction.calls == 10  # 5 iterations x 2 clusters


def test_abundance_threshold_changes_clusters(monkeypatch):
    monkeypatch.setattr(aa, 'MuonNuclearInteraction', FakeInteraction)
    # with a threshold above 65Cu abundance (~31%) only 63Cu is kept
    results = execute_auto_analysis(
        cu_mu(), convergence_check=True, abundance_threshold=50.)
    assert [r['cluster_isotopes']['Cu'] for r in results] == [63]
