"""Tests of the command line interfaces: `python -m undi` and `python -m undi.kubo_toyabe`."""
import json

import numpy as np
import pytest

pytest.importorskip('ase')

from ase import Atoms
from ase.io import write

import undi.__main__ as undi_cli
from undi.kubo_toyabe import __main__ as kt_cli


@pytest.fixture
def structure(tmp_path):
    """Cu and a muon (last atom, labelled H)."""
    path = tmp_path / 'structure.xyz'
    write(path, Atoms('CuH', positions=[[0, 0, 0], [1.5, 0, 0]]))
    return str(path)


def test_kubo_toyabe_cli_writes_the_polarization(structure, tmp_path):
    output = tmp_path / 'KT.json'
    results = kt_cli.main([structure, '--output', str(output)])

    assert json.loads(output.read_text()) == results
    assert set(results) == {'t', 'KT'}
    assert len(results['t']) == len(results['KT']) == 1000
    assert results['t'][0] == 0. and np.isclose(results['t'][-1], 20.)  # microseconds
    assert np.isclose(results['KT'][0], 1.)  # full polarization at t=0
    assert np.isclose(results['KT'][-1], 1. / 3., atol=1e-2)  # the 1/3 tail


def test_kubo_toyabe_cli_time_grid(structure, tmp_path):
    results = kt_cli.main([structure, '--output', str(tmp_path / 'KT.json'), '--tmax', '5e-6', '--npoints', '11'])
    assert len(results['t']) == 11 and np.isclose(results['t'][-1], 5.)


@pytest.fixture
def analysis(monkeypatch):
    """Replace the analysis, to look at the arguments of the CLI without running UNDI."""
    calls = []

    def execute_auto_analysis(**kwargs):
        calls.append(kwargs)
        return [{'B_ext': kwargs['B_mod'], 't': [0.0, 1.0], 'signal_z_lf': [1.0, 0.5]}]

    monkeypatch.setattr(undi_cli, 'execute_auto_analysis', execute_auto_analysis)
    return calls


def test_convergence_check_is_a_flag(structure, analysis, monkeypatch, tmp_path):
    """`--convergence-check` alone turns it on, without it is off (it used to be `type=bool`, where 'False' is True)."""
    monkeypatch.chdir(tmp_path)
    undi_cli.main([structure, '--Bmod', '0.002', '--max-hdim', '100'])
    undi_cli.main([structure, '--convergence-check'])
    assert [call['convergence_check'] for call in analysis] == [False, True]
    assert analysis[0]['B_mod'] == 0.002 and analysis[0]['max_hdim'] == 100


def test_results_are_dumped_to_json(structure, analysis, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    results = undi_cli.main([structure, '--Bmod', '0.5'])
    assert json.loads((tmp_path / 'results.json').read_text()) == results
