"""The line that says which C++ module runs and with how many OpenMP threads (no extension needed)."""
from types import SimpleNamespace

import pytest

import undi.undi as undi_module


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(undi_module, '_REPORTED_BACKENDS', set())


def fake_module(max_threads=lambda: 4):
    attributes = {'__name__': 'fast_quantum', '__file__': '/env/lib/fast_quantum.cpython-312.so'}
    if max_threads:
        attributes['max_threads'] = max_threads
    return SimpleNamespace(**attributes)


@pytest.fixture
def cpp(monkeypatch):
    def use(module):
        monkeypatch.setattr(undi_module.importlib, 'import_module', lambda name: module)
    return use


def test_reports_the_module_the_threads_and_the_precision(capsys, cpp, monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '3')
    cpp(fake_module())
    undi_module._report_backend('fast', True)
    out = capsys.readouterr().out
    assert out.startswith('[undi] C++ module fast_quantum (/env/lib/fast_quantum.cpython-312.so)')
    assert 'OpenMP threads: 4 (OMP_NUM_THREADS=3)' in out
    assert 'precision: single' in out


def test_reports_once_per_configuration(capsys, cpp):
    cpp(fake_module())
    undi_module._report_backend('fast', False)
    undi_module._report_backend('fast', False)
    assert capsys.readouterr().out.count('[undi]') == 1
    undi_module._report_backend('fast', True)  # another configuration
    assert 'precision: single' in capsys.readouterr().out


def test_old_build_without_the_thread_count(capsys, cpp, monkeypatch):
    monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
    cpp(fake_module(max_threads=None))
    undi_module._report_backend('light', False)
    out = capsys.readouterr().out
    assert 'OpenMP threads: unknown (rebuild undi)' in out and 'OMP_NUM_THREADS=not set' in out
    assert 'precision: double' in out
