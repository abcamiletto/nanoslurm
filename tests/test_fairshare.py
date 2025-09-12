import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import nanoslurm.backend as backend
from nanoslurm.backend import fairshare_scores


def test_fairshare_scores_sprio(monkeypatch):
    monkeypatch.setattr(backend, "_which", lambda cmd: cmd == "sprio")
    monkeypatch.setattr(
        backend,
        "_run",
        lambda cmd, check=False: types.SimpleNamespace(stdout="alice 0.5\nbob 0.1\n"),
    )
    assert fairshare_scores() == {"alice": 0.5, "bob": 0.1}


def test_fairshare_scores_sshare(monkeypatch):
    monkeypatch.setattr(backend, "_which", lambda cmd: cmd == "sshare")
    monkeypatch.setattr(
        backend,
        "_run",
        lambda cmd, check=False: types.SimpleNamespace(stdout="carol 0.7\n"),
    )
    assert fairshare_scores() == {"carol": 0.7}


def test_fairshare_scores_missing(monkeypatch):
    monkeypatch.setattr(backend, "_which", lambda cmd: False)
    assert fairshare_scores() == {}
