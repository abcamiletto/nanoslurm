import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import nanoslurm.stats as stats
from nanoslurm.stats import fairshare_scores


def test_fairshare_scores_sprio(monkeypatch):
    monkeypatch.setattr(
        stats.B,
        "sprio",
        lambda **kwargs: [
            {"user": "alice", "fairshare": "0.5"},
            {"user": "bob", "fairshare": "0.1"},
        ],
    )
    assert fairshare_scores() == {"alice": 0.5, "bob": 0.1}


def test_fairshare_scores_sshare(monkeypatch):
    monkeypatch.setattr(
        stats.B,
        "sprio",
        lambda **kwargs: (_ for _ in ()).throw(stats.B.SlurmUnavailableError("missing")),
    )
    monkeypatch.setattr(
        stats.B,
        "sshare",
        lambda **kwargs: [{"user": "carol", "fairshare": "0.7"}],
    )
    assert fairshare_scores() == {"carol": 0.7}


def test_fairshare_scores_missing(monkeypatch):
    missing = lambda **kwargs: (_ for _ in ()).throw(stats.B.SlurmUnavailableError("missing"))
    monkeypatch.setattr(stats.B, "sprio", missing)
    monkeypatch.setattr(stats.B, "sshare", missing)
    assert fairshare_scores() == {}
