import types

import pytest

from nanoslurm.nanoslurm import SlurmUnavailableError
from nanoslurm.tui import (
    _cluster_job_state_counts,
    _cluster_top_users,
    _cluster_partition_counts,
    _list_jobs,
)


def test_list_jobs_no_squeue(monkeypatch):
    monkeypatch.setattr("nanoslurm.tui._which", lambda cmd: False)
    with pytest.raises(SlurmUnavailableError):
        _list_jobs()


def test_list_jobs_parse(monkeypatch):
    monkeypatch.setattr("nanoslurm.tui._which", lambda cmd: True)

    def fake_run(cmd, check=False):
        return types.SimpleNamespace(stdout="1|job1|RUNNING\n2|job2|PENDING\n")

    monkeypatch.setattr("nanoslurm.tui._run", fake_run)
    assert _list_jobs() == [("1", "job1", "RUNNING"), ("2", "job2", "PENDING")]


def test_cluster_stats_no_squeue(monkeypatch):
    monkeypatch.setattr("nanoslurm.tui._which", lambda cmd: False)
    with pytest.raises(SlurmUnavailableError):
        _cluster_job_state_counts()
    with pytest.raises(SlurmUnavailableError):
        _cluster_top_users()
    with pytest.raises(SlurmUnavailableError):
        _cluster_partition_counts()


def test_cluster_job_state_counts_parse(monkeypatch):
    monkeypatch.setattr("nanoslurm.tui._which", lambda cmd: True)

    def fake_run(cmd, check=False):
        return types.SimpleNamespace(stdout="RUNNING\nRUNNING\nPENDING\n")

    monkeypatch.setattr("nanoslurm.tui._run", fake_run)
    assert _cluster_job_state_counts() == [
        ("PENDING", 1, 33.3),
        ("RUNNING", 2, 66.7),
    ]


def test_cluster_top_users_parse(monkeypatch):
    monkeypatch.setattr("nanoslurm.tui._which", lambda cmd: True)

    def fake_run(cmd, check=False):
        return types.SimpleNamespace(stdout="alice\nbob\nalice\n")

    monkeypatch.setattr("nanoslurm.tui._run", fake_run)
    assert _cluster_top_users(limit=10) == [
        ("alice", 2, 66.7),
        ("bob", 1, 33.3),
    ]


def test_cluster_partition_counts_parse(monkeypatch):
    monkeypatch.setattr("nanoslurm.tui._which", lambda cmd: True)

    def fake_run(cmd, check=False):
        return types.SimpleNamespace(stdout="alpha\nbeta\nalpha\n")

    monkeypatch.setattr("nanoslurm.tui._run", fake_run)
    assert _cluster_partition_counts() == [
        ("alpha", 2, 66.7),
        ("beta", 1, 33.3),
    ]

