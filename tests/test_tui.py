import types

import pytest

from nanoslurm.nanoslurm import SlurmUnavailableError
from nanoslurm.tui import (
    _cluster_job_state_counts,
    _cluster_top_users,
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


def test_cluster_commands_with_clusters(monkeypatch):
    monkeypatch.setattr("nanoslurm.tui._which", lambda cmd: True)
    captured: list[list[str]] = []

    def fake_run(cmd, check=False):
        captured.append(cmd)
        return types.SimpleNamespace(stdout="")

    monkeypatch.setattr("nanoslurm.tui._run", fake_run)
    _cluster_job_state_counts(clusters="alpha,beta")
    _cluster_top_users(clusters="alpha,beta")
    assert [
        ["squeue", "-h", "-o", "%T", "-M", "alpha,beta"],
        ["squeue", "-h", "-o", "%u", "-M", "alpha,beta"],
    ] == captured

