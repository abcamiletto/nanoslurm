import types

from nanoslurm.tui import _list_jobs


def test_list_jobs_no_squeue(monkeypatch):
    monkeypatch.setattr("nanoslurm.tui._which", lambda cmd: False)
    assert _list_jobs() == []


def test_list_jobs_parse(monkeypatch):
    monkeypatch.setattr("nanoslurm.tui._which", lambda cmd: True)

    def fake_run(cmd, check=False):
        return types.SimpleNamespace(stdout="1|job1|RUNNING\n2|job2|PENDING\n")

    monkeypatch.setattr("nanoslurm.tui._run", fake_run)
    assert _list_jobs() == [("1", "job1", "RUNNING"), ("2", "job2", "PENDING")]

