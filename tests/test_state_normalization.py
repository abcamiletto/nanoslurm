from __future__ import annotations

import nanoslurm.job as job
import nanoslurm.job as job
import nanoslurm.stats as stats
from nanoslurm.backend import normalize_state


def test_normalize_state_cases():
    cases = {
        "RUNNING": "RUNNING",
        "RUNNING+": "RUNNING",
        "COMPLETED (exit 0)": "COMPLETED",
        "FAILED*": "FAILED",
        "PENDING+ (Resources)": "PENDING",
    }
    for raw, expected in cases.items():
        assert normalize_state(raw) == expected


def test_list_jobs_state_normalization(monkeypatch):
    def fake_squeue(*, fields, **kwargs):
        return [
            {
                "id": "1",
                "name": "n",
                "user": "u",
                "partition": "p",
                "state": "RUNNING* (Priority)",
                "submit": "",
                "start": "",
            }
        ]

    monkeypatch.setattr(job.B, "squeue", fake_squeue)
    rows = job.list_jobs()
    assert rows[0].last_status == "RUNNING"


def test_node_state_counts_normalization(monkeypatch):
    def fake_sinfo(*, fields, **kwargs):
        return [
            {"state": "idle*", "count": "5"},
            {"state": "alloc+", "count": "3"},
        ]

    monkeypatch.setattr(stats.B, "sinfo", fake_sinfo)
    result = stats.node_state_counts()
    assert result == {"idle": 5, "alloc": 3}


def test_job_history_state_normalization(monkeypatch):
    def fake_sacct(*, fields, **kwargs):
        return [
            {"user": "alice", "state": "COMPLETED"},
            {"user": "bob", "state": "FAILED+"},
            {"user": "bob", "state": "COMPLETED+"},
            {"user": "alice", "state": "FAILED (rc=1)"},
        ]

    monkeypatch.setattr(stats.B, "sacct", fake_sacct)
    result = stats.job_history()
    assert result == {
        "alice": {"completed": 1, "failed": 1},
        "bob": {"completed": 1, "failed": 1},
    }

