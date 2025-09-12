import pytest

from nanoslurm.nanoslurm import Job, SlurmUnavailableError, submit


def test_submit_no_sbatch(monkeypatch):
    monkeypatch.setattr("nanoslurm.nanoslurm._which", lambda cmd: False)
    with pytest.raises(SlurmUnavailableError):
        submit(["echo"], cluster="c", time="0:10:00", cpus=1, memory=1, gpus=0)


def test_job_status_no_slurm(monkeypatch):
    monkeypatch.setattr("nanoslurm.nanoslurm._which", lambda cmd: False)

    job = Job(id=1, name="j", stdout_path=None, stderr_path=None)
    with pytest.raises(SlurmUnavailableError):
        _ = job.status
