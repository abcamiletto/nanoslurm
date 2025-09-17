from __future__ import annotations

import os
import shlex
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from . import backend as B
from .utils.cmd import run_command

RUN_SH = Path(__file__).with_name("run.sh")

_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED", "BOOT_FAIL", "NODE_FAIL"}
_RUNNINGISH = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "STAGE_OUT", "SUSPENDED", "RESV_DEL_HOLD"}


def submit(
    command: Iterable[str] | str,
    *,
    name: str = "job",
    cluster: str,
    time: str,
    cpus: int,
    memory: int,
    gpus: int = 0,
    stdout_file: str | Path = "./slurm_logs/%j.txt",
    stderr_file: str | Path = "./slurm_logs/%j.err",
    signal: str = "SIGUSR1@90",
    workdir: str | Path = Path.cwd(),
) -> "Job":
    """Submit a job and return a :class:`Job` handle."""
    B.require("sbatch")
    if not RUN_SH.exists():
        raise FileNotFoundError(f"run.sh not found at {RUN_SH}")

    stdout_file = Path(stdout_file).expanduser()
    stderr_file = Path(stderr_file).expanduser()
    workdir = Path(workdir).expanduser()
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    stderr_file.parent.mkdir(parents=True, exist_ok=True)

    stamp = _timestamp_ms()
    full_name = f"{name}_{stamp}"

    args = [
        "bash",
        str(RUN_SH),
        "-n",
        full_name,
        "-c",
        cluster,
        "-t",
        time,
        "-p",
        str(cpus),
        "-m",
        str(memory),
    ]
    if gpus > 0:
        args += ["-g", str(gpus)]
    args += [
        "-o",
        str(stdout_file),
        "-e",
        str(stderr_file),
        "-s",
        signal,
        "-w",
        str(workdir),
        "--",
    ]

    cmd_str = command if isinstance(command, str) else " ".join(shlex.quote(c) for c in command)
    args.append(cmd_str)

    proc = run_command(args, check=False)
    out = proc.stdout.strip()
    err = proc.stderr.strip()

    job_id: int | None = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Submitted batch job "):
            try:
                job_id = int(s.split()[-1])
            except ValueError:
                pass
            break
    if job_id is None:
        raise RuntimeError(f"Could not parse job id.\nstdout:\n{out}\nstderr:\n{err}")

    return Job(
        id=job_id,
        name=full_name,
        user=os.environ.get("USER", ""),
        partition=cluster,
        stdout_path=Path(str(stdout_file).replace("%j", str(job_id))),
        stderr_path=Path(str(stderr_file).replace("%j", str(job_id))),
    )


@dataclass
class Job:
    """Handle to a submitted SLURM job."""

    id: int
    name: str
    user: str
    partition: str
    stdout_path: Path | None
    stderr_path: Path | None
    submit_time: datetime | None = None
    start_time: datetime | None = None
    last_status: str | None = None

    @property
    def status(self) -> str:
        """Return the current SLURM job status."""
        rows = B.squeue(fields=["state"], jobs=[self.id])
        token = rows[0].get("state", "") if rows else ""
        state = B.normalize_state(token) if token else "UNKNOWN"
        self.last_status = state
        return state

    @property
    def wait_time(self) -> float | None:
        """Return the wait time in seconds between submission and start."""
        if self.submit_time and self.start_time:
            return (self.start_time - self.submit_time).total_seconds()
        return None

    def info(self) -> dict[str, str]:
        return B.scontrol_show_job(self.id).data

    def is_running(self) -> bool:
        """Check if the job is in a non-terminal state."""
        return self.status in _RUNNINGISH

    def is_finished(self) -> bool:
        """Check if the job reached a terminal state."""
        return self.status in _TERMINAL

    def wait(self, poll_interval: float = 5.0, timeout: float | None = None) -> str:
        """Wait for the job to finish."""
        start = time.time()
        while True:
            s = self.status
            if s in _TERMINAL:
                return s
            if timeout is not None and (time.time() - start) > timeout:
                return s
            time.sleep(poll_interval)

    def cancel(self) -> None:
        """Cancel the job via ``scancel``."""
        B.scancel(self.id)

    def tail(self, n: int = 10) -> str:
        """Return the last *n* lines from the job's stdout file."""
        if not self.stdout_path:
            raise FileNotFoundError("stdout path unknown (pass stdout_file in submit())")
        if self.stdout_path.exists():
            text = self.stdout_path.read_text(encoding="utf-8", errors="replace")
            return "".join(text.splitlines(True)[-n:])
        raise FileNotFoundError(f"stdout file not found at: {self.stdout_path}")


def list_jobs(user: str | None = None) -> list[Job]:
    """List SLURM jobs as :class:`Job` instances."""
    rows_data = B.squeue(
        fields=["id", "name", "user", "partition", "state", "submit", "start"],
        users=[user] if user else None,
    )

    rows: list[Job] = []
    for r in rows_data:
        try:
            jid_int = int(r["id"])
        except (KeyError, ValueError):
            continue
        token = B.normalize_state(r.get("state", ""))
        rows.append(
            Job(
                id=jid_int,
                name=r.get("name", ""),
                user=r.get("user", ""),
                partition=r.get("partition", ""),
                stdout_path=None,
                stderr_path=None,
                submit_time=_parse_datetime(r.get("submit", "")),
                start_time=_parse_datetime(r.get("start", "")),
                last_status=token,
            )
        )
    return rows


def _parse_datetime(token: str) -> datetime | None:
    token = token.strip()
    if not token or token in {"N/A", "Unknown"}:
        return None
    try:
        return datetime.fromisoformat(token)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(token, fmt)
            except ValueError:
                pass
    return None


def _timestamp_ms() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]


__all__ = ["Job", "submit", "list_jobs"]
