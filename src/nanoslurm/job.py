from __future__ import annotations

import os
import shlex
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ._slurm import (
    SlurmUnavailableError,
    normalize_state,
    require as _require,
    sacct as _sacct,
    squeue as _squeue,
    which as _which,
)
from .utils import run_command as _run

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
    gpus: int,
    stdout_file: str | Path = "./slurm_logs/%j.txt",
    stderr_file: str | Path = "./slurm_logs/%j.err",
    signal: str = "SIGUSR1@90",
    workdir: str | Path = Path.cwd(),
) -> "Job":
    """Submit a job and return a :class:`Job` handle."""
    _require("sbatch")
    if not RUN_SH.exists():
        raise FileNotFoundError(f"run.sh not found at {RUN_SH}")

    stdout_file = Path(stdout_file).expanduser()
    stderr_file = Path(stderr_file).expanduser()
    workdir = Path(workdir).expanduser()
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    stderr_file.parent.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
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
        "-g",
        str(gpus),
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

    proc = _run(args, check=False)
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
        if not (_which("squeue") or _which("sacct")):
            raise SlurmUnavailableError("squeue or sacct not found on PATH")
        s = _status(self.id) or "UNKNOWN"
        self.last_status = s
        return s

    @property
    def wait_time(self) -> float | None:
        """Return the wait time in seconds between submission and start."""
        if self.submit_time and self.start_time:
            return (self.start_time - self.submit_time).total_seconds()
        return None

    def info(self) -> dict[str, str]:
        _require("scontrol")
        out = _run(["scontrol", "-o", "show", "job", str(self.id)], check=False).stdout.strip()
        info: dict[str, str] = {}
        if out:
            for token in out.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    info[k] = v
        return info

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
        _require("scancel")
        _run(["scancel", str(self.id)], check=False)

    def tail(self, n: int = 10) -> str:
        """Return the last *n* lines from the job's stdout file."""
        if not self.stdout_path:
            raise FileNotFoundError("stdout path unknown (pass stdout_file in submit())")
        try:
            return _run(["tail", "-n", str(n), str(self.stdout_path)], check=False).stdout
        except Exception:
            if self.stdout_path.exists():
                text = self.stdout_path.read_text(encoding="utf-8", errors="replace")
                return "".join(text.splitlines(True)[-n:])
            raise FileNotFoundError(f"stdout file not found at: {self.stdout_path}")


def list_jobs(user: str | None = None) -> list[Job]:
    """List SLURM jobs as :class:`Job` instances."""
    if not (_which("squeue") or _which("sacct")):
        raise SlurmUnavailableError("squeue or sacct command not found on PATH")

    if _which("squeue"):
        fetch, extra = _squeue, {}
    else:
        fetch, extra = _sacct, {"allocations": True}

    rows_data = fetch(
        fields=["id", "name", "user", "partition", "state", "submit", "start"],
        users=[user] if user else None,
        **extra,
    )

    rows: list[Job] = []
    for r in rows_data:
        try:
            jid_int = int(r["id"])
        except (KeyError, ValueError):
            continue
        token = normalize_state(r.get("state", ""))
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


def _status(job_id: int) -> str | None:
    if _which("squeue"):
        try:
            rows = _squeue(fields=["state"], jobs=[job_id])
            if rows:
                token = normalize_state(rows[0].get("state", ""))
                if token:
                    return token
        except SlurmUnavailableError:
            pass
    if _which("sacct"):
        try:
            rows = _sacct(fields=["state"], jobs=[job_id], allocations=True)
            for r in rows:
                token = normalize_state(r.get("state", ""))
                if token:
                    return token
        except SlurmUnavailableError:
            pass
    return None


def _parse_datetime(token: str) -> datetime | None:
    token = token.strip()
    if token in {"", "N/A", "Unknown"}:
        return None
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.fromisoformat(token) if fmt is None else datetime.strptime(token, fmt)
        except ValueError:
            continue
    return None


__all__ = ["Job", "SlurmUnavailableError", "submit", "list_jobs"]
