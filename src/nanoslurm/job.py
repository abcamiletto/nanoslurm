from __future__ import annotations

import os
import shlex
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

from ._slurm import (
    SlurmUnavailableError,
    normalize_state,
    require as _require,
    run as _run,
    sacct as _sacct,
    squeue as _squeue,
    which as _which,
)

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
    stdout_file: Union[str, Path] = "./slurm_logs/%j.txt",
    stderr_file: Union[str, Path] = "./slurm_logs/%j.err",
    signal: str = "SIGUSR1@90",
    workdir: Union[str, Path] = Path.cwd(),
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

    job_id: Optional[int] = None
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
    stdout_path: Optional[Path]
    stderr_path: Optional[Path]
    submit_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    last_status: Optional[str] = None

    @property
    def output_file(self) -> Optional[Path]:
        """Alias for ``stdout_path``."""
        return self.stdout_path

    @property
    def status(self) -> str:
        """Return the current SLURM job status."""
        if not (_which("squeue") or _which("sacct")):
            raise SlurmUnavailableError("squeue or sacct not found on PATH")
        s = _squeue_status(self.id)
        if not s:
            s = _sacct_status(self.id)
        s = s or "UNKNOWN"
        self.last_status = s
        return s

    @property
    def wait_time(self) -> Optional[float]:
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

    def wait(self, poll_interval: float = 5.0, timeout: Optional[float] = None) -> str:
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
        if not self.stdout_path.exists():
            time.sleep(0.2)
        if self.stdout_path.exists():
            try:
                return _run(["tail", "-n", str(n), str(self.stdout_path)], check=False).stdout
            except Exception:
                text = self.stdout_path.read_text(encoding="utf-8", errors="replace")
                return "".join(text.splitlines(True)[-n:])
        raise FileNotFoundError(f"stdout file not found at: {self.stdout_path}")


def list_jobs(user: Optional[str] = None) -> list[Job]:
    """List SLURM jobs as :class:`Job` instances."""
    if not (_which("squeue") or _which("sacct")):
        raise SlurmUnavailableError("squeue or sacct command not found on PATH")

    rows_data: list[dict[str, str]]
    if _which("squeue"):
        rows_data = _squeue(
            fields=["id", "name", "user", "partition", "state", "submit", "start"],
            users=[user] if user else None,
            runner=_run,
            which_func=_which,
        )
    else:
        rows_data = _sacct(
            fields=["id", "name", "user", "partition", "state", "submit", "start"],
            users=[user] if user else None,
            allocations=True,
            runner=_run,
            which_func=_which,
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


def _squeue_status(job_id: int) -> Optional[str]:
    try:
        rows = _squeue(fields=["state"], jobs=[job_id], runner=_run, which_func=_which)
    except SlurmUnavailableError:
        return None
    if rows:
        state = rows[0].get("state", "")
        return normalize_state(state)
    return None


def _sacct_status(job_id: int) -> Optional[str]:
    try:
        rows = _sacct(
            fields=["state"],
            jobs=[job_id],
            allocations=True,
            runner=_run,
            which_func=_which,
        )
    except SlurmUnavailableError:
        return None
    for r in rows:
        token = normalize_state(r.get("state", ""))
        if token:
            return token
    return None


def _parse_datetime(token: str) -> Optional[datetime]:
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


__all__ = ["Job", "SlurmUnavailableError", "submit", "list_jobs"]
