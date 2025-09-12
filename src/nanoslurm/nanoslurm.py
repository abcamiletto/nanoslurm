from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

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
    """Submit a job and return a Job handle.

    Args:
        command: Command to execute on the node. List (preferred) or raw shell string.
        name: Base job name; a timestamp suffix is appended for uniqueness.
        cluster: SLURM partition (required).
        time: Time limit in HH:MM:SS (required).
        cpus: CPU cores (required).
        memory: Memory in GB (required).
        gpus: Number of GPUs (required).
        stdout_file: Stdout path (supports %j).
        stderr_file: Stderr path (supports %j).
        signal: SBATCH --signal (e.g., "SIGUSR1@90").
        workdir: Working directory at runtime (`sbatch -D`).

    Returns:
        Job: Handle with id, name, and resolved log paths.

    Raises:
        FileNotFoundError: If run.sh is missing.
        RuntimeError: If job id cannot be parsed from sbatch output.
    """
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

    # Parse exactly: "Submitted batch job <id>"
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
        stdout_path=Path(str(stdout_file).replace("%j", str(job_id))),
        stderr_path=Path(str(stderr_file).replace("%j", str(job_id))),
    )


@dataclass
class Job:
    """Handle to a submitted SLURM job."""

    id: int
    name: str
    stdout_path: Optional[Path]
    stderr_path: Optional[Path]

    @property
    def output_file(self) -> Optional[Path]:
        """Alias for stdout_path."""
        return self.stdout_path

    @property
    def status(self) -> str:
        """Return SLURM job status."""
        s = _squeue_status(self.id)
        if s:
            return s
        s = _sacct_status(self.id)
        return s or "UNKNOWN"

    def info(self) -> dict[str, str]:
        if not _which("scontrol"):
            return {}
        out = _run(["scontrol", "-o", "show", "job", str(self.id)], check=False).stdout.strip()
        if not out:
            return {}

        info: dict[str, str] = {}
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
        """Cancel the job via scancel."""
        if not _which("scancel"):
            raise RuntimeError("scancel not found on PATH")
        _run(["scancel", str(self.id)], check=False)

    def tail(self, n: int = 10) -> str:
        """Return the last n lines from the job's stdout file."""
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


def _squeue_status(job_id: int) -> Optional[str]:
    if not _which("squeue"):
        return None
    out = _run(["squeue", "-j", str(job_id), "-h", "-o", "%T"], check=False).stdout.strip()
    if out:
        token = out.split()[0].split("+")[0].split("(")[0].rstrip("*")
        return token
    return None


def _sacct_status(job_id: int) -> Optional[str]:
    if not _which("sacct"):
        return None
    out = _run(["sacct", "-j", str(job_id), "-o", "State", "-n", "--parsable2", "-X"], check=False).stdout
    for line in out.splitlines():
        token = line.strip()
        if token:
            return token.split()[0].split("+")[0].split("(")[0]
    return None


def _run(cmd: Sequence[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)


def _which(name: str) -> bool:
    return any(
        (Path(p) / name).exists() and (Path(p) / name).is_file() and os.access(Path(p) / name, os.X_OK)
        for p in os.environ.get("PATH", "").split(os.pathsep)
    )


def _timestamp_ms() -> str:
    now = datetime.now()
    ms = int((time.time() - int(time.time())) * 1000)
    return now.strftime("%Y-%m-%d_%H-%M-%S.") + f"{ms:03d}"
