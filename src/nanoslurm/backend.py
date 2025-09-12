from __future__ import annotations

import os
import shlex
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

RUN_SH = Path(__file__).with_name("run.sh")

_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED", "BOOT_FAIL", "NODE_FAIL"}
_RUNNINGISH = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "STAGE_OUT", "SUSPENDED", "RESV_DEL_HOLD"}


class SlurmUnavailableError(RuntimeError):
    """Raised when required SLURM commands are missing."""


def _require(cmd: str) -> None:
    """Ensure *cmd* exists on PATH, otherwise raise ``SlurmUnavailableError``."""
    if not _which(cmd):
        raise SlurmUnavailableError(f"Required command '{cmd}' not found. Is this a SLURM environment?")


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
        SlurmUnavailableError: If ``sbatch`` is unavailable.
    """
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
        """Alias for stdout_path."""
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
        """Cancel the job via scancel."""
        _require("scancel")
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


def list_jobs(user: Optional[str] = None) -> list[Job]:
    """List SLURM jobs as :class:`Job` instances.

    Args:
        user: If provided, limit to jobs belonging to *user*.
    """
    if not (_which("squeue") or _which("sacct")):
        raise SlurmUnavailableError("squeue or sacct command not found on PATH")

    use_squeue = _which("squeue")
    if use_squeue:
        cmd = ["squeue", "-h", "-o", "%i|%j|%u|%P|%T|%V|%S"]
        if user:
            cmd.extend(["-u", user])
    else:
        cmd = [
            "sacct",
            "-n",
            "-o",
            "JobIDRaw,JobName,User,Partition,State,Submit,Start",
            "--parsable2",
            "-X",
        ]
        if user:
            cmd.extend(["-u", user])

    out = _run(cmd, check=False).stdout
    rows: list[Job] = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 7:
            jid, name, usr, part, status, submit, start = parts
            try:
                jid_int = int(jid)
            except ValueError:
                continue
            token = status.split()[0].split("+")[0].split("(")[0].rstrip("*")
            rows.append(
                Job(
                    id=jid_int,
                    name=name,
                    user=usr,
                    partition=part,
                    stdout_path=None,
                    stderr_path=None,
                    submit_time=_parse_datetime(submit),
                    start_time=_parse_datetime(start),
                    last_status=token,
                )
            )
    return rows


def node_state_counts() -> dict[str, int]:
    """Return a mapping of node state to count.

    Runs ``sinfo`` to obtain node information and aggregates the number of
    nodes reported in each state. Requires that the ``sinfo`` command is
    available on ``PATH``.
    """
    if not _which("sinfo"):
        raise SlurmUnavailableError("sinfo command not found on PATH")
    out = _run(["sinfo", "-h", "-o", "%T|%D"], check=False).stdout
    counts: Counter[str] = Counter()
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) != 2:
            continue
        state, count = parts
        token = state.split()[0].split("+")[0].split("(")[0].rstrip("*")
        try:
            counts[token] += int(count)
        except ValueError:
            continue
    return dict(counts)


def recent_completions(span: str = "day", count: int = 7) -> list[tuple[str, int]]:
    """Return counts of recently completed jobs grouped by *span*.

    Args:
        span: Group results by ``"day"`` or ``"week"``.
        count: Number of periods to return.

    Returns:
        List of (period, job_count) tuples sorted chronologically.
    """
    _require("sacct")
    if span not in {"day", "week"}:
        raise ValueError("span must be 'day' or 'week'")

    delta = timedelta(days=count if span == "day" else count * 7)
    start = (datetime.now() - delta).strftime("%Y-%m-%d")
    cmd = [
        "sacct",
        "--state=CD",
        "--noheader",
        "--parsable2",
        "--format=End",
        f"--starttime={start}",
        "-X",
    ]
    out = _run(cmd, check=False).stdout
    counts: Counter[str] = Counter()
    for line in out.splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            dt = datetime.strptime(token.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        if span == "week":
            year, week, _ = dt.isocalendar()
            key = f"{year}-W{week:02d}"
        else:
            key = dt.strftime("%Y-%m-%d")
        counts[key] += 1
    items = sorted(counts.items())
    return items[-count:]


def _parse_gpu(gres: str) -> int:
    """Extract total GPU count from a SLURM GRES string."""
    total = 0
    for token in gres.split(","):
        token = token.strip().split("(")[0]
        if token.startswith("gpu:"):
            try:
                total += int(token.split(":")[-1])
            except ValueError:
                pass
    return total


def _partition_caps() -> dict[str, dict[str, int]]:
    """Return total CPUs/GPUs available in each partition."""
    _require("sinfo")
    out = _run(["sinfo", "-ah", "-o", "%P|%C|%G|%D"], check=False).stdout
    caps: dict[str, dict[str, int]] = {}
    for line in out.splitlines():
        part, c_field, g_field, d_field = (line + "|||").split("|")[:4]
        part = part.rstrip("*")
        cpus = 0
        if c_field:
            try:
                cpus = int(c_field.split("/")[-1])
            except ValueError:
                pass
        gpus_per_node = _parse_gpu(g_field)
        nodes = 0
        if d_field:
            try:
                nodes = int(d_field)
            except ValueError:
                pass
        caps[part] = {"cpus": cpus, "gpus": gpus_per_node * nodes}
    return caps


def partition_utilization() -> dict[str, float]:
    """Return per-partition utilization percentage based on running jobs."""
    caps = _partition_caps()
    _require("squeue")
    out = _run(["squeue", "-h", "-t", "RUNNING", "-o", "%P|%C|%b"], check=False).stdout
    usage: dict[str, dict[str, int]] = {}
    for line in out.splitlines():
        part, c_field, g_field = (line + "||").split("|")[:3]
        cpus = 0
        if c_field:
            try:
                cpus = int(c_field)
            except ValueError:
                pass
        gpus = _parse_gpu(g_field)
        u = usage.setdefault(part, {"cpus": 0, "gpus": 0})
        u["cpus"] += cpus
        u["gpus"] += gpus
    utils: dict[str, float] = {}
    for part, cap in caps.items():
        use = usage.get(part, {})
        cpu_total = cap.get("cpus", 0)
        gpu_total = cap.get("gpus", 0)
        cpu_pct = use.get("cpus", 0) / cpu_total if cpu_total else 0.0
        gpu_pct = use.get("gpus", 0) / gpu_total if gpu_total else 0.0
        utils[part] = max(cpu_pct, gpu_pct) * 100
    return utils


def fairshare_scores() -> dict[str, float]:
    """Return a mapping of users to their fair-share scores.

    The function attempts to query :command:`sprio` first and falls back to
    :command:`sshare`. If neither command is available an empty mapping is
    returned.
    """
    cmd: Optional[Sequence[str]] = None
    if _which("sprio"):
        cmd = ["sprio", "-o", "user,fairshare", "-n"]
    elif _which("sshare"):
        cmd = ["sshare", "-o", "user,fairshare", "-n"]
    if not cmd:
        return {}

    out = _run(cmd, check=False).stdout
    scores: dict[str, float] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            user, val = parts[0], parts[1]
            try:
                scores[user] = float(val)
            except ValueError:
                continue
    return scores


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


def job_history() -> dict[str, dict[str, int]]:
    """Return per-user job completion statistics for the last 24 hours.

    Uses ``sacct`` with an explicit 24-hour window to gather counts of
    completed and failed jobs for all users. If ``sacct`` is unavailable,
    an empty mapping is returned.
    """
    if not _which("sacct"):
        return {}

    now = datetime.now()
    start = now - timedelta(hours=24)
    cmd = [
        "sacct",
        "-a",
        "-X",
        "-n",
        "--parsable2",
        "-S",
        start.strftime("%Y-%m-%dT%H:%M:%S"),
        "-E",
        now.strftime("%Y-%m-%dT%H:%M:%S"),
        "-o",
        "User,State",
    ]
    out = _run(cmd, check=False).stdout
    stats: dict[str, dict[str, int]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        user, state = parts[0], parts[1]
        if not user:
            continue
        token = state.split()[0].split("+")[0].split("(")[0].rstrip("*")
        entry = stats.setdefault(user, {"completed": 0, "failed": 0})
        if token == "COMPLETED":
            entry["completed"] += 1
        elif token in _TERMINAL:
            entry["failed"] += 1
    return stats


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
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return True
    return False


def _timestamp_ms() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]


__all__ = [
    "Job",
    "SlurmUnavailableError",
    "submit",
    "list_jobs",
    "node_state_counts",
    "recent_completions",
    "fairshare_scores",
    "job_history",
    "partition_utilization",
]

