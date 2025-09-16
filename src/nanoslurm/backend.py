"""SLURM command wrappers used throughout :mod:`nanoslurm`."""

from __future__ import annotations

from dataclasses import dataclass
from shutil import which
from typing import Sequence

from .utils.cmd import run_command


class SlurmUnavailableError(RuntimeError):
    """Raised when a required SLURM command is missing."""


def available(cmd: str) -> bool:
    """Return ``True`` if *cmd* can be located on ``PATH``."""

    return which(cmd) is not None


def require(cmd: str) -> None:
    """Raise :class:`SlurmUnavailableError` if *cmd* is not available."""

    if not available(cmd):
        raise SlurmUnavailableError(
            f"Required command '{cmd}' not found. Is this a SLURM environment?"
        )


def normalize_state(state: str) -> str:
    """Return the base SLURM state token for *state*."""

    token = state.strip().split()[0] if state else ""
    token = token.split("+", 1)[0]
    token = token.split("(", 1)[0]
    token = token.rstrip("*")
    return token


def _table(
    cmd: Sequence[str],
    keys: Sequence[str],
    sep: str | None,
    *,
    runner=run_command,
) -> list[dict[str, str]]:
    out = runner(cmd, check=False).stdout
    rows: list[dict[str, str]] = []
    for line in out.splitlines():
        parts = line.split(sep) if sep else line.split()
        if len(parts) != len(keys):
            continue
        rows.append({k: v for k, v in zip(keys, parts)})
    return rows


# ---------------------------------------------------------------------------
# squeue

SQUEUE_FIELDS = {
    "id": "%i",
    "name": "%j",
    "user": "%u",
    "partition": "%P",
    "state": "%T",
    "submit": "%V",
    "start": "%S",
    "cpus": "%C",
    "gres": "%b",
    "nodelist": "%R",
}


def squeue(
    *,
    fields: Sequence[str] = ("id", "name", "user", "state"),
    jobs: Sequence[int] | None = None,
    users: Sequence[str] | None = None,
    partitions: Sequence[str] | None = None,
    states: Sequence[str] | None = None,
    sort: str | None = None,
    runner=run_command,
) -> list[dict[str, str]]:
    """Return rows from ``squeue``.

    If the real ``squeue`` command is not available this function falls back
    to :func:`sacct` provided all requested *fields* are supported there.
    """

    if available("squeue"):
        cmd = ["squeue", "-h"]
        if jobs:
            cmd += ["-j", ",".join(map(str, jobs))]
        if users:
            cmd += ["-u", ",".join(users)]
        if partitions:
            cmd += ["-p", ",".join(partitions)]
        if states:
            cmd += ["-t", ",".join(states)]
        if sort:
            cmd += ["--sort", sort]

        fmt = "|".join(SQUEUE_FIELDS[f] for f in fields)
        cmd += ["-o", fmt]
        return _table(cmd, list(fields), "|", runner=runner)

    if not all(field in SACCT_FIELDS for field in fields):
        raise SlurmUnavailableError("squeue command not found and fallback is unavailable")

    return sacct(
        fields=fields,
        jobs=jobs,
        users=users,
        partitions=partitions,
        states=states,
        allocations=True,
        runner=runner,
    )


# ---------------------------------------------------------------------------
# sacct

SACCT_FIELDS = {
    "id": "JobIDRaw",
    "name": "JobName",
    "user": "User",
    "partition": "Partition",
    "state": "State",
    "submit": "Submit",
    "start": "Start",
    "end": "End",
}


def sacct(
    *,
    fields: Sequence[str] = ("id", "name", "user", "state"),
    jobs: Sequence[int] | None = None,
    users: Sequence[str] | None = None,
    partitions: Sequence[str] | None = None,
    states: Sequence[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    all_users: bool = False,
    allocations: bool = False,
    runner=run_command,
) -> list[dict[str, str]]:
    require("sacct")

    cmd = ["sacct", "-n"]
    if allocations:
        cmd.append("-X")
    if all_users:
        cmd.append("-a")
    if jobs:
        cmd += ["-j", ",".join(map(str, jobs))]
    if users:
        cmd += ["-u", ",".join(users)]
    if partitions:
        cmd += ["-p", ",".join(partitions)]
    if states:
        cmd += ["-s", ",".join(states)]
    if start_time:
        cmd += ["-S", start_time]
    if end_time:
        cmd += ["-E", end_time]

    fmt = ",".join(SACCT_FIELDS[f] for f in fields)
    cmd += ["-o", fmt, "--parsable2"]
    return _table(cmd, list(fields), "|", runner=runner)


# ---------------------------------------------------------------------------
# sinfo

SINFO_FIELDS = {
    "part": "%P",
    "state": "%T",
    "count": "%D",
    "cpus": "%C",
    "gres": "%G",
    "nodes": "%D",
}


def sinfo(
    *,
    fields: Sequence[str] = ("state", "count"),
    partitions: Sequence[str] | None = None,
    states: Sequence[str] | None = None,
    all_partitions: bool = False,
    runner=run_command,
) -> list[dict[str, str]]:
    require("sinfo")

    cmd = ["sinfo", "-h"]
    if partitions:
        cmd += ["-p", ",".join(partitions)]
    if states:
        cmd += ["-t", ",".join(states)]
    if all_partitions:
        cmd.append("-a")

    fmt = "|".join(SINFO_FIELDS[f] for f in fields)
    cmd += ["-o", fmt]
    return _table(cmd, list(fields), "|", runner=runner)


# ---------------------------------------------------------------------------
# sprio and sshare

SPRIO_FIELDS = {
    "job_id": "jobid",
    "user": "user",
    "priority": "priority",
    "fairshare": "fairshare",
}


def sprio(
    *,
    fields: Sequence[str] = ("job_id", "user", "priority"),
    jobs: Sequence[int] | None = None,
    users: Sequence[str] | None = None,
    runner=run_command,
) -> list[dict[str, str]]:
    if not available("sprio"):
        raise SlurmUnavailableError("sprio command not found on PATH")

    cmd = ["sprio", "-n"]
    if jobs:
        cmd += ["-j", ",".join(map(str, jobs))]
    if users:
        cmd += ["-u", ",".join(users)]

    fmt = ",".join(SPRIO_FIELDS[f] for f in fields)
    cmd += ["-o", fmt]
    return _table(cmd, list(fields), None, runner=runner)


SSHARE_FIELDS = {
    "user": "user",
    "account": "account",
    "fairshare": "fairshare",
}


def sshare(
    *,
    fields: Sequence[str] = ("user", "fairshare"),
    users: Sequence[str] | None = None,
    accounts: Sequence[str] | None = None,
    runner=run_command,
) -> list[dict[str, str]]:
    if not available("sshare"):
        raise SlurmUnavailableError("sshare command not found on PATH")

    cmd = ["sshare", "-n"]
    if users:
        cmd += ["-u", ",".join(users)]
    if accounts:
        cmd += ["-A", ",".join(accounts)]

    fmt = ",".join(SSHARE_FIELDS[f] for f in fields)
    cmd += ["-o", fmt]
    return _table(cmd, list(fields), None, runner=runner)


# ---------------------------------------------------------------------------
# Misc helpers

def scancel(job_id: int, *, runner=run_command) -> None:
    """Cancel *job_id* via ``scancel``."""

    require("scancel")
    runner(["scancel", str(job_id)], check=False)


@dataclass(slots=True)
class ControlInfo:
    """Structured data parsed from ``scontrol show job`` output."""

    data: dict[str, str]


def scontrol_show_job(job_id: int, *, runner=run_command) -> ControlInfo:
    """Return information about *job_id* via ``scontrol``."""

    require("scontrol")
    out = runner(["scontrol", "-o", "show", "job", str(job_id)], check=False).stdout.strip()
    info: dict[str, str] = {}
    if out:
        for token in out.split():
            if "=" in token:
                key, value = token.split("=", 1)
                info[key] = value
    return ControlInfo(info)


__all__ = [
    "ControlInfo",
    "SlurmUnavailableError",
    "available",
    "normalize_state",
    "require",
    "sacct",
    "scancel",
    "scontrol_show_job",
    "sinfo",
    "squeue",
    "sprio",
    "sshare",
]

