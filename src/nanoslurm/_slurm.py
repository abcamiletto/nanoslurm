from __future__ import annotations

"""Thin wrappers around SLURM commands with explicit keyword-based options."""

from shutil import which
from typing import Sequence

from .utils import run_command


class SlurmUnavailableError(RuntimeError):
    """Raised when required SLURM commands are missing."""


def require(cmd: str) -> None:
    if not which(cmd):
        raise SlurmUnavailableError(
            f"Required command '{cmd}' not found. Is this a SLURM environment?"
        )
def normalize_state(state: str) -> str:
    """Normalize a SLURM state string.

    Removes common qualifiers such as ``+``, ``*`` and parenthetical
    annotations and strips trailing tokens after the first whitespace.
    """
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
        parts = line.split() if sep is None else line.split(sep)
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
    check: bool = True,
) -> list[dict[str, str]]:
    if check:
        require("squeue")

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
    check: bool = True,
) -> list[dict[str, str]]:
    if check:
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
    check: bool = True,
) -> list[dict[str, str]]:
    if check:
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
# sprio

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
    check: bool = True,
) -> list[dict[str, str]]:
    if check:
        require("sprio")

    cmd = ["sprio", "-n"]
    if jobs:
        cmd += ["-j", ",".join(map(str, jobs))]
    if users:
        cmd += ["-u", ",".join(users)]

    fmt = ",".join(SPRIO_FIELDS[f] for f in fields)
    cmd += ["-o", fmt]
    return _table(cmd, list(fields), None, runner=runner)


# ---------------------------------------------------------------------------
# sshare

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
    check: bool = True,
) -> list[dict[str, str]]:
    if check:
        require("sshare")

    cmd = ["sshare", "-n"]
    if users:
        cmd += ["-u", ",".join(users)]
    if accounts:
        cmd += ["-A", ",".join(accounts)]

    fmt = ",".join(SSHARE_FIELDS[f] for f in fields)
    cmd += ["-o", fmt]
    return _table(cmd, list(fields), None, runner=runner)


__all__ = [
    "SlurmUnavailableError",
    "normalize_state",
    "require",
    "squeue",
    "sacct",
    "sinfo",
    "sprio",
    "sshare",
]
