from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence, List, Dict, Optional


class SlurmUnavailableError(RuntimeError):
    """Raised when required SLURM commands are missing."""


def which(name: str) -> bool:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return True
    return False


def require(cmd: str, which_func=which) -> None:
    if not which_func(cmd):
        raise SlurmUnavailableError(
            f"Required command '{cmd}' not found. Is this a SLURM environment?"
        )


def run(cmd: Sequence[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
    )


def _table(
    cmd: Sequence[str],
    keys: Sequence[str],
    sep: Optional[str],
    *,
    runner=run,
) -> List[Dict[str, str]]:
    out = runner(cmd, check=False).stdout
    rows: List[Dict[str, str]] = []
    for line in out.splitlines():
        if sep is None:
            parts = line.split()
        else:
            parts = line.split(sep)
        if len(parts) != len(keys):
            continue
        rows.append({k: v for k, v in zip(keys, parts)})
    return rows


def squeue(
    fields: Mapping[str, str],
    args: Sequence[str] = (),
    *,
    runner=run,
    which_func=which,
    check: bool = True,
) -> List[Dict[str, str]]:
    if check:
        require("squeue", which_func=which_func)
    fmt = "|".join(fields.values())
    cmd = ["squeue", "-h", "-o", fmt, *args]
    return _table(cmd, list(fields.keys()), "|", runner=runner)


def sacct(
    fields: Mapping[str, str],
    args: Sequence[str] = (),
    *,
    runner=run,
    which_func=which,
    check: bool = True,
) -> List[Dict[str, str]]:
    if check:
        require("sacct", which_func=which_func)
    fmt = ",".join(fields.values())
    cmd = ["sacct", "-n", "-o", fmt, "--parsable2", *args]
    return _table(cmd, list(fields.keys()), "|", runner=runner)


def sinfo(
    fields: Mapping[str, str],
    args: Sequence[str] = (),
    *,
    runner=run,
    which_func=which,
    check: bool = True,
) -> List[Dict[str, str]]:
    if check:
        require("sinfo", which_func=which_func)
    fmt = "|".join(fields.values())
    cmd = ["sinfo", "-h", "-o", fmt, *args]
    return _table(cmd, list(fields.keys()), "|", runner=runner)


def sprio(
    fields: Mapping[str, str],
    args: Sequence[str] = (),
    *,
    runner=run,
    which_func=which,
    check: bool = True,
) -> List[Dict[str, str]]:
    if check and not which_func("sprio"):
        raise SlurmUnavailableError("sprio command not found on PATH")
    fmt = ",".join(fields.values())
    cmd = ["sprio", "-n", "-o", fmt, *args]
    return _table(cmd, list(fields.keys()), None, runner=runner)


def sshare(
    fields: Mapping[str, str],
    args: Sequence[str] = (),
    *,
    runner=run,
    which_func=which,
    check: bool = True,
) -> List[Dict[str, str]]:
    if check and not which_func("sshare"):
        raise SlurmUnavailableError("sshare command not found on PATH")
    fmt = ",".join(fields.values())
    cmd = ["sshare", "-n", "-o", fmt, *args]
    return _table(cmd, list(fields.keys()), None, runner=runner)


__all__ = [
    "SlurmUnavailableError",
    "run",
    "which",
    "require",
    "squeue",
    "sacct",
    "sinfo",
    "sprio",
    "sshare",
]
