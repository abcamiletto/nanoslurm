"""Backward compatible exports for the legacy :mod:`nanoslurm._slurm` module."""

from __future__ import annotations

from shutil import which

from .backend import (
    ControlInfo,
    SlurmUnavailableError,
    available,
    normalize_state,
    require,
    sacct,
    scancel,
    scontrol_show_job,
    sinfo,
    squeue,
    sprio,
    sshare,
)

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
    "which",
]

