"""Public API for the :mod:`nanoslurm` package."""

import sys

if not sys.platform.startswith("linux"):
    raise OSError("nanoslurm is only supported on Linux")

from .nanoslurm import Job, submit

__all__ = ["Job", "submit"]
