"""Public API for the :mod:`nanoslurm` package."""

import sys

if not sys.platform.startswith("linux"):
    raise OSError("nanoslurm is only supported on Linux")

from .defaults import DEFAULTS, KEY_TYPES, load_defaults, save_defaults
from .job import Job, list_jobs, submit

__all__ = [
    "Job",
    "submit",
    "list_jobs",
    "DEFAULTS",
    "KEY_TYPES",
    "load_defaults",
    "save_defaults",
]
