"""Public API for the :mod:`nanoslurm` package."""

import sys

if not sys.platform.startswith("linux"):
    raise OSError("nanoslurm is only supported on Linux")

from .backend import Job, list_jobs, submit
from .defaults import DEFAULTS, KEY_TYPES, load_defaults, save_defaults

__all__ = [
    "Job",
    "submit",
    "list_jobs",
    "DEFAULTS",
    "KEY_TYPES",
    "load_defaults",
    "save_defaults",
]
