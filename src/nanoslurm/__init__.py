"""Public API for the :mod:`nanoslurm` package."""

import sys

if not sys.platform.startswith("linux"):
    raise OSError("nanoslurm is only supported on Linux")

from .defaults import DEFAULTS, KEY_TYPES, load_defaults, save_defaults
from .nanoslurm import Job, submit

__all__ = [
    "Job",
    "submit",
    "DEFAULTS",
    "KEY_TYPES",
    "load_defaults",
    "save_defaults",
]
