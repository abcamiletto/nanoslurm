"""Command utilities for nanoslurm."""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Sequence, Optional


logger = logging.getLogger(__name__)


def run_command(
    cmd: Sequence[str],
    *,
    check: bool = True,
    retries: int = 0,
    retry_delay: float = 0.0,
    log: Optional[logging.Logger] = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run *cmd* with ``subprocess.run`` and optional retries.

    Parameters
    ----------
    cmd:
        Command arguments passed to :func:`subprocess.run`.
    check:
        If ``True`` (default), raise :class:`subprocess.CalledProcessError` on
        non-zero exit codes. Behaviour matches :func:`subprocess.run`.
    retries:
        Number of additional attempts to run the command after a failure. A
        value of ``0`` disables retries.
    retry_delay:
        Seconds to sleep between retries.
    log:
        Optional :class:`logging.Logger` to use for messages. If omitted, a
        module-level logger is used.
    **kwargs:
        Additional keyword arguments forwarded to :func:`subprocess.run`.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the executed command.
    """

    log = log or logger
    attempt = 0
    while True:
        log.debug("Running command: %s", cmd)
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=check,
                **kwargs,
            )
            log.debug("Command succeeded: %s", cmd)
            return proc
        except subprocess.CalledProcessError as exc:  # pragma: no cover - requires failing command
            log.warning(
                "Command failed with return code %s on attempt %s/%s",  # type: ignore[str-bytes-safe]
                exc.returncode,
                attempt + 1,
                retries + 1,
            )
            if attempt >= retries:
                log.debug("No retries left; raising error")
                raise
            attempt += 1
            if retry_delay > 0:
                time.sleep(retry_delay)
