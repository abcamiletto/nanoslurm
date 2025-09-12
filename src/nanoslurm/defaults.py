from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml
from platformdirs import user_config_dir

# Allowed keys and their types for default configuration
KEY_TYPES: Dict[str, type] = {
    "name": str,
    "cluster": str,
    "time": str,
    "cpus": int,
    "memory": int,
    "gpus": int,
    "stdout_file": str,
    "stderr_file": str,
    "signal": str,
    "workdir": str,
}

# Minimal built-in defaults; most values must be supplied via CLI or config
DEFAULTS: Dict[str, object] = {
    "name": "job",
    "stdout_file": "./slurm_logs/%j.txt",
    "stderr_file": "./slurm_logs/%j.err",
    "signal": "SIGUSR1@90",
    "workdir": ".",
}

CONFIG_PATH = Path(user_config_dir("nanoslurm")) / "config.yaml"


def load_defaults() -> Dict[str, object]:
    """Return defaults merged with any saved configuration."""
    data = DEFAULTS.copy()
    if CONFIG_PATH.exists():
        try:
            loaded = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            if isinstance(loaded, dict):
                data.update(loaded)
        except Exception:
            pass
    return data


def save_defaults(cfg: Dict[str, object]) -> None:
    """Persist provided configuration to :data:`CONFIG_PATH`."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))


KEY_HELP = ", ".join(f"{k} ({t.__name__})" for k, t in KEY_TYPES.items())
