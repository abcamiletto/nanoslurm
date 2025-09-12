from __future__ import annotations

import shlex
from pathlib import Path
from typing import Optional

import typer
import yaml
from platformdirs import user_config_dir
from rich.console import Console
from rich.table import Table

from .nanoslurm import submit

app = typer.Typer(help="Submit and manage jobs with nanoslurm")
console = Console()

# Allowed keys and their types for default configuration
KEY_TYPES: dict[str, type] = {
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
DEFAULTS: dict[str, object] = {
    "name": "job",
    "stdout_file": "./slurm_logs/%j.txt",
    "stderr_file": "./slurm_logs/%j.err",
    "signal": "SIGUSR1@90",
    "workdir": ".",
}

CONFIG_PATH = Path(user_config_dir("nanoslurm")) / "config.yaml"


def _load_defaults() -> dict[str, object]:
    data = DEFAULTS.copy()
    if CONFIG_PATH.exists():
        try:
            loaded = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            if isinstance(loaded, dict):
                data.update(loaded)
        except Exception:
            pass
    return data


def _save_defaults(cfg: dict[str, object]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))


@app.command()
def run(
    command: Optional[list[str]] = typer.Argument(None, help="Command to execute", show_default=False),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Base job name"),
    cluster: Optional[str] = typer.Option(None, "--cluster", "-c", help="SLURM partition"),
    time: Optional[str] = typer.Option(None, "--time", "-t", help="HH:MM:SS time limit"),
    cpus: Optional[int] = typer.Option(None, "--cpus", "-p", help="CPU cores"),
    memory: Optional[int] = typer.Option(None, "--memory", "-m", help="Memory in GB"),
    gpus: Optional[int] = typer.Option(None, "--gpus", "-g", help="GPUs"),
    stdout_file: Optional[str] = typer.Option(None, "--stdout-file", "-o", help="Stdout file"),
    stderr_file: Optional[str] = typer.Option(None, "--stderr-file", "-e", help="Stderr file"),
    signal: Optional[str] = typer.Option(None, "--signal", "-s", help="Signal spec"),
    workdir: Optional[str] = typer.Option(None, "--workdir", "-w", help="Working directory"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Prompt for missing values interactively"),
) -> None:
    """Submit a job using nanoslurm."""
    defaults = _load_defaults()
    values: dict[str, object] = {
        "name": name or defaults.get("name"),
        "cluster": cluster or defaults.get("cluster"),
        "time": time or defaults.get("time"),
        "cpus": cpus or defaults.get("cpus"),
        "memory": memory or defaults.get("memory"),
        "gpus": gpus or defaults.get("gpus"),
        "stdout_file": stdout_file or defaults.get("stdout_file"),
        "stderr_file": stderr_file or defaults.get("stderr_file"),
        "signal": signal or defaults.get("signal"),
        "workdir": workdir or defaults.get("workdir"),
    }

    if interactive:
        if not command:
            cmd_str = typer.prompt("command")
            command = shlex.split(cmd_str)
        for key, val in list(values.items()):
            if val is None:
                prompt = key.replace("_", " ")
                if KEY_TYPES[key] is int:
                    values[key] = typer.prompt(prompt, type=int)
                else:
                    values[key] = typer.prompt(prompt)
    else:
        if not command:
            raise typer.BadParameter("COMMAND required unless --interactive is used")
        missing = [k for k, v in values.items() if v is None]
        if missing:
            raise typer.BadParameter(f"Missing options: {', '.join(missing)}")

    job = submit(command, **values)  # type: ignore[arg-type]
    console.print(f"[green]Submitted job {job.id} ({job.name})[/green]")
    if job.stdout_path:
        console.print(f"stdout: {job.stdout_path}")
    if job.stderr_path:
        console.print(f"stderr: {job.stderr_path}")


defaults_app = typer.Typer(help="Manage default settings")
app.add_typer(defaults_app, name="defaults")


@defaults_app.command("show")
def defaults_show() -> None:
    """Display current defaults."""
    cfg = _load_defaults()
    table = Table("key", "value")
    for k, v in cfg.items():
        table.add_row(k, str(v))
    console.print(table)


@defaults_app.command("set")
def defaults_set(key: str, value: str) -> None:
    """Set a default value."""
    if key not in KEY_TYPES:
        raise typer.BadParameter(f"Unknown key: {key}")
    cfg = _load_defaults()
    typ = KEY_TYPES[key]
    cfg[key] = typ(value) if typ is int else value
    _save_defaults(cfg)
    console.print(f"[green]{key} set to {cfg[key]}[/green]")


@defaults_app.command("reset")
def defaults_reset() -> None:
    """Clear all saved defaults."""
    if CONFIG_PATH.exists():
        CONFIG_PATH.unlink()
    _save_defaults(DEFAULTS.copy())
    console.print("[green]Defaults reset[/green]")


@defaults_app.command("edit")
def defaults_edit() -> None:
    """Edit defaults in your configured editor."""
    content = yaml.safe_dump(_load_defaults(), sort_keys=False)
    result = typer.edit(content, extension=".yaml")
    if result is None:
        console.print("[yellow]No changes made[/yellow]")
        raise typer.Exit()
    try:
        data = yaml.safe_load(result) or {}
        if not isinstance(data, dict):
            raise ValueError
    except Exception as exc:
        console.print(f"[red]Invalid YAML: {exc}[/red]")
        raise typer.Exit(code=1)
    _save_defaults(data)
    console.print("[green]Defaults updated[/green]")


if __name__ == "__main__":
    app()

