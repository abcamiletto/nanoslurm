from __future__ import annotations

import shlex
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .defaults import (
    CONFIG_PATH,
    DEFAULTS,
    KEY_HELP,
    KEY_TYPES,
    load_defaults,
    save_defaults,
)
from .job import submit

app = typer.Typer(help="Submit and manage jobs with nanoslurm")
console = Console()


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
    defaults = load_defaults()
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


@app.command("monitor")
def monitor() -> None:
    """Launch the job monitor TUI."""
    from .tui import MonitorApp

    MonitorApp().run()


defaults_app = typer.Typer(help="Manage default settings")
app.add_typer(defaults_app, name="defaults")


@defaults_app.command("show")
def defaults_show() -> None:
    """Display current defaults."""
    cfg = load_defaults()
    table = Table("key", "value")
    for k, v in cfg.items():
        table.add_row(k, str(v))
    console.print(table)


@defaults_app.command("set")
def defaults_set(
    key: str = typer.Argument(..., help=f"Configuration key to set. Options: {KEY_HELP}"),
    value: str = typer.Argument(..., help="Value to store for the given key"),
) -> None:
    """Set a default value.

    Provides clearer feedback if an unknown key is supplied or if the value
    cannot be converted to the expected type.
    """
    if key not in KEY_TYPES:
        allowed = ", ".join(KEY_TYPES)
        raise typer.BadParameter(f"Unknown key: {key}. Allowed keys: {allowed}")
    cfg = load_defaults()
    typ = KEY_TYPES[key]
    if typ is int:
        try:
            cfg[key] = int(value)
        except ValueError:
            raise typer.BadParameter(f"{key} expects type int") from None
    else:
        cfg[key] = value
    save_defaults(cfg)
    console.print(f"[green]{key} set to {cfg[key]}[/green]")


@defaults_app.command("reset")
def defaults_reset() -> None:
    """Clear all saved defaults."""
    if CONFIG_PATH.exists():
        CONFIG_PATH.unlink()
    save_defaults(DEFAULTS.copy())
    console.print("[green]Defaults reset[/green]")


@defaults_app.command("edit")
def defaults_edit() -> None:
    """Edit defaults in your configured editor."""
    content = yaml.safe_dump(load_defaults(), sort_keys=False)
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
    save_defaults(data)
    console.print("[green]Defaults updated[/green]")


if __name__ == "__main__":
    app()
