from __future__ import annotations

import os

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header

from .nanoslurm import _run, _which


class JobApp(App):
    """Textual app to display current user's SLURM jobs."""

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:  # pragma: no cover - Textual composition
        yield Header()
        self.table: DataTable = DataTable()
        yield self.table
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - runtime hook
        self.table.add_columns("ID", "Name", "State")
        self.refresh_table()
        self.set_interval(2.0, self.refresh_table)

    def refresh_table(self) -> None:  # pragma: no cover - runtime hook
        rows = _list_jobs()
        self.table.clear()
        for row in rows:
            self.table.add_row(*row)


def _list_jobs() -> list[tuple[str, str, str]]:
    """Return a list of (id, name, state) for current user's jobs."""
    if not _which("squeue"):
        return []
    user = os.environ.get("USER", "")
    out = _run(["squeue", "-u", user, "-h", "-o", "%i|%j|%T"], check=False).stdout
    rows: list[tuple[str, str, str]] = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 3:
            rows.append(tuple(parts))
    return rows


__all__ = ["JobApp"]
