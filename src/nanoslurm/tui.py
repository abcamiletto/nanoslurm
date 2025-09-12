from __future__ import annotations

import os

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header

from .nanoslurm import SlurmUnavailableError, _run, _which


class JobApp(App):
    """Textual app to display current user's SLURM jobs."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("h", "cursor_left", "Left"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("l", "cursor_right", "Right"),
    ]

    def compose(self) -> ComposeResult:  # pragma: no cover - Textual composition
        yield Header()
        self.table: DataTable = DataTable()
        yield self.table
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - runtime hook
        self.table.add_columns("ID", "Name", "State")
        self.table.show_cursor = True
        self.table.cursor_type = "row"
        self.refresh_table()
        self.set_interval(2.0, self.refresh_table)
        self.set_focus(self.table)

    def action_cursor_left(self) -> None:  # pragma: no cover - Textual action
        self.table.action_cursor_left()

    def action_cursor_right(self) -> None:  # pragma: no cover - Textual action
        self.table.action_cursor_right()

    def action_cursor_up(self) -> None:  # pragma: no cover - Textual action
        self.table.action_cursor_up()

    def action_cursor_down(self) -> None:  # pragma: no cover - Textual action
        self.table.action_cursor_down()

    def refresh_table(self) -> None:  # pragma: no cover - runtime hook
        rows = _list_jobs()
        self.table.clear()
        for row in rows:
            self.table.add_row(*row)


def _list_jobs() -> list[tuple[str, str, str]]:
    """Return a list of (id, name, state) for current user's jobs."""
    if not _which("squeue"):
        raise SlurmUnavailableError("squeue command not found on PATH")
    user = os.environ.get("USER", "")
    out = _run(["squeue", "-u", user, "-h", "-o", "%i|%j|%T"], check=False).stdout
    rows: list[tuple[str, str, str]] = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 3:
            rows.append(tuple(parts))
    return rows


__all__ = ["JobApp"]
