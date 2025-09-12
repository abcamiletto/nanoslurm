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


class ClusterApp(App):
    """Textual app to display cluster-wide job statistics."""

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:  # pragma: no cover - Textual composition
        yield Header()
        self.state_table: DataTable = DataTable()
        yield self.state_table
        self.partition_table: DataTable = DataTable()
        yield self.partition_table
        self.user_table: DataTable = DataTable()
        yield self.user_table
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - runtime hook
        self.state_table.add_columns("State", "Count", "Percent")
        self.partition_table.add_columns("Partition", "Jobs", "Percent")
        self.user_table.add_columns("User", "Jobs", "Percent")
        self.refresh_tables()
        self.set_interval(2.0, self.refresh_tables)

    def refresh_tables(self) -> None:  # pragma: no cover - runtime hook
        states = _cluster_job_state_counts()
        partitions = _cluster_partition_counts()
        users = _cluster_top_users()
        self.state_table.clear()
        for state, count, pct in states:
            self.state_table.add_row(state, str(count), f"{pct:.1f}%")
        self.partition_table.clear()
        for part, count, pct in partitions:
            self.partition_table.add_row(part, str(count), f"{pct:.1f}%")
        self.user_table.clear()
        for user, count, pct in users:
            self.user_table.add_row(user, str(count), f"{pct:.1f}%")


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


def _cluster_job_state_counts() -> list[tuple[str, int, float]]:
    """Return a list of (state, count, percent) for all jobs on the cluster."""
    if not _which("squeue"):
        raise SlurmUnavailableError("squeue command not found on PATH")
    out = _run(["squeue", "-h", "-o", "%T"], check=False).stdout
    counts: dict[str, int] = {}
    for line in out.splitlines():
        line = line.strip()
        if line:
            counts[line] = counts.get(line, 0) + 1
    total = sum(counts.values()) or 1
    return sorted(
        [(state, count, round(count / total * 100, 1)) for state, count in counts.items()],
        key=lambda x: x[0],
    )


def _cluster_top_users(limit: int = 5) -> list[tuple[str, int, float]]:
    """Return the top users by job count."""
    if not _which("squeue"):
        raise SlurmUnavailableError("squeue command not found on PATH")
    out = _run(["squeue", "-h", "-o", "%u"], check=False).stdout
    counts: dict[str, int] = {}
    for line in out.splitlines():
        line = line.strip()
        if line:
            counts[line] = counts.get(line, 0) + 1
    total = sum(counts.values()) or 1
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    result: list[tuple[str, int, float]] = []
    for user, count in items[:limit]:
        result.append((user, count, round(count / total * 100, 1)))
    return result


def _cluster_partition_counts() -> list[tuple[str, int, float]]:
    """Return a list of (partition, count, percent) for all jobs."""
    if not _which("squeue"):
        raise SlurmUnavailableError("squeue command not found on PATH")
    out = _run(["squeue", "-h", "-o", "%P"], check=False).stdout
    counts: dict[str, int] = {}
    for line in out.splitlines():
        line = line.strip()
        if line:
            counts[line] = counts.get(line, 0) + 1
    total = sum(counts.values()) or 1
    return sorted(
        [
            (part, count, round(count / total * 100, 1))
            for part, count in counts.items()
        ],
        key=lambda x: x[0],
    )


__all__ = ["JobApp", "ClusterApp"]
