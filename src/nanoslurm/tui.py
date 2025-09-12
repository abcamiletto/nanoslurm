from __future__ import annotations

import os
from collections import Counter

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header

from .backend import list_jobs, partition_utilization

BASE_CSS = ""



class JobApp(App):
    """Textual app to display current user's SLURM jobs."""

    CSS = BASE_CSS
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
        rows = list_jobs(os.environ.get("USER"))
        self.table.clear()
        for job in rows:
            self.table.add_row(str(job.id), job.name, job.last_status or job.status)


class ClusterApp(App):
    """Textual app to display cluster-wide job statistics."""

    CSS = BASE_CSS
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
        self.partition_table.add_columns("Partition", "Jobs", "Percent", "Util%")
        self.user_table.add_columns("User", "Jobs", "Percent")
        self.refresh_tables()
        self.set_interval(2.0, self.refresh_tables)

    def refresh_tables(self) -> None:  # pragma: no cover - runtime hook
        job_list = list_jobs()
        total = len(job_list) or 1
        state_counts = Counter(job.last_status for job in job_list)
        part_counts = Counter(job.partition for job in job_list)
        user_counts = Counter(job.user for job in job_list)

        state_rows = sorted(
            (state, cnt, round(cnt / total * 100, 1)) for state, cnt in state_counts.items()
        )
        part_rows = sorted(
            (part, cnt, round(cnt / total * 100, 1)) for part, cnt in part_counts.items()
        )
        try:
            util_map = partition_utilization()
        except Exception:  # pragma: no cover - runtime environment
            util_map = {}
        user_rows = []
        for user, cnt in sorted(user_counts.items(), key=lambda x: (-x[1], x[0]))[:5]:
            user_rows.append((user, cnt, round(cnt / total * 100, 1)))

        self.state_table.clear()
        for state, count, pct in state_rows:
            self.state_table.add_row(state, str(count), f"{pct:.1f}%")
        self.partition_table.clear()
        for part, count, pct in part_rows:
            util = util_map.get(part, 0.0)
            self.partition_table.add_row(part, str(count), f"{pct:.1f}%", f"{util:.1f}%")
        self.user_table.clear()
        for user, count, pct in user_rows:
            self.user_table.add_row(user, str(count), f"{pct:.1f}%")


__all__ = ["JobApp", "ClusterApp"]
