from __future__ import annotations

import os
from collections import Counter

from textual.app import App, ComposeResult
from textual.containers import Center
from textual.widgets import DataTable, Footer, Header, TabbedContent, TabPane

from .backend import list_jobs

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
        self.tabs = TabbedContent()
        self.partition_tables: dict[str, DataTable] = {}
        with self.tabs:
            with TabPane("Summary"):
                self.state_table = DataTable()
                self.partition_table = DataTable()
                self.user_table = DataTable()
                yield Center(self.state_table)
                yield Center(self.partition_table)
                yield Center(self.user_table)
        yield self.tabs
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - runtime hook
        self.state_table.add_columns("State", "Jobs", "Share%")
        self.partition_table.add_columns("Partition", "Jobs", "Running", "Pending", "Share%")
        self.user_table.add_columns("User", "Jobs", "Running", "Pending", "Share%")
        self.refresh_tables()
        self.set_interval(2.0, self.refresh_tables)

    def refresh_tables(self) -> None:  # pragma: no cover - runtime hook
        job_list = list_jobs()
        total = len(job_list) or 1

        state_counts = Counter(job.last_status for job in job_list)

        part_stats: dict[str, Counter] = {}
        user_stats: dict[str, Counter] = {}
        for job in job_list:
            part = part_stats.setdefault(job.partition, Counter())
            usr = user_stats.setdefault(job.user, Counter())
            part["jobs"] += 1
            usr["jobs"] += 1
            part[job.last_status] += 1
            usr[job.last_status] += 1

        self.state_table.clear()
        for state, cnt in sorted(state_counts.items()):
            self.state_table.add_row(state, str(cnt), f"{cnt / total * 100:.1f}%")

        self.partition_table.clear()
        for part, cnts in sorted(part_stats.items()):
            jobs = cnts["jobs"]
            running = cnts.get("RUNNING", 0)
            pending = cnts.get("PENDING", 0)
            share = jobs / total * 100
            self.partition_table.add_row(part, str(jobs), str(running), str(pending), f"{share:.1f}%")

            if part not in self.partition_tables:
                table = DataTable()
                table.add_columns("User", "Jobs", "Running", "Pending", "Share%")
                pane = TabPane(part, Center(table))
                self.tabs.add_pane(pane)
                self.partition_tables[part] = table

        for part, table in self.partition_tables.items():
            u_stats: dict[str, Counter] = {}
            for job in job_list:
                if job.partition != part:
                    continue
                stats = u_stats.setdefault(job.user, Counter())
                stats["jobs"] += 1
                stats[job.last_status] += 1
            total_part = sum(s["jobs"] for s in u_stats.values()) or 1
            table.clear()
            for user, cnts in sorted(u_stats.items(), key=lambda x: (-x[1]["jobs"], x[0])):
                jobs = cnts["jobs"]
                running = cnts.get("RUNNING", 0)
                pending = cnts.get("PENDING", 0)
                share = jobs / total_part * 100
                table.add_row(user, str(jobs), str(running), str(pending), f"{share:.1f}%")

        self.user_table.clear()
        for user, cnts in sorted(user_stats.items(), key=lambda x: (-x[1]["jobs"], x[0])):
            jobs = cnts["jobs"]
            running = cnts.get("RUNNING", 0)
            pending = cnts.get("PENDING", 0)
            share = jobs / total * 100
            self.user_table.add_row(user, str(jobs), str(running), str(pending), f"{share:.1f}%")


__all__ = ["JobApp", "ClusterApp"]
