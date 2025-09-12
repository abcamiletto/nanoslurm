from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import timedelta

from textual.app import App, ComposeResult
from textual.containers import Center
from textual.widgets import DataTable, Footer, Header, TabbedContent, TabPane

from .backend import (
    fairshare_scores,
    job_history,
    list_jobs,
    node_state_counts,
    partition_utilization,
    recent_completions,
)

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
        with self.tabs:
            with TabPane("Summary"):
                self.node_table = DataTable()
                self.state_table = DataTable()
                self.partition_table = DataTable()
                self.user_table = DataTable()
                yield Center(self.node_table)
                yield Center(self.state_table)
                yield Center(self.partition_table)
                yield Center(self.user_table)
        yield self.tabs
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - runtime hook
        self.node_table.add_columns("State", "Nodes", "Percent")
        self.state_table.add_columns("State", "Jobs", "Percent")
        self.partition_table.add_columns("Partition", "Jobs", "Running", "Pending", "Share%", "Util%")
        self.user_table.add_columns(
            "User", "Jobs", "Running", "Pending", "Share%", "FairShare", "Succeeded (24h)", "Failed (24h)"
        )
        self.partition_tables: dict[str, DataTable] = {}
        self.refresh_tables()
        self.set_interval(2.0, self.refresh_tables)

    def refresh_tables(self) -> None:  # pragma: no cover - runtime hook
        node_counts = node_state_counts()
        total_nodes = sum(node_counts.values()) or 1

        job_list = list_jobs()
        total_jobs = len(job_list) or 1

        state_counts = Counter(job.last_status for job in job_list)

        part_stats: defaultdict[str, Counter] = defaultdict(Counter)
        user_stats: defaultdict[str, Counter] = defaultdict(Counter)
        for job in job_list:
            part_stats[job.partition]["jobs"] += 1
            part_stats[job.partition][job.last_status] += 1
            user_stats[job.user]["jobs"] += 1
            user_stats[job.user][job.last_status] += 1

        try:
            util_map = partition_utilization()
        except Exception:  # pragma: no cover - runtime environment
            util_map = {}

        shares = fairshare_scores()
        history = job_history()

        self.node_table.clear()
        for state, count in sorted(node_counts.items()):
            pct = count / total_nodes * 100
            self.node_table.add_row(state, str(count), f"{pct:.1f}%")

        self.state_table.clear()
        for state, cnt in sorted(state_counts.items()):
            pct = cnt / total_jobs * 100
            self.state_table.add_row(state, str(cnt), f"{pct:.1f}%")

        self.partition_table.clear()
        for part, stats in sorted(part_stats.items()):
            jobs = stats["jobs"]
            running = stats.get("RUNNING", 0)
            pending = stats.get("PENDING", 0)
            share = jobs / total_jobs * 100
            util = util_map.get(part, 0.0)
            self.partition_table.add_row(
                part, str(jobs), str(running), str(pending), f"{share:.1f}%", f"{util:.1f}%"
            )
            if part not in self.partition_tables:
                table = DataTable()
                table.add_columns("User", "Jobs", "Running", "Pending", "Share%")
                pane = TabPane(part, Center(table))
                self.tabs.add_pane(pane)
                self.partition_tables[part] = table

        for part, table in self.partition_tables.items():
            u_stats: defaultdict[str, Counter] = defaultdict(Counter)
            for job in job_list:
                if job.partition != part:
                    continue
                u_stats[job.user]["jobs"] += 1
                u_stats[job.user][job.last_status] += 1
            total_part = sum(s["jobs"] for s in u_stats.values()) or 1
            table.clear()
            for user, cnts in sorted(u_stats.items(), key=lambda x: (-x[1]["jobs"], x[0])):
                jobs = cnts["jobs"]
                running = cnts.get("RUNNING", 0)
                pending = cnts.get("PENDING", 0)
                share = jobs / total_part * 100
                table.add_row(user, str(jobs), str(running), str(pending), f"{share:.1f}%")

        self.user_table.clear()
        for user, cnts in sorted(user_stats.items(), key=lambda x: (-x[1]["jobs"], x[0]))[:5]:
            jobs = cnts["jobs"]
            running = cnts.get("RUNNING", 0)
            pending = cnts.get("PENDING", 0)
            share = jobs / total_jobs * 100
            fs = shares.get(user)
            fs_str = f"{fs:.3f}" if isinstance(fs, float) else "N/A"
            hist = history.get(user, {"completed": 0, "failed": 0})
            self.user_table.add_row(
                user,
                str(jobs),
                str(running),
                str(pending),
                f"{share:.1f}%",
                fs_str,
                str(hist.get("completed", 0)),
                str(hist.get("failed", 0)),
            )


class SummaryApp(App):
    """Textual app to display recent job completions."""

    CSS = BASE_CSS
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:  # pragma: no cover - Textual composition
        yield Header()
        self.day_table: DataTable = DataTable()
        yield self.day_table
        self.week_table: DataTable = DataTable()
        yield self.week_table
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - runtime hook
        self.day_table.add_columns("Day", "Jobs", "Spark")
        self.week_table.add_columns("Week", "Jobs", "Spark")
        self.refresh_tables()
        self.set_interval(60.0, self.refresh_tables)

    def refresh_tables(self) -> None:  # pragma: no cover - runtime hook
        day_rows = recent_completions("day", 7)
        week_rows = recent_completions("week", 8)

        def _add_rows(table: DataTable, rows: list[tuple[str, int]]) -> None:
            table.clear()
            if not rows:
                return
            max_count = max(cnt for _, cnt in rows) or 1
            levels = "▁▂▃▄▅▆▇█"
            for label, cnt in rows:
                idx = int(cnt / max_count * (len(levels) - 1))
                table.add_row(label, str(cnt), levels[idx])

        _add_rows(self.day_table, day_rows)
        _add_rows(self.week_table, week_rows)


__all__ = ["JobApp", "ClusterApp", "SummaryApp"]

