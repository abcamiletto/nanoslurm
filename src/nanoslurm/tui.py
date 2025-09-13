from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import timedelta

from textual.app import App, ComposeResult
from textual.containers import Grid, Vertical
from textual.widgets import DataTable, Footer, Header, TabbedContent, TabPane

from .job import list_jobs
from .stats import (
    fairshare_scores,
    job_history,
    node_state_counts,
    partition_node_state_counts,
    partition_utilization,
    recent_completions,
)

BASE_CSS = """
Screen {
    padding: 1;
}
#summary-grid {
    layout: grid;
    grid-size: 2;
    grid-gutter: 1;
}
#summary-grid DataTable {
    width: 100%;
}
.partition-pane {
    layout: vertical;
}
.partition-pane DataTable {
    width: 100%;
}
"""


class DashboardApp(App):
    """Combined TUI showing user jobs and cluster statistics."""

    CSS = BASE_CSS
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("h", "cursor_left", "Left"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("l", "cursor_right", "Right"),
    ]

    def __init__(self, **kwargs):
        kwargs.setdefault("ansi_color", True)
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:  # pragma: no cover - Textual composition
        yield Header()
        with TabbedContent() as self.main_tabs:
            with TabPane("Jobs"):
                self.job_table = DataTable()
                yield self.job_table
            with TabPane("Cluster"):
                self.cluster_tabs = TabbedContent()
                with self.cluster_tabs:
                    with TabPane("Summary"):
                        with Grid(id="summary-grid"):
                            self.partition_table = DataTable()
                            self.user_table = DataTable()
                            self.state_table = DataTable()
                            self.node_table = DataTable()
                            yield self.partition_table
                            yield self.user_table
                            yield self.state_table
                            yield self.node_table
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - runtime hook
        self.job_table.add_columns("ID", "Name", "State")
        self.job_table.show_cursor = True
        self.job_table.cursor_type = "row"

        self.node_table.add_columns("State", "Nodes", "Percent")
        self.state_table.add_columns("State", "Jobs", "Percent")
        self.partition_table.add_columns("Partition", "Jobs", "Running", "Pending", "Share%", "Util%")
        self.user_table.add_columns(
            "User",
            "Jobs",
            "Running",
            "Pending",
            "Share%",
            "FairShare",
            "Succeeded (24h)",
            "Failed (24h)",
        )

        self.partition_tables: dict[str, dict[str, DataTable]] = {}

        self.refresh_jobs()
        self.refresh_cluster()
        self.set_interval(2.0, self.refresh_jobs)
        self.set_interval(2.0, self.refresh_cluster)
        self.set_focus(self.job_table)

    def _focused_table(self) -> DataTable | None:
        widget = self.focused
        return widget if isinstance(widget, DataTable) else None

    def action_cursor_left(self) -> None:  # pragma: no cover - Textual action
        if (table := self._focused_table()) is not None:
            table.action_cursor_left()

    def action_cursor_right(self) -> None:  # pragma: no cover - Textual action
        if (table := self._focused_table()) is not None:
            table.action_cursor_right()

    def action_cursor_up(self) -> None:  # pragma: no cover - Textual action
        if (table := self._focused_table()) is not None:
            table.action_cursor_up()

    def action_cursor_down(self) -> None:  # pragma: no cover - Textual action
        if (table := self._focused_table()) is not None:
            table.action_cursor_down()

    def refresh_jobs(self) -> None:  # pragma: no cover - runtime hook
        rows = list_jobs(os.environ.get("USER"))
        self.job_table.clear()
        for job in rows:
            self.job_table.add_row(str(job.id), job.name, job.last_status or job.status)

    def refresh_cluster(self) -> None:  # pragma: no cover - runtime hook
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

        try:
            node_map = partition_node_state_counts()
        except Exception:  # pragma: no cover - runtime environment
            node_map = {}

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
                stats_table = DataTable()
                stats_table.add_columns("Metric", "Value")
                user_table = DataTable()
                user_table.add_columns("User", "Jobs", "Running", "Pending", "Share%")
                pane = TabPane(part, Vertical(stats_table, user_table, classes="partition-pane"))
                self.cluster_tabs.add_pane(pane)
                self.partition_tables[part] = {"stats": stats_table, "users": user_table}

        for part, tables in self.partition_tables.items():
            u_stats: defaultdict[str, Counter] = defaultdict(Counter)
            for job in job_list:
                if job.partition != part:
                    continue
                u_stats[job.user]["jobs"] += 1
                u_stats[job.user][job.last_status] += 1
            total_part = sum(s["jobs"] for s in u_stats.values()) or 1
            user_table = tables["users"]
            user_table.clear()
            for user, cnts in sorted(u_stats.items(), key=lambda x: (-x[1]["jobs"], x[0])):
                jobs = cnts["jobs"]
                running = cnts.get("RUNNING", 0)
                pending = cnts.get("PENDING", 0)
                share = jobs / total_part * 100
                user_table.add_row(user, str(jobs), str(running), str(pending), f"{share:.1f}%")

            stats_table = tables["stats"]
            stats_table.clear()
            counts = node_map.get(part)
            if counts:
                free_nodes = counts.get("IDLE", 0)
                total_part_nodes = sum(counts.values())
                stats_table.add_row("free nodes", str(free_nodes))
                stats_table.add_row("total nodes", str(total_part_nodes))
            else:
                stats_table.add_row("free nodes", "unavailable")
            util = util_map.get(part)
            if util is not None:
                stats_table.add_row("util%", f"{util:.1f}%")
            else:
                stats_table.add_row("util%", "unavailable")

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

    def __init__(self, **kwargs):
        kwargs.setdefault("ansi_color", True)
        super().__init__(**kwargs)

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


__all__ = ["DashboardApp", "SummaryApp"]

