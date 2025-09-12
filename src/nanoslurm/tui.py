from __future__ import annotations

import os
from collections import Counter

from textual.app import App, ComposeResult
from textual.containers import Center
from textual.widgets import DataTable, Footer, Header, TabbedContent, TabPane

from .backend import fairshare_scores, list_jobs

from .backend import list_jobs, node_state_counts, recent_completions
from .backend import list_jobs, partition_utilization, recent_completions

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
                self.node_table = DataTable()
                self.state_table = DataTable()
                self.partition_table = DataTable()
                self.user_table = DataTable()
                yield Center(self.node_table)
                self.state_table = DataTable()
                self.partition_table = DataTable()
                self.user_table = DataTable()
                yield Center(self.state_table)
                yield Center(self.partition_table)
                yield Center(self.user_table)
        yield self.tabs
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - runtime hook
        self.state_table.add_columns("State", "Count", "Percent")
        self.partition_table.add_columns("Partition", "Jobs", "Percent")
        self.user_table.add_columns("User", "Jobs", "Percent", "FairShare")
        self.node_table.add_columns("State", "Nodes", "Percent")
        self.state_table.add_columns("State", "Jobs", "Share%")
        self.partition_table.add_columns("Partition", "Jobs", "Running", "Pending", "Share%")
        self.state_table.add_columns("State", "Jobs", "Share%")
        self.partition_table.add_columns(
            "Partition", "Jobs", "Running", "Pending", "Share%", "Util%"
        )


        self.user_table.add_columns("User", "Jobs", "Running", "Pending", "Share%")
        self.refresh_tables()
        self.set_interval(2.0, self.refresh_tables)

    def refresh_tables(self) -> None:  # pragma: no cover - runtime hook
        node_counts = node_state_counts()
        total_nodes = sum(node_counts.values()) or 1
        node_rows = sorted(
            (state, cnt, round(cnt / total_nodes * 100, 1)) for state, cnt in node_counts.items()
        )

        job_list = list_jobs()
        total = len(job_list) or 1

        state_counts = Counter(job.last_status for job in job_list)
        part_counts = Counter(job.partition for job in job_list)
        user_counts = Counter(job.user for job in job_list)
        shares = fairshare_scores()

        state_rows = sorted(
            (state, cnt, round(cnt / total * 100, 1)) for state, cnt in state_counts.items()
        )
        part_rows = sorted(
            (part, cnt, round(cnt / total * 100, 1)) for part, cnt in part_counts.items()
        )
        user_rows = []
        for user, cnt in sorted(user_counts.items(), key=lambda x: (-x[1], x[0]))[:5]:
            fs = shares.get(user)
            user_rows.append((user, cnt, round(cnt / total * 100, 1), fs))


        part_stats: dict[str, Counter] = {}
        user_stats: dict[str, Counter] = {}
        for job in job_list:
            part = part_stats.setdefault(job.partition, Counter())
            usr = user_stats.setdefault(job.user, Counter())
            part["jobs"] += 1
            usr["jobs"] += 1
            part[job.last_status] += 1
            usr[job.last_status] += 1

        self.node_table.clear()
        for state, count, pct in node_rows:
            self.node_table.add_row(state, str(count), f"{pct:.1f}%")

        try:
            util_map = partition_utilization()
        except Exception:  # pragma: no cover - runtime environment
            util_map = {}


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

            util = util_map.get(part, 0.0)
            self.partition_table.add_row(
                part,
                str(jobs),
                str(running),
                str(pending),
                f"{share:.1f}%",
                f"{util:.1f}%",
            )

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
        for user, count, pct, fs in user_rows:
            fs_str = f"{fs:.3f}" if isinstance(fs, float) else "N/A"
            self.user_table.add_row(user, str(count), f"{pct:.1f}%", fs_str)
        for user, cnts in sorted(user_stats.items(), key=lambda x: (-x[1]["jobs"], x[0])):
            jobs = cnts["jobs"]
            running = cnts.get("RUNNING", 0)
            pending = cnts.get("PENDING", 0)
            share = jobs / total * 100
            self.user_table.add_row(user, str(jobs), str(running), str(pending), f"{share:.1f}%")


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
