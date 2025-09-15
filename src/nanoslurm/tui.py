from __future__ import annotations

from collections import Counter, defaultdict

from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.widgets import DataTable, Footer, Header

from .job import list_jobs
from .stats import fairshare_scores, node_state_counts, partition_utilization

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
"""


class StatsApp(App):
    """Textual app to display cluster statistics."""

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
        self.node_table.add_columns("State", "Nodes", "Percent")
        self.state_table.add_columns("State", "Jobs", "Percent")
        self.partition_table.add_columns("Partition", "Jobs", "Running", "Pending", "Share%", "Util%")
        self.user_table.add_columns("User", "Jobs", "Running", "Pending", "Share%", "FairShare")

        self.refresh_stats()
        self.set_interval(2.0, self.refresh_stats)
        self.set_focus(self.partition_table)

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

    def refresh_stats(self) -> None:  # pragma: no cover - runtime hook
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
                part,
                str(jobs),
                str(running),
                str(pending),
                f"{share:.1f}%",
                f"{util:.1f}%",
            )

        self.user_table.clear()
        for user, cnts in sorted(user_stats.items(), key=lambda x: (-x[1]["jobs"], x[0]))[:5]:
            jobs = cnts["jobs"]
            running = cnts.get("RUNNING", 0)
            pending = cnts.get("PENDING", 0)
            share = jobs / total_jobs * 100
            fs = shares.get(user)
            fs_str = f"{fs:.3f}" if isinstance(fs, float) else "N/A"
            self.user_table.add_row(
                user,
                str(jobs),
                str(running),
                str(pending),
                f"{share:.1f}%",
                fs_str,
            )


__all__ = ["StatsApp"]
