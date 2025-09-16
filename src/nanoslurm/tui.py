"""Textual monitor for viewing SLURM jobs managed by :mod:`nanoslurm`."""

from __future__ import annotations

from datetime import datetime, timedelta

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Static,
    TabbedContent,
    TabPane,
)

from .job import Job, list_jobs

LOGO = r"""
███╗   ██╗ █████╗ ███╗   ██╗ ██████╗ ███████╗██╗     ██╗   ██╗██████╗ ███╗   ███╗
████╗  ██║██╔══██╗████╗  ██║██╔═══██╗██╔════╝██║     ██║   ██║██╔══██╗████╗ ████║
██╔██╗ ██║███████║██╔██╗ ██║██║   ██║███████╗██║     ██║   ██║██████╔╝██╔████╔██║
██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║╚════██║██║     ██║   ██║██╔══██╗██║╚██╔╝██║
██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝███████║███████╗╚██████╔╝██║  ██║██║ ╚═╝ ██║
╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝
"""


def _format_datetime(value: datetime | None) -> str:
    if not value:
        return "-"
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return str(timedelta(seconds=int(seconds)))


def _partition_tab_id(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name)
    slug = "-".join(filter(None, slug.split("-")))
    return f"partition-{slug or 'default'}"


class WelcomeScreen(Screen[None]):
    """Landing screen that displays an ASCII logo and navigation options."""

    BINDINGS = [
        ("j", "show_jobs", "Monitor jobs"),
        ("p", "show_partitions", "Partition stats"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="welcome"):
            yield Static(LOGO, id="logo")
            yield Static("Welcome to the nanoslurm monitor", id="tagline")
            with Horizontal(id="menu"):
                yield Button("Monitor jobs", id="show-jobs", variant="primary")
                yield Button("Partition stats", id="show-partitions")
        yield Footer()

    @on(Button.Pressed, "#show-jobs")
    def _on_show_jobs_pressed(self, _: Button.Pressed) -> None:
        self.app.action_show_jobs()

    @on(Button.Pressed, "#show-partitions")
    def _on_show_partitions_pressed(self, _: Button.Pressed) -> None:
        self.app.action_show_partitions()

    def action_show_jobs(self) -> None:
        self.app.action_show_jobs()

    def action_show_partitions(self) -> None:
        self.app.action_show_partitions()


class JobsScreen(Screen[None]):
    """Screen that displays live job information."""

    BINDINGS = [("r", "refresh_now", "Refresh")]

    def __init__(self) -> None:
        super().__init__()
        self._jobs: dict[str, Job] = {}
        self._selected_key: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="jobs-root"):
            self.summary = Static("Loading jobs…", id="job-summary")
            yield self.summary
            with Horizontal(id="job-layout"):
                self.table = DataTable(id="jobs-table")
                yield self.table
                self.details = Static("Select a job to view details.", id="job-details")
                yield self.details
        yield Footer()

    def on_mount(self) -> None:
        self.table.add_columns("ID", "Name", "Partition", "State")
        self.table.cursor_type = "row"
        self.set_interval(2.0, self.refresh_jobs)
        self.refresh_jobs()

    def on_show(self) -> None:
        self.table.focus()

    def action_refresh_now(self) -> None:
        self.refresh_jobs()

    def refresh_jobs(self) -> None:
        jobs = list_jobs()
        self._jobs = {str(job.id): job for job in jobs}

        self.table.clear()
        for job in jobs:
            key = str(job.id)
            self.table.add_row(
                str(job.id),
                job.name,
                job.partition or "-",
                job.last_status or "UNKNOWN",
                key=key,
            )

        self._update_summary(jobs)

        if not jobs:
            self.details.update("No jobs in the queue.")
            self._selected_key = None
            return

        previous = self._selected_key if self._selected_key in self._jobs else str(jobs[0].id)
        self._selected_key = previous

        row_index = next((idx for idx, job in enumerate(jobs) if str(job.id) == previous), 0)
        self.table.move_cursor(row=row_index)
        self._update_details(self._jobs[previous])

    @on(DataTable.RowHighlighted, "#jobs-table")
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        key = event.row_key.value if event.row_key else None
        if not key:
            return
        job = self._jobs.get(key)
        if job is None:
            return
        self._selected_key = key
        self._update_details(job)

    def _update_summary(self, jobs: list[Job]) -> None:
        if not jobs:
            self.summary.update("No jobs found.")
            return
        running = sum(1 for job in jobs if job.last_status == "RUNNING")
        pending = sum(1 for job in jobs if job.last_status == "PENDING")
        others = len(jobs) - running - pending
        self.summary.update(
            f"Jobs: {len(jobs)} · Running: {running} · Pending: {pending} · Other: {others}"
        )

    def _update_details(self, job: Job) -> None:
        lines = [
            f"[b]{job.name}[/b]",
            "",
            f"ID: {job.id}",
            f"User: {job.user or '-'}",
            f"Partition: {job.partition or '-'}",
            f"State: {job.last_status or 'UNKNOWN'}",
            f"Submitted: {_format_datetime(job.submit_time)}",
            f"Started: {_format_datetime(job.start_time)}",
        ]
        wait = job.wait_time
        if wait is not None:
            lines.append(f"Queue wait: {_format_duration(wait)}")
        self.details.update("\n".join(lines))


class PartitionScreen(Screen[None]):
    """Screen that shows per-user and per-partition cluster statistics."""

    BINDINGS = [("r", "refresh_now", "Refresh")]

    def __init__(self) -> None:
        super().__init__()
        self._active_tab: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="partition-root"):
            self.status = Static("Gathering cluster statistics…", id="partition-status")
            yield self.status
            with Horizontal(id="partition-layout"):
                with Vertical(id="user-column"):
                    yield Static("User activity", id="user-title")
                    self.user_table = DataTable(id="user-table")
                    yield self.user_table
                with Vertical(id="partition-column"):
                    self.tabs = TabbedContent(id="partition-tabs")
                    yield self.tabs
        yield Footer()

    async def on_mount(self) -> None:
        self.user_table.add_columns("User", "Total", "Running", "Pending")
        self.set_interval(5.0, self.refresh_data)
        await self.refresh_data()

    async def action_refresh_now(self) -> None:
        await self.refresh_data()

    async def refresh_data(self) -> None:
        jobs = list_jobs()

        user_stats: dict[str, dict[str, int]] = {}
        partition_counts: dict[str, dict[str, int]] = {}
        partition_users: dict[str, set[str]] = {}

        for job in jobs:
            user = job.user or "(unknown)"
            part_key = job.partition or ""
            status = (job.last_status or "UNKNOWN").upper()

            user_entry = user_stats.setdefault(user, {"total": 0, "running": 0, "pending": 0})
            user_entry["total"] += 1
            if status == "RUNNING":
                user_entry["running"] += 1
            elif status == "PENDING":
                user_entry["pending"] += 1

            part_entry = partition_counts.setdefault(
                part_key, {"running": 0, "pending": 0, "other": 0}
            )
            if status == "RUNNING":
                part_entry["running"] += 1
            elif status == "PENDING":
                part_entry["pending"] += 1
            else:
                part_entry["other"] += 1

            partition_users.setdefault(part_key, set()).add(user)

        self.user_table.clear()
        for user in sorted(user_stats):
            counts = user_stats[user]
            self.user_table.add_row(
                user,
                str(counts["total"]),
                str(counts["running"]),
                str(counts["pending"]),
            )

        total_jobs = len(jobs)
        if total_jobs == 0:
            self.status.update("No jobs detected on the cluster.")
        else:
            partitions = len(partition_counts) or 0
            label = "partition" if partitions == 1 else "partitions"
            self.status.update(f"Tracking {total_jobs} jobs across {partitions} {label}.")

        await self.tabs.clear_panes()

        if not partition_counts:
            placeholder = TabPane(
                "Partitions",
                Static("No partition data available.", classes="partition-details"),
                id="partition-placeholder",
            )
            await self.tabs.add_pane(placeholder)
            self.tabs.active = "partition-placeholder"
            self._active_tab = "partition-placeholder"
            return

        available_ids: list[str] = []
        for part_key in sorted(partition_counts):
            label = part_key or "(none)"
            tab_id = _partition_tab_id(label)
            available_ids.append(tab_id)
            summary = self._format_partition_summary(
                label,
                partition_counts[part_key],
                partition_users.get(part_key, set()),
            )
            pane = TabPane(label, Static(summary, classes="partition-details"), id=tab_id)
            await self.tabs.add_pane(pane)

        current = self._active_tab if self._active_tab in available_ids else available_ids[0]
        self._active_tab = current
        self.tabs.active = current

    @on(TabbedContent.TabActivated, "#partition-tabs")
    def _on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self._active_tab = event.tab.id if event.tab else None

    def _format_partition_summary(
        self,
        label: str,
        counts: dict[str, int],
        users: set[str],
    ) -> str:
        total = counts["running"] + counts["pending"] + counts["other"]
        lines = [
            f"[b]{label}[/b]",
            "",
            f"Total jobs: {total}",
            f"Running: {counts['running']}",
            f"Pending: {counts['pending']}",
        ]
        if counts["other"]:
            lines.append(f"Other states: {counts['other']}")
        lines.extend(
            [
                "",
                f"Active users: {len(users)}",
            ]
        )
        return "\n".join(lines)


class MonitorApp(App[None]):
    """Application orchestrating the monitor screens."""

    def __init__(self, **kwargs):
        kwargs.setdefault("ansi_color", True)
        super().__init__(**kwargs)

    TITLE = LOGO
    SUB_TITLE = ""

    CSS = """
    Screen {
        padding: 1;
    }

    #welcome {
        height: 1fr;
        align: center middle;
    }

    #logo {
        content-align: center middle;
        text-style: bold;
    }

    #tagline {
        content-align: center middle;
        color: $text-muted;
        margin-top: 1;
    }

    #menu {
        content-align: center middle;
    }

    Button {
        min-width: 24;
        margin: 0 1;
    }

    #jobs-root, #partition-root {
        height: 1fr;
    }

    #job-layout, #partition-layout {
        height: 1fr;
    }

    #jobs-table, #user-table {
        width: 1.3fr;
    }

    #job-details, TabPane {
        border: panel;
        padding: 1 2;
        width: 1fr;
        height: 1fr;
        overflow-y: auto;
    }

    #user-column, #partition-column {
        width: 1fr;
    }

    #user-title {
        margin-bottom: 1;
        text-style: bold;
    }

    #partition-tabs {
        height: 1fr;
    }

    .partition-details {
        height: 100%;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "go_home", "Home"),
        ("j", "show_jobs", "Jobs"),
        ("p", "show_partitions", "Partitions"),
    ]

    def on_mount(self) -> None:
        self.install_screen(WelcomeScreen(), name="welcome")
        self.install_screen(JobsScreen(), name="jobs")
        self.install_screen(PartitionScreen(), name="partitions")
        self.push_screen("welcome")

    def action_go_home(self) -> None:
        self.switch_screen("welcome")

    def action_show_jobs(self) -> None:
        self.switch_screen("jobs")

    def action_show_partitions(self) -> None:
        self.switch_screen("partitions")


__all__ = ["MonitorApp"]
