"""Textual monitor for viewing SLURM jobs managed by :mod:`nanoslurm`."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Static

from .job import Job, list_jobs


class MonitorApp(App):
    """Minimal TUI that lists jobs and auto-refreshes."""

    CSS = """
    Screen {
        padding: 1;
    }

    #summary {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_now", "Refresh"),
    ]

    def compose(self) -> ComposeResult:  # pragma: no cover - Textual composition
        yield Header()
        with Vertical():
            self.summary = Static(id="summary")
            yield self.summary
            self.table = DataTable(id="jobs")
            yield self.table
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - runtime hook
        self.table.add_columns("ID", "Name", "User", "Partition", "State", "Submitted", "Started")
        self.refresh_jobs()
        self.set_interval(2.0, self.refresh_jobs)

    def action_refresh_now(self) -> None:  # pragma: no cover - Textual action
        self.refresh_jobs()

    def refresh_jobs(self) -> None:  # pragma: no cover - runtime hook
        jobs = list_jobs()
        self._update_summary(jobs)
        self.table.clear()
        for job in jobs:
            self.table.add_row(*self._row(job))

    def _update_summary(self, jobs: list[Job]) -> None:
        if not jobs:
            self.summary.update("No jobs found.")
            return
        running = sum(1 for job in jobs if job.last_status == "RUNNING")
        pending = sum(1 for job in jobs if job.last_status == "PENDING")
        self.summary.update(f"Jobs: {len(jobs)} · Running: {running} · Pending: {pending}")

    @staticmethod
    def _row(job: Job) -> tuple[str, ...]:
        submit = job.submit_time.isoformat(sep=" ") if job.submit_time else "-"
        start = job.start_time.isoformat(sep=" ") if job.start_time else "-"
        return (
            str(job.id),
            job.name,
            job.user,
            job.partition,
            job.last_status or "UNKNOWN",
            submit,
            start,
        )


__all__ = ["MonitorApp"]

