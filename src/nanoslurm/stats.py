"""Helper functions for computing SLURM cluster statistics."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from . import backend as B
from .job import _TERMINAL


def node_state_counts() -> dict[str, int]:
    """Return counts of nodes grouped by normalized state."""

    try:
        rows = B.sinfo(fields=["state", "count"])
    except B.SlurmUnavailableError:
        return {}

    counts: Counter[str] = Counter()
    for row in rows:
        token = B.normalize_state(row.get("state", ""))
        try:
            counts[token] += int(row.get("count", "0"))
        except ValueError:
            continue
    return dict(counts)


def partition_node_state_counts() -> dict[str, dict[str, int]]:
    """Return node state counts grouped by partition."""

    try:
        rows = B.sinfo(fields=["part", "state", "count"], all_partitions=True)
    except B.SlurmUnavailableError:
        return {}

    counts: dict[str, Counter[str]] = {}
    for row in rows:
        part = row.get("part", "").rstrip("*")
        token = B.normalize_state(row.get("state", ""))
        try:
            value = int(row.get("count", "0"))
        except ValueError:
            continue
        counts.setdefault(part, Counter())[token] += value
    return {part: dict(counter) for part, counter in counts.items()}


def recent_completions(span: str = "day", count: int = 7) -> list[tuple[str, int]]:
    """Return recent completion counts grouped by *span*."""

    if span not in {"day", "week"}:
        raise ValueError("span must be 'day' or 'week'")

    delta = timedelta(days=count if span == "day" else count * 7)
    start = (datetime.now() - delta).strftime("%Y-%m-%d")
    try:
        rows = B.sacct(
            fields=["end"],
            states=["CD"],
            start_time=start,
            allocations=True,
        )
    except B.SlurmUnavailableError:
        return []

    counts: Counter[str] = Counter()
    for row in rows:
        token = row.get("end", "").strip()
        if not token:
            continue
        try:
            dt = datetime.strptime(token.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        key = dt.strftime("%Y-%m-%d")
        if span == "week":
            year, week, _ = dt.isocalendar()
            key = f"{year}-W{week:02d}"
        counts[key] += 1
    items = sorted(counts.items())
    return items[-count:]


def _parse_gpu(gres: str) -> int:
    total = 0
    for token in gres.split(","):
        token = token.strip().split("(")[0]
        if token.startswith("gpu:"):
            try:
                total += int(token.split(":")[-1])
            except ValueError:
                continue
    return total


def _partition_caps() -> dict[str, dict[str, int]]:
    try:
        rows = B.sinfo(fields=["part", "cpus", "gres", "nodes"], all_partitions=True)
    except B.SlurmUnavailableError:
        return {}

    caps: dict[str, dict[str, int]] = {}
    for row in rows:
        part = row.get("part", "").rstrip("*")
        cpus = 0
        cpu_field = row.get("cpus", "")
        if cpu_field:
            try:
                cpus = int(cpu_field.split("/")[-1])
            except ValueError:
                cpus = 0
        gpus_per_node = _parse_gpu(row.get("gres", ""))
        nodes = 0
        node_field = row.get("nodes", "")
        if node_field:
            try:
                nodes = int(node_field)
            except ValueError:
                nodes = 0
        caps[part] = {"cpus": cpus, "gpus": gpus_per_node * nodes}
    return caps


def partition_utilization() -> dict[str, float]:
    """Return per-partition utilization percentages based on running jobs."""

    caps = _partition_caps()
    if not caps:
        return {}

    try:
        rows = B.squeue(fields=["partition", "cpus", "gres"], states=["RUNNING"])
    except B.SlurmUnavailableError:
        return {}

    usage: dict[str, dict[str, int]] = {}
    for row in rows:
        part = row.get("partition", "")
        try:
            cpus = int(row.get("cpus", "0"))
        except ValueError:
            cpus = 0
        gpus = _parse_gpu(row.get("gres", ""))
        use = usage.setdefault(part, {"cpus": 0, "gpus": 0})
        use["cpus"] += cpus
        use["gpus"] += gpus

    utilization: dict[str, float] = {}
    for part, cap in caps.items():
        totals = usage.get(part, {})
        cpu_cap = cap.get("cpus", 0)
        gpu_cap = cap.get("gpus", 0)
        cpu_pct = totals.get("cpus", 0) / cpu_cap if cpu_cap else 0.0
        gpu_pct = totals.get("gpus", 0) / gpu_cap if gpu_cap else 0.0
        utilization[part] = max(cpu_pct, gpu_pct) * 100
    return utilization


def fairshare_scores() -> dict[str, float]:
    """Return user fair-share scores from ``sprio`` or ``sshare``."""

    rows: list[dict[str, str]]
    try:
        rows = B.sprio(fields=["user", "fairshare"])
    except B.SlurmUnavailableError:
        try:
            rows = B.sshare(fields=["user", "fairshare"])
        except B.SlurmUnavailableError:
            return {}

    scores: dict[str, float] = {}
    for row in rows:
        user = row.get("user", "")
        value = row.get("fairshare", "")
        if not user:
            continue
        try:
            scores[user] = float(value)
        except ValueError:
            continue
    return scores


def job_history() -> dict[str, dict[str, int]]:
    """Return per-user job completion statistics for the last 24 hours."""

    now = datetime.now()
    start = now - timedelta(hours=24)

    try:
        rows = B.sacct(
            fields=["user", "state"],
            all_users=True,
            allocations=True,
            start_time=start.strftime("%Y-%m-%dT%H:%M:%S"),
            end_time=now.strftime("%Y-%m-%dT%H:%M:%S"),
        )
    except B.SlurmUnavailableError:
        return {}

    stats: dict[str, dict[str, int]] = {}
    for row in rows:
        user = row.get("user", "")
        if not user:
            continue
        token = B.normalize_state(row.get("state", ""))
        entry = stats.setdefault(user, {"completed": 0, "failed": 0})
        if token == "COMPLETED":
            entry["completed"] += 1
        elif token in _TERMINAL:
            entry["failed"] += 1
    return stats


__all__ = [
    "node_state_counts",
    "partition_utilization",
    "fairshare_scores",
    "recent_completions",
    "job_history",
]

