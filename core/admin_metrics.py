"""Read-only presentation of existing metrics, with explicit time boundaries."""
from __future__ import annotations

import asyncio
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

LIMITS = (100, 250, 500, 1000, 2000, 5000, 10000, 30000, 60000, 120000)
CST = timezone(timedelta(hours=8))


def timing_summary(raw: dict) -> dict:
    count = int(raw.get("count", 0))
    buckets = raw.get("buckets", {})
    cumulative = 0
    p95 = None
    overflow = False
    if count:
        for limit in LIMITS:
            cumulative += int(buckets.get(f"le_{limit}", 0))
            if cumulative >= math.ceil(count * .95):
                p95 = limit
                break
        if p95 is None and int(buckets.get("gt_120000", 0)):
            p95, overflow = 120000, True
    return {"count": count, "average_ms": round(raw.get("sum_ms", 0) / count, 1) if count else None,
            "p95_ms": p95, "p95_overflow": overflow,
            "buckets": [{"label": f"≤ {n / 1000:g} s", "count": int(buckets.get(f"le_{n}", 0))} for n in LIMITS]
                       + [{"label": "> 120 s", "count": int(buckets.get("gt_120000", 0))}]}


def build_dashboard(telemetry: dict, attribution: dict, now: datetime | None = None) -> dict:
    now = (now or datetime.now(CST)).astimezone(CST)
    earliest = now.date() - timedelta(days=13)
    rows = [row for row in attribution.get("records", [])
            if earliest.isoformat() <= row.get("day", "") <= now.date().isoformat()]
    endpoints = defaultdict(lambda: {"count": 0, "sum_ms": 0, "errors": 0, "buckets": Counter()})
    hours = Counter()
    statuses = Counter()
    for row in rows:
        count = int(row.get("count", 0))
        endpoint = endpoints[row["endpoint"]]
        endpoint["count"] += count
        endpoint["sum_ms"] += row.get("sum_ms", 0)
        endpoint["errors"] += count if row["status_class"] in {"4xx", "5xx"} else 0
        endpoint["buckets"][row["latency_bucket"]] += count
        hours[row["day"] + "T" + str(row["hour"]).zfill(2) + ":00"] += count
        statuses[row["status_class"]] += count
    total = sum(statuses.values())
    first_hour = min(hours) if hours else None
    series = []
    if first_hour:
        cursor = datetime.fromisoformat(first_hour).replace(tzinfo=CST)
        while cursor <= now:
            label = cursor.strftime("%Y-%m-%dT%H:00")
            # A missing observation is not proof of zero traffic (restart/threshold).
            series.append({"hour": label, "count": hours.get(label)})
            cursor += timedelta(hours=1)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "telemetry_since": telemetry.get("enabled_at"),
        "counters": telemetry.get("counters", {}),
        "ui_latency": timing_summary(telemetry.get("timings_ms", {}).get("ui_search_latency", {})),
        "rest": {"retention_days": 14, "first_hour": first_hour,
                 "last_hour": max(hours) if hours else None,
                 "count": total, "error_percent": round(100 * (statuses["4xx"] + statuses["5xx"]) / total, 2) if total else None,
                 "statuses": dict(statuses), "hours": series,
                 "endpoints": [{"endpoint": name, **timing_summary(raw), "errors": raw["errors"]}
                               for name, raw in sorted(endpoints.items())]},
    }


async def read_dashboard() -> dict:
    # Imports are lazy: the standalone test portal never loads the search engine.
    from core import telemetry, traffic_attribution
    telemetry_snapshot = telemetry.get_snapshot()
    attribution_snapshot = traffic_attribution.get_admin_snapshot()
    return await asyncio.to_thread(build_dashboard, telemetry_snapshot, attribution_snapshot)
