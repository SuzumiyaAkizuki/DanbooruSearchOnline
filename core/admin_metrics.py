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
            "distribution_available": bool(count and sum(buckets.values()) == count),
            "buckets": [{"label": f"≤ {n / 1000:g} s", "count": int(buckets.get(f"le_{n}", 0))} for n in LIMITS]
                       + [{"label": "> 120 s", "count": int(buckets.get("gt_120000", 0))}]}


def summarize_window(rows: list[dict], now: datetime, hours: int) -> dict:
    cutoff = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours - 1)
    selected = [row for row in rows if cutoff.strftime("%Y-%m-%dT%H") <= row["day"] + "T" + str(row["hour"]).zfill(2)
                <= now.strftime("%Y-%m-%dT%H")]
    endpoints = defaultdict(lambda: {"count": 0, "sum_ms": 0, "errors": 0, "buckets": Counter(), "statuses": Counter()})
    hourly = defaultdict(lambda: {"count": 0, "endpoints": Counter(), "peak_in_flight": 0, "peak_per_minute": 0})
    sources, clients, statuses = Counter(), Counter(), Counter()
    for row in selected:
        count = int(row["count"])
        entry = endpoints[row["endpoint"]]
        entry["count"] += count
        entry["sum_ms"] += row["sum_ms"]
        entry["buckets"][row["latency_bucket"]] += count
        entry["statuses"][row["status_class"]] += count
        entry["errors"] += count if row["status_class"] in {"4xx", "5xx"} else 0
        hour = hourly[row["day"] + "T" + str(row["hour"]).zfill(2) + ":00"]
        hour["count"] += count
        hour["endpoints"][row["endpoint"]] += count
        for peak in ("peak_in_flight", "peak_per_minute"):
            hour[peak] = max(hour[peak], int(row.get(peak, 0)))
        statuses[row["status_class"]] += count
        clients[row.get("client_family", "unknown")] += count
        kind = row.get("source_kind", "unavailable")
        name, site = row.get("source_name", ""), row.get("source_site", "")
        label = (f"未声明 · {name or 'unknown'}" if kind == "anonymous" else
                 f"声明 · {name or site or '未命名'}" if kind == "declared" else
                 f"{kind} · {site}" if kind in {"origin", "referer"} else
                 "其他合并来源" if kind == "other" else "无来源维度")
        sources[label] += count
    series = []
    cursor = cutoff
    while cursor <= now:
        label = cursor.strftime("%Y-%m-%dT%H:00")
        series.append({"hour": label, **hourly.get(label, {"count": None, "endpoints": {}, "peak_in_flight": None, "peak_per_minute": None})})
        cursor += timedelta(hours=1)
    total = sum(statuses.values())
    slow = sum(sum(raw["buckets"][bucket] for bucket in ("le_10000", "le_30000", "le_60000", "le_120000", "gt_120000")) for raw in endpoints.values())
    return {
        "requested_hours": hours, "count": total, "observed_hours": len(hourly),
        "first_hour": min(hourly) if hourly else None, "last_hour": max(hourly) if hourly else None,
        "statuses": dict(statuses), "error_percent": round(100 * (statuses["4xx"] + statuses["5xx"]) / total, 2) if total else None,
        "slow_count": slow, "slow_percent": round(100 * slow / total, 2) if total else None,
        "peak_in_flight": max((item["peak_in_flight"] for item in hourly.values()), default=None),
        "peak_per_minute": max((item["peak_per_minute"] for item in hourly.values()), default=None),
        "hours": series,
        "sources": [{"label": label, "count": count} for label, count in sources.most_common(8)]
                   + ([{"label": "其余来源合计", "count": sum(count for _, count in sources.most_common()[8:])}] if len(sources) > 8 else []),
        "clients": [{"label": label, "count": count} for label, count in clients.most_common()],
        "endpoints": [{"endpoint": name, **timing_summary(raw), "errors": raw["errors"],
                       "client_errors": raw["statuses"]["4xx"], "server_errors": raw["statuses"]["5xx"],
                       "slow_percent": round(100 * sum(raw["buckets"][bucket] for bucket in ("le_10000", "le_30000", "le_60000", "le_120000", "gt_120000")) / raw["count"], 2) if raw["count"] else None}
                      for name, raw in sorted(endpoints.items())],
    }


def summarize_ui_performance(snapshot: dict, now: datetime) -> dict:
    """Present existing process windows; never infer search history from totals."""
    windows = []
    for item in snapshot.get("windows", []):
        try:
            at = datetime.fromisoformat(item["recorded_at"])
            if at.tzinfo is None or at > now:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        windows.append((at, item))
    windows.sort(key=lambda pair: pair[0])
    selected = [item for _, item in windows[-180:]]
    latest = selected[-1] if selected else None
    return {"window_count": len(selected), "latest": latest,
            "stale": bool(windows and (now - windows[-1][0]).total_seconds() > 300),
            "lag_trend": [{"at": item["recorded_at"],
                           "average_ms": item.get("metrics", {}).get("event_loop_lag", {}).get("avg_ms")}
                          for item in selected]}


def build_dashboard(telemetry: dict, attribution: dict, now: datetime | None = None,
                    ui_performance: dict | None = None) -> dict:
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
    counters = telemetry.get("counters", {})
    ui_search = counters.get("ui_search", 0)
    attempts = counters.get("engine_cold_start_attempt", 0)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "telemetry_since": telemetry.get("enabled_at"),
        "counters": telemetry.get("counters", {}),
        "ui_latency": timing_summary(telemetry.get("timings_ms", {}).get("ui_search_latency", {})),
        "ui_performance": summarize_ui_performance(ui_performance or {}, now),
        "quality": {"selection_percent": round(100 * counters.get("ui_search_with_selection_session", 0) / ui_search, 2) if ui_search else None,
                    "copy_events_per_search": round(100 * (counters.get("ui_copy_all", 0) + counters.get("ui_copy_selected", 0)) / ui_search, 2) if ui_search else None,
                    "zero_percent": round(100 * counters.get("ui_zero_result", 0) / ui_search, 3) if ui_search else None,
                    "repeat_percent": round(100 * counters.get("ui_repeat_search_60s", 0) / ui_search, 2) if ui_search else None,
                    "cold_attempts": attempts, "cold_failures": counters.get("engine_cold_start_failure", 0),
                    "cold_successes": counters.get("engine_cold_start_success", 0)},
        "rest_windows": {str(hours): summarize_window(rows, now, hours) for hours in (24, 168, 336)},
        "rest": {"retention_days": 14, "first_hour": first_hour,
                 "last_hour": max(hours) if hours else None,
                 "count": total, "error_percent": round(100 * (statuses["4xx"] + statuses["5xx"]) / total, 2) if total else None,
                 "statuses": dict(statuses), "hours": series,
                 "endpoints": [{"endpoint": name, **timing_summary(raw), "errors": raw["errors"]}
                               for name, raw in sorted(endpoints.items())]},
    }


async def read_dashboard() -> dict:
    # Imports are lazy: the standalone test portal never loads the search engine.
    from core import telemetry, traffic_attribution, ui_performance
    telemetry_snapshot = telemetry.get_snapshot()
    attribution_snapshot = traffic_attribution.get_admin_snapshot()
    performance_snapshot = ui_performance.get_snapshot()
    return await asyncio.to_thread(build_dashboard, telemetry_snapshot, attribution_snapshot,
                                   ui_performance=performance_snapshot)
