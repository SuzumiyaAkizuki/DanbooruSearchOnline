"""Versioned product telemetry and feedback persistence.

The legacy ``count.json`` remains available for historical display. New product
metrics use an independent file so UI, REST, MCP and real copy actions are not
mixed with the old cumulative counters.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

from platform_utils import CounterConfig, get_counter_cfg, read_bytes, upload_bytes


TELEMETRY_SCHEMA_VERSION = 2
TELEMETRY_ENABLED_AT = "2026-07-25T00:00:00+08:00"
FEEDBACK_SCHEMA_VERSION = 1
TELEMETRY_FILE = "telemetry_v2.json"

EVENT_NAMES = frozenset({
    "ui_visit",
    "ui_search",
    "ui_zero_result",
    "ui_search_with_selection_session",
    "ui_repeat_search_60s",
    "ui_copy_selected",
    "ui_copy_all",
    "rest_search",
    "rest_related",
    "rest_artists",
    "mcp_search_tags",
    "mcp_get_related_tags",
    "mcp_get_artist_recommendations",
    "mcp_get_artist_profile",
    "mcp_get_anima_format",
    "mcp_get_newbie_format",
    "feedback_search_bad_case",
    "feedback_translation_error",
    "engine_cold_start_attempt",
    "engine_cold_start_success",
    "engine_cold_start_failure",
})

TIMING_NAMES = frozenset({
    "ui_search_latency",
    "search_to_first_selection",
    "search_to_first_copy",
    "engine_cold_start",
})

FEEDBACK_TYPES = frozenset({"search_bad_case", "translation_error"})
TIMING_BUCKET_LIMITS_MS = (100, 250, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 60_000, 120_000)
MAX_FEEDBACKS = 500
SYNC_INTERVAL = 1_800
SYNC_THRESHOLD = 200


class TelemetryDataError(ValueError):
    """Raised when a remote telemetry snapshot is not safe to merge."""


_memory_counters: Counter[str] = Counter()
_dirty_counters: Counter[str] = Counter()
_memory_timings: dict[str, dict[str, Any]] = {}
_dirty_timings: dict[str, dict[str, Any]] = {}
_memory_feedbacks: list[dict[str, Any]] = []
_dirty_feedbacks: list[dict[str, Any]] = []
_last_sync: float = 0.0
_sync_lock: Optional[asyncio.Lock] = None


def _get_sync_lock() -> asyncio.Lock:
    global _sync_lock
    if _sync_lock is None:
        _sync_lock = asyncio.Lock()
    return _sync_lock


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _bucket_name(duration_ms: float) -> str:
    for limit in TIMING_BUCKET_LIMITS_MS:
        if duration_ms <= limit:
            return f"le_{limit}"
    return f"gt_{TIMING_BUCKET_LIMITS_MS[-1]}"


def _new_timing() -> dict[str, Any]:
    return {"count": 0, "sum_ms": 0, "buckets": {}}


def _sanitize_counters(value: Any) -> Counter[str]:
    result: Counter[str] = Counter()
    if not isinstance(value, dict):
        return result
    for key, raw_count in value.items():
        if key not in EVENT_NAMES:
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count >= 0:
            result[key] = count
    return result


def _sanitize_timings(value: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(value, dict):
        return result
    valid_buckets = {
        *(f"le_{limit}" for limit in TIMING_BUCKET_LIMITS_MS),
        f"gt_{TIMING_BUCKET_LIMITS_MS[-1]}",
    }
    for metric, raw in value.items():
        if metric not in TIMING_NAMES or not isinstance(raw, dict):
            continue
        try:
            count = max(0, int(raw.get("count", 0)))
            sum_ms = max(0, int(raw.get("sum_ms", 0)))
        except (TypeError, ValueError):
            continue
        buckets: Counter[str] = Counter()
        raw_buckets = raw.get("buckets", {})
        if isinstance(raw_buckets, dict):
            for bucket, raw_count in raw_buckets.items():
                if bucket not in valid_buckets:
                    continue
                try:
                    bucket_count = int(raw_count)
                except (TypeError, ValueError):
                    continue
                if bucket_count >= 0:
                    buckets[bucket] = bucket_count
        result[metric] = {
            "count": count,
            "sum_ms": sum_ms,
            "buckets": dict(buckets),
        }
    return result


def _sanitize_feedbacks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict):
            continue
        feedback_id = str(raw.get("feedback_id") or "").strip()
        feedback_type = str(raw.get("feedback_type") or "").strip()
        if (
            raw.get("schema_version") != FEEDBACK_SCHEMA_VERSION
            or not feedback_id
            or feedback_id in seen
            or feedback_type not in FEEDBACK_TYPES
        ):
            continue
        try:
            json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError):
            continue
        seen.add(feedback_id)
        result.append(dict(raw))
        if len(result) >= MAX_FEEDBACKS:
            break
    return result


def _parse_remote(raw: bytes | None) -> tuple[Counter[str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if raw is None:
        return Counter(), {}, []
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TelemetryDataError("telemetry_v2 JSON is invalid") from exc
    if not isinstance(data, dict) or data.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
        raise TelemetryDataError("unsupported telemetry schema_version")
    return (
        _sanitize_counters(data.get("counters")),
        _sanitize_timings(data.get("timings_ms")),
        _sanitize_feedbacks(data.get("feedbacks")),
    )


def _read_remote() -> tuple[Counter[str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    cfg = get_counter_cfg()
    if not cfg.available:
        return Counter(), {}, []
    raw = read_bytes(TELEMETRY_FILE, cfg)
    return _parse_remote(raw)


def _merge_timings(
    base: dict[str, dict[str, Any]],
    additions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = _sanitize_timings(base)
    for metric, addition in _sanitize_timings(additions).items():
        target = merged.setdefault(metric, _new_timing())
        target["count"] += addition["count"]
        target["sum_ms"] += addition["sum_ms"]
        buckets = Counter(target.get("buckets", {}))
        buckets.update(addition.get("buckets", {}))
        target["buckets"] = dict(buckets)
    return merged


def _merge_feedbacks(remote: list[dict[str, Any]], additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined = _sanitize_feedbacks(additions) + _sanitize_feedbacks(remote)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in combined:
        feedback_id = item["feedback_id"]
        if feedback_id in seen:
            continue
        seen.add(feedback_id)
        result.append(item)
        if len(result) >= MAX_FEEDBACKS:
            break
    return result


def _sync_remote_task(
    counter_adds: dict[str, int],
    timing_adds: dict[str, dict[str, Any]],
    feedback_adds: list[dict[str, Any]],
) -> tuple[bool, Counter[str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    cfg: CounterConfig = get_counter_cfg()
    if not cfg.available:
        return False, Counter(), {}, []
    try:
        raw = read_bytes(TELEMETRY_FILE, cfg)
        remote_counters, remote_timings, remote_feedbacks = _parse_remote(raw)
    except Exception as exc:
        print(f"[Telemetry] 远端数据不可合并，中止同步以保护数据: {exc}", flush=True)
        return False, Counter(), {}, []

    counters = Counter(remote_counters)
    counters.update(_sanitize_counters(counter_adds))
    timings = _merge_timings(remote_timings, timing_adds)
    feedbacks = _merge_feedbacks(remote_feedbacks, feedback_adds)
    content = json.dumps({
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "enabled_at": TELEMETRY_ENABLED_AT,
        "updated_at": _utc_now_iso(),
        "counters": {
            event_name: int(counters.get(event_name, 0))
            for event_name in sorted(EVENT_NAMES)
        },
        "timings_ms": timings,
        "feedbacks": feedbacks,
    }, ensure_ascii=False, indent=2).encode("utf-8")
    commit_msg = (
        f"Telemetry v2: events={sum(counters.values())} | "
        f"feedbacks={len(feedbacks)}"
    )
    try:
        ok = upload_bytes(content, TELEMETRY_FILE, cfg, commit_msg, retries=3, retry_delay=1.0)
    except Exception as exc:
        print(f"[Telemetry] 远端写入异常: {exc}", flush=True)
        return False, Counter(), {}, []
    if not ok:
        return False, Counter(), {}, []
    return True, counters, timings, feedbacks


async def _perform_sync() -> None:
    global _last_sync
    lock = _get_sync_lock()
    if lock.locked():
        return
    async with lock:
        dirty_timing_count = sum(item.get("count", 0) for item in _dirty_timings.values())
        if not (_dirty_counters or dirty_timing_count or _dirty_feedbacks):
            return

        counter_adds = dict(_dirty_counters)
        timing_adds = {
            metric: {
                "count": item["count"],
                "sum_ms": item["sum_ms"],
                "buckets": dict(item.get("buckets", {})),
            }
            for metric, item in _dirty_timings.items()
        }
        feedback_adds = list(_dirty_feedbacks)
        _dirty_counters.clear()
        _dirty_timings.clear()
        _dirty_feedbacks.clear()

        loop = asyncio.get_running_loop()
        try:
            success, counters, timings, feedbacks = await loop.run_in_executor(
                None,
                _sync_remote_task,
                counter_adds,
                timing_adds,
                feedback_adds,
            )
        except Exception as exc:
            _dirty_counters.update(counter_adds)
            restored = _merge_timings(_dirty_timings, timing_adds)
            _dirty_timings.clear()
            _dirty_timings.update(restored)
            _dirty_feedbacks[:0] = feedback_adds
            print(f"[Telemetry] 同步任务异常，增量已回滚: {exc}", flush=True)
            return
        if success:
            _last_sync = time.time()
            counters.update(_dirty_counters)
            timings = _merge_timings(timings, _dirty_timings)
            feedbacks = _merge_feedbacks(feedbacks, _dirty_feedbacks)
            _memory_counters.clear()
            _memory_counters.update(counters)
            _memory_timings.clear()
            _memory_timings.update(timings)
            _memory_feedbacks.clear()
            _memory_feedbacks.extend(feedbacks)
            return

        _dirty_counters.update(counter_adds)
        restored = _merge_timings(_dirty_timings, timing_adds)
        _dirty_timings.clear()
        _dirty_timings.update(restored)
        _dirty_feedbacks[:0] = feedback_adds


def _check_sync() -> None:
    cfg = get_counter_cfg()
    if not cfg.available:
        return
    dirty_count = (
        sum(_dirty_counters.values())
        + sum(item.get("count", 0) for item in _dirty_timings.values())
        + len(_dirty_feedbacks)
    )
    if time.time() - _last_sync > SYNC_INTERVAL or dirty_count >= SYNC_THRESHOLD:
        asyncio.create_task(_perform_sync())


async def init() -> None:
    """Load the v2 snapshot without touching legacy counters."""
    global _last_sync
    cfg = get_counter_cfg()
    if not cfg.available:
        print(f"[Telemetry] 未配置持久化（platform={cfg.platform}），仅使用内存统计。", flush=True)
        return
    try:
        loop = asyncio.get_running_loop()
        counters, timings, feedbacks = await loop.run_in_executor(None, _read_remote)
    except Exception as exc:
        print(f"[Telemetry] 启动读取失败，本次不覆盖远端数据: {exc}", flush=True)
        return
    counters.update(_dirty_counters)
    timings = _merge_timings(timings, _dirty_timings)
    feedbacks = _merge_feedbacks(feedbacks, _dirty_feedbacks)
    _memory_counters.clear()
    _memory_counters.update(counters)
    _memory_timings.clear()
    _memory_timings.update(timings)
    _memory_feedbacks.clear()
    _memory_feedbacks.extend(feedbacks)
    _last_sync = time.time()
    print(
        f"[Telemetry] v2 初始化完成：事件={sum(counters.values())}, "
        f"反馈={len(feedbacks)}",
        flush=True,
    )


async def increment(event_name: str, amount: int = 1) -> int:
    if event_name not in EVENT_NAMES:
        raise ValueError(f"unknown telemetry event: {event_name}")
    amount = int(amount)
    if amount <= 0:
        raise ValueError("telemetry increment amount must be positive")
    _memory_counters[event_name] += amount
    _dirty_counters[event_name] += amount
    _check_sync()
    return _memory_counters[event_name]


async def record_timing(metric_name: str, duration_ms: float) -> None:
    if metric_name not in TIMING_NAMES:
        raise ValueError(f"unknown telemetry timing: {metric_name}")
    duration_ms = float(duration_ms)
    if not math.isfinite(duration_ms) or duration_ms < 0:
        raise ValueError("duration_ms must be a finite non-negative number")
    rounded = int(round(duration_ms))
    bucket = _bucket_name(duration_ms)
    for target_map in (_memory_timings, _dirty_timings):
        target = target_map.setdefault(metric_name, _new_timing())
        target["count"] += 1
        target["sum_ms"] += rounded
        buckets = Counter(target.get("buckets", {}))
        buckets[bucket] += 1
        target["buckets"] = dict(buckets)
    _check_sync()


async def add_feedback(
    *,
    feedback_type: str,
    query: str,
    search_settings: dict[str, Any] | None,
    app_version: str,
    platform: str = "",
    details: str = "",
    tag: str = "",
    current_cn_name: str = "",
    suggested_cn_name: str = "",
    category: str = "",
) -> dict[str, Any]:
    if feedback_type not in FEEDBACK_TYPES:
        raise ValueError(f"unsupported feedback_type: {feedback_type}")
    entry = {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "feedback_id": f"fb_{uuid.uuid4().hex}",
        "feedback_type": feedback_type,
        "query": str(query or "")[:4_000],
        "tag": str(tag or "")[:300],
        "expected": "",
        "details": str(details or "")[:2_000],
        "search_settings": dict(search_settings or {}),
        "app_version": str(app_version or "unknown")[:100],
        "platform": str(platform or "")[:50],
        "created_at": _utc_now_iso(),
    }
    if feedback_type == "translation_error":
        entry.update({
            "current_cn_name": str(current_cn_name or "")[:1_000],
            "suggested_cn_name": str(suggested_cn_name or "")[:1_000],
            "category": str(category or "")[:100],
        })
    _memory_feedbacks.insert(0, entry)
    del _memory_feedbacks[MAX_FEEDBACKS:]
    _dirty_feedbacks.insert(0, entry)
    await increment(f"feedback_{feedback_type}")
    print(
        f"[Telemetry] feedback accepted: type={feedback_type}, "
        f"id={entry['feedback_id']}",
        flush=True,
    )
    return dict(entry)


def _timing_p95_ms(timing: dict[str, Any]) -> int | None:
    count = int(timing.get("count", 0))
    if count <= 0:
        return None
    target = max(1, math.ceil(count * 0.95))
    cumulative = 0
    buckets = timing.get("buckets", {})
    for limit in TIMING_BUCKET_LIMITS_MS:
        cumulative += int(buckets.get(f"le_{limit}", 0))
        if cumulative >= target:
            return limit
    return TIMING_BUCKET_LIMITS_MS[-1]


def get_snapshot() -> dict[str, Any]:
    timings: dict[str, Any] = {}
    for metric, raw in _sanitize_timings(_memory_timings).items():
        timings[metric] = {
            **raw,
            "average_ms": round(raw["sum_ms"] / raw["count"], 1) if raw["count"] else None,
            "p95_ms": _timing_p95_ms(raw),
        }
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "enabled_at": TELEMETRY_ENABLED_AT,
        "counters": {
            event_name: int(_memory_counters.get(event_name, 0))
            for event_name in sorted(EVENT_NAMES)
        },
        "timings_ms": timings,
        "feedback_count": len(_memory_feedbacks),
    }


def get_feedbacks() -> list[dict[str, Any]]:
    return [dict(item) for item in _memory_feedbacks]


async def force_sync() -> None:
    await _perform_sync()
