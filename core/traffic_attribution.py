"""Privacy-preserving, service-level attribution for public REST traffic.

This module deliberately avoids request bodies, IP addresses, cookies,
authorization values, complete user agents, complete referrers, and any
long-lived per-user identifier.  Only low-cardinality aggregates are written
to OSS.  Attribution is observational: missing or invalid source declarations
never affect request handling.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import re
import time
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlsplit

from platform_utils import (
    PLATFORM,
    CounterConfig,
    get_counter_cfg,
    read_bytes,
    upload_bytes,
)


ATTRIBUTION_SCHEMA_VERSION = 1
ATTRIBUTION_RETENTION_DAYS = 14
ATTRIBUTION_MIN_SOURCE_REQUESTS = 5
ATTRIBUTION_MAX_IDENTIFIED_SOURCES_PER_DAY = 50
ATTRIBUTION_MAX_PENDING_SOURCES_PER_DAY = 200
ATTRIBUTION_SYNC_INTERVAL = 1_800
ATTRIBUTION_SYNC_THRESHOLD = 200

ATTRIBUTION_RESPONSE_HEADERS = {
    "X-DanbooruSearch-Attribution": "optional-during-observation",
    "X-DanbooruSearch-Client-Hint": (
        "Declare X-DanbooruSearch-Client and X-DanbooruSearch-Site; "
        "undeclared requests may be rate-limited in the future"
    ),
}

_CST = timezone(timedelta(hours=8))
_CLIENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,63}")
_IDENTIFIED_SOURCE_KINDS = frozenset({"declared", "origin", "referer"})
_SOURCE_KINDS = frozenset({"declared", "origin", "referer", "anonymous", "other"})
_ENDPOINTS = frozenset({"search", "related", "artists", "health"})
_STATUS_CLASSES = frozenset({"2xx", "3xx", "4xx", "5xx"})
_CLIENT_FAMILIES = frozenset({"browser", "python", "node", "curl", "java", "unknown"})
_LATENCY_BUCKETS = (100, 250, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 60_000, 120_000)
_LIMIT_BUCKETS = frozenset({"none", "1_20", "21_100", "101_plus"})
_TOP_K_BUCKETS = frozenset({"none", "1_10", "11_50", "51_plus"})
_MIN_COOC_BUCKETS = frozenset({"none", "1_3", "4_10", "11_plus"})
_GROUP_MODES = frozenset({"none", "off", "expand", "diverse"})
_SEGMENTATION_MODES = frozenset({"none", "on", "off"})
_PARAMETER_BUCKETS = frozenset(
    f"{min_cooc}:{group_mode}:{segmentation}"
    for min_cooc in _MIN_COOC_BUCKETS
    for group_mode in _GROUP_MODES
    for segmentation in _SEGMENTATION_MODES
)

# Common multi-label public suffixes used only when tldextract is unavailable.
# Production installs tldextract with its bundled PSL snapshot; the fallback
# keeps local/offline behavior conservative and deterministic.
_FALLBACK_MULTI_LABEL_SUFFIXES = frozenset({
    "co.jp", "co.kr", "co.uk", "com.au", "com.br", "com.cn", "com.hk",
    "com.sg", "com.tw", "net.au", "net.cn", "org.au", "org.cn", "org.uk",
})
_SHARED_HOSTING_SUFFIXES = (
    ".hf.space",
    ".ms.show",
    ".github.io",
    ".pages.dev",
    ".vercel.app",
    ".netlify.app",
)

_RecordKey = tuple[str, str, str, str, str, str, str, str, str, str, str, str]
_MetricMap = dict[_RecordKey, dict[str, int]]


class AttributionDataError(ValueError):
    """Raised when a persisted attribution snapshot is unsafe to merge."""


@dataclass(frozen=True)
class RequestObservation:
    source_kind: str
    source_name: str
    source_site: str
    client_family: str
    peak_in_flight: int
    context_token: Token


_memory_records: _MetricMap = {}
_dirty_records: _MetricMap = {}
_pending_records: dict[tuple[str, str, str, str], _MetricMap] = {}
_pending_totals: dict[tuple[str, str, str, str], int] = {}
_minute_counts: dict[tuple[str, str, str, str, str, str], int] = {}
_active_requests = 0
_enabled_at = ""
_last_sync = 0.0
_sync_lock: Optional[asyncio.Lock] = None
_safe_parameter_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "danbooru_rest_safe_attribution_parameters",
    default=None,
)


def _get_sync_lock() -> asyncio.Lock:
    global _sync_lock
    if _sync_lock is None:
        _sync_lock = asyncio.Lock()
    return _sync_lock


def _now() -> datetime:
    return datetime.now(_CST)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _attribution_file() -> str:
    platform = PLATFORM if PLATFORM in {"hf", "ms"} else "local"
    return f"traffic_attribution_v1_{platform}.json"


def _sanitize_client_name(value: Any) -> str:
    text = str(value or "").strip()
    if not _CLIENT_RE.fullmatch(text):
        return ""
    return text.lower()


def _fallback_registered_domain(host: str) -> str:
    for suffix in _SHARED_HOSTING_SUFFIXES:
        if host.endswith(suffix) and host != suffix.lstrip("."):
            return host
    labels = host.split(".")
    if len(labels) < 2:
        return ""
    last_two = ".".join(labels[-2:])
    if last_two in _FALLBACK_MULTI_LABEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


def _registered_domain(value: Any) -> str:
    """Return a normalized registrable domain without path, query, or port."""
    raw = str(value or "").strip()
    if not raw or raw.lower() == "null" or any(ch.isspace() for ch in raw):
        return ""
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    try:
        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError:
        return ""
    if len(host) > 253 or host == "localhost" or "." not in host:
        return ""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return ""
    for suffix in _SHARED_HOSTING_SUFFIXES:
        if host.endswith(suffix) and host != suffix.lstrip("."):
            return host
    try:
        import tldextract

        extracted = tldextract.TLDExtract(
            suffix_list_urls=(),
            include_psl_private_domains=True,
        )(host)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
    except Exception:
        pass
    return _fallback_registered_domain(host)


def _client_family(user_agent: Any) -> str:
    value = str(user_agent or "").lower()
    if not value:
        return "unknown"
    if "python-requests" in value or "python-httpx" in value or "aiohttp" in value:
        return "python"
    if "axios" in value or "undici" in value or "node-fetch" in value or value.startswith("node"):
        return "node"
    if "curl/" in value:
        return "curl"
    if "java/" in value or "okhttp" in value:
        return "java"
    if "mozilla/" in value or "chrome/" in value or "safari/" in value or "firefox/" in value:
        return "browser"
    return "unknown"


def classify_source(headers: Mapping[str, str]) -> tuple[str, str, str, str]:
    """Classify a request using service-level hints only."""
    family = _client_family(headers.get("user-agent"))
    declared_name = _sanitize_client_name(headers.get("x-danboorusearch-client"))
    declared_site = _registered_domain(headers.get("x-danboorusearch-site"))
    if declared_name or declared_site:
        return "declared", declared_name, declared_site, family
    origin = _registered_domain(headers.get("origin"))
    if origin:
        return "origin", "", origin, family
    referer = _registered_domain(headers.get("referer"))
    if referer:
        return "referer", "", referer, family
    return "anonymous", family, "", family


def _numeric_bucket(value: Any, *, first: int, second: int, names: tuple[str, str, str]) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "none"
    if number <= first:
        return names[0]
    if number <= second:
        return names[1]
    return names[2]


def _parameter_buckets(values: Mapping[str, Any] | None) -> tuple[str, str, str, str]:
    values = values or {}
    limit_bucket = _numeric_bucket(
        values.get("limit"), first=20, second=100, names=("1_20", "21_100", "101_plus")
    )
    top_k_bucket = _numeric_bucket(
        values.get("top_k"), first=10, second=50, names=("1_10", "11_50", "51_plus")
    )
    min_cooc_bucket = _numeric_bucket(
        values.get("min_cooc"), first=3, second=10, names=("1_3", "4_10", "11_plus")
    )
    group_mode = str(values.get("group_mode") or "none").lower()
    if group_mode not in _GROUP_MODES:
        group_mode = "none"
    segmentation = values.get("use_segmentation")
    segmentation_mode = "none" if segmentation is None else ("on" if bool(segmentation) else "off")
    return limit_bucket, top_k_bucket, min_cooc_bucket, f"{group_mode}:{segmentation_mode}"


def note_safe_parameters(**values: Any) -> None:
    """Attach an allowlisted, content-free parameter summary to this request."""
    context = _safe_parameter_context.get()
    if context is None:
        return
    for key in ("limit", "top_k", "min_cooc", "group_mode", "use_segmentation"):
        if key in values:
            context[key] = values[key]


def _latency_bucket(duration_ms: float) -> str:
    for limit in _LATENCY_BUCKETS:
        if duration_ms <= limit:
            return f"le_{limit}"
    return f"gt_{_LATENCY_BUCKETS[-1]}"


def _status_class(status_code: int) -> str:
    if 200 <= status_code < 300:
        return "2xx"
    if 300 <= status_code < 400:
        return "3xx"
    if 400 <= status_code < 500:
        return "4xx"
    return "5xx"


def _empty_metric() -> dict[str, int]:
    return {"count": 0, "sum_ms": 0, "peak_in_flight": 0, "peak_per_minute": 0}


def _merge_metric(target: dict[str, int], addition: Mapping[str, Any]) -> None:
    target["count"] += max(0, int(addition.get("count", 0)))
    target["sum_ms"] += max(0, int(addition.get("sum_ms", 0)))
    target["peak_in_flight"] = max(
        target["peak_in_flight"], max(0, int(addition.get("peak_in_flight", 0)))
    )
    target["peak_per_minute"] = max(
        target["peak_per_minute"], max(0, int(addition.get("peak_per_minute", 0)))
    )


def _add_record(target: _MetricMap, key: _RecordKey, metric: Mapping[str, Any]) -> None:
    row = target.setdefault(key, _empty_metric())
    _merge_metric(row, metric)


def _merge_record_maps(base: _MetricMap, additions: _MetricMap) -> _MetricMap:
    merged: _MetricMap = {key: dict(metric) for key, metric in base.items()}
    for key, metric in additions.items():
        _add_record(merged, key, metric)
    return merged


def _other_key(key: _RecordKey) -> _RecordKey:
    values = list(key)
    values[2] = "other"
    values[3] = ""
    values[4] = ""
    return tuple(values)  # type: ignore[return-value]


def _flush_pending(*, before_day: str | None = None, all_sources: bool = False) -> None:
    identities = list(_pending_records)
    for identity in identities:
        day = identity[0]
        if not all_sources and (before_day is None or day >= before_day):
            continue
        rows = _pending_records.pop(identity)
        _pending_totals.pop(identity, None)
        for key, metric in rows.items():
            other = _other_key(key)
            _add_record(_memory_records, other, metric)
            _add_record(_dirty_records, other, metric)


def _identified_identities_for_day(day: str) -> set[tuple[str, str, str, str]]:
    return {
        (key[0], key[2], key[3], key[4])
        for key in _memory_records
        if key[0] == day and key[2] in _IDENTIFIED_SOURCE_KINDS
    }


def _prune_expired_records() -> None:
    earliest = (_now().date() - timedelta(days=ATTRIBUTION_RETENTION_DAYS - 1)).isoformat()
    for records in (_memory_records, _dirty_records):
        for key in [key for key in records if key[0] < earliest]:
            records.pop(key, None)
    for identity in [identity for identity in _pending_records if identity[0] < earliest]:
        _pending_records.pop(identity, None)
        _pending_totals.pop(identity, None)


def _record_with_threshold(key: _RecordKey, metric: Mapping[str, Any]) -> None:
    day, _, source_kind, source_name, source_site, *_ = key
    if source_kind not in _IDENTIFIED_SOURCE_KINDS:
        _add_record(_memory_records, key, metric)
        _add_record(_dirty_records, key, metric)
        return
    identity = (day, source_kind, source_name, source_site)
    known_identities = _identified_identities_for_day(day)
    if identity in known_identities:
        _add_record(_memory_records, key, metric)
        _add_record(_dirty_records, key, metric)
        return
    if len(known_identities) >= ATTRIBUTION_MAX_IDENTIFIED_SOURCES_PER_DAY:
        other = _other_key(key)
        _add_record(_memory_records, other, metric)
        _add_record(_dirty_records, other, metric)
        return
    if identity not in _pending_records:
        pending_count = sum(1 for candidate in _pending_records if candidate[0] == day)
        if pending_count >= ATTRIBUTION_MAX_PENDING_SOURCES_PER_DAY:
            other = _other_key(key)
            _add_record(_memory_records, other, metric)
            _add_record(_dirty_records, other, metric)
            return
    pending = _pending_records.setdefault(identity, {})
    _add_record(pending, key, metric)
    _pending_totals[identity] = _pending_totals.get(identity, 0) + int(metric.get("count", 0))
    if _pending_totals[identity] < ATTRIBUTION_MIN_SOURCE_REQUESTS:
        return
    promoted = _pending_records.pop(identity)
    _pending_totals.pop(identity, None)
    for pending_key, pending_metric in promoted.items():
        destination_key = (
            pending_key
            if len(known_identities) < ATTRIBUTION_MAX_IDENTIFIED_SOURCES_PER_DAY
            else _other_key(pending_key)
        )
        _add_record(_memory_records, destination_key, pending_metric)
        _add_record(_dirty_records, destination_key, pending_metric)


def start_request(headers: Mapping[str, str]) -> RequestObservation:
    global _active_requests
    source_kind, source_name, source_site, family = classify_source(headers)
    _active_requests += 1
    token = _safe_parameter_context.set({})
    return RequestObservation(
        source_kind=source_kind,
        source_name=source_name,
        source_site=source_site,
        client_family=family,
        peak_in_flight=_active_requests,
        context_token=token,
    )


async def finish_request(
    observation: RequestObservation,
    *,
    endpoint: str,
    status_code: int,
    duration_ms: float,
) -> None:
    """Finish one observation. This function never controls API access."""
    global _active_requests, _enabled_at
    safe_parameters = _safe_parameter_context.get() or {}
    try:
        _safe_parameter_context.reset(observation.context_token)
    finally:
        _active_requests = max(0, _active_requests - 1)
    if endpoint not in _ENDPOINTS:
        return
    try:
        duration_ms = float(duration_ms)
    except (TypeError, ValueError):
        duration_ms = 0.0
    if not math.isfinite(duration_ms) or duration_ms < 0:
        duration_ms = 0.0
    now = _now()
    day = now.date().isoformat()
    hour = f"{now.hour:02d}"
    _flush_pending(before_day=day)
    _prune_expired_records()
    limit_bucket, top_k_bucket, min_cooc_bucket, mode_bucket = _parameter_buckets(safe_parameters)
    minute = now.strftime("%Y-%m-%dT%H:%M")
    minute_key = (
        minute,
        observation.source_kind,
        observation.source_name,
        observation.source_site,
        observation.client_family,
        endpoint,
    )
    _minute_counts[minute_key] = _minute_counts.get(minute_key, 0) + 1
    stale_minutes = [key for key in _minute_counts if not key[0].startswith(day)]
    for stale in stale_minutes:
        _minute_counts.pop(stale, None)
    key: _RecordKey = (
        day,
        hour,
        observation.source_kind,
        observation.source_name,
        observation.source_site,
        endpoint,
        _status_class(int(status_code)),
        observation.client_family,
        _latency_bucket(duration_ms),
        limit_bucket,
        top_k_bucket,
        f"{min_cooc_bucket}:{mode_bucket}",
    )
    metric = {
        "count": 1,
        "sum_ms": int(round(duration_ms)),
        "peak_in_flight": observation.peak_in_flight,
        "peak_per_minute": _minute_counts[minute_key],
    }
    _record_with_threshold(key, metric)
    if not _enabled_at:
        _enabled_at = _utc_now_iso()
    _check_sync()


def _valid_day(value: Any) -> str:
    text = str(value or "")
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return ""
    earliest = _now().date() - timedelta(days=ATTRIBUTION_RETENTION_DAYS - 1)
    if parsed < earliest or parsed > _now().date():
        return ""
    return text


def _sanitize_record(raw: Any) -> tuple[_RecordKey, dict[str, int]] | None:
    if not isinstance(raw, dict):
        return None
    day = _valid_day(raw.get("day"))
    hour = str(raw.get("hour") or "")
    source_kind = str(raw.get("source_kind") or "")
    endpoint = str(raw.get("endpoint") or "")
    status_class = str(raw.get("status_class") or "")
    client_family = str(raw.get("client_family") or "")
    latency_bucket = str(raw.get("latency_bucket") or "")
    limit_bucket = str(raw.get("limit_bucket") or "")
    top_k_bucket = str(raw.get("top_k_bucket") or "")
    parameter_bucket = str(raw.get("parameter_bucket") or "")
    if (
        not day
        or hour not in {f"{value:02d}" for value in range(24)}
        or source_kind not in _SOURCE_KINDS
        or endpoint not in _ENDPOINTS
        or status_class not in _STATUS_CLASSES
        or client_family not in _CLIENT_FAMILIES
        or latency_bucket not in {*(f"le_{value}" for value in _LATENCY_BUCKETS), f"gt_{_LATENCY_BUCKETS[-1]}"}
        or limit_bucket not in _LIMIT_BUCKETS
        or top_k_bucket not in _TOP_K_BUCKETS
        or parameter_bucket not in _PARAMETER_BUCKETS
    ):
        return None
    source_name = ""
    source_site = ""
    if source_kind == "declared":
        source_name = _sanitize_client_name(raw.get("source_name"))
        source_site = _registered_domain(raw.get("source_site"))
        if not source_name and not source_site:
            return None
    elif source_kind in {"origin", "referer"}:
        source_site = _registered_domain(raw.get("source_site"))
        if not source_site:
            return None
    elif source_kind == "anonymous":
        source_name = str(raw.get("source_name") or "")
        if source_name not in _CLIENT_FAMILIES:
            return None
    try:
        metric = {
            "count": max(0, int(raw.get("count", 0))),
            "sum_ms": max(0, int(raw.get("sum_ms", 0))),
            "peak_in_flight": max(0, int(raw.get("peak_in_flight", 0))),
            "peak_per_minute": max(0, int(raw.get("peak_per_minute", 0))),
        }
    except (TypeError, ValueError):
        return None
    if metric["count"] <= 0:
        return None
    key: _RecordKey = (
        day,
        hour,
        source_kind,
        source_name,
        source_site,
        endpoint,
        status_class,
        client_family,
        latency_bucket,
        limit_bucket,
        top_k_bucket,
        parameter_bucket,
    )
    return key, metric


def _parse_remote(raw: bytes | None) -> tuple[str, _MetricMap]:
    if raw is None:
        return "", {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttributionDataError("traffic attribution JSON is invalid") from exc
    if not isinstance(data, dict) or data.get("schema_version") != ATTRIBUTION_SCHEMA_VERSION:
        raise AttributionDataError("unsupported traffic attribution schema_version")
    expected_platform = PLATFORM if PLATFORM in {"hf", "ms"} else "local"
    if data.get("platform") != expected_platform:
        raise AttributionDataError("traffic attribution platform mismatch")
    records: _MetricMap = {}
    identified_by_day: dict[str, set[tuple[str, str, str, str]]] = {}
    for raw_record in data.get("records", []):
        sanitized = _sanitize_record(raw_record)
        if sanitized is None:
            continue
        key, metric = sanitized
        destination_key = key
        if key[2] in _IDENTIFIED_SOURCE_KINDS:
            identity = (key[0], key[2], key[3], key[4])
            identities = identified_by_day.setdefault(key[0], set())
            if identity not in identities:
                if len(identities) >= ATTRIBUTION_MAX_IDENTIFIED_SOURCES_PER_DAY:
                    destination_key = _other_key(key)
                else:
                    identities.add(identity)
        _add_record(records, destination_key, metric)
    return str(data.get("enabled_at") or "")[:40], records


def _serialize(enabled_at: str, records: _MetricMap) -> bytes:
    rows = []
    for key in sorted(records):
        (
            day,
            hour,
            source_kind,
            source_name,
            source_site,
            endpoint,
            status_class,
            client_family,
            latency_bucket,
            limit_bucket,
            top_k_bucket,
            parameter_bucket,
        ) = key
        rows.append({
            "day": day,
            "hour": hour,
            "source_kind": source_kind,
            "source_name": source_name,
            "source_site": source_site,
            "endpoint": endpoint,
            "status_class": status_class,
            "client_family": client_family,
            "latency_bucket": latency_bucket,
            "limit_bucket": limit_bucket,
            "top_k_bucket": top_k_bucket,
            "parameter_bucket": parameter_bucket,
            **records[key],
        })
    platform = PLATFORM if PLATFORM in {"hf", "ms"} else "local"
    return json.dumps({
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "enabled_at": enabled_at,
        "updated_at": _utc_now_iso(),
        "platform": platform,
        "retention_days": ATTRIBUTION_RETENTION_DAYS,
        "source_persistence": {
            "min_requests_per_day": ATTRIBUTION_MIN_SOURCE_REQUESTS,
            "max_identified_sources_per_day": ATTRIBUTION_MAX_IDENTIFIED_SOURCES_PER_DAY,
            "max_pending_sources_per_day": ATTRIBUTION_MAX_PENDING_SOURCES_PER_DAY,
            "overflow_bucket": "other",
        },
        "privacy": {
            "stores_ip": False,
            "stores_request_body": False,
            "stores_query_or_tags": False,
            "stores_cookie_or_authorization": False,
            "stores_complete_user_agent_or_referrer": False,
        },
        "records": rows,
    }, ensure_ascii=False, indent=2).encode("utf-8")


def _sync_remote_task(additions: _MetricMap, enabled_at: str) -> tuple[bool, str, _MetricMap]:
    cfg: CounterConfig = get_counter_cfg()
    if not cfg.available:
        return False, enabled_at, {}
    try:
        remote_enabled_at, remote_records = _parse_remote(read_bytes(_attribution_file(), cfg))
    except Exception as exc:
        print(f"[TrafficAttribution] 远端数据不可合并，中止同步以保护数据: {exc}", flush=True)
        return False, enabled_at, {}
    merged = _merge_record_maps(remote_records, additions)
    chosen_enabled_at = remote_enabled_at or enabled_at or _utc_now_iso()
    try:
        ok = upload_bytes(
            _serialize(chosen_enabled_at, merged),
            _attribution_file(),
            cfg,
            f"Traffic attribution v1: records={len(merged)}",
            retries=3,
            retry_delay=1.0,
        )
    except Exception as exc:
        print(f"[TrafficAttribution] 远端写入异常: {exc}", flush=True)
        return False, enabled_at, {}
    return ok, chosen_enabled_at, merged if ok else {}


async def _perform_sync() -> None:
    global _dirty_records, _memory_records, _enabled_at, _last_sync
    lock = _get_sync_lock()
    if lock.locked():
        return
    async with lock:
        if not _dirty_records:
            return
        additions = {key: dict(metric) for key, metric in _dirty_records.items()}
        _dirty_records.clear()
        loop = asyncio.get_running_loop()
        try:
            success, enabled_at, merged = await loop.run_in_executor(
                None,
                _sync_remote_task,
                additions,
                _enabled_at,
            )
        except Exception as exc:
            _dirty_records = _merge_record_maps(additions, _dirty_records)
            print(f"[TrafficAttribution] 同步任务异常，增量已回滚: {exc}", flush=True)
            return
        if not success:
            _dirty_records = _merge_record_maps(additions, _dirty_records)
            return
        _enabled_at = enabled_at
        _memory_records = _merge_record_maps(merged, _dirty_records)
        _last_sync = time.time()


def _check_sync() -> None:
    cfg = get_counter_cfg()
    if not cfg.available:
        return
    dirty_count = sum(metric.get("count", 0) for metric in _dirty_records.values())
    if time.time() - _last_sync > ATTRIBUTION_SYNC_INTERVAL or dirty_count >= ATTRIBUTION_SYNC_THRESHOLD:
        asyncio.create_task(_perform_sync())


async def init() -> None:
    """Load this platform's attribution snapshot without touching telemetry v2."""
    global _memory_records, _enabled_at, _last_sync
    cfg = get_counter_cfg()
    if not cfg.available:
        print(
            f"[TrafficAttribution] 未配置持久化（platform={PLATFORM}），仅使用内存统计。",
            flush=True,
        )
        return
    loop = asyncio.get_running_loop()
    try:
        enabled_at, records = await loop.run_in_executor(
            None,
            lambda: _parse_remote(read_bytes(_attribution_file(), cfg)),
        )
    except Exception as exc:
        print(f"[TrafficAttribution] 启动读取失败，本次不覆盖远端数据: {exc}", flush=True)
        return
    _enabled_at = enabled_at or _enabled_at or _utc_now_iso()
    _memory_records = _merge_record_maps(records, _dirty_records)
    _last_sync = time.time()
    print(
        f"[TrafficAttribution] v1 初始化完成：platform={PLATFORM}, records={len(records)}",
        flush=True,
    )


def get_snapshot() -> dict[str, Any]:
    """Return the persisted-safe in-memory view; sub-threshold sources are omitted."""
    return json.loads(_serialize(_enabled_at or _utc_now_iso(), _memory_records).decode("utf-8"))


def get_admin_snapshot() -> dict[str, Any]:
    """Copy only overview fields on the event loop; no I/O or source identities.

    The detached snapshot can be aggregated in a worker without racing mutable
    request counters. Pending, sub-threshold records remain excluded.
    """
    return {"records": [
        {"day": key[0], "hour": key[1], "endpoint": key[5],
         "status_class": key[6], "latency_bucket": key[8],
         "count": metric["count"], "sum_ms": metric["sum_ms"]}
        for key, metric in _memory_records.items()
    ]}


async def force_sync() -> None:
    _flush_pending(all_sources=True)
    _prune_expired_records()
    await _perform_sync()


def _reset_for_tests() -> None:
    """Reset module state for isolated unit tests."""
    global _active_requests, _enabled_at, _last_sync, _sync_lock
    _memory_records.clear()
    _dirty_records.clear()
    _pending_records.clear()
    _pending_totals.clear()
    _minute_counts.clear()
    _active_requests = 0
    _enabled_at = ""
    _last_sync = 0.0
    _sync_lock = None
    _safe_parameter_context.set(None)
