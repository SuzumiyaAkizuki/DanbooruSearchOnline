"""Versioned browser-workspace data helpers.

This module is deliberately UI-framework agnostic.  NiceGUI owns the actual
``localStorage`` I/O; the helpers here define and validate the payload that is
stored there so migrations can be exercised without starting the web service.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from typing import Any


WORKSPACE_SCHEMA_VERSION = 1
HISTORY_SCHEMA_VERSION = 2
LEGACY_HISTORY_SCHEMA_VERSION = 1
FAVORITES_SCHEMA_VERSION = 1
BACKUP_SCHEMA_VERSION = 1
WORKSPACE_STORAGE_KEY = "danbooru_workspace_v1"
HISTORY_STORAGE_KEY = "danbooru_search_history_v1"
FAVORITES_STORAGE_KEY = "danbooru_favorites_v1"

LEGACY_STAGED_STORAGE_KEY = "danbooru_staged_tags"
SUPPORTED_PROMPT_FORMATS = {"sdxl", "nai", "anima"}
ARTIST_SELECTION_ORIGINS = {
    "artist_search",
    "artist_recommendation",
    "prompt_import_artist",
}
MAX_WORKSPACE_JSON_BYTES = 1_000_000
MAX_COLLECTION_JSON_BYTES = 4_000_000
MAX_BACKUP_JSON_BYTES = 12_000_000
MAX_SELECTED_TAGS = 2_000
MAX_QUERIES = 100
MAX_DISMISSED = 2_000
MAX_HISTORY_ITEMS = 100
MAX_FAVORITES = 200


class WorkspaceDataError(ValueError):
    """Raised when a persisted workspace cannot be safely consumed."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_workspace(*, prompt_format: str = "sdxl") -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "workspace_id": f"ws_{uuid.uuid4().hex}",
        "title": "",
        "queries": [],
        "selected": [],
        "dismissed": [],
        "prompt_format": (
            prompt_format if prompt_format in SUPPORTED_PROMPT_FORMATS else "sdxl"
        ),
        "updated_at": now,
    }


def _decode_json_object(
    raw: Any,
    *,
    label: str,
    max_bytes: int = MAX_WORKSPACE_JSON_BYTES,
) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        raise WorkspaceDataError(f"{label} is empty")
    if len(raw.encode("utf-8")) > max_bytes:
        raise WorkspaceDataError(f"{label} exceeds the size limit")
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise WorkspaceDataError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise WorkspaceDataError(f"{label} must be a JSON object")
    return value


def _clean_text(value: Any, *, max_length: int = 500) -> str:
    if value is None:
        return ""
    return str(value).strip()[:max_length]


def _clean_weight(value: Any) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(weight):
        return 1.0
    return round(min(5.0, max(0.1, weight)), 1)


def normalize_workspace(value: Any) -> tuple[dict[str, Any], list[str]]:
    """Validate a workspace and return a safe canonical copy plus warnings."""
    data = _decode_json_object(value, label="workspace")
    if data.get("schema_version") != WORKSPACE_SCHEMA_VERSION:
        raise WorkspaceDataError("unsupported workspace schema_version")

    warnings: list[str] = []
    workspace_id = _clean_text(data.get("workspace_id"), max_length=100)
    if not workspace_id:
        workspace_id = f"ws_{uuid.uuid4().hex}"
        warnings.append("workspace_id_missing")

    queries: list[dict[str, Any]] = []
    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list):
        warnings.append("queries_invalid")
        raw_queries = []
    for item in raw_queries[:MAX_QUERIES]:
        if not isinstance(item, dict):
            warnings.append("query_entry_invalid")
            continue
        query = _clean_text(item.get("query"), max_length=4_000)
        if not query:
            warnings.append("query_entry_empty")
            continue
        settings = item.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}
            warnings.append("query_settings_invalid")
        queries.append({
            "query": query,
            "searched_at": _clean_text(item.get("searched_at"), max_length=100)
            or utc_now_iso(),
            "settings": settings,
        })

    selected: list[dict[str, Any]] = []
    seen_tags: set[str] = set()
    raw_selected = data.get("selected", [])
    if not isinstance(raw_selected, list):
        warnings.append("selected_invalid")
        raw_selected = []
    for item in raw_selected[:MAX_SELECTED_TAGS]:
        if not isinstance(item, dict):
            warnings.append("selected_entry_invalid")
            continue
        tag = _clean_text(item.get("tag"), max_length=300)
        if not tag or tag in seen_tags:
            warnings.append("selected_entry_empty_or_duplicate")
            continue
        seen_tags.add(tag)
        selected.append({
            "tag": tag,
            "cn_name": _clean_text(item.get("cn_name"), max_length=1_000),
            "weight": _clean_weight(item.get("weight", 1.0)),
            "origin": _clean_text(item.get("origin"), max_length=100) or "unknown",
            "source": _clean_text(item.get("source"), max_length=1_000),
            "added_at": _clean_text(item.get("added_at"), max_length=100)
            or utc_now_iso(),
        })

    raw_dismissed = data.get("dismissed", [])
    dismissed: list[Any] = []
    if isinstance(raw_dismissed, list):
        for item in raw_dismissed[:MAX_DISMISSED]:
            if isinstance(item, str):
                dismissed.append(item[:300])
            elif isinstance(item, dict):
                try:
                    json.dumps(item, ensure_ascii=False)
                except (TypeError, ValueError):
                    warnings.append("dismissed_entry_invalid")
                    continue
                dismissed.append(item)
    if not isinstance(raw_dismissed, list):
        warnings.append("dismissed_invalid")

    prompt_format = data.get("prompt_format", "sdxl")
    if prompt_format not in SUPPORTED_PROMPT_FORMATS:
        prompt_format = "sdxl"
        warnings.append("prompt_format_invalid")

    normalized = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "title": _clean_text(data.get("title"), max_length=200),
        "queries": queries,
        "selected": selected,
        "dismissed": dismissed,
        "prompt_format": prompt_format,
        "updated_at": _clean_text(data.get("updated_at"), max_length=100)
        or utc_now_iso(),
    }
    return normalized, warnings


def migrate_legacy_workspace(
    legacy_staged: Any,
    legacy_config: Any = None,
) -> tuple[dict[str, Any], list[str]]:
    """Convert the pre-P0 selected-tag payload into ``WorkspaceState`` v1."""
    warnings: list[str] = []
    try:
        staged = _decode_json_object(legacy_staged, label="legacy staged tags")
    except WorkspaceDataError:
        staged = {}
        if legacy_staged:
            warnings.append("legacy_staged_invalid")

    try:
        config = _decode_json_object(legacy_config, label="legacy config")
    except WorkspaceDataError:
        config = {}
        if legacy_config:
            warnings.append("legacy_config_invalid")

    prompt_format = config.get("prompt_format", "sdxl")
    workspace = new_workspace(prompt_format=prompt_format)
    raw_tags = staged.get("tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []
        warnings.append("legacy_tags_invalid")
    raw_weights = staged.get("weights", {})
    if not isinstance(raw_weights, dict):
        raw_weights = {}
        warnings.append("legacy_weights_invalid")

    now = utc_now_iso()
    seen: set[str] = set()
    for raw_tag in raw_tags[:MAX_SELECTED_TAGS]:
        tag = _clean_text(raw_tag, max_length=300)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        workspace["selected"].append({
            "tag": tag,
            "cn_name": "",
            "weight": _clean_weight(raw_weights.get(tag, 1.0)),
            "origin": "legacy_migration",
            "source": "",
            "added_at": now,
        })
    return workspace, warnings


def sync_selected_entries(
    workspace: dict[str, Any],
    tags: list[str],
    weights: dict[str, float],
    cn_names: dict[str, str] | None = None,
    metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a workspace copy whose selected entries match the live UI state."""
    normalized, _ = normalize_workspace(workspace)
    old_by_tag = {item["tag"]: item for item in normalized["selected"]}
    cn_names = cn_names or {}
    metadata = metadata or {}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    now = utc_now_iso()
    for raw_tag in tags[:MAX_SELECTED_TAGS]:
        tag = _clean_text(raw_tag, max_length=300)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        old = old_by_tag.get(tag, {})
        meta = metadata.get(tag, {})
        selected.append({
            "tag": tag,
            "cn_name": _clean_text(cn_names.get(tag) or old.get("cn_name"), max_length=1_000),
            "weight": _clean_weight(weights.get(tag, old.get("weight", 1.0))),
            "origin": _clean_text(
                meta.get("origin") or old.get("origin"), max_length=100
            ) or "existing_selection",
            "source": _clean_text(
                meta.get("source") or old.get("source"), max_length=1_000
            ),
            "added_at": _clean_text(old.get("added_at"), max_length=100) or now,
        })
    normalized["selected"] = selected
    normalized["updated_at"] = now
    return normalized


def dump_workspace(workspace: dict[str, Any]) -> str:
    normalized, _ = normalize_workspace(workspace)
    try:
        raw = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise WorkspaceDataError("workspace contains non-serializable data") from exc
    if len(raw.encode("utf-8")) > MAX_WORKSPACE_JSON_BYTES:
        raise WorkspaceDataError("workspace exceeds the size limit")
    return raw


def clone_workspace(workspace: dict[str, Any]) -> dict[str, Any]:
    """Return a detached, validated WorkspaceState copy."""
    return normalize_workspace(dump_workspace(workspace))[0]


def append_workspace_query(
    workspace: dict[str, Any],
    query: str,
    settings: dict[str, Any],
    *,
    searched_at: str | None = None,
) -> dict[str, Any]:
    normalized = clone_workspace(workspace)
    query = _clean_text(query, max_length=4_000)
    if not query:
        return normalized
    record = {
        "query": query,
        "searched_at": searched_at or utc_now_iso(),
        "settings": settings if isinstance(settings, dict) else {},
    }
    normalized["queries"] = (normalized["queries"] + [record])[-MAX_QUERIES:]
    normalized["updated_at"] = record["searched_at"]
    return normalized


def workspace_signature(workspace: dict[str, Any]) -> str:
    normalized = clone_workspace(workspace)
    minimal = {
        "workspace_id": normalized["workspace_id"],
        "title": normalized["title"],
        "selected": [
            (item["tag"], item["weight"], item["origin"], item["source"])
            for item in normalized["selected"]
        ],
        "prompt_format": normalized["prompt_format"],
        "queries": normalized["queries"],
        "dismissed": normalized["dismissed"],
    }
    return json.dumps(minimal, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _history_signature(query: str, settings: dict[str, Any]) -> str:
    normalized_query = " ".join(query.split()).casefold()
    return json.dumps(
        [normalized_query, settings],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _history_workspace_snapshot(
    workspace: dict[str, Any],
    query: str,
    settings: dict[str, Any],
    searched_at: str,
) -> dict[str, Any]:
    """Create a restorable workspace snapshot without accumulated query history."""
    snapshot = clone_workspace(workspace)
    snapshot["queries"] = [{
        "query": query,
        "searched_at": searched_at,
        "settings": settings,
    }]
    snapshot["updated_at"] = searched_at
    return snapshot


def empty_history() -> dict[str, Any]:
    return {"schema_version": HISTORY_SCHEMA_VERSION, "items": []}


def normalize_history(value: Any) -> tuple[dict[str, Any], list[str]]:
    if value in (None, ""):
        return empty_history(), []
    data = _decode_json_object(
        value,
        label="history",
        max_bytes=MAX_COLLECTION_JSON_BYTES,
    )
    schema_version = data.get("schema_version")
    if schema_version not in {LEGACY_HISTORY_SCHEMA_VERSION, HISTORY_SCHEMA_VERSION}:
        raise WorkspaceDataError("unsupported history schema_version")
    raw_items = data.get("items", [])
    if not isinstance(raw_items, list):
        raise WorkspaceDataError("history items must be a list")

    items: list[dict[str, Any]] = []
    warnings: list[str] = []
    if schema_version == LEGACY_HISTORY_SCHEMA_VERSION:
        warnings.append("history_schema_migrated")
    seen: set[str] = set()
    for raw in raw_items:
        if len(items) >= MAX_HISTORY_ITEMS:
            break
        if not isinstance(raw, dict):
            warnings.append("history_entry_invalid")
            continue
        query = _clean_text(raw.get("query"), max_length=4_000)
        settings = raw.get("settings", {})
        if not query or not isinstance(settings, dict):
            warnings.append("history_entry_invalid")
            continue
        searched_at = _clean_text(raw.get("searched_at"), max_length=100) or utc_now_iso()
        try:
            workspace, workspace_warnings = normalize_workspace(raw.get("workspace"))
        except WorkspaceDataError:
            warnings.append("history_workspace_invalid")
            continue
        snapshot = _history_workspace_snapshot(
            workspace,
            query,
            settings,
            searched_at,
        )
        if workspace["queries"] != snapshot["queries"]:
            if "history_workspace_queries_compacted" not in warnings:
                warnings.append("history_workspace_queries_compacted")
        signature = _history_signature(query, settings)
        if signature in seen:
            warnings.append("history_entry_duplicate")
            continue
        seen.add(signature)
        warnings.extend(workspace_warnings)
        items.append({
            "history_id": _clean_text(raw.get("history_id"), max_length=100)
            or f"hist_{uuid.uuid4().hex}",
            "query": query,
            "searched_at": searched_at,
            "settings": settings,
            "workspace_id": snapshot["workspace_id"],
            "workspace": snapshot,
        })
    return {"schema_version": HISTORY_SCHEMA_VERSION, "items": items}, warnings


def add_history_entry(
    history: dict[str, Any],
    query: str,
    settings: dict[str, Any],
    workspace: dict[str, Any],
    *,
    searched_at: str | None = None,
) -> dict[str, Any]:
    normalized_history, _ = normalize_history(history)
    query = _clean_text(query, max_length=4_000)
    if not query:
        return normalized_history
    settings = settings if isinstance(settings, dict) else {}
    searched_at = searched_at or utc_now_iso()
    normalized_workspace = _history_workspace_snapshot(
        workspace,
        query,
        settings,
        searched_at,
    )
    signature = _history_signature(query, settings)
    remaining = [
        item for item in normalized_history["items"]
        if _history_signature(item["query"], item["settings"]) != signature
    ]
    entry = {
        "history_id": f"hist_{uuid.uuid4().hex}",
        "query": query,
        "searched_at": searched_at,
        "settings": settings,
        "workspace_id": normalized_workspace["workspace_id"],
        "workspace": normalized_workspace,
    }
    normalized_history["items"] = [entry] + remaining[:MAX_HISTORY_ITEMS - 1]
    return normalized_history


def merge_history(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    current_items = normalize_history(current)[0]["items"]
    incoming_items = normalize_history(incoming)[0]["items"]
    combined = sorted(
        incoming_items + current_items,
        key=lambda item: item.get("searched_at", ""),
        reverse=True,
    )
    result = empty_history()
    seen: set[str] = set()
    for item in combined:
        signature = _history_signature(item["query"], item["settings"])
        if signature in seen:
            continue
        seen.add(signature)
        result["items"].append(item)
        if len(result["items"]) >= MAX_HISTORY_ITEMS:
            break
    return result


def empty_favorites() -> dict[str, Any]:
    return {"schema_version": FAVORITES_SCHEMA_VERSION, "items": []}


def _normalize_favorite(raw: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(raw, dict):
        raise WorkspaceDataError("favorite must be an object")
    name = _clean_text(raw.get("name"), max_length=200)
    if not name:
        raise WorkspaceDataError("favorite name is required")
    selected_workspace = new_workspace(prompt_format=raw.get("prompt_format", "sdxl"))
    selected_workspace["selected"] = raw.get("selected", [])
    selected_workspace, warnings = normalize_workspace(selected_workspace)
    created_at = _clean_text(raw.get("created_at"), max_length=100) or utc_now_iso()
    return {
        "favorite_id": _clean_text(raw.get("favorite_id"), max_length=100)
        or f"fav_{uuid.uuid4().hex}",
        "name": name,
        "selected": selected_workspace["selected"],
        "prompt_format": selected_workspace["prompt_format"],
        "source_query": _clean_text(raw.get("source_query"), max_length=4_000),
        "notes": _clean_text(raw.get("notes"), max_length=2_000),
        "created_at": created_at,
        "updated_at": _clean_text(raw.get("updated_at"), max_length=100) or created_at,
    }, warnings


def normalize_favorites(value: Any) -> tuple[dict[str, Any], list[str]]:
    if value in (None, ""):
        return empty_favorites(), []
    data = _decode_json_object(
        value,
        label="favorites",
        max_bytes=MAX_COLLECTION_JSON_BYTES,
    )
    if data.get("schema_version") != FAVORITES_SCHEMA_VERSION:
        raise WorkspaceDataError("unsupported favorites schema_version")
    raw_items = data.get("items", [])
    if not isinstance(raw_items, list):
        raise WorkspaceDataError("favorites items must be a list")
    items: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    for raw in raw_items:
        if len(items) >= MAX_FAVORITES:
            break
        try:
            item, item_warnings = _normalize_favorite(raw)
        except WorkspaceDataError:
            warnings.append("favorite_entry_invalid")
            continue
        if item["favorite_id"] in seen_ids:
            warnings.append("favorite_id_duplicate")
            continue
        seen_ids.add(item["favorite_id"])
        warnings.extend(item_warnings)
        items.append(item)
    return {"schema_version": FAVORITES_SCHEMA_VERSION, "items": items}, warnings


def favorite_from_workspace(
    workspace: dict[str, Any],
    name: str,
    *,
    notes: str = "",
    favorite_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    normalized = clone_workspace(workspace)
    source_query = normalized["queries"][-1]["query"] if normalized["queries"] else ""
    now = utc_now_iso()
    favorite, _ = _normalize_favorite({
        "favorite_id": favorite_id or f"fav_{uuid.uuid4().hex}",
        "name": name,
        "selected": normalized["selected"],
        "prompt_format": normalized["prompt_format"],
        "source_query": source_query,
        "notes": notes,
        "created_at": created_at or now,
        "updated_at": now,
    })
    return favorite


def replace_with_favorite(favorite: dict[str, Any]) -> dict[str, Any]:
    favorite, _ = _normalize_favorite(favorite)
    workspace = new_workspace(prompt_format=favorite["prompt_format"])
    workspace["title"] = favorite["name"]
    now = utc_now_iso()
    workspace["selected"] = [
        {
            **item,
            "origin": (
                item.get("origin")
                if item.get("origin") in ARTIST_SELECTION_ORIGINS
                else "favorite_restore"
            ),
            "source": favorite["name"],
            "added_at": now,
        }
        for item in favorite["selected"]
    ]
    if favorite["source_query"]:
        workspace["queries"] = [{
            "query": favorite["source_query"],
            "searched_at": now,
            "settings": {},
        }]
    workspace["updated_at"] = now
    return normalize_workspace(workspace)[0]


def merge_favorite_into_workspace(
    workspace: dict[str, Any],
    favorite: dict[str, Any],
) -> dict[str, Any]:
    normalized = clone_workspace(workspace)
    favorite, _ = _normalize_favorite(favorite)
    existing = {item["tag"] for item in normalized["selected"]}
    now = utc_now_iso()
    for item in favorite["selected"]:
        if item["tag"] in existing:
            continue
        existing.add(item["tag"])
        normalized["selected"].append({
            **item,
            "origin": (
                item.get("origin")
                if item.get("origin") in ARTIST_SELECTION_ORIGINS
                else "favorite_restore"
            ),
            "source": favorite["name"],
            "added_at": now,
        })
    normalized["updated_at"] = now
    return normalize_workspace(normalized)[0]


def merge_workspaces(
    current: dict[str, Any],
    incoming: dict[str, Any],
    *,
    origin: str = "backup_import",
    source: str = "JSON backup",
) -> dict[str, Any]:
    """Merge selected tags without overwriting current weights or format."""
    normalized = clone_workspace(current)
    incoming = clone_workspace(incoming)
    existing = {item["tag"] for item in normalized["selected"]}
    now = utc_now_iso()
    for item in incoming["selected"]:
        if item["tag"] in existing:
            continue
        existing.add(item["tag"])
        normalized["selected"].append({
            **item,
            "origin": origin,
            "source": source,
            "added_at": now,
        })
    normalized["queries"] = (
        normalized["queries"] + incoming["queries"]
    )[-MAX_QUERIES:]
    normalized["dismissed"] = (
        normalized["dismissed"] + [
            item for item in incoming["dismissed"]
            if item not in normalized["dismissed"]
        ]
    )[:MAX_DISMISSED]
    normalized["updated_at"] = now
    return normalize_workspace(normalized)[0]


def merge_favorites(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    current_items = normalize_favorites(current)[0]["items"]
    incoming_items = normalize_favorites(incoming)[0]["items"]
    current_ids = {item["favorite_id"] for item in current_items}
    merged = list(current_items)
    merged.extend(item for item in incoming_items if item["favorite_id"] not in current_ids)
    merged.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return {"schema_version": FAVORITES_SCHEMA_VERSION, "items": merged[:MAX_FAVORITES]}


def dump_collection(
    value: dict[str, Any],
    *,
    label: str,
    max_bytes: int = MAX_COLLECTION_JSON_BYTES,
) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise WorkspaceDataError(f"{label} contains non-serializable data") from exc
    if len(raw.encode("utf-8")) > max_bytes:
        raise WorkspaceDataError(f"{label} exceeds the size limit")
    return raw


def build_backup(
    *,
    config: dict[str, Any],
    workspace: dict[str, Any],
    history: dict[str, Any],
    favorites: dict[str, Any],
) -> dict[str, Any]:
    backup = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "exported_at": utc_now_iso(),
        "config": config if isinstance(config, dict) else {},
        "workspace": clone_workspace(workspace),
        "history": normalize_history(history)[0],
        "favorites": normalize_favorites(favorites)[0],
    }
    dump_collection(backup, label="backup", max_bytes=MAX_BACKUP_JSON_BYTES)
    return backup


def normalize_backup(value: Any) -> tuple[dict[str, Any], list[str]]:
    data = _decode_json_object(
        value,
        label="backup",
        max_bytes=MAX_BACKUP_JSON_BYTES,
    )
    if data.get("schema_version") != BACKUP_SCHEMA_VERSION:
        raise WorkspaceDataError("unsupported backup schema_version")
    workspace, workspace_warnings = normalize_workspace(data.get("workspace"))
    history, history_warnings = normalize_history(data.get("history"))
    favorites, favorite_warnings = normalize_favorites(data.get("favorites"))
    config = data.get("config", {})
    if not isinstance(config, dict):
        config = {}
    return {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "exported_at": _clean_text(data.get("exported_at"), max_length=100),
        "config": config,
        "workspace": workspace,
        "history": history,
        "favorites": favorites,
    }, workspace_warnings + history_warnings + favorite_warnings
