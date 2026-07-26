"""Prompt import parsing and conservative workspace grouping helpers."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .workspace import clone_workspace, utc_now_iso


MAX_PROMPT_IMPORT_TOKENS = 2_000

WORKSPACE_GROUP_ORDER = (
    "主体/角色",
    "外貌",
    "服装",
    "动作/姿势",
    "表情",
    "场景/背景",
    "构图/镜头",
    "画师",
    "其他",
)

_GROUP_RULES: dict[str, set[str]] = {
    "主体/角色": {
        "tag_group:birds", "tag_group:cats", "tag_group:character_count",
        "tag_group:dogs", "tag_group:family_relationships", "tag_group:groups",
        "tag_group:jobs", "tag_group:legendary_creatures", "tag_group:people",
    },
    "外貌": {
        "tag_group:ass", "tag_group:body_parts", "tag_group:breasts_tags",
        "tag_group:ears_tags", "tag_group:eyes_tags", "tag_group:feet",
        "tag_group:hair", "tag_group:hair_color", "tag_group:hair_styles",
        "tag_group:hands", "tag_group:makeup", "tag_group:piercings",
        "tag_group:pussy", "tag_group:shoulders", "tag_group:skin_color",
        "tag_group:wings",
    },
    "服装": {
        "tag_group:accessories", "tag_group:attire", "tag_group:eyewear",
        "tag_group:fashion_style", "tag_group:handwear", "tag_group:headwear",
        "tag_group:legwear", "tag_group:neck_and_neckwear", "tag_group:nudity",
        "tag_group:patterns", "tag_group:sexual_attire", "tag_group:sleeves",
    },
    "动作/姿势": {
        "tag_group:bdsm_and_torture", "tag_group:covering", "tag_group:dances",
        "tag_group:gestures", "tag_group:holding_tags", "tag_group:posture",
        "tag_group:sex_acts", "tag_group:sexual_positions",
        "tag_group:simulated_sex_acts", "tag_group:sports",
        "tag_group:verbs_and_gerunds",
    },
    "表情": {"tag_group:face_tags"},
    "场景/背景": {
        "tag_group:backgrounds", "tag_group:fire", "tag_group:flowers",
        "tag_group:holidays_and_celebrations", "tag_group:lighting",
        "tag_group:locations", "tag_group:real_world_locations",
        "tag_group:theme", "tag_group:water",
    },
    "构图/镜头": {
        "tag_group:artistic_license", "tag_group:focus_tags",
        "tag_group:image_composition", "tag_group:visual_aesthetic",
    },
}

# 分类优先级与展示顺序不同：更具体的 Tag Group 先于宽泛的外貌/面部集合。
_GROUP_CLASSIFICATION_PRIORITY = (
    "主体/角色",
    "服装",
    "动作/姿势",
    "场景/背景",
    "构图/镜头",
    "外貌",
    "表情",
)


@dataclass(frozen=True)
class ParsedPromptToken:
    raw: str
    name: str
    weight: float = 1.0
    is_artist: bool = False


@dataclass(frozen=True)
class PromptResolvedItem:
    original: str
    tag: str
    weight: float
    matched_by: str
    is_artist: bool = False
    cn_name: str = ""
    category: str = "Other"


@dataclass(frozen=True)
class PromptCorrection:
    original: str
    canonical: str
    matched_by: str


@dataclass(frozen=True)
class PromptPendingItem:
    original: str
    normalized: str
    weight: float
    reason: str
    is_artist: bool = False
    candidates: tuple[str, ...] = ()
    alias_target: str = ""


@dataclass
class PromptImportResult:
    parsed_count: int = 0
    items: list[PromptResolvedItem] = field(default_factory=list)
    corrections: list[PromptCorrection] = field(default_factory=list)
    pending: list[PromptPendingItem] = field(default_factory=list)
    duplicate_count: int = 0


@dataclass
class WorkspaceCanonicalizationResult:
    workspace: dict[str, Any]
    corrections: list[PromptCorrection] = field(default_factory=list)
    pending: list[PromptPendingItem] = field(default_factory=list)
    duplicate_count: int = 0


def _clean_weight(value: float) -> float:
    return round(min(5.0, max(0.1, float(value))), 1)


def _unescape_tag_name(value: str) -> str:
    return re.sub(r"\\([(){}\[\]])", r"\1", value).strip()


def _extract_explicit_weight(value: str) -> tuple[str, float | None]:
    nai = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*::\s*(.+?)\s*::\s*", value)
    if nai:
        return nai.group(2).strip(), float(nai.group(1))

    weighted = re.fullmatch(
        r"\s*\(\s*(.+)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*",
        value,
    )
    if weighted:
        return weighted.group(1).strip(), float(weighted.group(2))
    return value.strip(), None


def parse_prompt_text(text: str) -> list[ParsedPromptToken]:
    """Parse the supported comma/newline Prompt forms without resolving names."""
    if not isinstance(text, str):
        return []

    parsed: list[ParsedPromptToken] = []
    for raw_part in re.split(r"[,，\r\n]+", text)[:MAX_PROMPT_IMPORT_TOKENS]:
        raw = raw_part.strip()
        if not raw:
            continue

        value, explicit_weight = _extract_explicit_weight(raw)
        emphasis = 1.0
        while len(value) >= 2:
            if value.startswith("{") and value.endswith("}"):
                emphasis *= 1.1
                value = value[1:-1].strip()
                continue
            if value.startswith("[") and value.endswith("]"):
                emphasis *= 0.9
                value = value[1:-1].strip()
                continue
            break

        # 支持如 {(tag:1.2)} 这类组合写法。
        value, nested_weight = _extract_explicit_weight(value)
        weight = explicit_weight if explicit_weight is not None else nested_weight
        weight = (weight if weight is not None else 1.0) * emphasis

        value = _unescape_tag_name(value)
        is_artist = value.startswith("@")
        if is_artist:
            value = value[1:].strip()
        if not value:
            continue
        parsed.append(ParsedPromptToken(
            raw=raw,
            name=value,
            weight=_clean_weight(weight),
            is_artist=is_artist,
        ))
    return parsed


def _normalized_compare_name(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def resolve_prompt_text(
    text: str,
    *,
    resolve_tag: Callable[[str], dict[str, Any]],
    resolve_artist: Callable[[str], dict[str, Any]],
    lookup_tag: Callable[[str], dict[str, Any] | None],
    allow_nsfw: bool,
) -> PromptImportResult:
    """Resolve Prompt tokens through the existing canonical resolvers."""
    tokens = parse_prompt_text(text)
    result = PromptImportResult(parsed_count=len(tokens))
    seen: set[str] = set()

    for token in tokens:
        resolved = resolve_artist(token.name) if token.is_artist else resolve_tag(token.name)
        key = "artist" if token.is_artist else "tag"
        canonical = str(resolved.get(key) or "").strip()
        matched_by = str(resolved.get("matched_by") or "not_found")

        if not canonical:
            result.pending.append(PromptPendingItem(
                original=token.raw,
                normalized=_normalized_compare_name(token.name),
                weight=token.weight,
                reason=matched_by,
                is_artist=token.is_artist,
                candidates=tuple(str(item) for item in resolved.get("candidates", [])[:5]),
                alias_target=str(resolved.get("alias_target") or ""),
            ))
            continue

        metadata = {} if token.is_artist else (lookup_tag(canonical) or {})
        if not token.is_artist and not allow_nsfw and str(metadata.get("nsfw", "0")) == "1":
            result.pending.append(PromptPendingItem(
                original=token.raw,
                normalized=_normalized_compare_name(token.name),
                weight=token.weight,
                reason="nsfw_filtered",
                candidates=(canonical,),
            ))
            continue

        if canonical in seen:
            result.duplicate_count += 1
            continue
        seen.add(canonical)

        result.items.append(PromptResolvedItem(
            original=token.raw,
            tag=canonical,
            weight=token.weight,
            matched_by=matched_by,
            is_artist=token.is_artist,
            cn_name="画师标签" if token.is_artist else str(metadata.get("cn_name") or ""),
            category="Artist" if token.is_artist else str(metadata.get("category") or "Other"),
        ))

        if token.name.strip() != canonical:
            result.corrections.append(PromptCorrection(
                original=token.name,
                canonical=canonical,
                matched_by=matched_by,
            ))

    return result


def pending_to_workspace_entry(
    item: PromptPendingItem,
    *,
    source: str = "Prompt 导入",
) -> dict[str, Any]:
    return {
        "kind": "prompt_import_pending",
        "pending_id": f"pending_{uuid.uuid4().hex}",
        "original": item.original,
        "normalized": item.normalized,
        "weight": item.weight,
        "reason": item.reason,
        "is_artist": item.is_artist,
        "candidates": list(item.candidates),
        "alias_target": item.alias_target,
        "source": source,
        "added_at": utc_now_iso(),
    }


def canonicalize_workspace_tags(
    workspace: dict[str, Any],
    *,
    resolve_tag: Callable[[str], dict[str, Any]],
    resolve_artist: Callable[[str], dict[str, Any]],
    lookup_tag: Callable[[str], dict[str, Any] | None],
    artist_origins: set[str] | frozenset[str],
    source: str,
) -> WorkspaceCanonicalizationResult:
    """Normalize tags when loading historical or externally stored workspaces."""
    normalized = clone_workspace(workspace)
    selected: list[dict[str, Any]] = []
    corrections: list[PromptCorrection] = []
    pending: list[PromptPendingItem] = []
    seen: set[str] = set()
    duplicate_count = 0
    existing_pending = {
        (
            item.get("normalized"),
            bool(item.get("is_artist")),
            item.get("reason"),
            item.get("alias_target"),
        )
        for item in normalized.get("dismissed", [])
        if isinstance(item, dict) and item.get("kind") == "prompt_import_pending"
    }

    for entry in normalized.get("selected", []):
        original = str(entry.get("tag") or "").strip()
        is_artist = entry.get("origin") in artist_origins
        resolved = resolve_artist(original) if is_artist else resolve_tag(original)
        key = "artist" if is_artist else "tag"
        canonical = str(resolved.get(key) or "").strip()
        matched_by = str(resolved.get("matched_by") or "not_found")
        if not canonical:
            unresolved = PromptPendingItem(
                original=original,
                normalized=_normalized_compare_name(original),
                weight=float(entry.get("weight", 1.0)),
                reason=matched_by,
                is_artist=is_artist,
                candidates=tuple(str(item) for item in resolved.get("candidates", [])[:5]),
                alias_target=str(resolved.get("alias_target") or ""),
            )
            pending_key = (
                unresolved.normalized,
                unresolved.is_artist,
                unresolved.reason,
                unresolved.alias_target,
            )
            if pending_key in existing_pending:
                duplicate_count += 1
            else:
                existing_pending.add(pending_key)
                pending.append(unresolved)
            continue
        if canonical in seen:
            duplicate_count += 1
            continue
        seen.add(canonical)

        updated = dict(entry)
        updated["tag"] = canonical
        if not is_artist:
            metadata = lookup_tag(canonical) or {}
            if metadata.get("cn_name"):
                updated["cn_name"] = str(metadata["cn_name"])
        selected.append(updated)

        if original != canonical:
            corrections.append(PromptCorrection(
                original=original,
                canonical=canonical,
                matched_by=matched_by,
            ))

    normalized["selected"] = selected
    if pending:
        normalized["dismissed"] = list(normalized.get("dismissed", [])) + [
            pending_to_workspace_entry(item, source=source) for item in pending
        ]
    normalized = clone_workspace(normalized)
    return WorkspaceCanonicalizationResult(
        workspace=normalized,
        corrections=corrections,
        pending=pending,
        duplicate_count=duplicate_count,
    )


def classify_workspace_tag(
    *,
    category: str,
    tag_groups: set[str] | list[str] | tuple[str, ...],
    is_artist: bool = False,
) -> str:
    """Map only stable category/Tag Group evidence; otherwise return 其他."""
    if is_artist or category == "Artist":
        return "画师"
    if category == "Character":
        return "主体/角色"

    groups = set(tag_groups)
    for dimension in _GROUP_CLASSIFICATION_PRIORITY:
        if groups.intersection(_GROUP_RULES[dimension]):
            return dimension
    return "其他"
