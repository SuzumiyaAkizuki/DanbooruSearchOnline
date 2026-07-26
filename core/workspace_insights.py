"""Explainable workspace candidate reasons and concept-coverage helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


COVERED = "covered"
CANDIDATE_UNSELECTED = "candidate_unselected"
UNCOVERED = "uncovered"


@dataclass(frozen=True)
class ConceptCoverageItem:
    segment: str
    status: str
    candidate_tags: tuple[str, ...] = ()
    selected_tags: tuple[str, ...] = ()


def _clean_sources(sources: Iterable[Any], *, limit: int = 3) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for source in sources:
        value = str(source or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def semantic_candidate_reason(source: Any, layer: Any = "") -> str:
    source_text = str(source or "").strip()
    if str(layer or "") == "artist":
        return f"匹配画师名称：{source_text}" if source_text else "来自画师名称匹配"
    return f"匹配输入：{source_text}" if source_text else "来自语义搜索"


def related_candidate_reason(sources: Iterable[Any]) -> str:
    values = _clean_sources(sources)
    if not values:
        return "来自标签共现推荐"
    return f"与已选 {'、'.join(values)} 经常共同出现"


def tag_group_candidate_reason(group_name: Any, sources: Iterable[Any]) -> str:
    group_text = str(group_name or "").strip()
    values = _clean_sources(sources)
    if values and group_text:
        return f"与已选 {'、'.join(values)} 属于同一标签组：{group_text}"
    if values:
        return f"与已选 {'、'.join(values)} 属于同一标签组"
    if group_text:
        return f"来自标签组：{group_text}"
    return "来自标签组扩展"


def artist_candidate_reason(sources: Iterable[Any]) -> str:
    values = _clean_sources(sources)
    if not values:
        return "根据当前已选视觉标签推荐"
    return f"根据已选 {'、'.join(values)} 推荐"


def selected_tag_reason(origin: Any, source: Any) -> str:
    origin_text = str(origin or "")
    source_text = str(source or "").strip()
    if origin_text == "semantic_search":
        return f"来自搜索：{source_text}" if source_text else "来自语义搜索"
    if origin_text == "related_recommendation":
        return f"来自关联推荐：{source_text}" if source_text else "来自关联推荐"
    if origin_text == "tag_group":
        return f"来自标签组：{source_text}" if source_text else "来自标签组"
    if origin_text in {"artist_search", "artist_recommendation"}:
        return f"根据已选 {source_text} 推荐" if source_text else "根据当前已选视觉标签推荐"
    if origin_text in {"prompt_import", "prompt_import_artist"}:
        return "来自导入 Prompt"
    if origin_text == "favorite_restore":
        return f"来自收藏：{source_text}" if source_text else "来自收藏恢复"
    if origin_text == "backup_import":
        return "来自 JSON 备份"
    return f"来源：{source_text}" if source_text else "来源信息不可用"


def compute_concept_coverage(
    segments: Iterable[Any],
    results: Iterable[dict[str, Any]],
    selected_tags: Iterable[Any],
    *,
    min_score: float = 0.45,
) -> list[ConceptCoverageItem]:
    """Approximate coverage using the engine's real per-result source field."""
    ordered_segments = _clean_sources(segments, limit=1_000)
    selected = {str(tag) for tag in selected_tags if str(tag or "").strip()}
    result_rows = list(results)
    coverage: list[ConceptCoverageItem] = []

    for segment in ordered_segments:
        candidates: list[str] = []
        seen_candidates: set[str] = set()
        for row in result_rows:
            if str(row.get("source") or "") != segment:
                continue
            try:
                score = float(row.get("final_score", 0.0))
            except (TypeError, ValueError):
                continue
            if score < min_score:
                continue
            tag = str(row.get("tag") or "").strip()
            if not tag or tag in seen_candidates:
                continue
            seen_candidates.add(tag)
            candidates.append(tag)

        selected_candidates = [tag for tag in candidates if tag in selected]
        if selected_candidates:
            status = COVERED
        elif candidates:
            status = CANDIDATE_UNSELECTED
        else:
            status = UNCOVERED
        coverage.append(ConceptCoverageItem(
            segment=segment,
            status=status,
            candidate_tags=tuple(candidates),
            selected_tags=tuple(selected_candidates),
        ))
    return coverage
