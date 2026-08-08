"""推荐区域共享的状态与调度规则，不包含 NiceGUI 渲染。"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


def recommendation_seed_tags(controller: Any, selected_tags: list[str]) -> list[str]:
    """排除画师标签，避免把画师作为关联/同类推荐的种子。"""
    artist_tags = (
        set(controller._current_artist_rec_tags)
        | set(controller._artist_result_tags)
        | set(controller._workspace_artist_tags)
    )
    if controller.result_table is not None:
        for row in controller.result_table.rows:
            if row.get('layer') == 'artist' and row.get('tag'):
                artist_tags.add(row['tag'])
    return [tag for tag in selected_tags if tag not in artist_tags]


def merge_related_results(
    current_related: list,
    related: list | None,
    selected_tags: set[str],
) -> list:
    """刷新时保留仍被选中的旧关联标签，防止已选项从面板消失。"""
    related = list(related or [])
    new_tags = {item.tag for item in related}
    preserved = [
        item for item in current_related
        if item.tag in selected_tags and item.tag not in new_tags
    ]
    return related + preserved


def queue_latest_recommendation_request(
    controller: Any,
    selected_tags: list[str],
    show_nsfw: bool,
    scopes: set[str],
) -> bool:
    """合并同一选择快照的范围；返回是否需要启动消费任务。"""
    selected_tags = recommendation_seed_tags(controller, selected_tags)
    requested_scopes = frozenset(scopes)
    pending = controller._pending_recommendation_request
    if (
        pending is not None
        and pending['selected_tags'] == selected_tags
        and pending['show_nsfw'] == show_nsfw
    ):
        requested_scopes = pending['scopes'] | requested_scopes

    controller._recommendation_generation += 1
    controller._pending_recommendation_request = {
        'generation': controller._recommendation_generation,
        'selected_tags': list(selected_tags),
        'show_nsfw': show_nsfw,
        'scopes': requested_scopes,
    }
    return controller._recommendation_task is None or controller._recommendation_task.done()


def page_count(item_count: int, page_size: int) -> int:
    """返回分页数量；空列表没有有效页。"""
    if item_count <= 0 or page_size <= 0:
        return 0
    return (item_count + page_size - 1) // page_size


def clamp_page(page: int, total_pages: int) -> int:
    """把外部页码约束在有效范围内。"""
    return max(1, min(page, total_pages)) if total_pages > 0 else 0


def page_items(items: list, page: int, page_size: int) -> list:
    """返回有效页中的项目，空结果或无效页都返回空列表。"""
    total_pages = page_count(len(items), page_size)
    current_page = clamp_page(page, total_pages)
    if current_page == 0:
        return []
    start = (current_page - 1) * page_size
    return items[start:start + page_size]


def set_paginated_recommendation_page(
    controller: Any,
    page: int,
    *,
    state_prefix: str,
    render_page: Callable[[], None],
    motion_element_id: str,
) -> None:
    """切换关联或画师推荐页，并同步翻页按钮的状态。"""
    page_count_value = getattr(controller, f'{state_prefix}_page_count')
    if page_count_value < 1:
        return
    previous_page = getattr(controller, f'{state_prefix}_page')
    current_page = clamp_page(page, page_count_value)
    setattr(controller, f'{state_prefix}_page', current_page)
    render_page()
    motion_class = (
        'motion-recommendation-enter-right'
        if current_page >= previous_page
        else 'motion-recommendation-enter-left'
    )
    controller._replay_motion(motion_element_id, motion_class)

    label = getattr(controller, f'{state_prefix}_page_label')
    if label is not None:
        label.text = f'{current_page} / {page_count_value}'
    previous_button = getattr(controller, f'{state_prefix}_prev_button')
    if previous_button is not None:
        (previous_button.disable if current_page == 1 else previous_button.enable)()
    next_button = getattr(controller, f'{state_prefix}_next_button')
    if next_button is not None:
        (next_button.disable if current_page == page_count_value else next_button.enable)()


async def consume_latest_recommendation_requests(
    controller: Any,
    *,
    debounce_seconds: float,
    fetch: Callable[[dict], Awaitable[dict]],
    apply: Callable[[dict, dict], Awaitable[None]],
    client_alive: Callable[[], bool],
    report_error: Callable[[Exception], None],
) -> None:
    """顺序消费最新快照，计算完成后丢弃已经过期的结果。"""
    while controller._pending_recommendation_request is not None:
        await asyncio.sleep(debounce_seconds)
        request = controller._pending_recommendation_request
        controller._pending_recommendation_request = None
        try:
            result = await fetch(request)
        except Exception as exc:
            report_error(exc)
            continue
        if request['generation'] != controller._recommendation_generation:
            continue
        if not client_alive():
            return
        await apply(request, result)
