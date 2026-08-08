"""推荐区域的 NiceGUI 容器与分页控件渲染。"""

from typing import Any

from nicegui import ui

from webui.constants import (
    ARTIST_REC_LIMIT,
    ARTIST_REC_PAGE_SIZE,
    RELATED_REC_PAGE_SIZE,
)
from webui.recommendations import page_count


def render_related_list(controller: Any, related: list, show_nsfw: bool) -> None:
    """保存关联推荐快照，构建分页控件并渲染第一页。"""
    if controller.related_list_container is None or controller.related_pagination is None:
        return
    controller.related_list_container.clear()
    controller.related_pagination.clear()
    controller._related_checkboxes.clear()
    controller._related_page = 1
    controller._related_page_label = None
    controller._related_prev_button = None
    controller._related_next_button = None
    controller._related_results = [
        item for item in related
        if not (item.nsfw == '1' and not show_nsfw)
    ]
    controller._related_show_nsfw = show_nsfw
    controller._related_page_count = page_count(
        len(controller._related_results), RELATED_REC_PAGE_SIZE
    )

    if not controller._related_results:
        controller._render_related_page()
        controller._replay_motion(
            'danbooru-related-recommendations', 'motion-recommendation-enter-right'
        )
        return

    if controller._related_page_count > 1:
        with controller.related_pagination:
            with ui.row().classes('w-full items-center justify-center gap-2 px-3 py-2'):
                controller._related_prev_button = ui.button(
                    '‹',
                    on_click=lambda: controller._set_related_page(
                        controller._related_page - 1
                    ),
                ).props('flat dense round color=grey-7')
                controller._related_page_label = ui.label().classes(
                    'text-xs text-gray-600 min-w-12 text-center'
                )
                controller._related_next_button = ui.button(
                    '›',
                    on_click=lambda: controller._set_related_page(
                        controller._related_page + 1
                    ),
                ).props('flat dense round color=grey-7')

    controller._set_related_page(1)


def render_artist_recommendations(
    controller: Any,
    artist_results: list,
    top_tags: dict | None = None,
    show_nsfw: bool = True,
) -> None:
    """保存画师推荐快照，构建分页控件并渲染第一页。"""
    if controller.artist_rec_list is None or controller.artist_rec_pagination is None:
        return
    controller.artist_rec_list.clear()
    controller.artist_rec_pagination.clear()
    controller._artist_rec_checkboxes.clear()
    controller._artist_rec_rows.clear()
    controller._artist_rec_page = 1
    controller._artist_rec_page_label = None
    controller._artist_rec_prev_button = None
    controller._artist_rec_next_button = None
    controller._artist_rec_results = list(artist_results[:ARTIST_REC_LIMIT])
    controller._artist_rec_top_tags = dict(top_tags or {})
    controller._artist_rec_show_nsfw = show_nsfw
    controller._artist_rec_page_count = page_count(
        len(controller._artist_rec_results), ARTIST_REC_PAGE_SIZE
    )
    controller._current_artist_rec_tags = {
        result.artist for result in controller._artist_rec_results
    }
    controller._artist_rec_sources = {
        result.artist: '、'.join(result.sources[:3])
        for result in controller._artist_rec_results
    }

    if not controller._artist_rec_results:
        controller._render_artist_rec_page()
        controller._replay_motion(
            'danbooru-artist-recommendations', 'motion-recommendation-enter-right'
        )
        return

    if controller._artist_rec_page_count > 1:
        with controller.artist_rec_pagination:
            with ui.row().classes('w-full items-center justify-center gap-2 px-3 py-2'):
                controller._artist_rec_prev_button = ui.button(
                    '‹',
                    on_click=lambda: controller._set_artist_rec_page(
                        controller._artist_rec_page - 1
                    ),
                ).props('flat dense round color=grey-7')
                controller._artist_rec_page_label = ui.label().classes(
                    'text-xs text-gray-600 min-w-12 text-center'
                )
                controller._artist_rec_next_button = ui.button(
                    '›',
                    on_click=lambda: controller._set_artist_rec_page(
                        controller._artist_rec_page + 1
                    ),
                ).props('flat dense round color=grey-7')

    controller._set_artist_rec_page(1)
