"""推荐区域的 NiceGUI 容器与分页控件渲染。"""

from typing import Any

from nicegui import ui

from webui.constants import (
    ARTIST_REC_LIMIT,
    ARTIST_REC_PAGE_SIZE,
    GROUP_RENDER_TAG_LIMIT,
    RELATED_REC_PAGE_SIZE,
)
from core.workspace_insights import (
    artist_candidate_reason,
    related_candidate_reason,
    tag_group_candidate_reason,
)
from webui.helpers import (
    group_names_key,
    group_scroll_dom_id,
    limit_group_render_tags,
    should_group_start_expanded,
)
from webui.recommendations import page_count, page_items


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


def render_related_page(controller: Any) -> None:
    """重建当前关联推荐页，节点数量固定不超过 10 条。"""
    controller.related_list_container.clear()
    controller._related_checkboxes.clear()

    if not controller._related_results:
        with controller.related_list_container:
            ui.label('暂无推荐').classes('text-sm text-gray-400 italic p-4')
        return

    selected_now = set(controller._get_selected_tags())
    page_results = page_items(
        controller._related_results,
        controller._related_page,
        RELATED_REC_PAGE_SIZE,
    )

    with controller.related_list_container:
        for r in page_results:
            tag = r.tag
            cn_first = r.cn_name.split(',')[0].strip() if r.cn_name else ''
            is_selected = tag in selected_now
            score_pct = f'+{r.cooc_score * 100:.0f}%'

            wiki_text = controller._lookup_tag_wiki(tag)

            sources_str = '、'.join(
                s.replace('tag_group:', '') for s in r.sources
            ) if r.sources else '—'
            CAT_LABEL = {'General': '通用', 'Character': '角色', 'Copyright': '作品'}
            cat_label = CAT_LABEL.get(r.category, '')
            tooltip_html = ''
            if wiki_text:
                prefix = f'<span style="opacity:0.7;margin-right:4px;">[{cat_label}]</span>' if cat_label else ''
                tooltip_html += f'<div style="margin-bottom:6px;">{prefix}{wiki_text}</div>'
            tooltip_html += (
                f'<div style="opacity:0.85;">'
                f'{r.cn_name}<br>'
                f'共现: {r.cooc_count:,}  相关度: {r.cooc_score:.2f}<br>'
                f'来自选中: {sources_str}'
                f'</div>'
            )

            # 行背景色按分类区分
            CAT_BG = {
                'General':   'background-color: rgba(59,130,246,0.06);',   # 淡蓝
                'Character': 'background-color: rgba(34,197,94,0.06);',    # 淡绿
                'Copyright': 'background-color: rgba(168,85,247,0.06);',   # 淡紫
            }
            row_bg = CAT_BG.get(r.category, '')

            # 整行容器，tooltip 挂在行上
            with ui.row().classes(
                'w-full flex-nowrap items-stretch gap-0 overflow-hidden '
                'related-item recommendation-row'
            ).style(row_bg):
                # 整行 wiki tooltip
                if tooltip_html:
                    with ui.tooltip().props('content-class="bg-black text-white shadow-4" max-width="500px"'):
                        ui.html(tooltip_html).style('font-size:14px;line-height:1.5;max-width:480px;')

                # Checkbox 单元格
                with ui.element('div').classes(
                    'recommendation-cell flex-none justify-center px-2 py-2'
                ):
                    cb = ui.checkbox(
                        '', value=is_selected,
                        on_change=lambda e, t=tag: controller._on_related_checkbox_change(t, e.value)
                    ).props('dense').classes('flex-none')
                    controller._related_checkboxes[tag] = cb

                # 标签名（可点击跳转）+ 中文名
                with ui.element('div').classes(
                    'recommendation-cell '
                    'flex-1 min-w-0 overflow-hidden px-3 py-2'
                ):
                    with ui.column().classes('w-full gap-0 min-w-0 overflow-hidden'):
                        with ui.row().classes('w-full flex-nowrap items-center gap-1 min-w-0 overflow-hidden'):
                            link = ui.link(
                                tag,
                                f'https://danbooru.donmai.us/wiki_pages/{tag}',
                                new_tab=True
                            ).classes(
                                'tag-link text-primary font-bold text-xs flex-1 min-w-0 truncate'
                            )
                            link.on('click', controller._mark_interaction)
                            if r.sources and r.sources[0].startswith('tag_group:'):
                                group_display = r.sources[0].replace('tag_group:', '')
                                ui.label(group_display).classes(
                                    'text-xs text-orange-500 font-bold bg-orange-50 px-1 rounded'
                                )

                        if cn_first:
                            ui.label(cn_first).classes('w-full text-xs text-gray-500 truncate')
                        ui.label(
                            related_candidate_reason(r.sources)
                        ).classes('w-full text-xs text-slate-500 truncate')

                # 关联分数单元格
                with ui.element('div').classes(
                    'recommendation-cell '
                    'flex-none justify-end min-w-16 px-3 py-2'
                ):
                    score_color = 'green' if r.cooc_score > 0.6 else ('teal' if r.cooc_score > 0.3 else 'grey')
                    ui.label(score_pct).classes(
                        f'text-sm font-bold text-{score_color}-600 whitespace-nowrap'
                    )

def render_artist_page(controller: Any) -> None:
    """重建当前画师页，节点数量固定不超过 ARTIST_REC_PAGE_SIZE。"""
    if controller.artist_rec_list is None:
        return
    controller.artist_rec_list.clear()
    controller._artist_rec_checkboxes.clear()
    controller._artist_rec_rows.clear()

    if not controller._artist_rec_results:
        with controller.artist_rec_list:
            ui.label('暂无推荐画师').classes('text-sm text-gray-400 italic p-4')
        return

    selected_now = set(controller._get_selected_tags())
    page_results = page_items(
        controller._artist_rec_results,
        controller._artist_rec_page,
        ARTIST_REC_PAGE_SIZE,
    )

    with controller.artist_rec_list:
        for r in page_results:
            artist = r.artist
            is_selected = artist in selected_now
            # 归一化：除以命中标签数，cap 到 100%
            normalized = min(r.score / max(r.hit_count, 1), 1.0)
            score_pct = f'+{normalized * 100:.0f}%'
            reason = artist_candidate_reason(r.sources)
            post_str = f'{r.post_count:,}' if r.post_count else '—'

            # tooltip：画师擅长标签
            tag_list = controller._artist_rec_top_tags.get(artist, [])
            tooltip_html = f'<div><b>{artist}</b><br>这位画师经常画:<br>'
            if tag_list:
                for t in tag_list[:10]:
                    tooltip_html += f'  · {t}<br>'
            else:
                tooltip_html += '  (无数据)'
            tooltip_html += '</div>'

            with ui.row().classes(
                'w-full flex-nowrap items-stretch gap-0 overflow-hidden '
                'related-item recommendation-row'
            ).style('background: rgba(244,114,182,0.04);') as row:
                controller._artist_rec_rows.append(row)
                # tooltip
                with ui.tooltip().props('content-class="bg-black text-white shadow-4" max-width="400px"'):
                    ui.html(tooltip_html).style('font-size:14px;line-height:1.5;max-width:380px;')

                # Checkbox 单元格
                with ui.element('div').classes(
                    'recommendation-cell flex-none justify-center px-2 py-2'
                ):
                    cb = ui.checkbox(
                        '', value=is_selected,
                        on_change=lambda e, t=artist: controller._on_artist_rec_checkbox_change(t, e.value)
                    ).props('dense')
                    controller._artist_rec_checkboxes[artist] = cb

                # 画师名 + 信息
                with ui.element('div').classes(
                    'recommendation-cell '
                    'flex-grow min-w-0 overflow-hidden px-3 py-2'
                ):
                    with ui.column().classes('w-full gap-0 min-w-0'):
                        ui.link(
                            artist,
                            f'https://danbooru.donmai.us/posts?tags={artist}',
                            new_tab=True,
                        ).classes('text-primary font-bold text-xs')
                        ui.label(reason).classes('text-xs text-slate-500')
                        ui.label(f'作品 {post_str}').classes('text-xs text-gray-400')

                # 分值单元格
                with ui.element('div').classes(
                    'recommendation-cell '
                    'flex-none justify-end min-w-16 px-3 py-2'
                ):
                    score_color = 'green' if normalized > 0.6 else ('teal' if normalized > 0.3 else 'grey')
                    ui.label(score_pct).classes(
                        f'text-sm font-bold text-{score_color}-600 whitespace-nowrap'
                    )

def render_group_expansion(controller: Any, group_data: list, selected_tags: list[str], show_nsfw: bool) -> None:
    """渲染 Group 同类扩展区域。"""
    if controller.group_expansion_container is None:
        return
    controller.group_expansion_container.clear()
    controller._group_checkboxes.clear()
    controller._group_candidate_sources.clear()
    group_key = group_names_key(group_data)
    if group_key != controller._group_render_key:
        controller._group_render_key = group_key
        controller._group_render_limits.clear()
        controller._group_expanded_names.clear()
        controller._group_scroll_positions.clear()

    if not group_data:
        with controller.group_expansion_container:
            ui.label('已选标签无分组信息').classes('text-sm text-gray-400 italic p-2')
        controller._replay_motion('danbooru-group-expansion', 'motion-refresh-enter')
        return

    # 行背景色按分类区分（与关联推荐一致）
    CAT_BG = {
        'General':   'background-color: rgba(59,130,246,0.06);',
        'Character': 'background-color: rgba(34,197,94,0.06);',
        'Copyright': 'background-color: rgba(168,85,247,0.06);',
    }
    CAT_LABEL = {'General': '通用', 'Character': '角色', 'Copyright': '作品'}

    selected_now = set(controller._get_selected_tags())

    with controller.group_expansion_container:
        for group_info in group_data:
            group_name = group_info['group']
            group_cn = group_info.get('group_cn_name', group_name.replace('tag_group:', ''))
            group_sources = list(group_info.get('sources') or [])
            group_reason = tag_group_candidate_reason(group_cn, group_sources)
            group_source_detail = group_cn
            if group_sources:
                group_source_detail += f"；触发标签：{'、'.join(group_sources[:3])}"
            tags = group_info['tags']
            visible_limit = controller._group_render_limits.get(group_name, GROUP_RENDER_TAG_LIMIT)
            visible_tags, hidden_count = limit_group_render_tags(tags, visible_limit)
            scroll_id = group_scroll_dom_id(group_name)

            expansion = ui.expansion(
                f'{group_cn} ({len(tags)} 个标签)',
                icon='label',
                value=should_group_start_expanded(group_name, controller._group_expanded_names),
            ).classes('w-full').props('dense')
            expansion.on(
                'update:model-value',
                lambda e, g=group_name: controller._on_group_expansion_change(g, e),
            )
            with expansion:
                with ui.element('div').props(
                    f'id="{scroll_id}" data-danbooru-group-scroll="1"'
                ).classes('w-full grid grid-cols-2 gap-1 p-1').style('max-height: 600px; overflow-y: auto;'):
                    for t in visible_tags:
                        tag = t['tag']
                        controller._group_candidate_sources.setdefault(tag, group_source_detail)
                        cn_first = t['cn_name'].split(',')[0].strip() if t['cn_name'] else ''
                        cn_full = t.get('cn_name', '')
                        cat = t['category']
                        wiki_text = str(t.get('wiki', ''))
                        row_bg = CAT_BG.get(cat, '')
                        is_selected = tag in selected_now

                        cat_label = CAT_LABEL.get(cat, '')
                        tooltip_html = ''
                        if wiki_text:
                            prefix = f'<span style="opacity:0.7;margin-right:4px;">[{cat_label}]</span>' if cat_label else ''
                            tooltip_html += f'<div style="margin-bottom:6px;">{prefix}{wiki_text}</div>'
                        if cn_full:
                            tooltip_html += f'<div style="opacity:0.85;">{cn_full}</div>'

                        with ui.row().classes(
                            'w-full min-w-0 flex-nowrap items-center gap-1.5 px-2 py-1.5 '
                            'rounded overflow-hidden related-item'
                        ).style(row_bg):
                            if tooltip_html:
                                with ui.tooltip().props('content-class="bg-black text-white shadow-4" max-width="500px"'):
                                    ui.html(tooltip_html).style('font-size:14px;line-height:1.5;max-width:480px;')

                            # 复选框
                            cb = ui.checkbox(
                                '', value=is_selected,
                                on_change=lambda e, t=tag: controller._on_group_checkbox_change(t, e.value),
                            ).props('dense').classes('flex-none')
                            controller._group_checkboxes[tag] = cb

                            # 标签名 + 中文名（与关联推荐对齐方式一致）
                            with ui.column().classes('flex-1 gap-0 min-w-0 overflow-hidden'):
                                link = ui.link(
                                    tag,
                                    f'https://danbooru.donmai.us/wiki_pages/{tag}',
                                    new_tab=True,
                                ).classes(
                                    'tag-link w-full min-w-0 text-primary font-bold text-xs truncate'
                                )
                                if cn_first:
                                    ui.label(cn_first).classes(
                                        'w-full text-xs text-gray-500 truncate'
                                    )
                                ui.label(group_reason).classes(
                                    'w-full text-xs text-slate-500 truncate'
                                )

                            # 热度
                            count = t['post_count']
                            if count > 0:
                                if count >= 10000:
                                    count_str = f'{count/1000:.0f}k'
                                elif count >= 1000:
                                    count_str = f'{count/1000:.1f}k'
                                else:
                                    count_str = str(count)
                                ui.label(count_str).classes(
                                    'flex-none ml-auto self-center text-sm font-bold '
                                    'text-grey-600 whitespace-nowrap'
                                )
                    if hidden_count > 0:
                        async def _load_more(
                            g=group_name,
                            total=len(tags),
                            gd=group_data,
                            st=list(selected_tags),
                            sn=show_nsfw,
                        ):
                            await controller._load_more_group_tags(g, total, gd, st, sn)

                        ui.button(
                            f'加载更多（剩余 {hidden_count} 个）',
                            icon='expand_more',
                            on_click=_load_more,
                        ).props('dense flat color=primary').classes('col-span-2 text-xs')
    controller._restore_group_scroll_positions()
    controller._replay_motion('danbooru-group-expansion', 'motion-refresh-enter')
