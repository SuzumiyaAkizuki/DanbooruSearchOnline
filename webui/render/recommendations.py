"""推荐区域的 NiceGUI 容器与分页控件渲染。"""

from typing import Any
from types import SimpleNamespace

from core.ui_performance import measure

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
    group_scroll_dom_id,
    limit_group_render_tags,
)
from webui.recommendations import page_count, page_items


@measure('related_render')
def render_related_list(controller: Any, related: list, show_nsfw: bool) -> None:
    """保存关联推荐快照，构建分页控件并渲染第一页。"""
    if controller.related_list_container is None or controller.related_pagination is None:
        return
    results = [item for item in related if not (item.nsfw == '1' and not show_nsfw)]
    signature = tuple((r.tag, r.cn_name, r.category, r.nsfw, r.cooc_count,
                       r.cooc_score, tuple(r.sources), controller._lookup_tag_wiki(r.tag)) for r in results)
    if (getattr(controller, '_related_signature', None) == signature
            and controller.related_list_container.default_slot.children):
        _sync_checks(controller, controller._related_checkboxes)
        return
    controller._related_signature = None
    controller.related_list_container.clear()
    controller.related_pagination.clear()
    controller._related_checkboxes.clear()
    controller._related_page = 1
    controller._related_page_label = None
    controller._related_prev_button = None
    controller._related_next_button = None
    controller._related_results = results
    controller._related_show_nsfw = show_nsfw
    controller._related_page_count = page_count(
        len(controller._related_results), RELATED_REC_PAGE_SIZE
    )

    if not controller._related_results:
        controller._render_related_page()
        controller._replay_motion(
            'danbooru-related-recommendations', 'motion-recommendation-enter-right'
        )
        controller._related_signature = signature
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
    controller._related_signature = signature


@measure('artist_render')
def render_artist_recommendations(
    controller: Any,
    artist_results: list,
    top_tags: dict | None = None,
    show_nsfw: bool = True,
) -> None:
    """保存画师推荐快照，构建分页控件并渲染第一页。"""
    if controller.artist_rec_list is None or controller.artist_rec_pagination is None:
        return
    signature = (tuple((r.artist, r.score, r.cooc_count, r.post_count, tuple(r.sources), r.hit_count)
                       for r in artist_results[:ARTIST_REC_LIMIT]),
                 tuple((r.artist, tuple((top_tags or {}).get(r.artist, [])[:10]))
                       for r in artist_results[:ARTIST_REC_LIMIT]), show_nsfw)
    if (getattr(controller, '_artist_signature', None) == signature
            and controller.artist_rec_list.default_slot.children):
        _sync_checks(controller, controller._artist_rec_checkboxes)
        return
    controller._artist_signature = None
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
        controller._artist_signature = signature
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
    controller._artist_signature = signature


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
                'General':   'background-color: var(--general-bg);',   # 淡蓝
                'Character': 'background-color: var(--character-bg);',    # 淡绿
                'Copyright': 'background-color: var(--copyright-bg);',   # 淡紫
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
                        on_change=lambda e, t=tag: _checkbox_changed(controller, '_on_related_checkbox_change', t, e.value)
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
            ).style('background: var(--artist-bg);') as row:
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
                        on_change=lambda e, t=artist: _checkbox_changed(controller, '_on_artist_rec_checkbox_change', t, e.value)
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

def _checkbox_changed(controller, method, tag, value):
    if not getattr(controller, '_syncing_recommendation_checks', False):
        getattr(controller, method)(tag, value)


def _sync_checks(controller, checks):
    previous = getattr(controller, '_syncing_recommendation_checks', False)
    controller._syncing_recommendation_checks = True
    try:
        selected = set(controller._get_selected_tags())
        for tag, checkbox in checks.items():
            if not checkbox.is_deleted and checkbox.value != (tag in selected):
                checkbox.set_value(tag in selected)
    finally:
        controller._syncing_recommendation_checks = previous


def _group_reason(info):
    name = info.get('group_cn_name', info['group'].replace('tag_group:', ''))
    return tag_group_candidate_reason(name, list(info.get('sources') or []))


def _group_changed(controller, name, event):
    controller._on_group_expansion_change(name, event)
    view = getattr(controller, '_group_views', {}).get(name)
    if view is not None and name in controller._group_expanded_names:
        render_group_page(controller, name)


def clear_group_expansion(controller):
    controller.group_expansion_container.clear()
    controller._group_views = {}
    controller._group_checkboxes.clear()
    controller._group_candidate_sources.clear()
    controller._group_render_limits.clear()
    controller._group_expanded_names.clear()
    controller._group_scroll_positions.clear()


@measure('group_render')
def render_group_expansion(controller: Any, group_data: list, selected_tags: list[str], show_nsfw: bool) -> None:
    root = controller.group_expansion_container
    if root is None:
        return
    views = getattr(controller, '_group_views', {})
    controller._group_views = views
    names = {info['group'] for info in group_data}
    for name in list(views):
        if name not in names or views[name].root.is_deleted:
            if not views[name].root.is_deleted:
                views[name].root.delete()
            del views[name]
            controller._group_render_limits.pop(name, None)
            controller._group_expanded_names.discard(name)
    if not views:
        root.clear()
    if not group_data:
        with root:
            ui.label('已选标签无分组信息').classes('text-sm text-gray-400 italic p-2')
    controller._group_checkboxes.clear()
    controller._group_candidate_sources.clear()
    for index, info in enumerate(group_data):
        name = info['group']
        view = views.get(name)
        if view is None:
            with root:
                expansion = ui.expansion('', icon='label', value=False).classes('w-full').props('dense')
                expansion.on('update:model-value', lambda e, g=name: _group_changed(controller, g, e))
            view = SimpleNamespace(root=expansion, body=None, rows={}, more=None, info=info,
                                   selected_tags=list(selected_tags), show_nsfw=show_nsfw)
            views[name] = view
        view.info, view.selected_tags, view.show_nsfw = info, list(selected_tags), show_nsfw
        label = f"{info.get('group_cn_name', name.replace('tag_group:', ''))} ({len(info['tags'])} 个标签)"
        if view.root.text != label:
            view.root.set_text(label)
        if root.default_slot.children[index] is not view.root:
            view.root.move(root, index)
        if view.body is not None or name in controller._group_expanded_names:
            render_group_page(controller, name)
        source = info.get('group_cn_name', name.replace('tag_group:', ''))
        sources = list(info.get('sources') or [])
        if sources:
            source += f"；触发标签：{'、'.join(sources[:3])}"
        for tag in view.rows:
            controller._group_candidate_sources.setdefault(tag, source)
        for tag, (_, checkbox, _) in view.rows.items():
            controller._group_checkboxes[tag] = checkbox
    controller._restore_group_scroll_positions()


@measure('group_page_render')
def render_group_page(controller, name):
    view = getattr(controller, '_group_views', {}).get(name)
    if view is None or view.root.is_deleted:
        return
    if view.body is None:
        with view.root:
            view.body = ui.element('div').props(
                f'id="{group_scroll_dom_id(name)}" data-danbooru-group-scroll="1"'
            ).classes('w-full grid grid-cols-2 gap-1 p-1').style('max-height: 600px; overflow-y: auto;')
    limit = controller._group_render_limits.get(name, GROUP_RENDER_TAG_LIMIT)
    tags, hidden = limit_group_render_tags(view.info['tags'], limit)
    reason = _group_reason(view.info)
    desired = {t['tag'] for t in tags}
    for tag in list(view.rows):
        if tag not in desired:
            row, checkbox, _ = view.rows.pop(tag)
            row.delete()
            if controller._group_checkboxes.get(tag) is checkbox:
                controller._group_checkboxes.pop(tag, None)
    for index, tag_info in enumerate(tags):
        tag = tag_info['tag']
        signature = (tag_info['cn_name'], tag_info['category'], tag_info['post_count'],
                     str(tag_info.get('wiki', '')), reason)
        old = view.rows.get(tag)
        if old is None or old[2] != signature:
            if old is not None:
                old[0].delete()
            with view.body:
                row, checkbox = _create_group_row(controller, tag_info, reason)
            view.rows[tag] = (row, checkbox, signature)
        row, checkbox, _ = view.rows[tag]
        if view.body.default_slot.children[index] is not row:
            row.move(view.body, index)
        controller._group_checkboxes[tag] = checkbox
        source = view.info.get('group_cn_name', name.replace('tag_group:', ''))
        sources = list(view.info.get('sources') or [])
        if sources:
            source += f"；触发标签：{'、'.join(sources[:3])}"
        controller._group_candidate_sources.setdefault(tag, source)
    _sync_checks(controller, {tag: item[1] for tag, item in view.rows.items()})
    if hidden:
        if view.more is None:
            async def load_more():
                await controller._load_more_group_tags(name, len(view.info['tags']),
                    [view.info], view.selected_tags, view.show_nsfw)
            with view.body:
                view.more = ui.button('', icon='expand_more', on_click=load_more).props(
                    'dense flat color=primary').classes('col-span-2 text-xs')
        text = f'加载更多（剩余 {hidden} 个）'
        if view.more.text != text:
            view.more.set_text(text)
        if view.body.default_slot.children[-1] is not view.more:
            view.more.move(view.body)
    elif view.more is not None:
        view.more.delete()
        view.more = None


def _create_group_row(controller, t, group_reason):
    CAT_BG = {'General': 'background-color: var(--general-bg);',
              'Character': 'background-color: var(--character-bg);',
              'Copyright': 'background-color: var(--copyright-bg);'}
    CAT_LABEL = {'General': '通用', 'Character': '角色', 'Copyright': '作品'}
    selected_now = set(controller._get_selected_tags())
    tag = t['tag']
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
    ).style(row_bg) as row:
        if tooltip_html:
            with ui.tooltip().props('content-class="bg-black text-white shadow-4" max-width="500px"'):
                ui.html(tooltip_html).style('font-size:14px;line-height:1.5;max-width:480px;')

        # 复选框
        cb = ui.checkbox(
            '', value=is_selected,
            on_change=lambda e, t=tag: _checkbox_changed(controller, '_on_group_checkbox_change', t, e.value),
        ).props('dense').classes('flex-none')

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
    return row, cb
