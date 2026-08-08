"""工作区工具栏和已选标签区域的基础视图。"""

from typing import Any

from nicegui import ui

from core.prompt_import import (
    PromptImportResult,
    WORKSPACE_GROUP_ORDER,
    WorkspaceCanonicalizationResult,
    pending_to_workspace_entry,
)
from core.workspace_insights import selected_tag_reason
from webui.helpers import format_selected_tag_label


def build_workspace_toolbar(controller: Any) -> None:
    """构建工作区历史、收藏、导入与备份入口。"""
    with ui.element('div').classes(
        'w-full bg-slate-50 border-b border-slate-200 px-4 py-2'
    ):
        with ui.row().classes('w-full items-center gap-2 flex-wrap'):
            ui.icon('workspaces', color='primary')
            ui.label('标签工作区').classes('section-heading mr-2')
            controller.undo_btn = ui.button(
                '撤销', icon='undo', on_click=controller._undo_workspace,
            ).props('dense flat color=grey-7')
            controller.redo_btn = ui.button(
                '恢复', icon='redo', on_click=controller._redo_workspace,
            ).props('dense flat color=grey-7')
            controller.undo_btn.disable()
            controller.redo_btn.disable()
            ui.separator().props('vertical').classes('h-7 mx-1')
            ui.button('历史', icon='history', on_click=controller._open_history_dialog).props(
                'dense flat color=primary'
            )
            controller.history_count_label = ui.label('0').classes('text-xs text-gray-500 -ml-2')
            ui.button('收藏', icon='star_outline', on_click=controller._open_favorites_dialog).props(
                'dense flat color=amber-8'
            )
            controller.favorites_count_label = ui.label('0').classes('text-xs text-gray-500 -ml-2')
            ui.button('保存收藏', icon='bookmark_add', on_click=controller._open_save_favorite_dialog).props(
                'dense flat color=teal-7'
            )
            ui.button('导入 Prompt', icon='playlist_add', on_click=controller._open_prompt_import_dialog).props(
                'dense flat color=purple-7'
            )
            ui.button('备份 / 迁移', icon='swap_horiz', on_click=controller._open_backup_dialog).props(
                'dense flat color=grey-7'
            )
    controller._update_workspace_counts()


def build_selection_bar(controller: Any) -> None:
    """构建已选标签栏及其操作入口。"""
    controller.selection_bar_card = ui.element('div').classes('w-full bg-blue-50 p-4')
    with controller.selection_bar_card:
        with ui.row().classes('w-full items-center justify-between'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('check_circle', color='primary')
                ui.label('已选标签').classes('font-bold text-primary')
                controller.selection_count_label = ui.label('0').classes(
                    'bg-primary text-white px-2 rounded-full text-sm'
                )
                with ui.icon('info_outline', size='sm', color='grey').classes('cursor-help'):
                    with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                        ui.html(
                            '点击 <b>−</b> / <b>+</b> 可调整标签权重（步长 0.1，范围 0.1~1.9）。<br>'
                            '权重 1.0 时输出原始标签；其余输出 <code>(tag:1.2)</code> 格式。'
                        ).style('font-size:14px;line-height:1.6;')

            with ui.row().classes('items-center gap-2'):
                with ui.button('没搜到？', icon='help_outline').props(
                    'dense flat color=grey-6'
                ).classes('text-sm') as bad_button:
                    with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                        ui.html(
                            '点击此处以反馈失败案例。<br>您的搜索词将被匿名收集用于优化引擎（不包含个人隐私）。'
                        ).style('font-size:14px;line-height:1.5;')
                controller.bad_case_btn = bad_button
                controller.bad_case_btn.disable()
                controller.bad_case_btn.on_click(controller.report_bad_case)
                controller.format_toggle_btn = ui.button('SDXL', icon='swap_horiz').props(
                    'dense flat color=grey-7'
                ).classes('text-xs font-mono')
                with controller.format_toggle_btn:
                    with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                        ui.html(
                            '切换复制格式：<br>'
                            '<b>SDXL</b>：<code>(tag:1.2)</code><br>'
                            '<b>NAI</b>：<code>1.2::tag::</code><br>'
                            '<b>Anima</b>：<code>(tag:1.5)</code> 下划线→空格'
                        ).style('font-size:13px;line-height:1.7;')
                controller.format_toggle_btn.on_click(controller._toggle_prompt_format)
                clear_button = ui.button('清空已选', icon='delete_sweep').props(
                    'dense flat color=red-7'
                ).classes('text-xs')
                clear_button.on_click(controller._clear_all_staged)
                copy_button = ui.button('复制选中', icon='content_copy').props(
                    'dense unelevated color=primary'
                )
                copy_button.on_click(controller.copy_selection)

        controller.selected_chips_container = ui.element('div').classes(
            'w-full mt-2 min-h-10 p-1 rounded bg-white border border-blue-100'
        ).props('id="danbooru-selected-chips"')
        controller.prompt_pending_container = ui.column().classes('w-full gap-2 mt-2')


def show_prompt_import_summary(controller: Any, result: PromptImportResult, added_count: int, duplicate_count: int) -> None:
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-3xl max-h-[85vh]'):
        ui.label('Prompt 导入结果').classes('text-lg font-bold')
        ui.label(
            f'解析 {result.parsed_count} 项 · 新增 {added_count} 项 · '
            f'纠正 {len(result.corrections)} 项 · 重复合并 {duplicate_count} 项 · '
            f'待确认 {len(result.pending)} 项'
        ).classes('text-sm text-gray-700 bg-slate-50 rounded p-3')
        with ui.scroll_area().classes('w-full max-h-[58vh]'):
            if result.corrections:
                ui.label('名称规范化').classes('font-bold text-teal-700 mt-2')
                for correction in result.corrections:
                    ui.label(
                        f'{correction.original} → {correction.canonical}'
                    ).classes('text-sm font-mono text-teal-800')
            if result.pending:
                ui.label('未加入，已进入待确认区').classes(
                    'font-bold text-orange-700 mt-3'
                )
                for item in result.pending:
                    pending_record = pending_to_workspace_entry(item)
                    ui.label(
                        f'{item.original}：{controller._prompt_pending_reason(pending_record)}'
                    ).classes('text-sm text-orange-800 break-all')
        with ui.row().classes('w-full justify-end'):
            ui.button('知道了', on_click=dialog.close).props('unelevated color=primary')
    dialog.open()

def render_prompt_pending(controller: Any) -> None:
    if controller.prompt_pending_container is None:
        return
    controller.prompt_pending_container.clear()
    pending_items = [
        item for item in controller.workspace_state.get('dismissed', [])
        if isinstance(item, dict) and item.get('kind') == 'prompt_import_pending'
    ]
    if not pending_items:
        return

    with controller.prompt_pending_container:
        with ui.expansion(
            f'待确认内容（{len(pending_items)}）', icon='help_outline', value=True,
        ).classes('w-full bg-orange-50 border border-orange-200 rounded'):
            ui.label(
                '以下内容不会出现在复制结果中。可选择可靠候选，或从待确认区移除。'
            ).classes('text-xs text-orange-700 mb-2')
            for item in pending_items:
                with ui.row().classes(
                    'w-full items-center justify-between gap-2 border-t border-orange-100 py-2'
                ):
                    with ui.column().classes('gap-0 min-w-0 flex-grow'):
                        ui.label(str(item.get('original') or '')).classes(
                            'text-sm font-mono text-gray-800 break-all'
                        )
                        ui.label(controller._prompt_pending_reason(item)).classes(
                            'text-xs text-orange-700'
                        )
                    with ui.row().classes('gap-1 flex-wrap justify-end'):
                        if item.get('reason') not in {'alias_target_missing', 'nsfw_filtered'}:
                            for candidate in item.get('candidates', [])[:5]:
                                ui.button(
                                    str(candidate),
                                    on_click=lambda i=item, c=str(candidate):
                                        controller._accept_prompt_candidate(i, c),
                                ).props('flat dense color=teal-7').classes('text-xs font-mono')
                        ui.button(
                            icon='close',
                            on_click=lambda i=item: controller._remove_prompt_pending(i),
                        ).props('flat round dense color=grey-6')

def show_workspace_canonicalization(controller: Any, result: WorkspaceCanonicalizationResult, label: str) -> None:
    if not result.corrections and not result.pending and not result.duplicate_count:
        return
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-2xl'):
        ui.label(f'{label}标签规范化结果').classes('text-lg font-bold')
        ui.label(
            f'纠正 {len(result.corrections)} 项 · '
            f'重复合并 {result.duplicate_count} 项 · '
            f'待确认 {len(result.pending)} 项'
        ).classes('text-sm text-gray-600')
        if result.corrections:
            with ui.column().classes('w-full gap-1'):
                for correction in result.corrections:
                    ui.label(
                        f'{correction.original} → {correction.canonical}'
                    ).classes('text-sm font-mono text-teal-800')
        if result.pending:
            ui.label('未识别内容已移入工作区待确认区，不会进入复制结果。').classes(
                'text-sm text-orange-700'
            )
        with ui.row().classes('w-full justify-end'):
            ui.button('知道了', on_click=dialog.close).props('unelevated color=primary')
    dialog.open()

def render_selected_chips(controller: Any) -> None:
    """按稳定 Tag Group 规则渲染；复制顺序仍使用原始选择顺序。"""
    if controller.selected_chips_container is None:
        return
    tags = controller._get_selected_tags()
    previous_tags = set(controller._rendered_selected_chip_tags)
    controller.selected_chips_container.clear()
    if not tags:
        controller._rendered_selected_chip_tags.clear()
        with controller.selected_chips_container:
            ui.label('暂无已选标签').classes('text-xs text-gray-400 italic p-2 self-center')
        return

    grouped: dict[str, list[str]] = {name: [] for name in WORKSPACE_GROUP_ORDER}
    for tag in tags:
        grouped[controller._workspace_group_for_tag(tag)].append(tag)

    with controller.selected_chips_container:
        step = 0.5 if controller.prompt_format == 'anima' else 0.1
        for group_name in WORKSPACE_GROUP_ORDER:
            group_tags = grouped[group_name]
            if not group_tags:
                continue
            with ui.element('div').classes('w-full px-1 py-1'):
                ui.label(f'{group_name} · {len(group_tags)}').classes(
                    'text-xs font-bold text-slate-500 mb-1'
                )
                with ui.row().classes('w-full gap-1 flex-wrap'):
                    for tag in group_tags:
                        controller._render_selected_tag_chip(
                            tag, step, animate=tag not in previous_tags,
                        )
    controller._rendered_selected_chip_tags = set(tags)

def render_selected_tag_chip(controller: Any, tag: str, step: float, *, animate: bool = False) -> None:
    w = controller.tag_weights.get(tag, 1.0)
    extra_cls = 'boosted' if w > 1.0 else ('reduced' if w < 1.0 else '')
    w_str = f'{w:.1f}'
    display_label = format_selected_tag_label(tag, controller._get_cn_name_for_tag(tag))
    motion_cls = ' motion-chip-enter' if animate else ''
    with ui.element('div').classes(f'weight-chip{motion_cls} {extra_cls}'):
        metadata = controller._pending_selection_meta.get(tag, {})
        reason = selected_tag_reason(
            metadata.get('origin'),
            metadata.get('source'),
        )
        with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
            ui.label(reason).style('font-size:13px;')
        with ui.element('button').classes('weight-btn').props(f'title="移除 {tag}"').on(
            'click', lambda t=tag: controller._remove_selected_tag(t)
        ):
            ui.html('&times;')
        with ui.element('button').classes('weight-btn').on(
            'click', lambda t=tag, s=step: controller._adjust_weight(t, -s)
        ):
            ui.html('&minus;')
        ui.label(display_label).style(
            'font-family:Consolas,Monaco,monospace;font-size:12px;'
            'color:#2c5282;max-width:240px;overflow:hidden;'
            'text-overflow:ellipsis;white-space:nowrap;'
        )
        if w != 1.0:
            ui.label(w_str).classes('weight-label').style(
                'color:#e65100;font-weight:bold;'
            )
        plus_btn = ui.element('button').classes('weight-btn').on(
            'click', lambda t=tag, s=step: controller._adjust_weight(t, +s)
        )
        if controller.prompt_format == 'anima':
            with plus_btn:
                with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                    ui.html('Anima模型所需要的权重数值较大').style('font-size:12px;')
        with plus_btn:
            ui.html('&plus;')
