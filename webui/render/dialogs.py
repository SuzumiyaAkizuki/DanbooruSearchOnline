"""通用帮助、历史与备份弹窗视图。"""

import json
from typing import Any

from nicegui import ui

from webui.helpers import format_history_settings, format_history_time


def build_sponsor_dialog(
    controller: Any,
    *,
    title: str,
    image_url: str,
    toolchain_url: str,
    ui_text: dict,
) -> None:
    """构建低干扰的赞赏弹窗。"""
    with ui.dialog() as controller.sponsor_dialog, ui.card().classes('w-full max-w-sm'):
        with ui.column().classes('w-full items-center gap-2 text-center'):
            ui.label(title).classes('text-base font-bold text-gray-800')
            ui.label(ui_text['sponsor']['body']).classes('text-sm text-gray-600 leading-relaxed')
            ui.image(image_url).classes('w-60 max-w-full rounded border border-gray-200')
            ui.label('微信赞赏码').classes('text-xs text-gray-400')
            ui.link(
                ui_text['sponsor']['toolchain_prompt'],
                toolchain_url,
                new_tab=True,
            ).classes('text-xs text-blue-500 hover:text-blue-700 hover:underline')
        with ui.row().classes('w-full justify-end'):
            ui.button('关闭', on_click=controller.sponsor_dialog.close).props('flat color=grey-7')


def build_help_dialog(controller: Any, *, ui_text: dict, sponsor_notice_text: str) -> None:
    """构建帮助、文档与本地数据说明弹窗。"""
    from platform_utils import PLATFORM

    alternate_url = (
        'https://www.modelscope.cn/studios/SAkizuki/DanbooruSearchOnline'
        if PLATFORM == 'hf'
        else 'https://huggingface.co/spaces/SAkizuki/DanbooruSearch'
    )
    with ui.dialog() as controller.help_dialog, ui.card().classes(
        'w-full max-w-3xl max-h-[90vh] p-0 gap-0'
    ):
        with ui.row().classes('w-full items-center justify-between px-5 py-4 border-b border-slate-200'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('help_outline', color='primary')
                ui.label('帮助 / 关于').classes('text-lg font-bold text-slate-800')
            ui.button(icon='close', on_click=controller.help_dialog.close).props(
                'flat dense round color=grey-7'
            )

        with ui.scroll_area().classes('w-full h-[72vh]'):
            with ui.column().classes('w-full gap-5 px-5 py-4'):
                with ui.column().classes('help-section'):
                    with ui.element('div').classes('help-section-heading'):
                        ui.label(ui_text['help']['update_title']).classes('help-section-heading-title')
                        ui.label(ui_text['help']['update_summary']).classes('help-section-heading-subtitle')
                    ui.markdown(ui_text['help']['guide_markdown']).classes('help-content')

                with ui.column().classes('help-section'):
                    with ui.element('div').classes(
                        'help-section-heading help-section-heading--documentation'
                    ):
                        ui.label(ui_text['documentation']['title']).classes('help-section-heading-title')
                        ui.label(ui_text['documentation']['subtitle']).classes('help-section-heading-subtitle')
                    with ui.column().classes('help-content gap-3'):
                        with ui.row().classes('w-full gap-3 flex-wrap'):
                            for link in ui_text['documentation']['links']:
                                link_url = alternate_url if link['url'] == '{alternate_url}' else link['url']
                                ui.link(link['label'], link_url, new_tab=True).classes('help-link')
                        ui.markdown(ui_text['documentation']['copyright_markdown']).classes(
                            'help-content'
                        )

                with ui.column().classes('help-section'):
                    with ui.element('div').classes(
                        'help-section-heading help-section-heading--notice'
                    ):
                        ui.label(ui_text['notice']['title']).classes('help-section-heading-title')
                        ui.label(ui_text['notice']['subtitle']).classes('help-section-heading-subtitle')
                    ui.markdown(ui_text['notice']['body_markdown']).classes('help-content')

                with ui.element('div').classes(
                    'w-full rounded-lg border border-red-200 bg-red-50 px-4 py-3'
                ):
                    ui.label('本地个人数据').classes('text-sm font-bold text-red-900')
                    ui.label(
                        '历史、收藏、当前工作区和搜索配置只保存在这个浏览器中。'
                    ).classes('text-xs text-red-800 mt-1')
                    ui.button(
                        '删除所有本地个人数据',
                        icon='delete_forever',
                        on_click=controller._confirm_delete_all_personal_data,
                    ).props('outline color=negative no-caps').classes('mt-3')

                with ui.row().classes('w-full items-center justify-between gap-3 flex-wrap'):
                    ui.label('DanbooruSearch 将持续免费开放。').classes('text-xs text-slate-500')
                    ui.button(
                        sponsor_notice_text,
                        icon='volunteer_activism',
                        on_click=controller.sponsor_dialog.open,
                    ).props('flat dense no-caps color=grey-7').classes('text-xs')


def open_history_dialog(controller: Any) -> None:
    """显示本地搜索历史及其恢复操作。"""
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl max-h-[85vh]'):
        with ui.row().classes('w-full items-center justify-between'):
            ui.label('搜索历史').classes('text-lg font-bold')
            with ui.row().classes('gap-2'):
                if controller.search_history.get('items'):
                    ui.button(
                        '清空全部',
                        icon='delete_sweep',
                        on_click=lambda: controller._confirm_clear_history(dialog),
                    ).props('flat dense color=red-7')
                ui.button(icon='close', on_click=dialog.close).props('flat round dense')

        with ui.scroll_area().classes('w-full h-[65vh]'):
            items = controller.search_history.get('items', [])
            if not items:
                ui.label('暂无搜索历史').classes('text-sm text-gray-400 p-6')
            for item in items:
                with ui.card().classes('w-full mb-2 p-3 border border-gray-200 shadow-none'):
                    with ui.row().classes('w-full items-start justify-between gap-3'):
                        with ui.column().classes('gap-1 flex-grow min-w-0'):
                            ui.label(item['query']).classes('font-medium text-gray-800 break-all')
                            selected_count = len(item['workspace'].get('selected', []))
                            ui.label(
                                f"{format_history_time(item.get('searched_at'))} · "
                                f"工作区内有 {selected_count} 个标签"
                            ).classes('text-xs text-gray-400')
                            ui.label(
                                format_history_settings(item.get('settings'))
                            ).classes('text-xs text-gray-400')
                        with ui.row().classes('gap-1 flex-wrap justify-end'):
                            ui.button(
                                '重新搜索',
                                icon='search',
                                on_click=lambda value=item, current=dialog: controller._history_research(
                                    value, current
                                ),
                            ).props('flat dense color=primary')
                            ui.button(
                                '恢复工作区',
                                icon='restore',
                                on_click=lambda value=item, current=dialog: controller._history_restore(
                                    value, current
                                ),
                            ).props('flat dense color=teal-7')
                            ui.button(
                                '追加查询',
                                icon='playlist_add',
                                on_click=lambda value=item, current=dialog: controller._history_append(
                                    value, current
                                ),
                            ).props('flat dense color=purple-7')
                            ui.button(
                                icon='delete_outline',
                                on_click=lambda value=item, current=dialog: controller._delete_history_entry(
                                    value, current
                                ),
                            ).props('flat round dense color=red-6')
    dialog.open()


def open_backup_dialog(controller: Any, *, description: str) -> None:
    """显示备份导出与二次确认导入入口。"""
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-2xl'):
        ui.label('本地数据备份与迁移').classes('text-lg font-bold')
        ui.label(description).classes('text-sm text-orange-700 bg-orange-50 rounded p-3')
        ui.button('导出完整 JSON', icon='download', on_click=controller._export_backup).props(
            'unelevated color=primary'
        ).classes('w-full')
        ui.separator()
        import_mode = ui.select(
            {
                'merge': '合并：保留当前权重和配置',
                'overwrite': '覆盖：使用备份中的全部数据',
                'favorites_only': '只导入收藏',
            },
            value='merge',
            label='导入方式',
        ).props('outlined').classes('w-full')
        pending_import = {'raw': None, 'name': ''}
        pending_label = ui.label(
            '选择文件后，请检查文件名和导入方式，再点击“确认导入”。'
        ).classes('text-sm text-gray-500')
        confirm_import_button = None

        async def handle_upload(event: Any) -> None:
            try:
                raw = await event.file.text('utf-8')
                json.loads(raw)
                pending_import['raw'] = raw
                pending_import['name'] = event.file.name
                pending_label.text = f'已选择：{event.file.name}；尚未执行导入'
                confirm_import_button.enable()
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                pending_import['raw'] = None
                pending_import['name'] = ''
                pending_label.text = '文件读取失败，请重新选择有效的 JSON 文件。'
                confirm_import_button.disable()
                ui.notify(f'文件读取失败：{exc}', type='negative', timeout=5000)
            except Exception as exc:
                pending_import['raw'] = None
                pending_import['name'] = ''
                pending_label.text = '文件读取失败，请重新选择。'
                confirm_import_button.disable()
                print(f'[UI] JSON 文件读取异常: {exc}', flush=True)
                ui.notify('文件读取失败，请检查文件格式', type='negative')

        async def confirm_import() -> None:
            raw = pending_import.get('raw')
            if not isinstance(raw, str):
                ui.notify('请先选择 JSON 文件', type='warning')
                return
            confirm_import_button.disable()
            try:
                await controller._import_backup_text(raw, import_mode.value)
                dialog.close()
            except (ValueError,) as exc:
                ui.notify(f'导入失败：{exc}', type='negative', timeout=5000)
                confirm_import_button.enable()
            except Exception as exc:
                print(f'[UI] JSON 导入异常: {exc}', flush=True)
                ui.notify('导入失败，请检查文件格式', type='negative')
                confirm_import_button.enable()

        ui.upload(
            label='选择 JSON 文件',
            on_upload=handle_upload,
            auto_upload=True,
            max_file_size=12_000_000,
            on_rejected=lambda: ui.notify('文件过大，仅支持 12 MB 以内的 JSON', type='warning'),
        ).props('accept=.json').classes('w-full')
        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('关闭', on_click=dialog.close).props('flat')
            confirm_import_button = ui.button(
                '确认导入', icon='check', on_click=confirm_import,
            ).props('unelevated color=primary')
            confirm_import_button.disable()
    dialog.open()


def open_favorites_dialog(controller: Any) -> None:
    """显示收藏列表与加载、合并、导出等操作入口。"""
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-5xl max-h-[88vh]'):
        with ui.row().classes('w-full items-center justify-between'):
            ui.label('收藏').classes('text-lg font-bold')
            ui.button(icon='close', on_click=dialog.close).props('flat round dense')
        with ui.scroll_area().classes('w-full h-[70vh]'):
            items = controller.favorites.get('items', [])
            if not items:
                ui.label('暂无收藏').classes('text-sm text-gray-400 p-6')
            for item in items:
                with ui.card().classes('w-full mb-2 p-3 border border-amber-100 shadow-none'):
                    with ui.row().classes('w-full items-start justify-between gap-3'):
                        with ui.column().classes('gap-1 flex-grow min-w-0'):
                            ui.label(item['name']).classes('font-bold text-gray-800')
                            ui.label(
                                f"{len(item['selected'])} 个标签 · {item['prompt_format'].upper()} · "
                                f"更新于 {item['updated_at']}"
                            ).classes('text-xs text-gray-400')
                            if item.get('source_query'):
                                ui.label(f"来源：{item['source_query']}").classes(
                                    'text-xs text-gray-500 break-all'
                                )
                            if item.get('notes'):
                                ui.label(item['notes']).classes('text-xs text-gray-600')
                        with ui.row().classes('gap-1 flex-wrap justify-end max-w-xl'):
                            ui.button(
                                '替换载入', icon='file_open',
                                on_click=lambda value=item, current=dialog: controller._load_favorite(
                                    value, False, current
                                ),
                            ).props('flat dense color=teal-7')
                            ui.button(
                                '合并', icon='merge',
                                on_click=lambda value=item, current=dialog: controller._load_favorite(
                                    value, True, current
                                ),
                            ).props('flat dense color=primary')
                            ui.button(
                                '复制', icon='content_copy',
                                on_click=lambda value=item: controller._copy_favorite(value),
                            ).props('flat dense color=grey-7')
                            ui.button(
                                '重命名', icon='edit',
                                on_click=lambda value=item, current=dialog: controller._rename_favorite(
                                    value, current
                                ),
                            ).props('flat dense color=grey-7')
                            ui.button(
                                '覆盖', icon='save',
                                on_click=lambda value=item, current=dialog: controller._overwrite_favorite(
                                    value, current
                                ),
                            ).props('flat dense color=amber-8')
                            ui.button(
                                '导出', icon='download',
                                on_click=lambda value=item: controller._export_favorite(value),
                            ).props('flat dense color=purple-7')
                            ui.button(
                                icon='delete_outline',
                                on_click=lambda value=item, current=dialog: controller._confirm_delete_favorite(
                                    value, current
                                ),
                            ).props('flat round dense color=red-6')
    dialog.open()


def open_save_favorite_dialog(controller: Any) -> None:
    """显示保存当前工作区为收藏的表单。"""
    if not controller._get_selected_tags():
        ui.notify('当前工作区没有可收藏的标签', type='warning')
        return
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
        ui.label('保存当前工作区为收藏').classes('text-lg font-bold')
        name_input = ui.input('收藏名称').props('outlined maxlength=80').classes('w-full')
        notes_input = ui.textarea('备注（可选）').props(
            'outlined autogrow maxlength=500'
        ).classes('w-full')

        def save() -> None:
            name = (name_input.value or '').strip()
            if not name:
                ui.notify('请输入收藏名称', type='warning')
                return
            if any(item['name'] == name for item in controller.favorites.get('items', [])):
                ui.notify('已存在同名收藏，请在收藏列表中使用“覆盖”', type='warning')
                return
            if controller._save_current_workspace_as_favorite(
                name, (notes_input.value or '').strip()
            ):
                dialog.close()

        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('取消', on_click=dialog.close).props('flat')
            ui.button('保存', icon='bookmark_add', on_click=save).props(
                'unelevated color=teal-7'
            )
    dialog.open()


def open_prompt_import_dialog(controller: Any, *, description: str) -> None:
    """显示 Prompt 粘贴、解析和导入入口。"""
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-3xl'):
        ui.label('导入 Prompt').classes('text-lg font-bold')
        ui.label(description).classes('text-sm text-gray-600')
        prompt_input = ui.textarea(
            label='粘贴 Prompt',
            placeholder='1girl, (white_serafuku:1.2), {rain}, @artist_name',
        ).props('outlined autogrow maxlength=20000').classes('w-full min-h-48')
        import_button = None

        async def submit_import() -> None:
            text = str(prompt_input.value or '').strip()
            if not text:
                ui.notify('请先粘贴 Prompt', type='warning')
                return
            import_button.disable()
            try:
                result = await controller._resolve_prompt_import_text(text)
                if result.parsed_count == 0:
                    ui.notify('没有解析到可导入内容', type='warning')
                    import_button.enable()
                    return
                added_count, duplicate_count = controller._apply_prompt_import_result(result)
                dialog.close()
                controller._show_prompt_import_summary(result, added_count, duplicate_count)
            except Exception as exc:
                print(f'[UI] Prompt 导入异常: {exc}', flush=True)
                ui.notify('Prompt 导入失败，请检查输入内容', type='negative')
                import_button.enable()

        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('取消', on_click=dialog.close).props('flat')
            import_button = ui.button(
                '解析并导入', icon='playlist_add', on_click=submit_import,
            ).props('unelevated color=purple-7')
    dialog.open()


def confirm_clear_history(controller: Any, parent_dialog: Any) -> None:
    """确认清空历史，明确不影响收藏与当前工作区。"""
    with ui.dialog() as confirm, ui.card():
        ui.label('确定清空全部搜索历史吗？收藏和当前工作区不会受到影响。')
        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('取消', on_click=confirm.close).props('flat')

            def clear() -> None:
                controller._clear_all_history()
                confirm.close()
                parent_dialog.close()
                ui.notify('搜索历史已清空', type='positive')

            ui.button('清空', on_click=clear).props('unelevated color=red-7')
    confirm.open()


def confirm_delete_favorite(controller: Any, favorite: dict, parent_dialog: Any) -> None:
    """确认删除指定收藏。"""
    with ui.dialog() as dialog, ui.card():
        ui.label(f"确定删除收藏“{favorite['name']}”吗？")
        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('取消', on_click=dialog.close).props('flat')

            def delete() -> None:
                if not controller._delete_favorite(favorite):
                    return
                dialog.close()
                parent_dialog.close()
                ui.notify('收藏已删除', type='positive')

            ui.button('删除', on_click=delete).props('unelevated color=red-7')
    dialog.open()


def confirm_delete_all_personal_data(controller: Any, storage_keys: list[str]) -> None:
    """确认删除当前浏览器保存的个人数据，不涉及服务器或导出的备份文件。"""
    with ui.dialog() as confirm, ui.card().classes('w-full max-w-lg'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('warning', color='negative')
            ui.label('删除所有本地个人数据？').classes('text-lg font-bold text-red-900')
        ui.label(
            '将删除当前浏览器中保存的搜索配置、搜索输入、工作区、已选标签、权重、'
            '搜索历史、收藏以及旧版或损坏数据备份。'
        ).classes('text-sm text-slate-700 leading-relaxed')
        ui.label(
            '此操作无法撤销，但不会删除你已经下载到电脑上的 JSON 备份，也不会修改服务器端匿名聚合统计。'
        ).classes('text-xs text-red-700 bg-red-50 rounded p-3')

        async def delete_all_personal_data() -> None:
            delete_button.disable()
            keys_json = json.dumps(storage_keys, ensure_ascii=False)
            try:
                await ui.run_javascript(
                    f"""(() => {{
                        const keys = {keys_json};
                        keys.forEach((key) => localStorage.removeItem(key));
                        return keys.length;
                    }})()""",
                    timeout=5.0,
                )
                confirm.close()
                controller.help_dialog.close()
                ui.notify('本地个人数据已删除，页面即将刷新', type='positive', timeout=2000)
                await ui.run_javascript(
                    "(() => { setTimeout(() => window.location.reload(), 500); return true; })()",
                    timeout=5.0,
                )
            except Exception as exc:
                print(f'[UI] 删除本地个人数据失败: {exc}', flush=True)
                delete_button.enable()
                ui.notify('删除失败，请稍后重试', type='negative')

        with ui.row().classes('w-full justify-end gap-2 mt-2'):
            ui.button('取消', on_click=confirm.close).props('flat color=grey-7')
            delete_button = ui.button(
                '确认删除',
                icon='delete_forever',
                on_click=delete_all_personal_data,
            ).props('unelevated color=negative no-caps')
    confirm.open()


def open_rename_favorite_dialog(
    controller: Any,
    favorite: dict,
    parent_dialog: Any,
) -> None:
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-md'):
        ui.label('重命名收藏').classes('font-bold')
        name_input = ui.input('新名称', value=favorite['name']).props(
            'outlined maxlength=80'
        ).classes('w-full')

        def rename() -> None:
            error = controller._rename_favorite_to(
                favorite,
                str(name_input.value or '').strip(),
            )
            if error:
                ui.notify(error, type='warning')
                return
            dialog.close()
            parent_dialog.close()
            controller._open_favorites_dialog()

        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('取消', on_click=dialog.close).props('flat')
            ui.button('保存', on_click=rename).props('unelevated color=primary')
    dialog.open()


def open_search_feedback_dialog(
    controller: Any,
    *,
    query: str,
    privacy_text: str,
) -> None:
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
        ui.label('反馈搜索问题').classes('text-base font-bold text-gray-800')
        ui.label(f'当前搜索词：{query}').classes('text-sm text-gray-600')
        ui.label(privacy_text).classes(
            'text-xs text-slate-500 bg-slate-50 rounded p-2'
        )
        detail_input = ui.textarea(
            label='具体问题（可选）',
            placeholder='例如：结果偏题、缺少某个关键标签、召回了不相关角色/作品...',
        ).props('outlined autogrow maxlength=500 counter').classes('w-full')

        async def submit_feedback() -> None:
            submit_button.disable()
            try:
                await controller._submit_search_feedback(
                    query,
                    str(detail_input.value or '').strip(),
                )
                if controller.bad_case_btn is not None:
                    controller.bad_case_btn.disable()
                dialog.close()
                ui.notify('感谢反馈！我们会持续优化。', type='positive', timeout=3000)
            except Exception as exc:
                print(f'[UI] bad_case 记录异常: {exc}')
                submit_button.enable()
                ui.notify('记录失败，请稍后再试。', type='warning', timeout=3000)

        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('取消', on_click=dialog.close).props('flat color=grey-7')
            submit_button = ui.button(
                '提交反馈', on_click=submit_feedback,
            ).props('unelevated color=primary')
    dialog.open()


def open_translation_feedback_dialog(
    controller: Any,
    *,
    row: dict,
    privacy_text: str,
) -> None:
    tag = str(row.get('tag') or '').strip()
    current_cn_name = str(row.get('cn_name') or '').strip()
    current_cn_first = current_cn_name.split(',', 1)[0].strip()

    with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
        ui.label('反馈翻译错误').classes('text-base font-bold text-gray-800')
        ui.label(f'词条：{tag}').classes('text-sm font-mono text-gray-700')
        ui.label(
            f'当前中文名：{current_cn_first or current_cn_name or "（空）"}'
        ).classes('text-sm text-gray-600')
        ui.label(privacy_text).classes(
            'text-xs text-slate-500 bg-slate-50 rounded p-2'
        )
        suggested_input = ui.input(
            label='建议中文名（可选）',
            placeholder='如果有更合适的译名，可以填在这里',
        ).props('outlined maxlength=120 counter').classes('w-full')
        detail_input = ui.textarea(
            label='问题说明（可选）',
            placeholder='例如：含义不准确、作品/角色名误译、中文名缺失...',
        ).props('outlined autogrow maxlength=500 counter').classes('w-full')

        async def submit_feedback() -> None:
            submit_button.disable()
            try:
                await controller._submit_translation_feedback(
                    row,
                    str(suggested_input.value or '').strip(),
                    str(detail_input.value or '').strip(),
                )
                dialog.close()
                ui.notify('感谢反馈！这条翻译问题已记录。', type='positive', timeout=3000)
            except Exception as exc:
                print(f'[UI] translation_error 记录异常: {exc}')
                submit_button.enable()
                ui.notify('记录失败，请稍后再试。', type='warning', timeout=3000)

        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('取消', on_click=dialog.close).props('flat color=grey-7')
            submit_button = ui.button(
                '提交反馈', on_click=submit_feedback,
            ).props('unelevated color=primary')
    dialog.open()
