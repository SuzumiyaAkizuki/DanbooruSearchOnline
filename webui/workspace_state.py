"""工作区状态的可复用规则，不包含 NiceGUI 渲染。"""

from dataclasses import dataclass
from typing import Any

from core.workspace import (
    add_history_entry,
    append_workspace_query,
    clone_workspace,
    empty_history,
    merge_favorites,
    merge_history,
    merge_workspaces,
    normalize_favorites,
    sync_selected_entries,
    workspace_signature,
)
from platform_utils import nsfw_allowed
from webui.helpers import sanitize_restored_config


def set_selection_meta(controller: Any, tag: str, origin: str, source: str = '') -> None:
    """记录标签的来源，供写入工作区时补齐可追溯信息。"""
    controller._pending_selection_meta[tag] = {
        'origin': origin,
        'source': source,
    }


def push_undo_snapshot(controller: Any, *, limit: int = 30) -> bool:
    """在工作区内容变化前保存快照；相同快照不重复入栈。"""
    snapshot = clone_workspace(controller.workspace_state)
    signature = workspace_signature(snapshot)
    if controller._undo_stack and workspace_signature(controller._undo_stack[-1]) == signature:
        return False
    controller._undo_stack.append(snapshot)
    controller._undo_stack = controller._undo_stack[-limit:]
    controller._redo_stack.clear()
    controller._update_undo_buttons()
    return True


def selected_tags(controller: Any) -> list[str]:
    """合并结果表与工作区额外标签，稳定保留用户选择顺序。"""
    table_tags = [row['tag'] for row in controller.result_table.selected] if controller.result_table else []
    seen = set(table_tags)
    extra_pool = set(controller.chip_extra_selected)
    extra = [tag for tag in controller._selected_order if tag in extra_pool and tag not in seen]
    seen.update(extra)
    extra.extend(sorted(tag for tag in extra_pool if tag not in seen))
    return table_tags + extra


def collect_config_state(controller: Any, config_version: int) -> dict:
    """从页面控件采集可持久化的工作区配置。"""
    return {
        'version': config_version,
        'top_k': int(controller.input_top_k.value) if controller.input_top_k else 10,
        'limit': int(controller.input_limit.value) if controller.input_limit else 80,
        'popularity_weight': float(controller.input_weight.value) if controller.input_weight else 0.15,
        'show_nsfw': bool(controller.input_nsfw.value) if controller.input_nsfw else False,
        'use_segmentation': bool(controller.input_segment.value) if controller.input_segment else True,
        'selected_layers': dict(controller.selected_layers),
        'selected_cats': dict(controller.selected_cats),
        'sw_semantic': bool(controller.sw_semantic.value) if controller.sw_semantic else False,
        'sw_layer': bool(controller.sw_layer.value) if controller.sw_layer else False,
        'sw_source': bool(controller.sw_source.value) if controller.sw_source else False,
        'prompt_format': controller.prompt_format,
        'rows_per_page': controller._get_rows_per_page(),
        'search_query': controller.search_input.value if controller.search_input else '',
        'dismissed_announcement_version': controller.dismissed_announcement_version,
        'search_mode': controller.input_search_mode.value if controller.input_search_mode else '自定义',
        'group_mode': controller.input_group_mode.value if controller.input_group_mode else 'off',
        'max_per_group': int(controller.input_max_per_group.value) if controller.input_max_per_group else 2,
    }


def apply_config_state(
    controller: Any,
    config: object,
    announcement_version: str,
) -> None:
    """将已校验的配置按原有顺序同步回控件和工作区状态。"""
    config = sanitize_restored_config(config if isinstance(config, dict) else {})

    dismissed_version = config.get('dismissed_announcement_version', '')
    controller.dismissed_announcement_version = dismissed_version
    if controller.announcement_banner:
        controller.announcement_banner.set_visibility(
            dismissed_version != announcement_version
        )

    if controller.input_search_mode and 'search_mode' in config:
        controller.input_search_mode.set_value(config['search_mode'])
    if controller.input_top_k and 'top_k' in config:
        controller.input_top_k.set_value(config['top_k'])
    if controller.input_limit and 'limit' in config:
        controller.input_limit.set_value(config['limit'])
    if controller.input_weight and 'popularity_weight' in config:
        controller.input_weight.set_value(config['popularity_weight'])
    if controller.input_segment and 'use_segmentation' in config:
        controller.input_segment.set_value(config['use_segmentation'])
    if controller.input_group_mode and 'group_mode' in config:
        controller.input_group_mode.set_value(config['group_mode'])
    if controller.input_max_per_group and 'max_per_group' in config:
        controller.input_max_per_group.set_value(config['max_per_group'])
    if nsfw_allowed() and controller.input_nsfw and 'show_nsfw' in config:
        controller.input_nsfw.set_value(config['show_nsfw'])

    for layer, value in config.get('selected_layers', {}).items():
        if layer in controller.selected_layers:
            controller.selected_layers[layer] = bool(value)
            if layer in controller._layer_checkboxes:
                controller._layer_checkboxes[layer].set_value(bool(value))
    for category, value in config.get('selected_cats', {}).items():
        if category in controller.selected_cats:
            controller.selected_cats[category] = bool(value)
            if category in controller._cat_checkboxes:
                controller._cat_checkboxes[category].set_value(bool(value))

    if controller.sw_semantic and 'sw_semantic' in config:
        controller.sw_semantic.set_value(config['sw_semantic'])
    if controller.sw_layer and 'sw_layer' in config:
        controller.sw_layer.set_value(config['sw_layer'])
    if controller.sw_source and 'sw_source' in config:
        controller.sw_source.set_value(config['sw_source'])
    if 'prompt_format' in config:
        controller._apply_prompt_format(config['prompt_format'])
    if 'rows_per_page' in config:
        controller._set_rows_per_page(config['rows_per_page'])
    if controller.search_input and config.get('search_query'):
        controller.search_input.set_value(config['search_query'])
    controller._update_table_columns()


def current_search_settings(controller: Any) -> dict:
    """采集一条历史记录所需的搜索设置。"""
    return {
        'search_mode': controller.input_search_mode.value if controller.input_search_mode else '自定义',
        'top_k': int(controller.input_top_k.value) if controller.input_top_k else 10,
        'limit': int(controller.input_limit.value) if controller.input_limit else 80,
        'popularity_weight': float(controller.input_weight.value) if controller.input_weight else 0.15,
        'show_nsfw': bool(controller.input_nsfw.value) if controller.input_nsfw else False,
        'use_segmentation': bool(controller.input_segment.value) if controller.input_segment else True,
        'target_layers': [key for key, value in controller.selected_layers.items() if value],
        'target_categories': [key for key, value in controller.selected_cats.items() if value],
        'group_mode': controller.input_group_mode.value if controller.input_group_mode else 'off',
        'max_per_group': int(controller.input_max_per_group.value) if controller.input_max_per_group else 2,
    }


def apply_search_settings(
    controller: Any,
    settings: object,
    search_mode_options: tuple[str, ...],
) -> None:
    """恢复历史/收藏的搜索设置，并保留原有的最后保存动作。"""
    if not isinstance(settings, dict):
        return
    controller._applying_preset = True
    try:
        mode = settings.get('search_mode')
        if controller.input_search_mode and mode in search_mode_options:
            controller.input_search_mode.set_value(mode)
        if controller.input_top_k and isinstance(settings.get('top_k'), int):
            controller.input_top_k.set_value(settings['top_k'])
        if controller.input_limit and isinstance(settings.get('limit'), int):
            controller.input_limit.set_value(settings['limit'])
        if controller.input_weight and isinstance(settings.get('popularity_weight'), (int, float)):
            controller.input_weight.set_value(settings['popularity_weight'])
        if controller.input_segment and isinstance(settings.get('use_segmentation'), bool):
            controller.input_segment.set_value(settings['use_segmentation'])
        if controller.input_group_mode and settings.get('group_mode') in ('off', 'expand', 'diverse'):
            controller.input_group_mode.set_value(settings['group_mode'])
        if controller.input_max_per_group and isinstance(settings.get('max_per_group'), int):
            controller.input_max_per_group.set_value(settings['max_per_group'])
        if nsfw_allowed() and controller.input_nsfw and isinstance(settings.get('show_nsfw'), bool):
            controller.input_nsfw.set_value(settings['show_nsfw'])

        layers = settings.get('target_layers')
        if isinstance(layers, list):
            selected = set(layers)
            for layer in controller.selected_layers:
                value = layer in selected
                controller.selected_layers[layer] = value
                if layer in controller._layer_checkboxes:
                    controller._layer_checkboxes[layer].set_value(value)
        categories = settings.get('target_categories')
        if isinstance(categories, list):
            selected = set(categories)
            for category in controller.selected_cats:
                value = category in selected
                controller.selected_cats[category] = value
                if category in controller._cat_checkboxes:
                    controller._cat_checkboxes[category].set_value(value)
    finally:
        controller._applying_preset = False
    controller._save_config()


def apply_workspace_state(
    controller: Any,
    workspace: dict,
    artist_origins: set[str],
    *,
    persist: bool = True,
    refresh_recommendations: bool = True,
) -> None:
    """将版本化工作区同步至页面状态；渲染仍通过控制器回调完成。"""
    controller.workspace_state = clone_workspace(workspace)
    selected = controller.workspace_state['selected']
    tags = [item['tag'] for item in selected]
    tag_set = set(tags)
    controller._selected_order = list(tags)
    controller.tag_weights = {item['tag']: item.get('weight', 1.0) for item in selected}
    controller._pending_selection_meta = {
        item['tag']: {
            'origin': item.get('origin', 'unknown'),
            'source': item.get('source', ''),
        }
        for item in selected
    }
    controller._workspace_artist_tags = {
        item['tag'] for item in selected
        if item.get('origin') in artist_origins
    }

    table_tags = {row['tag'] for row in controller.result_table.rows} if controller.result_table else set()
    controller.chip_extra_selected.clear()
    controller.chip_extra_selected.update(tag for tag in tags if tag not in table_tags)
    if controller.result_table is not None:
        controller.result_table.selected = [
            row for row in controller.result_table.rows if row.get('tag') in tag_set
        ]
    controller._apply_prompt_format(controller.workspace_state.get('prompt_format', 'sdxl'))
    controller._render_selected_chips()
    controller._render_prompt_pending()
    controller._render_concept_coverage()
    if controller.selection_count_label is not None:
        controller.selection_count_label.text = str(len(tags))
    if controller.results_section is not None:
        controller.results_section.set_visibility(bool(tags) or bool(controller.full_table_data))

    if persist:
        controller._save_staged_tags()
        controller._save_config()
    if refresh_recommendations:
        show_nsfw = bool(controller.input_nsfw.value) if controller.input_nsfw else False
        controller._last_recommendation_seed_tags = []
        controller._refresh_recommendations_if_seed_changed(tags, show_nsfw)


def pop_undo_workspace(controller: Any) -> dict | None:
    """执行撤销栈的纯状态转移，返回需要应用的目标工作区。"""
    if not controller._undo_stack:
        return None
    controller._redo_stack.append(clone_workspace(controller.workspace_state))
    return controller._undo_stack.pop()


def pop_redo_workspace(controller: Any, *, limit: int = 30) -> dict | None:
    """执行重做栈的纯状态转移，返回需要应用的目标工作区。"""
    if not controller._redo_stack:
        return None
    controller._undo_stack.append(clone_workspace(controller.workspace_state))
    controller._undo_stack = controller._undo_stack[-limit:]
    return controller._redo_stack.pop()


def sync_workspace_selection(controller: Any, artist_origins: set[str]) -> list[str]:
    """把当前选择、权重和来源元数据写入版本化工作区状态。"""
    tags = controller._get_selected_tags()
    controller._selected_order = list(tags)
    cn_names = {tag: controller._get_cn_name_for_tag(tag) for tag in tags}
    controller.workspace_state['prompt_format'] = controller.prompt_format
    controller.workspace_state = sync_selected_entries(
        controller.workspace_state,
        tags,
        controller.tag_weights,
        cn_names,
        controller._pending_selection_meta,
    )
    controller._workspace_artist_tags = {
        item['tag'] for item in controller.workspace_state['selected']
        if item.get('origin') in artist_origins
    }
    return tags


def record_search_history_state(controller: Any, query: str) -> None:
    """将当前查询追加到工作区和历史；持久化仍由页面调用方执行。"""
    settings = controller._current_search_settings()
    controller.workspace_state = append_workspace_query(
        controller.workspace_state,
        query,
        settings,
    )
    controller._save_staged_tags()
    controller.search_history = add_history_entry(
        controller.search_history,
        query,
        settings,
        controller.workspace_state,
    )


def remove_history_entry(controller: Any, history_id: object) -> None:
    """按历史 ID 删除一条记录。"""
    controller.search_history['items'] = [
        entry for entry in controller.search_history.get('items', [])
        if entry.get('history_id') != history_id
    ]


def clear_history(controller: Any) -> None:
    """重置搜索历史，不影响当前工作区或收藏。"""
    controller.search_history = empty_history()


def replace_favorites_safely(controller: Any, favorites: dict) -> bool:
    """先尝试保存新收藏；失败时恢复内存中的旧收藏。"""
    previous = controller.favorites
    controller.favorites = favorites
    if controller._save_favorites():
        controller._update_workspace_counts()
        return True
    controller.favorites = previous
    controller._storage_session_dirty.discard('favorites')
    return False


@dataclass(frozen=True)
class BackupImportPlan:
    """规范化备份在页面应用前的纯状态变更计划。"""

    favorites: dict
    workspace: dict | None
    history: dict | None
    config: dict | None
    message: str


def merge_imported_favorite(current_favorites: dict, favorite: object) -> tuple[dict, list[str]]:
    """将单收藏导出内容规范化后合并到当前收藏。"""
    normalized, warnings = normalize_favorites({
        'schema_version': 1,
        'items': [favorite],
    })
    return merge_favorites(current_favorites, normalized), warnings


def build_backup_import_plan(
    current_workspace: dict,
    current_history: dict,
    current_favorites: dict,
    backup: dict,
    mode: str,
) -> BackupImportPlan:
    """生成备份导入方案；合并模式不改变当前配置和同标签权重。"""
    if mode == 'favorites_only':
        return BackupImportPlan(
            favorites=merge_favorites(current_favorites, backup['favorites']),
            workspace=None,
            history=None,
            config=None,
            message='收藏已合并导入',
        )
    if mode == 'overwrite':
        return BackupImportPlan(
            favorites=backup['favorites'],
            workspace=backup['workspace'],
            history=backup['history'],
            config=backup['config'],
            message='本地数据已由备份覆盖',
        )
    return BackupImportPlan(
        favorites=merge_favorites(current_favorites, backup['favorites']),
        workspace=merge_workspaces(current_workspace, backup['workspace']),
        history=merge_history(current_history, backup['history']),
        config=None,
        message='备份已合并；当前标签权重和配置保持不变',
    )
