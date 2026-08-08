"""
webui/controller.py
─────────────
NiceGUI 页面控制器。

▸ 编排用户动作、检索调用、页面状态与视图刷新。
▸ NiceGUI 控件创建集中在 webui.render。
▸ 启动、路由和应用挂载集中在 ui_nicegui.py。
"""
import asyncio
import logging
import re
import time
import json as _json
from datetime import timedelta, timezone

from nicegui import ui
from core import counter, telemetry
from core.engine import DanbooruTagger
from core.models import SearchRequest
from core.prompt_import import (
    PromptImportResult,
    WorkspaceCanonicalizationResult,
    canonicalize_workspace_tags,
    classify_workspace_tag,
    pending_to_workspace_entry,
    resolve_prompt_text,
)
from core.workspace import (
    FAVORITES_STORAGE_KEY,
    HISTORY_STORAGE_KEY,
    LEGACY_STAGED_STORAGE_KEY,
    WORKSPACE_STORAGE_KEY,
    WorkspaceDataError,
    build_backup,
    dump_collection,
    empty_favorites,
    empty_history,
    favorite_from_workspace,
    merge_favorite_into_workspace,
    new_workspace,
    normalize_backup,
    normalize_favorites,
    replace_with_favorite,
    utc_now_iso,
)
from platform_utils import nsfw_allowed
from webui.constants import (
    ANNOUNCEMENT_VERSION as _ANNOUNCEMENT_VERSION,
    ARTIST_ORIGINS as _ARTIST_ORIGINS,
    ARTIST_REC_LIMIT,
    CONFIG_LS_KEY as _CONFIG_LS_KEY,
    CONFIG_VERSION as _CONFIG_VERSION,
    GROUP_RENDER_TAG_LIMIT,
    HISTORY_PRE_COMPACTION_BACKUP_KEY as _HISTORY_PRE_COMPACTION_BACKUP_KEY,
    LOCAL_STORAGE_NAMES as _LOCAL_STORAGE_NAMES,
    LOCAL_STORAGE_RESTORE_RETRY_DELAYS as _LOCAL_STORAGE_RESTORE_RETRY_DELAYS,
    RECOMMENDATION_DEBOUNCE_SECONDS,
    SEARCH_MODE_OPTIONS as _SEARCH_MODE_OPTIONS,
    SEARCH_MODE_PRESETS as _SEARCH_MODE_PRESETS,
    SPONSOR_IMAGE_URL,
    SPONSOR_NOTICE_TEXT,
    SPONSOR_TITLE,
    SPONSOR_TOOLCHAIN_URL,
    UI_TEXT,
    WORKSPACE_SAVE_DEBOUNCE_SECONDS,
)
from webui.helpers import (
    apply_nsfw_filter as _apply_nsfw_filter,
    format_history_settings,
    format_history_time,
    format_selected_tag_label,
    format_tag_with_weight,
    get_git_commit,
    group_names_key,
    group_scroll_dom_id,
    limit_group_render_tags,
    next_group_render_limit,
    result_to_row as _result_to_row,
    scroll_state_restore_script,
    should_group_start_expanded,
)
from webui.styles import MOTION_STYLE
from webui.state import PageState, StateField
from webui.local_storage import backup_key, clear_restore_cache, finish_storage_restore_task, flush_storage_session_changes, pause_storage_restore, prepare_restore_snapshot, read_prepared_value, restore_staged_storage, restore_with_retries, schedule_workspace_persist, start_storage_restore_task, storage_keys, storage_listener_script, write_is_ready
from webui.workspace_state import apply_config_state, apply_search_settings, apply_workspace_state, build_backup_import_plan, clear_history, collect_config_state, current_search_settings, merge_imported_favorite, pop_redo_workspace, pop_undo_workspace, push_undo_snapshot, record_search_history_state, remove_history_entry, replace_favorites_safely, selected_tags, set_selection_meta, sync_workspace_selection
from webui.recommendations import clamp_page, consume_latest_recommendation_requests, merge_related_results, queue_latest_recommendation_request, recommendation_seed_tags, set_paginated_recommendation_page
from webui.render.results_panel import (
    filter_by_source,
    build_results_columns,
    render_concept_coverage,
    update_table_columns,
)
from webui.render.recommendations import (
    render_artist_page,
    render_artist_recommendations,
    render_group_expansion,
    render_related_list,
    render_related_page,
)
from webui.render.dialogs import (
    build_help_dialog,
    build_sponsor_dialog,
    confirm_clear_history,
    confirm_delete_all_personal_data,
    confirm_delete_favorite,
    open_backup_dialog,
    open_favorites_dialog,
    open_history_dialog,
    open_prompt_import_dialog,
    open_rename_favorite_dialog,
    open_save_favorite_dialog,
    open_search_feedback_dialog,
    open_translation_feedback_dialog,
)
from webui.render.workspace_panel import (
    build_selection_bar,
    build_workspace_toolbar,
    render_prompt_pending,
    render_selected_chips,
    render_selected_tag_chip,
    show_prompt_import_summary,
    show_workspace_canonicalization,
)
from webui.render.search_panel import build_search_panel
from webui.render.page import (
    build_page as render_page,
    build_release_announcement,
    render_service_status,
)


# 仅统计当前进程中已建立 Socket.IO 连接的 UI 页面，不等同于唯一用户数。
_ACTIVE_UI_CLIENT_IDS: set[str] = set()


def _mark_ui_session_active(client_id: str) -> None:
    _ACTIVE_UI_CLIENT_IDS.add(client_id)


def _mark_ui_session_inactive(client_id: str) -> None:
    _ACTIVE_UI_CLIENT_IDS.discard(client_id)


def _get_active_ui_session_count() -> int:
    return len(_ACTIVE_UI_CLIENT_IDS)


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
# suppress MCP streamable-HTTP transport noise ("No response returned" from Starlette middleware)
class _SuppressMCPNoise(logging.Filter):
    _MARKER = "No response returned"

    def filter(self, record: logging.LogRecord) -> bool:
        if self._MARKER in record.getMessage():
            return False
        if record.exc_info:
            import traceback
            tb_text = "".join(traceback.format_exception(*record.exc_info))
            if self._MARKER in tb_text:
                return False
        return True

logging.getLogger("uvicorn.error").addFilter(_SuppressMCPNoise())

# suppress MCP OAuth discovery 404 noise (clients probing .well-known OAuth endpoints)
class _SuppressOAuthNoise(logging.Filter):
    _MARKERS = (
        ".well-known/oauth-authorization-server",
        ".well-known/oauth-protected-resource",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if any(marker in record.getMessage() for marker in self._MARKERS):
            return False
        return True

logging.getLogger("uvicorn.access").addFilter(_SuppressOAuthNoise())
logging.getLogger("nicegui").addFilter(_SuppressOAuthNoise())

# ── 辅助函数 ───────────────────────────────────────────────────────────────────

_HISTORY_DISPLAY_TIMEZONE = timezone(timedelta(hours=8))


def _format_history_time(value: object) -> str:
    return format_history_time(value)


def _format_history_settings(settings: object) -> str:
    return format_history_settings(settings)

def _next_group_render_limit(current: int, total: int, page_size: int) -> int:
    return next_group_render_limit(current, total, page_size)


def _limit_group_render_tags(tags: list[dict], visible_limit: int | None = None) -> tuple[list[dict], int]:
    return limit_group_render_tags(tags, visible_limit)


def _should_group_start_expanded(group_name: str, expanded_groups: set[str]) -> bool:
    return should_group_start_expanded(group_name, expanded_groups)


def _group_names_key(group_data: list[dict]) -> tuple[str, ...]:
    return group_names_key(group_data)


def _group_scroll_dom_id(group_name: str) -> str:
    return group_scroll_dom_id(group_name)


def _scroll_state_restore_script(positions: dict[str, int]) -> str:
    return scroll_state_restore_script(positions)


def _get_git_commit() -> str:
    return get_git_commit()


def result_to_row(r, nsfw_visible: bool) -> dict:
    return _result_to_row(r, nsfw_visible)


def apply_nsfw_filter(rows: list[dict], show_nsfw: bool) -> list[dict]:
    return _apply_nsfw_filter(rows, show_nsfw)


def _format_tag_with_weight(tag: str, weight: float, fmt: str = 'sdxl') -> str:
    return format_tag_with_weight(tag, weight, fmt)


def _format_selected_tag_label(tag: str, cn_name: str = '') -> str:
    return format_selected_tag_label(tag, cn_name)


# ── UI 类 ─────────────────────────────────────────────────────────────────────

class DanbooruSearchUI:
    full_table_data = StateField()
    current_segments = StateField()
    current_keywords = StateField()
    current_cached_queries = StateField()
    current_filter_keyword = StateField()
    current_query_str = StateField()
    full_tags_str = StateField()
    full_tags_str_sfw = StateField()
    current_related = StateField()
    chip_extra_selected = StateField()
    _selected_order = StateField('selected_order')
    _rendered_selected_chip_tags = StateField('rendered_selected_chip_tags')
    workspace_state = StateField('workspace')
    search_history = StateField('history')
    favorites = StateField()
    _undo_stack = StateField('undo_stack')
    _redo_stack = StateField('redo_stack')
    _pending_selection_meta = StateField('pending_selection_meta')
    _workspace_artist_tags = StateField('workspace_artist_tags')
    tag_weights = StateField()
    prompt_format = StateField()
    selected_layers = StateField()
    selected_cats = StateField('selected_categories')
    _related_results = StateField('related_results')
    _related_show_nsfw = StateField('related_show_nsfw')
    _related_page = StateField('related_page')
    _related_page_count = StateField('related_page_count')
    _group_render_limits = StateField('group_render_limits')
    _group_expanded_names = StateField('group_expanded_names')
    _group_scroll_positions = StateField('group_scroll_positions')
    _group_render_key = StateField('group_render_key')
    _group_candidate_sources = StateField('group_candidate_sources')
    _artist_rec_rows = StateField('artist_rec_rows')
    _artist_rec_results = StateField('artist_rec_results')
    _artist_rec_top_tags = StateField('artist_rec_top_tags')
    _artist_rec_show_nsfw = StateField('artist_rec_show_nsfw')
    _artist_rec_sources = StateField('artist_rec_sources')
    _artist_rec_page = StateField('artist_rec_page')
    _artist_rec_page_count = StateField('artist_rec_page_count')
    _current_artist_rec_tags = StateField('current_artist_rec_tags')
    _artist_result_tags = StateField('artist_result_tags')
    _last_recommendation_seed_tags = StateField('last_recommendation_seed_tags')
    _pending_recommendation_request = StateField('pending_recommendation_request')
    _recommendation_generation = StateField('recommendation_generation')
    _storage_states = StateField('storage_states')
    _storage_session_dirty = StateField('storage_session_dirty')
    _storage_applying = StateField('storage_applying')
    _storage_raw_values = StateField('storage_raw_values')

    def __init__(self):
        self.state = PageState()
        self.search_count_label = None
        self.service_status_container = None
        self._service_status_task = None
        self._last_service_status_key = None
        self.current_search_interacted = True
        self._telemetry_search_started_at: float | None = None
        self._telemetry_selection_recorded = False
        self._telemetry_copy_timing_recorded = False
        self._telemetry_last_query = ''
        self._telemetry_last_search_at = 0.0

        self.full_table_data: list[dict] = []
        self.current_segments: list[str] = []   # 从句级原始片段，用于区分 chip 颜色
        self.current_keywords: list[str] = []
        self.current_cached_queries: set[str] = set()
        self.current_filter_keyword: str = 'ALL'  # 当前选中的分词筛选 keyword（NSFW 切换时复用）
        self.current_query_str: str = ""
        self.full_tags_str: str = ""
        self.full_tags_str_sfw: str = ""

        self.result_table = None           # 左栏表格
        self.related_list_container = None  # 右栏关联推荐列表
        self.group_expansion_container = None  # 左栏 Group 同类扩展（表格下方）
        self.artist_rec_pagination = None
        self.related_pagination = None
        self.client = None
        self._group_render_limits: dict[str, int] = {}
        self._group_expanded_names: set[str] = set()
        self._group_scroll_positions: dict[str, int] = {}
        self._group_render_key: tuple[str, ...] = ()
        self.results_section = None        # 整个结果区域（搜索前隐藏）
        self.coverage_container = None
        self.selection_count_label = None
        self.selected_display = None       # 已废弃 textarea，保留兼容
        self.selected_chips_container = None  # 已选标签 chip 容器
        self.prompt_pending_container = None
        self.current_related: list = []
        self.chip_extra_selected: set = set()
        self._selected_order: list[str] = []
        self._rendered_selected_chip_tags: set[str] = set()
        self.workspace_state: dict = new_workspace()
        self.search_history: dict = empty_history()
        self.favorites: dict = empty_favorites()
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._pending_selection_meta: dict[str, dict[str, str]] = {}
        self._workspace_artist_tags: set[str] = set()
        self.undo_btn = None
        self.redo_btn = None
        self.history_count_label = None
        self.favorites_count_label = None
        self._workspace_storage_listener_installed = False
        # 选择刷新只保留最新快照；运行中的线程不取消，完成后丢弃过期结果。
        self._recommendation_task = None  # type: asyncio.Task | None
        self._pending_recommendation_request = None
        self._recommendation_generation = 0
        self._workspace_save_task = None  # type: asyncio.Task | None
        self._coverage_render_task = None  # type: asyncio.Task | None
        self._storage_restore_task = None  # type: asyncio.Task | None
        self._storage_restore_started = False
        self._storage_restoring = False
        self._storage_failure_notified = False
        self._storage_states = {name: 'pending' for name in _LOCAL_STORAGE_NAMES}
        self._storage_session_dirty: set[str] = set()
        self._storage_applying: set[str] = set()
        self._storage_raw_values: dict[str, str | None] = {}

        # tag -> prompt 权重，范围 [0.1, 1.9]，默认 1.0
        self.tag_weights: dict[str, float] = {}
        # 复制格式：'sdxl'、'nai' 或 'anima'
        self.prompt_format: str = 'sdxl'
        self.format_toggle_btn = None

        self.init_banner = None
        self.input_top_k = None
        self.input_limit = None
        self.input_weight = None
        self.input_nsfw = None
        self.input_segment = None
        self.input_search_mode = None
        self.input_group_mode = None
        self.input_max_per_group = None
        self._applying_preset = False
        self.search_input = None
        self.keywords_container = None
        self.spinner = None
        self.search_btn = None

        self.selected_layers = {'英文': True, '中文扩展词': True, '释义': True, '中文核心词': True, 'artist': True}
        self.selected_cats = {'General': True, 'Copyright': True, 'Character': True}

        self.bad_case_btn = None

        self.announcement_banner = None
        self.dismissed_announcement_version = ''
        self.help_dialog = None
        self.sponsor_dialog = None

        # 表格显示选项开关
        self.sw_semantic = None
        self.sw_layer = None
        self.sw_source = None

        # 关联推荐的 checkbox 引用
        self._related_checkboxes: dict[str, ui.checkbox] = {}
        self._related_results: list = []
        self._related_show_nsfw = True
        self._related_page = 1
        self._related_page_count = 0
        self._related_page_label = None
        self._related_prev_button = None
        self._related_next_button = None
        # 同类标签的 checkbox 引用
        self._group_checkboxes: dict[str, ui.checkbox] = {}
        # 推荐画师的 checkbox 引用
        self._artist_rec_checkboxes: dict[str, ui.checkbox] = {}
        self._artist_rec_rows: list = []
        self._artist_rec_results: list = []
        self._artist_rec_top_tags: dict[str, list[str]] = {}
        self._artist_rec_show_nsfw = True
        self._artist_rec_sources: dict[str, str] = {}
        self._group_candidate_sources: dict[str, str] = {}
        self._artist_rec_page = 1
        self._artist_rec_page_count = 0
        self._artist_rec_page_label = None
        self._artist_rec_prev_button = None
        self._artist_rec_next_button = None
        # 当前推荐画师的标签名集合（用于 Anima 模式复制时加 @ 前缀）
        self._current_artist_rec_tags: set[str] = set()
        self._artist_result_tags: set[str] = set()
        self._last_recommendation_seed_tags: list[str] = []

        # 高级选项中各层/类型的 checkbox 引用，用于 restore 时同步控件状态
        self._layer_checkboxes: dict[str, ui.checkbox] = {}
        self._cat_checkboxes: dict[str, ui.checkbox] = {}

    def _update_footer_text(self):
        if self.search_count_label is not None and self._client_alive():
            try:
                total = counter.get()
                visits = counter.get_visits()
                commit = _get_git_commit()
                self.search_count_label.content = (
                    f'累计搜索 {total:,} 次 | 累计访问 {visits:,} 次 | '
                    f'<span class="font-mono text-gray-300">版本号: {commit}</span>'
                    f'<br>'
                    f'<a href="/api/docs" '
                    f'target="_blank" rel="noopener noreferrer" '
                    f'class="text-blue-400 hover:text-blue-600 hover:underline">使用 API 服务</a>'
                    f' | <a href="https://github.com/SuzumiyaAkizuki/DanbooruSearchOnline#mcp-接口" '
                    f'target="_blank" rel="noopener noreferrer" '
                    f'class="text-blue-400 hover:text-blue-600 hover:underline">使用 MCP 服务</a>'
                )
            except AttributeError:
                pass

    def _update_service_status(self):
        if self.service_status_container is None or not self._client_alive():
            return

        ready = DanbooruTagger.is_ready()
        online_sessions = _get_active_ui_session_count()
        load = DanbooruTagger.get_load_snapshot()
        active = load['active']
        waiting = load['waiting']
        capacity = load['capacity']
        busy = ready and (waiting > 0 or active >= capacity)
        status_key = (ready, busy, online_sessions, active, waiting, capacity)
        if status_key == self._last_service_status_key:
            return
        self._last_service_status_key = status_key

        render_service_status(self, {
            'ready': ready,
            'busy': busy,
            'online_sessions': online_sessions,
            'active': active,
            'waiting': waiting,
        })

    def _client_connected(self) -> bool:
        if not self._client_alive():
            return False
        client = self.client
        connection_state = getattr(client, 'has_socket_connection', None)
        if callable(connection_state):
            connection_state = connection_state()
        if connection_state is not None:
            return bool(connection_state)
        if hasattr(client, 'tab_id'):
            return client.tab_id is not None
        return True

    async def _service_status_loop(self):
        """Refresh service status without creating a NiceGUI timer element."""
        try:
            while self._client_alive():
                await asyncio.sleep(5.0)
                if not self._client_alive():
                    return
                if not self._client_connected():
                    continue
                try:
                    self._update_service_status()
                except RuntimeError:
                    # The page may be deleted between the connection check and render.
                    return
        except asyncio.CancelledError:
            return
        finally:
            if self._service_status_task is asyncio.current_task():
                self._service_status_task = None

    def _start_service_status_task(self):
        task = self._service_status_task
        if task is not None and not task.done():
            return
        self._service_status_task = asyncio.create_task(self._service_status_loop())

    def _build_sponsor_dialog(self):
        build_sponsor_dialog(
            self,
            title=SPONSOR_TITLE,
            image_url=SPONSOR_IMAGE_URL,
            toolchain_url=SPONSOR_TOOLCHAIN_URL,
            ui_text=UI_TEXT,
        )

    def _build_help_dialog(self):
        build_help_dialog(
            self,
            ui_text=UI_TEXT,
            sponsor_notice_text=SPONSOR_NOTICE_TEXT,
        )

    def _confirm_delete_all_personal_data(self):
        storage_keys = [
            _CONFIG_LS_KEY,
            WORKSPACE_STORAGE_KEY,
            HISTORY_STORAGE_KEY,
            FAVORITES_STORAGE_KEY,
            LEGACY_STAGED_STORAGE_KEY,
            f'{WORKSPACE_STORAGE_KEY}_corrupt_backup',
            f'{HISTORY_STORAGE_KEY}_corrupt_backup',
            f'{FAVORITES_STORAGE_KEY}_corrupt_backup',
            _HISTORY_PRE_COMPACTION_BACKUP_KEY,
        ]
        confirm_delete_all_personal_data(self, storage_keys)

    def _build_release_announcement(self):
        build_release_announcement(self)

    def _dismiss_release_announcement(self):
        self.dismissed_announcement_version = _ANNOUNCEMENT_VERSION
        if self.announcement_banner:
            self.announcement_banner.set_visibility(False)
        self._save_config()

    def _mark_interaction(self, e=None):
        if not self.current_search_interacted:
            self.current_search_interacted = True

            async def silent_success_update():
                try:
                    await counter.increment_success()
                except Exception:
                    pass
            asyncio.create_task(silent_success_update())

    def _record_first_selection_if_needed(
        self,
        previous_tags: list[str],
        current_tags: list[str],
    ) -> None:
        if self._telemetry_search_started_at is None or self._telemetry_selection_recorded:
            return
        if not (set(current_tags) - set(previous_tags)):
            return
        self._telemetry_selection_recorded = True
        duration_ms = (time.perf_counter() - self._telemetry_search_started_at) * 1000

        async def record_selection() -> None:
            try:
                await telemetry.increment("ui_search_with_selection_session")
                await telemetry.record_timing("search_to_first_selection", duration_ms)
            except Exception as exc:
                print(f"[UI] 首次选择统计失败: {exc}", flush=True)

        asyncio.create_task(record_selection())

    def _record_ui_copy(self, event_name: str) -> None:
        duration_ms: float | None = None
        if (
            self._telemetry_search_started_at is not None
            and not self._telemetry_copy_timing_recorded
        ):
            self._telemetry_copy_timing_recorded = True
            duration_ms = (time.perf_counter() - self._telemetry_search_started_at) * 1000

        async def record_copy() -> None:
            try:
                await counter.increment_copy()
                await telemetry.increment(event_name)
                if duration_ms is not None:
                    await telemetry.record_timing("search_to_first_copy", duration_ms)
            except Exception as exc:
                print(f"[UI] 复制统计失败: {exc}", flush=True)

        asyncio.create_task(record_copy())

    # ── 分页辅助 ──────────────────────────────────────────────────────────

    def _get_rows_per_page(self) -> int:
        if self.result_table is None:
            return 0
        p = self.result_table.pagination
        # pagination 可能是 int 或 dict
        if isinstance(p, dict):
            return int(p.get('rowsPerPage', 0))
        return int(p) if p else 0

    def _set_rows_per_page(self, value: int):
        if self.result_table is None:
            return
        allowed = {5, 7, 10, 15, 20, 25, 50, 0}  # 0 = All
        value = value if value in allowed else 0
        p = self.result_table.pagination
        if isinstance(p, dict):
            p['rowsPerPage'] = value
            self.result_table.pagination = p
        else:
            self.result_table.pagination = value

    # ── 配置持久化 ────────────────────────────────────────────────────────

    def _apply_prompt_format(self, prompt_format: str):
        """统一设置复制格式及按钮外观，不触发额外持久化。"""
        if prompt_format not in ('sdxl', 'nai', 'anima'):
            prompt_format = 'sdxl'
        self.prompt_format = prompt_format
        if not self.format_toggle_btn:
            return
        if prompt_format == 'nai':
            self.format_toggle_btn.text = 'NAI'
            self.format_toggle_btn.props('color=purple-7')
        elif prompt_format == 'anima':
            self.format_toggle_btn.text = 'Anima'
            self.format_toggle_btn.props('color=teal-7')
        else:
            self.format_toggle_btn.text = 'SDXL'
            self.format_toggle_btn.props('color=grey-7')

    def _collect_config_state(self) -> dict:
        return collect_config_state(self, _CONFIG_VERSION)

    def _storage_write_allowed(self, name: str) -> bool:
        """Block writes until that localStorage domain has been safely restored."""
        if not write_is_ready(name, self._storage_states, self._storage_applying) or not self._client_connected():
            if self._storage_restore_started:
                self._storage_session_dirty.add(name)
            return False
        return True

    def _save_config(self):
        """将当前控件状态序列化并写入 localStorage。"""
        if not self._storage_write_allowed('config'):
            return False
        cfg = self._collect_config_state()
        js = _json.dumps(cfg, ensure_ascii=False)
        try:
            self.client.run_javascript(
                f"localStorage.setItem('{_CONFIG_LS_KEY}', {_json.dumps(js)});"
            )
        except RuntimeError:
            self._storage_session_dirty.add('config')
            return False
        self._storage_session_dirty.discard('config')
        return True

    def _apply_config_state(self, cfg: dict):
        apply_config_state(self, cfg, _ANNOUNCEMENT_VERSION)

    # ══════════════════════════════════════════════════════════════════════
    # 页面构建
    # ══════════════════════════════════════════════════════════════════════

    def build_page(self):
        engine_ready = DanbooruTagger.is_ready()
        render_page(
            self,
            motion_style=MOTION_STYLE,
            sponsor_notice_text=SPONSOR_NOTICE_TEXT,
        )
        if not engine_ready:
            asyncio.ensure_future(self._hide_banner_when_ready())

    def _replay_motion(self, element_id: str, motion_class: str) -> None:
        """在客户端重放一次区域级动效；减少动态效果时保持静止。"""
        if not self._client_alive():
            return
        try:
            ui.run_javascript(f'''
                const element = document.getElementById('{element_id}');
                if (element && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {{
                    element.classList.remove(
                        'motion-results-enter',
                        'motion-secondary-enter',
                        'motion-refresh-enter',
                        'motion-recommendation-enter-right',
                        'motion-recommendation-enter-left',
                    );
                    void element.offsetWidth;
                    element.classList.add('{motion_class}');
                }}
            ''')
        except RuntimeError:
            pass

    # ── 搜索卡片 ─────────────────────────────────────────────────────────

    def _build_search_card(self):
        build_search_panel(self)


    # ── 工作区工具 ────────────────────────────────────────────────────────

    def _build_workspace_toolbar(self):
        build_workspace_toolbar(self)

    def _update_workspace_counts(self):
        if self.history_count_label is not None:
            self.history_count_label.text = str(len(self.search_history.get('items', [])))
        if self.favorites_count_label is not None:
            self.favorites_count_label.text = str(len(self.favorites.get('items', [])))

    def _current_search_settings(self) -> dict:
        return current_search_settings(self)

    def _apply_search_settings(self, settings: dict):
        apply_search_settings(self, settings, _SEARCH_MODE_OPTIONS)

    def _record_search_history(self, query: str):
        record_search_history_state(self, query)
        self._save_history()
        self._update_workspace_counts()

    def _open_history_dialog(self):
        open_history_dialog(self)

    async def _history_research(self, item: dict, dialog):
        dialog.close()
        self._apply_search_settings(item.get('settings', {}))
        self.search_input.set_value(item['query'])
        await self.perform_search()

    async def _history_restore(self, item: dict, dialog):
        normalized = await self._canonicalize_workspace_for_load(
            item['workspace'],
            source='历史工作区恢复',
        )
        self._push_undo_snapshot()
        self._apply_workspace_state(normalized.workspace)
        self.search_input.set_value(item['query'])
        dialog.close()
        ui.notify('已恢复历史工作区；未重新发起搜索', type='positive')
        self._show_workspace_canonicalization(normalized, '历史工作区')

    async def _history_append(self, item: dict, dialog):
        dialog.close()
        self.search_input.set_value(item['query'])
        await self.perform_search()

    def _delete_history_entry(self, item: dict, dialog):
        remove_history_entry(self, item.get('history_id'))
        self._save_history()
        self._update_workspace_counts()
        dialog.close()
        self._open_history_dialog()

    def _confirm_clear_history(self, parent_dialog):
        confirm_clear_history(self, parent_dialog)

    def _clear_all_history(self) -> None:
        clear_history(self)
        self._save_history()
        self._update_workspace_counts()

    def _open_prompt_import_dialog(self):
        open_prompt_import_dialog(
            self,
            description=UI_TEXT['dialogs']['prompt_import_description'],
        )

    async def _resolve_prompt_import_text(self, text: str) -> PromptImportResult:
        tagger = await DanbooruTagger.get_instance()
        allow_nsfw = bool(nsfw_allowed() and self.input_nsfw and self.input_nsfw.value)
        return await asyncio.to_thread(
            resolve_prompt_text,
            text,
            resolve_tag=tagger.resolve_tag_name,
            resolve_artist=tagger.resolve_artist_name,
            lookup_tag=tagger.get_tag_workspace_metadata,
            allow_nsfw=allow_nsfw,
        )

    def _apply_prompt_import_result(self, result: PromptImportResult) -> tuple[int, int]:
        current = self._get_selected_tags()
        existing = set(current)
        existing_pending = {
            (
                item.get('normalized'),
                bool(item.get('is_artist')),
                item.get('reason'),
                item.get('alias_target'),
            )
            for item in self.workspace_state.get('dismissed', [])
            if isinstance(item, dict) and item.get('kind') == 'prompt_import_pending'
        }
        pending_to_add = []
        duplicate_count = result.duplicate_count

        for pending in result.pending:
            key = (
                pending.normalized,
                pending.is_artist,
                pending.reason,
                pending.alias_target,
            )
            if key in existing_pending:
                duplicate_count += 1
                continue
            existing_pending.add(key)
            pending_to_add.append(pending_to_workspace_entry(pending))

        new_items = [item for item in result.items if item.tag not in existing]
        duplicate_count += len(result.items) - len(new_items)
        if new_items or pending_to_add:
            self._push_undo_snapshot()

        for item in new_items:
            existing.add(item.tag)
            current.append(item.tag)
            self.tag_weights[item.tag] = item.weight
            self._set_selection_meta(
                item.tag,
                'prompt_import_artist' if item.is_artist else 'prompt_import',
                item.original,
            )

        if pending_to_add:
            self.workspace_state['dismissed'] = (
                list(self.workspace_state.get('dismissed', [])) + pending_to_add
            )[-2000:]

        if new_items or pending_to_add:
            self._set_selected_tags(current, record_undo=False)
            self._render_prompt_pending()
        return len(new_items), duplicate_count

    @staticmethod
    def _prompt_pending_reason(item: dict) -> str:
        reason = item.get('reason')
        if reason == 'alias_target_missing':
            target = item.get('alias_target') or '未知目标'
            return f'Alias 指向 {target}，但目标不在当前标签库范围内'
        if reason == 'nsfw_filtered':
            return '当前 NSFW 设置不允许加入该标签'
        if reason == 'ambiguous_compact':
            return '存在多个可能的规范标签，请人工选择'
        if reason == 'not_found':
            return '当前标签库中无法唯一识别'
        if item.get('is_artist'):
            return '画师共现库中无法唯一识别'
        return f'无法识别（{reason or "unknown"}）'
    def _show_prompt_import_summary(self, result: PromptImportResult, added_count: int, duplicate_count: int):
        show_prompt_import_summary(self, result, added_count, duplicate_count)
    def _render_prompt_pending(self):
        render_prompt_pending(self)

    def _remove_prompt_pending(self, pending: dict):
        pending_id = pending.get('pending_id')
        self._push_undo_snapshot()
        self.workspace_state['dismissed'] = [
            item for item in self.workspace_state.get('dismissed', [])
            if not (
                isinstance(item, dict)
                and item.get('kind') == 'prompt_import_pending'
                and item.get('pending_id') == pending_id
            )
        ]
        self._save_staged_tags()
        self._render_prompt_pending()

    def _accept_prompt_candidate(self, pending: dict, candidate: str):
        tagger = DanbooruTagger._instance
        is_artist = bool(pending.get('is_artist'))
        if tagger is None:
            ui.notify('标签引擎尚未就绪，请稍后再试', type='warning')
            return
        if is_artist:
            resolved = tagger.resolve_artist_name(candidate)
            canonical = resolved.get('artist')
        else:
            resolved = tagger.resolve_tag_name(candidate)
            canonical = resolved.get('tag')
        if not canonical:
            ui.notify('该候选目前无法加入工作区', type='warning')
            return
        if not is_artist:
            metadata = tagger.get_tag_workspace_metadata(canonical) or {}
            nsfw_blocked = (
                str(metadata.get('nsfw', '0')) == '1'
                and not bool(nsfw_allowed() and self.input_nsfw and self.input_nsfw.value)
            )
            if nsfw_blocked:
                ui.notify('当前 NSFW 设置不允许加入该候选', type='warning')
                return

        self._push_undo_snapshot()
        self.workspace_state['dismissed'] = [
            item for item in self.workspace_state.get('dismissed', [])
            if not (
                isinstance(item, dict)
                and item.get('kind') == 'prompt_import_pending'
                and item.get('pending_id') == pending.get('pending_id')
            )
        ]
        current = self._get_selected_tags()
        if canonical not in current:
            current.append(canonical)
            self.tag_weights[canonical] = float(pending.get('weight', 1.0))
            self._set_selection_meta(
                canonical,
                'prompt_import_artist' if is_artist else 'prompt_import',
                str(pending.get('original') or ''),
            )
        self._set_selected_tags(current, record_undo=False)
        self._render_prompt_pending()
        ui.notify(
            f"已确认：{pending.get('normalized') or pending.get('original')} → {canonical}",
            type='positive',
        )

    async def _canonicalize_workspace_for_load(
        self,
        workspace: dict,
        *,
        source: str,
    ) -> WorkspaceCanonicalizationResult:
        tagger = await DanbooruTagger.get_instance()
        return await asyncio.to_thread(
            canonicalize_workspace_tags,
            workspace,
            resolve_tag=tagger.resolve_tag_name,
            resolve_artist=tagger.resolve_artist_name,
            lookup_tag=tagger.get_tag_workspace_metadata,
            artist_origins=_ARTIST_ORIGINS,
            source=source,
        )
    def _show_workspace_canonicalization(self, result: WorkspaceCanonicalizationResult, label: str):
        show_workspace_canonicalization(self, result, label)

    def _open_save_favorite_dialog(self):
        open_save_favorite_dialog(self)

    def _save_current_workspace_as_favorite(self, name: str, notes: str) -> bool:
        favorite = favorite_from_workspace(self.workspace_state, name, notes=notes)
        candidate = {
            'schema_version': 1,
            'items': [favorite] + self.favorites.get('items', [])[:199],
        }
        if self._replace_favorites_safely(candidate):
            ui.notify(f'已保存收藏：{name}', type='positive')
            return True
        return False

    def _open_favorites_dialog(self):
        open_favorites_dialog(self)

    async def _load_favorite(self, favorite: dict, merge: bool, dialog):
        if merge:
            workspace = merge_favorite_into_workspace(self.workspace_state, favorite)
            message = f"已合并收藏：{favorite['name']}"
        else:
            workspace = replace_with_favorite(favorite)
            message = f"已载入收藏：{favorite['name']}"
        normalized = await self._canonicalize_workspace_for_load(
            workspace,
            source=f"收藏恢复：{favorite['name']}",
        )
        self._push_undo_snapshot()
        self._apply_workspace_state(normalized.workspace)
        dialog.close()
        ui.notify(message, type='positive')
        self._show_workspace_canonicalization(normalized, f"收藏“{favorite['name']}”")

    def _copy_favorite(self, favorite: dict):
        parts: list[str] = []
        for item in favorite.get('selected', []):
            tag = item['tag']
            if favorite.get('prompt_format') == 'anima' and item.get('origin') in _ARTIST_ORIGINS:
                tag = f'@{tag}'
            parts.append(_format_tag_with_weight(
                tag,
                item.get('weight', 1.0),
                favorite.get('prompt_format', 'sdxl'),
            ))
        ui.clipboard.write(', '.join(parts))
        ui.notify(f"已复制收藏：{favorite['name']}", type='positive')

    def _rename_favorite(self, favorite: dict, parent_dialog):
        open_rename_favorite_dialog(self, favorite, parent_dialog)

    def _rename_favorite_to(self, favorite: dict, name: str) -> str:
        if not name:
            return '请输入名称'
        if any(
            item['favorite_id'] != favorite['favorite_id'] and item['name'] == name
            for item in self.favorites.get('items', [])
        ):
            return '已存在同名收藏'
        candidate = normalize_favorites(self.favorites)[0]
        for item in candidate['items']:
            if item['favorite_id'] == favorite['favorite_id']:
                item['name'] = name
                item['updated_at'] = utc_now_iso()
                break
        if not self._replace_favorites_safely(candidate):
            return '保存失败，请稍后重试'
        return ''

    def _overwrite_favorite(self, favorite: dict, parent_dialog):
        if not self._get_selected_tags():
            ui.notify('当前工作区没有标签，不能覆盖收藏', type='warning')
            return
        replacement = favorite_from_workspace(
            self.workspace_state,
            favorite['name'],
            notes=favorite.get('notes', ''),
            favorite_id=favorite['favorite_id'],
            created_at=favorite.get('created_at'),
        )
        candidate = {
            'schema_version': 1,
            'items': [
                replacement if item['favorite_id'] == favorite['favorite_id'] else item
                for item in self.favorites.get('items', [])
            ],
        }
        if not self._replace_favorites_safely(candidate):
            return
        parent_dialog.close()
        ui.notify(f"已覆盖收藏：{favorite['name']}", type='positive')

    def _export_favorite(self, favorite: dict):
        payload = {
            'schema_version': 1,
            'exported_at': favorite.get('updated_at', ''),
            'favorite': favorite,
        }
        raw = dump_collection(payload, label='favorite export').encode('utf-8')
        safe_name = re.sub(r'[^0-9A-Za-z\u4e00-\u9fff_-]+', '_', favorite['name'])[:60]
        ui.download(raw, filename=f'danbooru_favorite_{safe_name or "export"}.json', media_type='application/json')

    def _confirm_delete_favorite(self, favorite: dict, parent_dialog):
        confirm_delete_favorite(self, favorite, parent_dialog)

    def _delete_favorite(self, favorite: dict) -> bool:
        candidate = {
            'schema_version': 1,
            'items': [
                item for item in self.favorites.get('items', [])
                if item['favorite_id'] != favorite['favorite_id']
            ],
        }
        return self._replace_favorites_safely(candidate)

    def _open_backup_dialog(self):
        open_backup_dialog(
            self,
            description=UI_TEXT['dialogs']['backup_description'],
        )

    def _export_backup(self):
        backup = build_backup(
            config=self._collect_config_state(),
            workspace=self.workspace_state,
            history=self.search_history,
            favorites=self.favorites,
        )
        raw = _json.dumps(backup, ensure_ascii=False, indent=2).encode('utf-8')
        filename = f"danbooru_workspace_backup_{utc_now_iso()[:10]}.json"
        ui.download(raw, filename=filename, media_type='application/json')

    async def _import_backup_text(self, raw: str, mode: str):
        # 单个收藏导出文件也可直接从这里重新导入。
        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError as exc:
            raise WorkspaceDataError('文件不是有效 JSON') from exc
        if isinstance(parsed, dict) and 'favorite' in parsed:
            candidate, warnings = merge_imported_favorite(
                self.favorites,
                parsed['favorite'],
            )
            if not self._replace_favorites_safely(candidate):
                return
            ui.notify('收藏已导入', type='positive')
            if warnings:
                print(f'[UI] 收藏导入提示: {warnings}', flush=True)
            return

        backup, warnings = normalize_backup(parsed)
        workspace_normalization = None
        if mode != 'favorites_only':
            workspace_normalization = await self._canonicalize_workspace_for_load(
                backup['workspace'],
                source='JSON 备份导入',
            )
            backup['workspace'] = workspace_normalization.workspace
        plan = build_backup_import_plan(
            self.workspace_state,
            self.search_history,
            self.favorites,
            backup,
            mode,
        )
        if not self._replace_favorites_safely(plan.favorites):
            return
        if plan.workspace is not None:
            self._push_undo_snapshot()
            self.search_history = plan.history
            if plan.config is not None:
                self._apply_config_state(plan.config)
            self._apply_workspace_state(plan.workspace)
            self._save_history()
        message = plan.message
        self._update_workspace_counts()
        ui.notify(message, type='positive', timeout=4000)
        if workspace_normalization is not None:
            self._show_workspace_canonicalization(
                workspace_normalization,
                'JSON 备份工作区',
            )
        if warnings:
            print(f'[UI] 备份导入提示: {sorted(set(warnings))}', flush=True)

    # ── 已选标签栏 ────────────────────────────────────────────────────────

    def _build_selection_bar(self):
        build_selection_bar(self)
    def _render_selected_chips(self):
        render_selected_chips(self)

    def _workspace_group_for_tag(self, tag: str) -> str:
        is_artist = tag in self._workspace_artist_tags
        category = 'Artist' if is_artist else 'Other'
        groups: set[str] = set()

        tagger = DanbooruTagger._instance
        if tagger is not None:
            metadata = tagger.get_tag_workspace_metadata(tag)
            if metadata:
                category = str(metadata.get('category') or category)
                groups = set(metadata.get('groups') or [])
        elif self.result_table is not None:
            for row in self.result_table.rows:
                if row.get('tag') == tag:
                    category = str(row.get('category') or category)
                    break

        return classify_workspace_tag(
            category=category,
            tag_groups=groups,
            is_artist=is_artist,
        )
    def _render_selected_tag_chip(self, tag: str, step: float, *, animate: bool = False):
        render_selected_tag_chip(self, tag, step, animate=animate)

    def _adjust_weight(self, tag: str, delta: float):
        """调整单个标签权重。Anima 模式范围 [0.5, 5.0]，其他模式 [0.1, 1.9]。"""
        current = self.tag_weights.get(tag, 1.0)
        new_w = round(current + delta, 1)
        if self.prompt_format == 'anima':
            min_w, max_w = 0.5, 5.0
        else:
            min_w, max_w = 0.1, 1.9
        if new_w < min_w:
            ui.notify(f'权重范围为 {min_w} ~ {max_w}，已到达最小值', type='warning', timeout=2000)
            return
        if new_w > max_w:
            ui.notify(f'权重范围为 {min_w} ~ {max_w}，已到达最大值', type='warning', timeout=2000)
            return
        self._push_undo_snapshot()
        self.tag_weights[tag] = new_w
        self._save_staged_tags()
        self._render_selected_chips()

    def _get_cn_name_for_tag(self, tag: str) -> str:
        """尽量从当前 UI 数据中取标签中文名，用于已选区展示。"""
        if self.result_table is not None:
            for row in self.result_table.rows:
                if row.get('tag') == tag:
                    return str(row.get('cn_name') or '')

        for item in self.current_related:
            if getattr(item, 'tag', None) == tag:
                return str(getattr(item, 'cn_name', '') or '')

        for item in self.workspace_state.get('selected', []):
            if item.get('tag') == tag:
                return str(item.get('cn_name') or '')

        try:
            tagger = DanbooruTagger._instance
            if tagger and tagger.df is not None and tag in tagger._name_to_idx:
                idx = tagger._name_to_idx[tag]
                return str(tagger.df.iloc[idx].get('cn_name', '') or '')
        except Exception:
            pass
        return ''

    def _remove_selected_tag(self, tag: str):
        """从已选中移除标签（同步表格选中状态）。"""
        self._mark_interaction()
        current = self._get_selected_tags()
        if tag in current:
            current.remove(tag)
        self.tag_weights.pop(tag, None)
        self._set_selected_tags(current)

    # ── 备选区持久化 ─────────────────────────────────────────────────────

    _STAGED_LS_KEY = LEGACY_STAGED_STORAGE_KEY

    def _set_selection_meta(self, tag: str, origin: str, source: str = ''):
        set_selection_meta(self, tag, origin, source)

    def _push_undo_snapshot(self):
        push_undo_snapshot(self)

    def _update_undo_buttons(self):
        if self.undo_btn is not None:
            self.undo_btn.enable() if self._undo_stack else self.undo_btn.disable()
        if self.redo_btn is not None:
            self.redo_btn.enable() if self._redo_stack else self.redo_btn.disable()

    def _apply_workspace_state(
        self,
        workspace: dict,
        *,
        persist: bool = True,
        refresh_recommendations: bool = True,
    ):
        apply_workspace_state(
            self,
            workspace,
            _ARTIST_ORIGINS,
            persist=persist,
            refresh_recommendations=refresh_recommendations,
        )

    def _undo_workspace(self):
        target = pop_undo_workspace(self)
        if target is None:
            ui.notify('没有可撤销的操作', type='info', timeout=1500)
            return
        self._apply_workspace_state(target)
        self._update_undo_buttons()
        ui.notify('已撤销', type='positive', timeout=1500)

    def _redo_workspace(self):
        target = pop_redo_workspace(self)
        if target is None:
            ui.notify('没有可恢复的操作', type='info', timeout=1500)
            return
        self._apply_workspace_state(target)
        self._update_undo_buttons()
        ui.notify('已恢复', type='positive', timeout=1500)

    def _schedule_workspace_persist(self):
        schedule_workspace_persist(self, WORKSPACE_SAVE_DEBOUNCE_SECONDS)

    def _save_staged_tags(self):
        """将实时选择同步到版本化 WorkspaceState，并去抖写入 localStorage。"""
        sync_workspace_selection(self, _ARTIST_ORIGINS)
        self._schedule_workspace_persist()

    def _local_storage_keys(self) -> dict[str, str]:
        return storage_keys()

    async def _prepare_local_storage_restore(self, names: list[str]) -> dict:
        """Snapshot requested keys in-browser and compact legacy history in memory."""
        keys = {name: self._local_storage_keys()[name] for name in names}
        return await prepare_restore_snapshot(self.client, self._client_connected, keys)

    async def _read_local_storage_value(
        self,
        name: str,
        key: str,
        length,
    ) -> str | None:
        return await read_prepared_value(self.client, self._client_connected, name, key, length)

    def _clear_local_storage_restore_cache(self):
        if not self._client_connected():
            return
        clear_restore_cache(self.client)

    async def _backup_local_storage_key(self, source_key: str, backup_key: str) -> bool:
        """Copy an existing value inside the browser before replacing it."""
        if not self._client_connected():
            return False
        return await backup_key(self.client, source_key, backup_key)

    async def _backup_history_before_compaction(self) -> bool:
        """Preserve the original history in-browser before replacing it with v2."""
        return await self._backup_local_storage_key(
            HISTORY_STORAGE_KEY,
            _HISTORY_PRE_COMPACTION_BACKUP_KEY,
        )

    def _save_history(self):
        if not self._storage_write_allowed('history'):
            return False
        client = self.client
        while True:
            try:
                data = dump_collection(self.search_history, label='history')
                client.run_javascript(
                    f"localStorage.setItem('{HISTORY_STORAGE_KEY}', {_json.dumps(data)});"
                )
                self._storage_session_dirty.discard('history')
                return True
            except WorkspaceDataError as exc:
                if not self.search_history.get('items'):
                    print(f'[UI] 搜索历史保存失败: {exc}', flush=True)
                    return False
                self.search_history['items'].pop()
                print('[UI] 搜索历史超过本地大小限制，已移除最旧记录。', flush=True)
            except RuntimeError as exc:
                print(f'[UI] 搜索历史保存失败: {exc}', flush=True)
                self._storage_session_dirty.add('history')
                return False

    def _save_favorites(self):
        if not self._storage_write_allowed('favorites'):
            return False
        client = self.client
        try:
            data = dump_collection(self.favorites, label='favorites')
            client.run_javascript(
                f"localStorage.setItem('{FAVORITES_STORAGE_KEY}', {_json.dumps(data)});"
            )
            self._storage_session_dirty.discard('favorites')
            return True
        except (WorkspaceDataError, RuntimeError) as exc:
            print(f'[UI] 收藏保存失败: {exc}', flush=True)
            self._storage_session_dirty.add('favorites')
            return False

    def _replace_favorites_safely(self, favorites: dict) -> bool:
        if replace_favorites_safely(self, favorites):
            return True
        ui.notify('收藏数据过大或无法写入，操作已取消', type='negative')
        return False

    async def _restore_staged_tags(self) -> tuple[dict[str, str], set[str], list[str]]:
        """Run one restore attempt and return failures, required writes and warnings."""
        return await restore_staged_storage(
            self,
            _LOCAL_STORAGE_NAMES,
            _CONFIG_VERSION,
        )

    async def _restore_local_storage_with_retries(self):
        self._storage_restore_started = True
        was_cancelled = False
        try:
            result = await restore_with_retries(
                self,
                _LOCAL_STORAGE_RESTORE_RETRY_DELAYS,
            )
            if result.client_stopped:
                return
            if result.completed:
                flush_storage_session_changes(self)
                self._install_workspace_storage_listener()
                self._storage_failure_notified = False
                if result.warnings:
                    print(
                        f'[UI] 本地数据恢复提示: {sorted(set(result.warnings))}',
                        flush=True,
                    )
                return
            unresolved = sorted(
                name for name in _LOCAL_STORAGE_NAMES
                if self._storage_states.get(name) != 'ready'
            )
            if unresolved:
                client_id = str(getattr(self.client, 'id', 'unknown'))[:8]
                detail = '; '.join(
                    f'{name}={message}' for name, message in sorted(result.failures.items())
                )
                print(
                    f'[UI] localStorage 恢复未完成 (client={client_id}, '
                    f'keys={unresolved}): {detail}；未覆盖这些浏览器数据。',
                    flush=True,
                )
                if self._client_connected() and not self._storage_failure_notified:
                    self._storage_failure_notified = True
                    try:
                        with self.client:
                            ui.notify(
                                '本地工作区数据暂未恢复，已停止覆盖旧数据；连接恢复后会自动重试。',
                                type='warning',
                                timeout=5000,
                            )
                    except RuntimeError:
                        # The client may disconnect after the connection check.
                        pass
        except asyncio.CancelledError:
            was_cancelled = True
        finally:
            if finish_storage_restore_task(
                self,
                asyncio.current_task(),
                was_cancelled,
            ):
                self._start_storage_restore_task()

    def _start_storage_restore_task(self):
        return start_storage_restore_task(
            self,
            self._restore_local_storage_with_retries,
        )

    def _pause_storage_restore(self):
        pause_storage_restore(self)

    def _install_workspace_storage_listener(self):
        """其他标签页修改工作区时提示刷新，避免静默覆盖。"""
        if self._workspace_storage_listener_installed:
            return
        self._workspace_storage_listener_installed = True
        try:
            ui.run_javascript(storage_listener_script())
        except RuntimeError:
            pass

    def _clear_all_staged(self):
        """清空所有已选标签。"""
        self._mark_interaction()
        pending_items = [
            item for item in self.workspace_state.get('dismissed', [])
            if isinstance(item, dict) and item.get('kind') == 'prompt_import_pending'
        ]
        if self._get_selected_tags() or pending_items:
            self._push_undo_snapshot()
        if pending_items:
            self.workspace_state['dismissed'] = [
                item for item in self.workspace_state.get('dismissed', [])
                if not (
                    isinstance(item, dict)
                    and item.get('kind') == 'prompt_import_pending'
                )
            ]
        self.chip_extra_selected.clear()
        self._selected_order.clear()
        self.tag_weights.clear()
        self._pending_selection_meta.clear()
        self._workspace_artist_tags.clear()
        if self.result_table is not None:
            self.result_table.selected = []
        self._artist_rec_checkboxes.clear()
        self._current_artist_rec_tags.clear()
        self._artist_result_tags.clear()
        self._last_recommendation_seed_tags = []
        self._render_selected_chips()
        self._render_prompt_pending()
        self._render_concept_coverage()
        if self.selection_count_label is not None:
            self.selection_count_label.text = '0'
        show_nsfw_val = self.input_nsfw.value
        self._refresh_related([], show_nsfw_val)
        self._render_artist_rec([], {})
        # 清空 Group 同类扩展
        if self.group_expansion_container is not None:
            self.group_expansion_container.clear()
            with self.group_expansion_container:
                ui.label('请先搜索并勾选标签…').classes('text-sm text-gray-400 italic p-4')
        self._save_staged_tags()
        ui.notify('已清空所有已选标签', type='warning')

        # ── 两栏结果（CSS 强制并排）──────────────────────────────────────────
    def _build_results_columns(self):
        build_results_columns(self)

    def _set_related_page(self, page: int):
        """切换关联推荐页，只构建当前页的可见行。"""
        set_paginated_recommendation_page(
            self,
            page,
            state_prefix='_related',
            render_page=self._render_related_page,
            motion_element_id='danbooru-related-recommendations',
        )

    def _lookup_tag_wiki(self, tag: str) -> str:
        """在控制器层读取标签元数据，视图层不直接访问检索引擎。"""
        try:
            tagger = DanbooruTagger._instance
            if tagger and tagger.df is not None and tag in tagger._name_to_idx:
                index = tagger._name_to_idx[tag]
                return str(tagger.df.iloc[index].get('wiki', ''))
        except Exception:
            pass
        return ''

    def _render_related_page(self):
        render_related_page(self)

    def _render_related_list(self, related: list, show_nsfw: bool):
        """保存关联推荐快照，并仅渲染当前页。"""
        render_related_list(self, related, show_nsfw)

    # ══════════════════════════════════════════════════════════════════════
    # 交互逻辑
    # ══════════════════════════════════════════════════════════════════════

    async def _hide_banner_when_ready(self):
        while not DanbooruTagger.is_ready():
            if not self._client_alive():
                return
            await asyncio.sleep(1)
        if not self._client_alive():
            return
        if self.init_banner:
            self.init_banner.set_visibility(False)
        self._update_service_status()
        self._update_footer_text()
        # 工作区可能早于引擎恢复；引擎就绪后用真实 Category / Tag Group
        # 元数据重新分组，避免重启期间首次渲染的标签全部停留在“其他”。
        self._render_selected_chips()

    def _client_alive(self) -> bool:
        try:
            client = self.client
            if client is None and self.search_btn is not None:
                client = self.search_btn.client
            return client is not None and not bool(getattr(client, '_deleted', False))
        except (AttributeError, RuntimeError):
            return False

    def _dispose(self):
        """Stop page-owned background resources."""
        tasks = (
            self._service_status_task,
            self._storage_restore_task,
            self._workspace_save_task,
        )
        self._service_status_task = None
        self._storage_restore_task = None
        self._workspace_save_task = None
        for task in tasks:
            if task is not None and not task.done():
                task.cancel()

    # ── 分词筛选 ──────────────────────────────────────────────────────────

    def _filter_by_source(self, keyword: str):
        filter_by_source(self, keyword)
    def _render_concept_coverage(self):
        render_concept_coverage(self)

    def _schedule_concept_coverage_render(self):
        """让选择区先刷新，再在下一个事件循环周期重建查询理解。"""
        if self._coverage_render_task and not self._coverage_render_task.done():
            self._coverage_render_task.cancel()

        async def _render():
            await asyncio.sleep(0)
            if self._client_alive():
                self._render_concept_coverage()

        self._coverage_render_task = asyncio.ensure_future(_render())

    async def _search_uncovered_segment(self, segment: str):
        segment = str(segment or '').strip()
        if not segment:
            return
        self.search_input.set_value(segment)
        await self.perform_search()

    # ── 搜索 ──────────────────────────────────────────────────────────────

    async def perform_search(self):
        query = self.search_input.value.strip()
        if not query:
            return

        # 搜索前校验数值参数
        _err_fields = []
        if self.input_top_k and (self.input_top_k.value is None or str(self.input_top_k.value).strip() == ''):
            _err_fields.append('Top K')
        if self.input_limit and (self.input_limit.value is None or str(self.input_limit.value).strip() == ''):
            _err_fields.append('返回数量')
        if self.input_weight and (self.input_weight.value is None or str(self.input_weight.value).strip() == ''):
            _err_fields.append('热度权重')
        if _err_fields:
            ui.notify(f'请填写：{"、".join(_err_fields)}', type='negative', timeout=3000)
            return

        # 搜索前保存配置
        self._save_config()

        self.current_query_str = query
        self.search_btn.disable()
        self.spinner.classes(remove='hidden')
        ui.notify('正在搜索...', type='info')

        if self.bad_case_btn is not None:
            self.bad_case_btn.disable()

        target_layers_list = [k for k, v in self.selected_layers.items() if v]
        target_cats_list   = [k for k, v in self.selected_cats.items()   if v]

        if not target_layers_list:
            ui.notify('请至少选择一个匹配层！', type='warning')
            self.search_btn.enable()
            self.spinner.classes(add='hidden')
            return

        search_started_at = time.perf_counter()
        search_invoked_at = time.monotonic()
        normalized_telemetry_query = ' '.join(query.split()).casefold()
        await telemetry.increment("ui_search")
        if (
            normalized_telemetry_query == self._telemetry_last_query
            and search_invoked_at - self._telemetry_last_search_at <= 60
        ):
            await telemetry.increment("ui_repeat_search_60s")
        self._telemetry_last_query = normalized_telemetry_query
        self._telemetry_last_search_at = search_invoked_at

        try:
            tagger = await DanbooruTagger.get_instance()

            show_nsfw_val = self.input_nsfw.value

            request = SearchRequest(
                query=query,
                top_k=int(self.input_top_k.value),
                limit=int(self.input_limit.value),
                popularity_weight=float(self.input_weight.value),
                show_nsfw=show_nsfw_val,
                use_segmentation=self.input_segment.value if self.input_segment else True,
                target_layers=target_layers_list,
                target_categories=target_cats_list,
                group_mode=self.input_group_mode.value if self.input_group_mode else 'off',
                max_per_group=int(self.input_max_per_group.value) if self.input_max_per_group else 2,
            )
            response = await tagger.search_async(request)
            await telemetry.record_timing(
                "ui_search_latency",
                (time.perf_counter() - search_started_at) * 1000,
            )
            if not response.results:
                await telemetry.increment("ui_zero_result")
            self._telemetry_search_started_at = search_started_at
            self._telemetry_selection_recorded = False
            self._telemetry_copy_timing_recorded = False

            # 后台计数
            async def silent_counter_update():
                try:
                    await counter.increment()
                    if response.keywords:
                        await counter.add_keywords(response.keywords)
                    self._update_footer_text()
                except Exception as e:
                    print(f"[UI] 后台静默更新计数失败: {e}", flush=True)
            asyncio.create_task(silent_counter_update())

            if not self._client_alive():
                return

            table_data = [result_to_row(r, show_nsfw_val) for r in response.results]
            self._artist_result_tags = {row['tag'] for row in table_data if row.get('layer') == 'artist'}
            self.full_table_data = table_data
            self.full_tags_str = response.tags_all
            self.full_tags_str_sfw = response.tags_sfw
            self.current_segments = list(response.segments) if response.segments else []
            self.current_keywords = list(response.keywords) if response.keywords else []
            self.current_cached_queries = set(response.cached_queries or [])

            self.results_section.set_visibility(True)

            _saved_rpp = self._get_rows_per_page()
            self.result_table.rows = apply_nsfw_filter(table_data, show_nsfw_val)
            self._set_rows_per_page(_saved_rpp)
            all_selected = self._get_selected_tags()
            self._selected_order = list(all_selected)
            self.chip_extra_selected.clear()
            self.chip_extra_selected.update(all_selected)
            self.result_table.selected = []
            self._render_selected_chips()
            await self._update_selection_display(None)

            self._refresh_related([], show_nsfw_val)
            self._last_recommendation_seed_tags = []

            # 查询理解：分词来源筛选与概念覆盖共用同一组 chips
            self.current_filter_keyword = 'ALL'
            self._render_concept_coverage()
            self._record_search_history(query)
            self._replay_motion('danbooru-results-section', 'motion-results-enter')
            ui.notify(f'找到 {len(table_data)} 个标签', type='positive')
            self.current_search_interacted = False

            if self.bad_case_btn is not None:
                self.bad_case_btn.enable()

        except RuntimeError as e:
            if 'deleted' in str(e).lower() or 'client' in str(e).lower():
                return
            try:
                ui.notify(f'错误: {str(e)}', type='negative')
            except RuntimeError:
                pass
        except Exception as e:
            try:
                ui.notify(f'错误: {str(e)}', type='negative')
            except RuntimeError:
                pass
        finally:
            try:
                self.search_btn.enable()
                self.spinner.classes(add='hidden')
            except RuntimeError:
                pass

    # ── 选择管理 ──────────────────────────────────────────────────────────

    def _get_selected_tags(self) -> list[str]:
        return selected_tags(self)

    def _get_recommendation_seed_tags(self, selected_tags: list[str]) -> list[str]:
        return recommendation_seed_tags(self, selected_tags)

    def _refresh_recommendations_if_seed_changed(self, selected_tags: list[str], show_nsfw: bool):
        seed_tags = self._get_recommendation_seed_tags(selected_tags)
        if seed_tags == self._last_recommendation_seed_tags:
            return
        self._last_recommendation_seed_tags = list(seed_tags)
        self._refresh_related_from_selection(seed_tags, show_nsfw)
        self._refresh_group_from_selection(seed_tags, show_nsfw)
        self._refresh_artist_from_selection(seed_tags, show_nsfw)

    def _set_selected_tags(
        self,
        tags: list[str],
        skip_refresh: bool = False,
        record_undo: bool = True,
    ):
        tags = list(dict.fromkeys(tags))
        previous_tags = [item['tag'] for item in self.workspace_state.get('selected', [])]
        if record_undo and tags != previous_tags:
            self._push_undo_snapshot()
        self._selected_order = list(tags)
        tag_set = set(tags)
        table_tag_set = {row['tag'] for row in self.result_table.rows} if self.result_table else set()
        self.chip_extra_selected.clear()
        self.chip_extra_selected.update(t for t in tag_set if t not in table_tag_set)
        # clean up weights for deselected tags
        for t in list(self.tag_weights):
            if t not in tag_set:
                del self.tag_weights[t]
                self._pending_selection_meta.pop(t, None)

        if self.result_table is not None:
            self.result_table.selected = [row for row in self.result_table.rows if row.get('tag') in tag_set]

        # 同步推荐画师 checkbox
        for t, cb in self._artist_rec_checkboxes.items():
            cb.set_value(t in tag_set)

        all_tags = self._get_selected_tags()
        self._selected_order = list(all_tags)
        self._record_first_selection_if_needed(previous_tags, all_tags)
        if self.selection_count_label is not None:
            self.selection_count_label.text = str(len(all_tags))
        self._save_staged_tags()
        self._render_selected_chips()
        self._schedule_concept_coverage_render()
        # 显式刷新关联推荐和 Group 区域（不依赖 table.on('selection') 事件，
        # 因为在 chip 点击回调上下文中该事件可能不可靠）。
        # 从关联推荐/同类标签勾选时跳过，由各自动态刷新或手动按钮触发。
        if not skip_refresh:
            show_nsfw_val = self.input_nsfw.value
            self._refresh_recommendations_if_seed_changed(all_tags, show_nsfw_val)
            if not all_tags:
                self.chip_extra_selected.clear()

    async def _update_selection_display(self, _e):
        if self.result_table is None:
            return
        self._mark_interaction()

        all_tags = self._get_selected_tags()
        previous_tags = [item['tag'] for item in self.workspace_state.get('selected', [])]
        self._record_first_selection_if_needed(previous_tags, all_tags)
        if all_tags != previous_tags:
            self._push_undo_snapshot()
        existing = set(previous_tags)
        for row in self.result_table.rows:
            tag = row.get('tag')
            if not tag or tag in existing or tag not in all_tags:
                continue
            origin = 'artist_search' if row.get('layer') == 'artist' else 'semantic_search'
            self._set_selection_meta(tag, origin, str(row.get('source') or self.current_query_str))
        self._selected_order = list(all_tags)
        # clean up weights for deselected tags
        tag_set = set(all_tags)
        for t in list(self.tag_weights):
            if t not in tag_set:
                del self.tag_weights[t]
                self._pending_selection_meta.pop(t, None)
        # init weight for newly selected tags
        for t in all_tags:
            self.tag_weights.setdefault(t, 1.0)

        if self.selection_count_label is not None:
            self.selection_count_label.text = str(len(all_tags))
        self._render_selected_chips()

        # 先把选择区的轻量变化交给 WebSocket 刷出，再处理推荐和查询理解。
        self._save_staged_tags()
        await asyncio.sleep(0)

        # 同步当前画师页的 checkbox（当前页最多 8 个）。
        for t, cb in self._artist_rec_checkboxes.items():
            cb.set_value(t in tag_set)

        show_nsfw_val = self.input_nsfw.value
        self._refresh_recommendations_if_seed_changed(all_tags, show_nsfw_val)
        if not all_tags:
            self.chip_extra_selected.clear()
        self._render_concept_coverage()

    def _on_related_checkbox_change(self, tag: str, checked: bool):
        self._mark_interaction()
        current = self._get_selected_tags()
        if checked:
            if tag not in current:
                source = ''
                for item in self.current_related:
                    if getattr(item, 'tag', None) == tag:
                        source = ', '.join(getattr(item, 'sources', []) or [])
                        break
                self._set_selection_meta(tag, 'related_recommendation', source)
                current.append(tag)
                self.tag_weights.setdefault(tag, 1.0)
                self._set_selected_tags(current, skip_refresh=True)
                ui.notify(f'已添加 {tag}', type='positive', timeout=1500)
        else:
            if tag in current:
                current.remove(tag)
                self.tag_weights.pop(tag, None)
                self._set_selected_tags(current, skip_refresh=True)
                ui.notify(f'已移除 {tag}', type='warning', timeout=1500)
        # 刷新推荐画师
        show_nsfw_val = self.input_nsfw.value
        self._refresh_artist_from_selection(current, show_nsfw_val)

    def _on_group_checkbox_change(self, tag: str, checked: bool):
        """同类标签复选框变化回调。"""
        self._mark_interaction()
        current = self._get_selected_tags()
        if checked:
            if tag not in current:
                self._set_selection_meta(
                    tag,
                    'tag_group',
                    self._group_candidate_sources.get(tag, ''),
                )
                current.append(tag)
                self.tag_weights.setdefault(tag, 1.0)
                self._set_selected_tags(current, skip_refresh=True)
                ui.notify(f'已添加 {tag}', type='positive', timeout=1500)
        else:
            if tag in current:
                current.remove(tag)
                self.tag_weights.pop(tag, None)
                self._set_selected_tags(current, skip_refresh=True)
                ui.notify(f'已移除 {tag}', type='warning', timeout=1500)
        # 即刻刷新关联推荐 + 画师推荐
        show_nsfw_val = self.input_nsfw.value
        self._refresh_related_from_selection(current, show_nsfw_val)
        self._refresh_artist_from_selection(current, show_nsfw_val)

    def _on_artist_rec_checkbox_change(self, tag: str, checked: bool):
        """推荐画师复选框变化回调。"""
        self._mark_interaction()
        current = self._get_selected_tags()
        if checked:
            if tag not in current:
                self._set_selection_meta(
                    tag,
                    'artist_recommendation',
                    self._artist_rec_sources.get(tag, ''),
                )
                current.append(tag)
                self.tag_weights.setdefault(tag, 1.0)
                self._set_selected_tags(current, skip_refresh=True)
                ui.notify(f'已添加画师 {tag}', type='positive', timeout=1500)
        else:
            if tag in current:
                current.remove(tag)
                self.tag_weights.pop(tag, None)
                self._set_selected_tags(current, skip_refresh=True)
                ui.notify(f'已移除画师 {tag}', type='warning', timeout=1500)

    def _manual_refresh_related(self):
        """手动触发关联推荐列表的刷新"""
        self._mark_interaction()
        show_nsfw_val = self.input_nsfw.value
        all_tags = self._get_selected_tags()

        if all_tags:
            self._refresh_related_from_selection(all_tags, show_nsfw_val)
            self._refresh_artist_from_selection(all_tags, show_nsfw_val)
            ui.notify('已触发关联推荐更新', type='info', timeout=1500)
        else:
            self.chip_extra_selected.clear()
            self._refresh_related([], show_nsfw_val)
            ui.notify('已清空关联推荐', type='info', timeout=1500)

    def _manual_refresh_group(self):
        """手动触发同类扩展区域的刷新"""
        self._mark_interaction()
        show_nsfw_val = self.input_nsfw.value
        all_tags = self._get_selected_tags()

        if all_tags:
            self._refresh_group_from_selection(all_tags, show_nsfw_val)
            ui.notify('已触发同类标签更新', type='info', timeout=1500)
        else:
            if self.group_expansion_container is not None:
                self.group_expansion_container.clear()
                with self.group_expansion_container:
                    ui.label('请先搜索并勾选标签…').classes('text-sm text-gray-400 italic p-4')
            ui.notify('暂未选中标签', type='info', timeout=1500)

    # ── 关联推荐 ──────────────────────────────────────────────────────────

    def _refresh_related(self, related: list, show_nsfw: bool):
        selected_now = set(self._get_selected_tags())
        merged = merge_related_results(self.current_related, related, selected_now)

        self.current_related = merged
        if self.related_list_container is not None:
            self._render_related_list(merged, show_nsfw)

    def _refresh_related_from_selection(self, selected_tags: list[str], show_nsfw: bool):
        """请求刷新关联推荐；实际计算由统一的 latest-wins 调度器执行。"""
        self._queue_recommendation_refresh(
            selected_tags, show_nsfw, {'related'},
        )

    def _refresh_group_from_selection(self, selected_tags: list[str], show_nsfw: bool):
        """请求刷新同类扩展；实际计算由统一的 latest-wins 调度器执行。"""
        self._queue_recommendation_refresh(
            selected_tags, show_nsfw, {'group'},
        )

    def _refresh_artist_from_selection(self, selected_tags: list[str], show_nsfw: bool = True):
        """请求刷新画师推荐；实际计算由统一的 latest-wins 调度器执行。"""
        self._queue_recommendation_refresh(
            selected_tags, show_nsfw, {'artist'},
        )

    def _queue_recommendation_refresh(
        self,
        selected_tags: list[str],
        show_nsfw: bool,
        scopes: set[str],
    ):
        """合并同一选择快照的刷新范围，并确保运行中的线程不会被取消。"""
        if queue_latest_recommendation_request(
            self,
            selected_tags,
            show_nsfw,
            scopes,
        ):
            self._recommendation_task = asyncio.ensure_future(
                self._recommendation_worker()
            )

    async def _recommendation_worker(self):
        """连续消费最新选择快照；过期结果只计算不渲染。"""
        try:
            async def fetch(request: dict) -> dict:
                selected_tags = request['selected_tags']
                if not selected_tags:
                    return {
                        'related': [],
                        'groups': [],
                        'artists': [],
                        'artist_top_tags': {},
                    }
                tagger = await DanbooruTagger.get_instance()
                return await tagger.get_selection_recommendations_async(
                    selected_tags,
                    request['show_nsfw'],
                    request['scopes'],
                    related_limit=50,
                    artist_limit=ARTIST_REC_LIMIT,
                    artist_min_cooc=3,
                )

            async def apply(request: dict, result: dict) -> None:
                selected_tags = request['selected_tags']
                show_nsfw = request['show_nsfw']
                scopes = request['scopes']
                if 'related' in scopes:
                    self._refresh_related(result['related'], show_nsfw)
                if 'group' in scopes:
                    if selected_tags:
                        await self._capture_group_scroll_positions()
                        self._render_group_expansion(
                            result['groups'], selected_tags, show_nsfw,
                        )
                    elif self.group_expansion_container is not None:
                        self.group_expansion_container.clear()
                        with self.group_expansion_container:
                            ui.label('请先搜索并勾选标签…').classes(
                                'text-sm text-gray-400 italic p-4'
                            )
                if 'artist' in scopes:
                    self._render_artist_rec(
                        result['artists'],
                        result['artist_top_tags'],
                        show_nsfw,
                    )

            consume = consume_latest_recommendation_requests(
                self,
                debounce_seconds=RECOMMENDATION_DEBOUNCE_SECONDS,
                fetch=fetch,
                apply=apply,
                client_alive=self._client_alive,
                report_error=lambda exc: print(f'[UI] 推荐刷新失败: {exc}', flush=True),
            )
            await consume
        finally:
            self._recommendation_task = None

    def _set_artist_rec_page(self, page: int):
        """切换推荐画师页，只构建当前页的可见行。"""
        set_paginated_recommendation_page(
            self,
            page,
            state_prefix='_artist_rec',
            render_page=self._render_artist_rec_page,
            motion_element_id='danbooru-artist-recommendations',
        )
    def _render_artist_rec_page(self):
        render_artist_page(self)

    def _render_artist_rec(self, artist_results, top_tags=None, show_nsfw: bool = True):
        """保存推荐快照，并仅渲染当前画师页。"""
        render_artist_recommendations(self, artist_results, top_tags, show_nsfw)
    def _render_group_expansion(self, group_data: list, selected_tags: list[str], show_nsfw: bool):
        render_group_expansion(self, group_data, selected_tags, show_nsfw)

    def _on_group_expansion_change(self, group_name: str, event):
        value = getattr(event, 'args', None)
        if isinstance(value, dict):
            value = value.get('value', value.get('modelValue'))
        if bool(value):
            self._group_expanded_names.add(group_name)
        else:
            self._group_expanded_names.discard(group_name)

    async def _capture_group_scroll_positions(self, *, anchor_bottom: bool = False):
        client = self.client
        if client is None or getattr(client, '_deleted', False):
            return
        anchor_flag = 'true' if anchor_bottom else 'false'
        try:
            raw = await client.run_javascript(
                f"""
                const anchorBottom = {anchor_flag};
                const groupEntries = Array.from(document.querySelectorAll('[data-danbooru-group-scroll="1"]'))
                    .flatMap(el => {{
                        const top = Math.round(el.scrollTop || 0);
                        const entries = [[el.id, top]];
                        if (anchorBottom) {{
                            entries.push([`${{el.id}}__bottom__`, Math.round(el.scrollHeight - top)]);
                        }}
                        return entries;
                    }});
                JSON.stringify({{
                    __window__: Math.round(
                        window.scrollY ||
                        document.documentElement.scrollTop ||
                        document.body.scrollTop ||
                        0
                    ),
                    ...Object.fromEntries(groupEntries),
                }});
                """,
                timeout=1.0,
            )
        except Exception:
            return
        try:
            data = _json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, dict):
                self._group_scroll_positions = {
                    str(k): int(v) for k, v in data.items()
                    if str(k) and int(v) >= 0
                }
        except Exception:
            pass

    def _restore_group_scroll_positions(self):
        if not self._group_scroll_positions:
            return
        client = self.client
        if client is None or getattr(client, '_deleted', False):
            return
        client.run_javascript(_scroll_state_restore_script(self._group_scroll_positions), timeout=1.0)

    async def _load_more_group_tags(
        self,
        group_name: str,
        total: int,
        group_data: list,
        selected_tags: list[str],
        show_nsfw: bool,
    ):
        await self._capture_group_scroll_positions(anchor_bottom=True)
        current = self._group_render_limits.get(group_name, GROUP_RENDER_TAG_LIMIT)
        self._group_render_limits[group_name] = _next_group_render_limit(
            current,
            total,
            GROUP_RENDER_TAG_LIMIT,
        )
        self._group_expanded_names.add(group_name)
        self._render_group_expansion(group_data, selected_tags, show_nsfw)

    # ── 表格列动态更新 ──────────────────────────────────────────────────

    def _update_table_columns(self, e=None):
        update_table_columns(self, e)

    # ── 搜索模式 / 参数联动 ──────────────────────────────────────────────

    def _on_search_mode_change(self, _e=None):
        mode = self.input_search_mode.value if self.input_search_mode else None
        if not mode or mode == '自定义' or mode not in _SEARCH_MODE_PRESETS:
            return
        preset = _SEARCH_MODE_PRESETS[mode]
        self._applying_preset = True
        try:
            if self.input_top_k:
                self.input_top_k.set_value(preset['top_k'])
            if self.input_limit:
                self.input_limit.set_value(preset['limit'])
            if self.input_weight:
                self.input_weight.set_value(preset['popularity_weight'])
            if self.input_segment:
                self.input_segment.set_value(preset['use_segmentation'])
            if self.input_group_mode:
                self.input_group_mode.set_value(preset['group_mode'])
            if self.input_max_per_group:
                self.input_max_per_group.set_value(preset['max_per_group'])
        finally:
            self._applying_preset = False

    def _on_param_changed(self, _e=None):
        if not self._applying_preset and self.input_search_mode:
            if self.input_search_mode.value != '自定义':
                self.input_search_mode.set_value('自定义')

    # ── NSFW 切换 ─────────────────────────────────────────────────────────

    async def on_nsfw_toggle(self, e):
        show_nsfw_val = self.input_nsfw.value

        # 复用当前分词筛选：同时套用新 NSFW 状态并保持 chip 选中态
        self._filter_by_source(self.current_filter_keyword)
        if not show_nsfw_val:
            self.result_table.selected = [r for r in self.result_table.selected if r.get('nsfw') != '1']
        await self._update_selection_display(None)

    # ── 复制 / 反馈 ──────────────────────────────────────────────────────

    def _toggle_prompt_format(self):
        if self.prompt_format == 'sdxl':
            self._apply_prompt_format('nai')
        elif self.prompt_format == 'nai':
            self._apply_prompt_format('anima')
        else:
            self._apply_prompt_format('sdxl')
        self._save_config()
        self._save_staged_tags()
        self._render_selected_chips()

    def copy_selection(self):
        self._mark_interaction()
        tags = self._get_selected_tags()
        parts = []
        artist_tags = (
            set(self._current_artist_rec_tags)
            | set(self._artist_result_tags)
            | set(self._workspace_artist_tags)
        )
        for t in tags:
            w = self.tag_weights.get(t, 1.0)
            if self.prompt_format == 'anima' and t in artist_tags:
                parts.append(_format_tag_with_weight(f'@{t}', w, self.prompt_format))
            else:
                parts.append(_format_tag_with_weight(t, w, self.prompt_format))
        prompt = ', '.join(parts)
        ui.clipboard.write(prompt)
        fmt_label = {'sdxl': 'SDXL', 'nai': 'NAI', 'anima': 'Anima'}.get(self.prompt_format, 'SDXL')
        ui.notify(f'已复制选中标签（{fmt_label} 格式）!', type='positive')
        if prompt:
            self._record_ui_copy("ui_copy_selected")

    def _copy_all_tags(self):
        self._mark_interaction()
        show_nsfw_val = self.input_nsfw.value
        tags_str = self.full_tags_str if show_nsfw_val else self.full_tags_str_sfw
        if tags_str:
            tags_str = tags_str.replace('(', '\\(').replace(')', '\\)')
            ui.clipboard.write(tags_str)
            ui.notify('已复制全部标签!', type='positive')
            self._record_ui_copy("ui_copy_all")
        else:
            ui.notify('暂无标签可复制', type='warning')

    def _feedback_settings(self) -> dict:
        return {
            'top_k': int(self.input_top_k.value) if self.input_top_k else None,
            'limit': int(self.input_limit.value) if self.input_limit else None,
            'popularity_weight': float(self.input_weight.value) if self.input_weight else None,
            'show_nsfw': self.input_nsfw.value if self.input_nsfw else None,
            'use_segmentation': self.input_segment.value if self.input_segment else None,
            'search_mode': self.input_search_mode.value if self.input_search_mode else '自定义',
            'target_layers': [key for key, enabled in self.selected_layers.items() if enabled],
            'target_categories': [key for key, enabled in self.selected_cats.items() if enabled],
            'group_mode': self.input_group_mode.value if self.input_group_mode else 'off',
            'max_per_group': int(self.input_max_per_group.value) if self.input_max_per_group else 2,
        }

    def report_bad_case(self):
        from platform_utils import PLATFORM
        query = self.current_query_str.strip()
        if len(query) <= 1:
            ui.notify('搜索词太短，无法提交反馈。', type='warning', timeout=2000)
            return
        open_search_feedback_dialog(
            self,
            query=query,
            privacy_text=UI_TEXT['dialogs']['search_feedback_privacy'],
        )

    async def _submit_search_feedback(self, query: str, detail: str) -> None:
        from platform_utils import PLATFORM

        await telemetry.add_feedback(
            feedback_type='search_bad_case',
            query=query,
            search_settings=self._feedback_settings(),
            app_version=_get_git_commit(),
            platform=PLATFORM,
            details=detail,
        )

    def report_translation_error(self, e):
        raw_args = getattr(e, 'args', None)
        # print(f'[UI] translation_feedback event received: {raw_args!r}', flush=True)
        row = raw_args
        if isinstance(row, list) and row:
            row = row[0]
        if not isinstance(row, dict):
            print(f'[UI] translation_feedback invalid payload: {raw_args!r}', flush=True)
            ui.notify('无法读取当前词条信息。', type='warning', timeout=2000)
            return

        tag = str(row.get('tag') or '').strip()
        current_cn_name = str(row.get('cn_name') or '').strip()
        if not tag:
            ui.notify('无法读取当前词条。', type='warning', timeout=2000)
            return
        open_translation_feedback_dialog(
            self,
            row=row,
            privacy_text=UI_TEXT['dialogs']['translation_feedback_privacy'],
        )

    async def _submit_translation_feedback(
        self,
        row: dict,
        suggested: str,
        detail: str,
    ) -> None:
        from platform_utils import PLATFORM

        await telemetry.add_feedback(
            feedback_type='translation_error',
            query=self.current_query_str.strip(),
            search_settings=self._feedback_settings(),
            app_version=_get_git_commit(),
            platform=PLATFORM,
            details=detail,
            tag=str(row.get('tag') or '').strip(),
            current_cn_name=str(row.get('cn_name') or '').strip(),
            suggested_cn_name=suggested,
            category=str(row.get('category') or ''),
        )
