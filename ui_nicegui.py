"""
ui_nicegui.py
─────────────
NiceGUI 前端层（重构版）。

▸ 只负责渲染 / 交互。
▸ 调用 core.engine.DanbooruTagger，通过 core.models 的数据结构通信。
▸ 不包含任何算法逻辑。
▸ 平台相关配置（host/port/云端判断）统一由 platform_utils 提供。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
print("[UI] 脚本开始执行", flush=True)
import asyncio
import logging
import os
import re
import time
import json as _json
import subprocess
import traceback
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from fastapi.responses import PlainTextResponse

def _excepthook(exc_type, exc_value, exc_tb):
    print("[UI] FATAL ERROR ON STARTUP:", flush=True)
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _excepthook

from nicegui import ui, app, run
from core import counter, telemetry
from api_fastapi import app as api_app
from core.engine import DanbooruTagger
from core.models import RelatedTag, SearchRequest
from core.ui_text import load_ui_text
from core.prompt_import import (
    WORKSPACE_GROUP_ORDER,
    PromptImportResult,
    WorkspaceCanonicalizationResult,
    canonicalize_workspace_tags,
    classify_workspace_tag,
    pending_to_workspace_entry,
    resolve_prompt_text,
)
from core.workspace_insights import (
    CANDIDATE_UNSELECTED,
    COVERED,
    UNCOVERED,
    artist_candidate_reason,
    compute_concept_coverage,
    related_candidate_reason,
    selected_tag_reason,
    semantic_candidate_reason,
    tag_group_candidate_reason,
)
from core.workspace import (
    ARTIST_SELECTION_ORIGINS,
    FAVORITES_STORAGE_KEY,
    HISTORY_STORAGE_KEY,
    LEGACY_STAGED_STORAGE_KEY,
    WORKSPACE_STORAGE_KEY,
    WorkspaceDataError,
    add_history_entry,
    append_workspace_query,
    build_backup,
    clone_workspace,
    dump_collection,
    dump_workspace,
    empty_favorites,
    empty_history,
    favorite_from_workspace,
    merge_favorite_into_workspace,
    merge_favorites,
    merge_history,
    merge_workspaces,
    migrate_legacy_workspace,
    new_workspace,
    normalize_backup,
    normalize_favorites,
    normalize_history,
    normalize_workspace,
    replace_with_favorite,
    sync_selected_entries,
    utc_now_iso,
    workspace_signature,
)
from platform_utils import is_cloud, get_host_port, nsfw_allowed
from mcp_server import mcp


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

# ── 表格列定义 ─────────────────────────────────────────────────────────────────

TABLE_COLUMNS = [
    {'name': 'tag',         'label': '匹配标签', 'field': 'tag',         'align': 'left', 'sortable': True},
    {'name': 'cn_name',     'label': '含义',     'field': 'cn_name',     'align': 'left'},
    {'name': 'nsfw',        'label': '分级',     'field': 'nsfw',        'align': 'center', 'sortable': True},
    {'name': 'final_score', 'label': '综合分',   'field': 'final_score', 'sortable': True},
    {'name': 'count',       'label': '热度',     'field': 'count',       'sortable': True},
    {'name': 'reason',      'label': '推荐原因', 'field': 'reason',      'align': 'left'},
]

OPTIONAL_COLS = {
    'semantic': {'name': 'semantic_score', 'label': '语义分',   'field': 'semantic_score', 'sortable': True},
    'layer':    {'name': 'layer',          'label': '匹配层',   'field': 'layer'},
    'source':   {'name': 'source',         'label': '匹配来源', 'field': 'source'},
}

# localStorage key 与配置版本，版本变更时自动丢弃旧配置
_CONFIG_LS_KEY = 'danbooru_search_config'
_CONFIG_VERSION = 7
_ANNOUNCEMENT_VERSION = 'p0-workspace-2026-07'
_LOCAL_STORAGE_READ_CHUNK_CHARS = 200_000
_LOCAL_STORAGE_MAX_READ_CHARS = 4_000_000
_HISTORY_PRE_COMPACTION_BACKUP_KEY = f'{HISTORY_STORAGE_KEY}_pre_compaction_backup'
_LOCAL_STORAGE_RESTORE_CACHE = '__danbooruLocalStorageRestoreV2'
_LOCAL_STORAGE_RESTORE_RETRY_DELAYS = (0.0, 1.0, 3.0)
_LOCAL_STORAGE_NAMES = ('config', 'workspace', 'history', 'favorites', 'legacy')

SPONSOR_IMAGE_URL = "https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202501120027592.png"
SPONSOR_TOOLCHAIN_URL = "http://intro.sakizuki.site/index.html"
SPONSOR_NOTICE_TEXT = "喜欢的话，可以请作者喝杯咖啡"
SPONSOR_TITLE = "谢谢你愿意支持"
UI_TEXT = load_ui_text()


def _resolve_group_render_limit(default: int = 80) -> int:
    raw = os.environ.get('DANBOORU_GROUP_RENDER_LIMIT')
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


GROUP_RENDER_TAG_LIMIT = _resolve_group_render_limit()
ARTIST_REC_LIMIT = 64
ARTIST_REC_PAGE_SIZE = 8
RELATED_REC_PAGE_SIZE = 10
RECOMMENDATION_DEBOUNCE_SECONDS = 0.1
WORKSPACE_SAVE_DEBOUNCE_SECONDS = 0.3

# 搜索模式预设
_SEARCH_MODE_PRESETS: dict[str, dict] = {
    '精确查词': {'top_k': 20, 'limit': 10, 'popularity_weight': 0.15, 'use_segmentation': False, 'group_mode': 'off', 'max_per_group': 2},
    '概念扩展': {'top_k': 80, 'limit': 80, 'popularity_weight': 0.15, 'use_segmentation': True,  'group_mode': 'expand', 'max_per_group': 2},
    '描述查词': {'top_k': 20, 'limit': 20, 'popularity_weight': 0.15, 'use_segmentation': False, 'group_mode': 'off', 'max_per_group': 2},
    '完整场景': {'top_k': 5,  'limit': 80, 'popularity_weight': 0.15, 'use_segmentation': True,  'group_mode': 'diverse', 'max_per_group': 2},
}
_SEARCH_MODE_OPTIONS = ['自定义'] + list(_SEARCH_MODE_PRESETS.keys())
_ARTIST_ORIGINS = set(ARTIST_SELECTION_ORIGINS)


# ── 辅助函数 ───────────────────────────────────────────────────────────────────

_HISTORY_DISPLAY_TIMEZONE = timezone(timedelta(hours=8))


def _format_history_time(value: object) -> str:
    """将历史记录的 ISO 时间转成简短的北京时间。"""
    raw = str(value or '').strip()
    if not raw:
        return '--'
    try:
        parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_HISTORY_DISPLAY_TIMEZONE)
        return parsed.astimezone(_HISTORY_DISPLAY_TIMEZONE).strftime('%y-%m-%d %H:%M:%S')
    except ValueError:
        return raw


def _format_history_settings(settings: object) -> str:
    if not isinstance(settings, dict):
        settings = {}

    mode = settings.get('search_mode')
    preset = _SEARCH_MODE_PRESETS.get(mode)
    if preset and all(settings.get(key) == value for key, value in preset.items()):
        return f'预设：{mode}'

    top_k = settings.get('top_k', '--')
    limit = settings.get('limit', '--')
    segmentation = settings.get('use_segmentation')
    segmentation_text = '开启' if segmentation is True else '关闭' if segmentation is False else '--'
    return f'Top K：{top_k} · 数量上限：{limit} · 分词：{segmentation_text}'

def _sanitize_restored_config(cfg: dict) -> dict:
    """保留旧配置中的已知、安全字段；坏字段不得阻止页面启动。"""
    safe: dict = {}

    numeric_fields = {
        'top_k': (int, 1, 200),
        'limit': (int, 10, 500),
        'popularity_weight': (float, 0.0, 1.0),
        'max_per_group': (int, 1, 10),
    }
    for key, (caster, minimum, maximum) in numeric_fields.items():
        if key not in cfg:
            continue
        try:
            value = caster(cfg[key])
        except (TypeError, ValueError):
            continue
        if minimum <= value <= maximum:
            safe[key] = value

    for key in (
        'show_nsfw', 'use_segmentation', 'sw_semantic', 'sw_layer',
        'sw_source',
    ):
        if isinstance(cfg.get(key), bool):
            safe[key] = cfg[key]

    for key in ('selected_layers', 'selected_cats'):
        value = cfg.get(key)
        if isinstance(value, dict):
            safe[key] = {str(k): v for k, v in value.items() if isinstance(v, bool)}

    if cfg.get('prompt_format') in ('sdxl', 'nai', 'anima'):
        safe['prompt_format'] = cfg['prompt_format']
    if cfg.get('search_mode') in _SEARCH_MODE_OPTIONS:
        safe['search_mode'] = cfg['search_mode']
    if cfg.get('group_mode') in ('off', 'expand', 'diverse'):
        safe['group_mode'] = cfg['group_mode']
    if isinstance(cfg.get('rows_per_page'), int) and cfg['rows_per_page'] in {0, 5, 7, 10, 15, 20, 25, 50}:
        safe['rows_per_page'] = cfg['rows_per_page']
    if isinstance(cfg.get('search_query'), str):
        safe['search_query'] = cfg['search_query'][:4_000]
    if isinstance(cfg.get('dismissed_announcement_version'), str):
        safe['dismissed_announcement_version'] = cfg['dismissed_announcement_version'][:100]
    return safe

def _next_group_render_limit(current: int, total: int, page_size: int) -> int:
    if page_size <= 0:
        return total
    return min(total, max(page_size, current + page_size))


def _limit_group_render_tags(tags: list[dict], visible_limit: int | None = None) -> tuple[list[dict], int]:
    limit = GROUP_RENDER_TAG_LIMIT if visible_limit is None else visible_limit
    if limit <= 0:
        return tags, 0
    if len(tags) <= limit:
        return tags, 0
    return tags[:limit], len(tags) - limit


def _should_group_start_expanded(group_name: str, expanded_groups: set[str]) -> bool:
    return group_name in expanded_groups


def _group_names_key(group_data: list[dict]) -> tuple[str, ...]:
    return tuple(sorted({str(group.get('group', '')) for group in group_data}))


def _group_scroll_dom_id(group_name: str) -> str:
    safe_name = re.sub(r'[^0-9A-Za-z_-]+', '_', group_name)
    return f'group-scroll-{safe_name}'


def _scroll_state_restore_script(positions: dict[str, int]) -> str:
    js_positions = _json.dumps(positions)
    return f"""
        (() => {{
            const positions = {js_positions};
            const restore = () => {{
                const windowTop = positions.__window__;
                if (typeof windowTop === 'number') {{
                    window.scrollTo({{ top: windowTop, behavior: 'auto' }});
                    const root = document.scrollingElement || document.documentElement || document.body;
                    if (root) root.scrollTop = windowTop;
                }}
                for (const [id, top] of Object.entries(positions)) {{
                    if (id === '__window__') continue;
                    if (id.endsWith('__bottom__')) continue;
                    const el = document.getElementById(id);
                    if (!el) continue;
                    const bottom = positions[`${{id}}__bottom__`];
                    if (typeof bottom === 'number') {{
                        el.scrollTop = Math.max(0, el.scrollHeight - bottom);
                    }} else {{
                        el.scrollTop = top;
                    }}
                }}
            }};
            requestAnimationFrame(() => {{
                restore();
                requestAnimationFrame(restore);
            }});
            setTimeout(restore, 80);
        }})();
    """


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return os.environ.get('COMMIT_SHA', 'unknown')[:7]


def result_to_row(r, nsfw_visible: bool) -> dict:
    d = asdict(r)
    d['_nsfw_blocked'] = (r.nsfw == '1') and not nsfw_visible
    d['reason'] = semantic_candidate_reason(r.source, r.layer)
    return d


def apply_nsfw_filter(rows: list[dict], show_nsfw: bool) -> list[dict]:
    result = []
    for row in rows:
        r = dict(row)
        r['_nsfw_blocked'] = (r.get('nsfw') == '1') and not show_nsfw
        result.append(r)
    return result


def _format_tag_with_weight(tag: str, weight: float, fmt: str = 'sdxl') -> str:
    """格式化单个标签。
    sdxl:  (tag:1.2)  权重 1.0 时输出 tag
    nai:   1.2::tag:: 权重 1.0 时输出 tag
    anima: (tag:1.5)  权重 1.0 时输出 tag，下划线替换为空格
    所有模式均对标签名中的括号进行反斜杠转义。
    """
    tag = tag.replace('(', '\\(').replace(')', '\\)')
    if fmt == 'anima':
        tag = tag.replace('_', ' ')
    if weight == 1.0:
        return tag
    if fmt == 'nai':
        return f'{weight:.1f}::{tag}::'
    return f'({tag}:{weight:.1f})'


def _format_selected_tag_label(tag: str, cn_name: str = '') -> str:
    cn_first = (cn_name or '').split(',', 1)[0].strip()
    return f'{tag} | {cn_first}' if cn_first else tag


# ── UI 类 ─────────────────────────────────────────────────────────────────────

class DanbooruSearchUI:
    def __init__(self):
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

        self.service_status_container.clear()
        with self.service_status_container:
            if not ready:
                with ui.row().classes(
                    'w-full items-center gap-2 service-state-panel loading'
                ):
                    ui.spinner(size='18px', color='primary')
                    ui.label('引擎初始化中，请稍候…约需 5~10 分钟').classes('font-medium')
            else:
                with ui.row().classes(
                    f'w-full items-center gap-2 service-state-panel {"busy" if busy else "ready"}'
                ):
                    ui.icon(
                        'schedule' if busy else 'check_circle',
                        size='18px',
                        color='warning' if busy else 'positive',
                    )
                    parts = [
                        '服务繁忙' if busy else '服务可用',
                        f'{online_sessions} 个在线页面',
                    ]
                    if active > 0:
                        parts.append(f'正在处理 {active} 个任务')
                    if waiting > 0:
                        parts.append(f'等待 {waiting} 个')
                    ui.label(' · '.join(parts)).classes('font-medium')

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
        with ui.dialog() as self.sponsor_dialog, ui.card().classes('w-full max-w-sm'):
            with ui.column().classes('w-full items-center gap-2 text-center'):
                ui.label(SPONSOR_TITLE).classes('text-base font-bold text-gray-800')
                ui.label(UI_TEXT['sponsor']['body']).classes('text-sm text-gray-600 leading-relaxed')
                ui.image(SPONSOR_IMAGE_URL).classes('w-60 max-w-full rounded border border-gray-200')
                ui.label('微信赞赏码').classes('text-xs text-gray-400')
                ui.link(
                    UI_TEXT['sponsor']['toolchain_prompt'],
                    SPONSOR_TOOLCHAIN_URL,
                    new_tab=True,
                ).classes('text-xs text-blue-500 hover:text-blue-700 hover:underline')
            with ui.row().classes('w-full justify-end'):
                ui.button('关闭', on_click=self.sponsor_dialog.close).props('flat color=grey-7')

    def _build_help_dialog(self):
        from platform_utils import PLATFORM

        alternate_url = (
            'https://www.modelscope.cn/studios/SAkizuki/DanbooruSearchOnline'
            if PLATFORM == 'hf' else
            'https://huggingface.co/spaces/SAkizuki/DanbooruSearch'
        )
        with ui.dialog() as self.help_dialog, ui.card().classes('w-full max-w-3xl max-h-[90vh] p-0 gap-0'):
            with ui.row().classes('w-full items-center justify-between px-5 py-4 border-b border-slate-200'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('help_outline', color='primary')
                    ui.label('帮助 / 关于').classes('text-lg font-bold text-slate-800')
                ui.button(icon='close', on_click=self.help_dialog.close).props('flat dense round color=grey-7')

            with ui.scroll_area().classes('w-full h-[72vh]'):
                with ui.column().classes('w-full gap-5 px-5 py-4'):
                    with ui.column().classes('help-section'):
                        with ui.element('div').classes('help-section-heading'):
                            ui.label(UI_TEXT['help']['update_title']).classes(
                                'help-section-heading-title'
                            )
                            ui.label(UI_TEXT['help']['update_summary']).classes(
                                'help-section-heading-subtitle'
                            )
                        ui.markdown(UI_TEXT['help']['guide_markdown']).classes('help-content')

                    with ui.column().classes('help-section'):
                        with ui.element('div').classes(
                            'help-section-heading help-section-heading--documentation'
                        ):
                            ui.label(UI_TEXT['documentation']['title']).classes(
                                'help-section-heading-title'
                            )
                            ui.label(UI_TEXT['documentation']['subtitle']).classes(
                                'help-section-heading-subtitle'
                            )
                        with ui.column().classes('help-content gap-3'):
                            with ui.row().classes('w-full gap-3 flex-wrap'):
                                for link in UI_TEXT['documentation']['links']:
                                    link_url = (
                                        alternate_url
                                        if link['url'] == '{alternate_url}'
                                        else link['url']
                                    )
                                    ui.link(link['label'], link_url, new_tab=True).classes('help-link')
                            ui.markdown(
                                UI_TEXT['documentation']['copyright_markdown']
                            ).classes('help-content')

                    with ui.column().classes('help-section'):
                        with ui.element('div').classes(
                            'help-section-heading help-section-heading--notice'
                        ):
                            ui.label(UI_TEXT['notice']['title']).classes(
                                'help-section-heading-title'
                            )
                            ui.label(UI_TEXT['notice']['subtitle']).classes(
                                'help-section-heading-subtitle'
                            )
                        ui.markdown(UI_TEXT['notice']['body_markdown']).classes('help-content')

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
                            on_click=self._confirm_delete_all_personal_data,
                        ).props('outline color=negative no-caps').classes('mt-3')

                    with ui.row().classes('w-full items-center justify-between gap-3 flex-wrap'):
                        ui.label('DanbooruSearch 将持续免费开放。').classes('text-xs text-slate-500')
                        ui.button(
                            SPONSOR_NOTICE_TEXT,
                            icon='volunteer_activism',
                            on_click=self.sponsor_dialog.open,
                        ).props('flat dense no-caps color=grey-7').classes('text-xs')

    def _confirm_delete_all_personal_data(self):
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

            async def delete_all_personal_data():
                delete_btn.disable()
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
                keys_json = _json.dumps(storage_keys, ensure_ascii=False)
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
                    self.help_dialog.close()
                    ui.notify('本地个人数据已删除，页面即将刷新', type='positive', timeout=2000)
                    await ui.run_javascript(
                        "(() => { setTimeout(() => window.location.reload(), 500); return true; })()",
                        timeout=5.0,
                    )
                except Exception as exc:
                    print(f'[UI] 删除本地个人数据失败: {exc}', flush=True)
                    delete_btn.enable()
                    ui.notify('删除失败，请稍后重试', type='negative')

            with ui.row().classes('w-full justify-end gap-2 mt-2'):
                ui.button('取消', on_click=confirm.close).props('flat color=grey-7')
                delete_btn = ui.button(
                    '确认删除',
                    icon='delete_forever',
                    on_click=delete_all_personal_data,
                ).props('unelevated color=negative no-caps')
        confirm.open()

    def _build_release_announcement(self):
        self.announcement_banner = ui.element('div').classes(
            'w-full release-notice section-surface px-3 py-2'
        )
        with self.announcement_banner:
            with ui.row().classes('w-full items-center justify-between gap-2'):
                with ui.row().classes('items-center gap-2 min-w-0 flex-wrap'):
                    ui.icon('new_releases', size='18px', color='primary')
                    ui.label(
                        '新版已加入标签工作区、Prompt 导入、Alias 纠错和分渠道统计。'
                    ).classes('text-sm text-slate-700')
                    ui.button(
                        '查看详情', on_click=self.help_dialog.open,
                    ).props('flat dense no-caps color=primary').classes('text-xs')
                ui.button(
                    icon='close', on_click=self._dismiss_release_announcement,
                ).props('flat dense round color=grey-6')

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
        return {
            'version': _CONFIG_VERSION,
            'top_k': int(self.input_top_k.value) if self.input_top_k else 10,
            'limit': int(self.input_limit.value) if self.input_limit else 80,
            'popularity_weight': float(self.input_weight.value) if self.input_weight else 0.15,
            'show_nsfw': bool(self.input_nsfw.value) if self.input_nsfw else False,
            'use_segmentation': bool(self.input_segment.value) if self.input_segment else True,
            'selected_layers': dict(self.selected_layers),
            'selected_cats': dict(self.selected_cats),
            'sw_semantic': bool(self.sw_semantic.value) if self.sw_semantic else False,
            'sw_layer': bool(self.sw_layer.value) if self.sw_layer else False,
            'sw_source': bool(self.sw_source.value) if self.sw_source else False,
            'prompt_format': self.prompt_format,
            'rows_per_page': self._get_rows_per_page(),
            'search_query': self.search_input.value if self.search_input else '',
            'dismissed_announcement_version': self.dismissed_announcement_version,
            'search_mode': self.input_search_mode.value if self.input_search_mode else '自定义',
            'group_mode': self.input_group_mode.value if self.input_group_mode else 'off',
            'max_per_group': int(self.input_max_per_group.value) if self.input_max_per_group else 2,
        }

    def _storage_write_allowed(self, name: str) -> bool:
        """Block writes until that localStorage domain has been safely restored."""
        if name in self._storage_applying:
            return False
        ready = self._storage_states.get(name) == 'ready'
        if not ready or not self._client_connected():
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
        cfg = _sanitize_restored_config(cfg if isinstance(cfg, dict) else {})

        dismissed_version = cfg.get('dismissed_announcement_version', '')
        self.dismissed_announcement_version = dismissed_version
        if self.announcement_banner:
            self.announcement_banner.set_visibility(
                dismissed_version != _ANNOUNCEMENT_VERSION
            )

        # 模式可能触发预设填充，因此随后再覆盖各个具体参数。
        if self.input_search_mode and 'search_mode' in cfg:
            self.input_search_mode.set_value(cfg['search_mode'])
        if self.input_top_k and 'top_k' in cfg:
            self.input_top_k.set_value(cfg['top_k'])
        if self.input_limit and 'limit' in cfg:
            self.input_limit.set_value(cfg['limit'])
        if self.input_weight and 'popularity_weight' in cfg:
            self.input_weight.set_value(cfg['popularity_weight'])
        if self.input_segment and 'use_segmentation' in cfg:
            self.input_segment.set_value(cfg['use_segmentation'])
        if self.input_group_mode and 'group_mode' in cfg:
            self.input_group_mode.set_value(cfg['group_mode'])
        if self.input_max_per_group and 'max_per_group' in cfg:
            self.input_max_per_group.set_value(cfg['max_per_group'])
        if nsfw_allowed() and self.input_nsfw and 'show_nsfw' in cfg:
            self.input_nsfw.set_value(cfg['show_nsfw'])

        for layer, val in cfg.get('selected_layers', {}).items():
            if layer in self.selected_layers:
                self.selected_layers[layer] = bool(val)
                if layer in self._layer_checkboxes:
                    self._layer_checkboxes[layer].set_value(bool(val))
        for cat, val in cfg.get('selected_cats', {}).items():
            if cat in self.selected_cats:
                self.selected_cats[cat] = bool(val)
                if cat in self._cat_checkboxes:
                    self._cat_checkboxes[cat].set_value(bool(val))

        if self.sw_semantic and 'sw_semantic' in cfg:
            self.sw_semantic.set_value(cfg['sw_semantic'])
        if self.sw_layer and 'sw_layer' in cfg:
            self.sw_layer.set_value(cfg['sw_layer'])
        if self.sw_source and 'sw_source' in cfg:
            self.sw_source.set_value(cfg['sw_source'])
        if 'prompt_format' in cfg:
            self._apply_prompt_format(cfg['prompt_format'])
        if 'rows_per_page' in cfg:
            self._set_rows_per_page(cfg['rows_per_page'])
        if self.search_input and cfg.get('search_query'):
            self.search_input.set_value(cfg['search_query'])
        self._update_table_columns()

    # ══════════════════════════════════════════════════════════════════════
    # 页面构建
    # ══════════════════════════════════════════════════════════════════════

    def build_page(self):
        self.client = ui.context.client
        ui.colors(primary='#4A90E2', secondary='#5E6C84', accent='#FF6B6B')
        ui.add_head_html('''
            <meta name="description" content="基于语义匹配的 Danbooru 标签搜索引擎，支持中英双语描述、多维匹配、智能分词与共现关联推荐。">
            <meta name="keywords" content="Danbooru, AI绘画, Stable Diffusion, 提示词, 标签搜索, RAG, Prompt, NovelAI">
            <meta name="google-site-verification" content="cx4sl9Mb172GUFL556JFwKCP-pT3naQcmlMriy5B8ls" />

            <style>
                .nsfw-blur-cell      { filter: blur(8px); opacity: 0.5; transition: all 0.3s ease;
                                       pointer-events: none !important; user-select: none !important; }
                .nsfw-checkbox-disabled { pointer-events: none !important; opacity: 0.3 !important; }
                .nsfw-row-blocked    { cursor: not-allowed !important; }
                .result-table-flat {
                    border-radius: 4px;
                    box-shadow: none !important;
                }
                .result-table-flat .q-table th,
                .result-table-flat .q-table td {
                    border-color: var(--cell-border);
                }
                .recommendation-grid {
                    overflow: hidden;
                    background: #ffffff;
                    border-top: 1px solid var(--cell-border);
                    border-bottom: 1px solid var(--cell-border);
                }
                .recommendation-row + .recommendation-row {
                    border-top: 1px solid var(--cell-border);
                }
                .recommendation-cell {
                    align-self: stretch;
                    display: flex;
                    align-items: center;
                }
                .related-item { transition: background-color 0.15s ease; }
                .related-item:hover { background-color: rgba(74, 144, 226, 0.04); }
                .tag-link { text-decoration: none; font-family: 'Consolas', 'Monaco', 'Courier New', monospace; }
                .tag-link:hover { text-decoration: underline; }
                .weight-chip { display: inline-flex; align-items: center; gap: 2px;
                               border-radius: 16px; padding: 2px 6px 2px 4px;
                               background: #e3edf7; border: 1px solid #b3cde8;
                               font-size: 12px; margin: 3px; white-space: nowrap; }
                .weight-chip.boosted  { background: #fff3e0; border-color: #ffb74d; }
                .weight-chip.reduced  { background: #f3e5f5; border-color: #ce93d8; }
                .weight-btn { cursor: pointer; width: 18px; height: 18px; border-radius: 50%;
                              display: inline-flex; align-items: center; justify-content: center;
                              font-size: 13px; font-weight: bold; line-height: 1;
                              border: none; background: rgba(0,0,0,0.08);
                              color: #555; transition: background 0.15s; padding: 0; }
                .weight-btn:hover { background: rgba(0,0,0,0.18); }
                .weight-label { font-family: Consolas, Monaco, monospace; font-size: 11px;
                                color: #888; min-width: 28px; text-align: center; }

                :root {
                    --section-surface: #f8fafc;
                    --section-border: #dbe4ee;
                    --cell-border: #e2e8f0;
                    --section-heading: #334155;
                    --section-radius: 8px;
                }
                .product-search-card {
                    background: #ffffff;
                    border: 1px solid var(--section-border);
                    border-top: 3px solid #4a90e2;
                    border-radius: var(--section-radius);
                    box-shadow: none;
                }
                .section-surface {
                    box-sizing: border-box;
                    background: var(--section-surface);
                    border: 1px solid var(--section-border);
                    border-radius: var(--section-radius);
                    box-shadow: none !important;
                }
                .section-heading {
                    color: var(--section-heading);
                    font-size: 14px;
                    font-weight: 700;
                    line-height: 1.4;
                }
                .service-state-panel {
                    border-radius: 8px;
                    padding: 7px 10px;
                    font-size: 12px;
                }
                .service-state-panel.loading {
                    background: #eff6ff;
                    border: 1px solid #bfdbfe;
                    color: #1d4ed8;
                }
                .service-state-panel.ready {
                    background: #ecfdf5;
                    border: 1px solid #a7f3d0;
                    color: #047857;
                }
                .service-state-panel.busy {
                    background: #fff7ed;
                    border: 1px solid #fed7aa;
                    color: #c2410c;
                }
                .query-insight-panel {
                    padding: 10px 12px;
                }
                .homepage-support-note {
                    color: #64748b;
                    font-size: 12px;
                    line-height: 1.6;
                }
                .homepage-support-note a {
                    color: #4a90e2;
                    font-weight: 500;
                    text-decoration: none;
                }
                .homepage-support-note a:hover {
                    color: #2563a8;
                    text-decoration: underline;
                }
                .help-section {
                    width: 100%;
                    gap: 12px;
                    background: #ffffff;
                }
                .help-section-heading {
                    display: flex;
                    flex-direction: column;
                    width: 100%;
                    gap: 5px;
                    padding: 12px 16px;
                    border: 1px solid #b9d7f7;
                    border-radius: 8px;
                    background: #eef6ff;
                }
                .help-section-heading-title {
                    color: #174f91;
                    font-size: 15px;
                    font-weight: 700;
                    line-height: 1.4;
                }
                .help-section-heading-subtitle {
                    color: #2563a8;
                    font-size: 13px;
                    font-weight: 400;
                    line-height: 1.55;
                }
                .help-section-heading--documentation {
                    border-color: #a7e3c4;
                    background: #ecfdf5;
                }
                .help-section-heading--documentation .help-section-heading-title {
                    color: #047857;
                }
                .help-section-heading--documentation .help-section-heading-subtitle {
                    color: #16855f;
                }
                .help-section-heading--notice {
                    border-color: #fed7aa;
                    background: #fff7ed;
                }
                .help-section-heading--notice .help-section-heading-title {
                    color: #c2410c;
                }
                .help-section-heading--notice .help-section-heading-subtitle {
                    color: #b45309;
                }
                .help-content {
                    width: 100%;
                    color: #334155;
                    font-size: 14px;
                    line-height: 1.75;
                }
                .help-section > .help-content {
                    box-sizing: border-box;
                    padding-right: 16px;
                    padding-left: 16px;
                }
                .help-content h3,
                .help-content h4 {
                    color: #334155;
                }
                .help-link {
                    color: #2563a8;
                    font-size: 14px;
                    line-height: 1.6;
                }
                .help-link:hover {
                    color: #174f86;
                    text-decoration: underline;
                }

                /* 强制双栏并排 */
                .two-col-layout {
                    display: flex !important;
                    flex-wrap: nowrap !important;
                    align-items: flex-start !important;
                    gap: 16px !important;
                }
                .two-col-layout > .col-left {
                    flex: 0 0 62% !important;
                    min-width: 0 !important;
                    max-width: 62% !important;
                    overflow: hidden;
                }
                .two-col-layout > .col-right {
                    flex: 0 0 36% !important;
                    min-width: 0 !important;
                    max-width: 36% !important;
                    overflow: hidden;
                }

                /* 窄屏回退为上下排列 */
                @media (max-width: 900px) {
                    .two-col-layout {
                        flex-wrap: wrap !important;
                    }
                    .two-col-layout > .col-left,
                    .two-col-layout > .col-right {
                        flex: 1 1 100% !important;
                        max-width: 100% !important;
                    }
                }

            </style>
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-QPB7EEPR5G"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag(){dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', 'G-QPB7EEPR5G');
            </script>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    function openExternal(root) {
                        root.querySelectorAll('a[href^="http"]').forEach(function(a) {
                            a.setAttribute('target', '_blank');
                            a.setAttribute('rel', 'noopener noreferrer');
                        });
                    }
                    openExternal(document);
                    new MutationObserver(function(mutations) {
                        mutations.forEach(function(m) {
                            m.addedNodes.forEach(function(node) {
                                if (node.querySelectorAll) openExternal(node);
                            });
                        });
                    }).observe(document.body, { childList: true, subtree: true });
                });
            </script>
        ''')

        self._build_sponsor_dialog()
        self._build_help_dialog()

        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-3'):

            # ── 1. 搜索主路径 ──
            self._build_search_card()
            if not DanbooruTagger.is_ready():
                asyncio.ensure_future(self._hide_banner_when_ready())

            # ── 2. 紧凑版本公告 ──
            self._build_release_announcement()

            # ── 3. 工作区工具和已选标签（无需搜索即可恢复）──
            self.workspace_card = ui.card().classes(
                'w-full p-0 gap-0 overflow-hidden section-surface'
            )
            with self.workspace_card:
                self._build_workspace_toolbar()
                self._build_selection_bar()

            # ── 4~5. 搜索结果区域（搜索前隐藏）──
            self.results_section = ui.column().classes('w-full gap-4')
            self.results_section.set_visibility(False)

            with self.results_section:
                # ── 4. 查询理解：来源筛选 + 概念覆盖 ──
                self.coverage_container = ui.column().classes('w-full gap-2')

                # ── 5. 两栏结果 ──
                self._build_results_columns()

            # ── 6. 页脚 ──
            with ui.element('div').classes('w-full text-center py-4 mt-2'):
                self.search_count_label = ui.html('正在加载数据...').classes('text-xs text-gray-400')
                self._update_footer_text()
                ui.button(SPONSOR_NOTICE_TEXT, on_click=self.sponsor_dialog.open) \
                    .props('flat dense no-caps color=grey-6') \
                    .classes('text-xs mt-1')

    # ── 搜索卡片 ─────────────────────────────────────────────────────────

    def _build_search_card(self):
        with ui.card().classes('w-full product-search-card'):
            with ui.row().classes('w-full items-start justify-between gap-3 mb-1'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('search', size='2em', color='primary')
                    ui.label('Danbooru 标签模糊搜索').classes('text-2xl font-bold text-gray-800')
                ui.button(
                    '帮助 / 关于', icon='help_outline', on_click=self.help_dialog.open,
                ).props('flat dense no-caps color=grey-7').classes('text-sm')
            ui.label(
                '基于语义匹配的标签搜索引擎，支持多维匹配与共现关联推荐。'
            ).classes('text-sm text-gray-500 -mt-1 mb-1')
            with ui.row().classes(
                'w-full items-center justify-between gap-x-4 gap-y-1 flex-wrap mb-3'
            ):
                ui.link(
                    '查看工具链介绍 / 使用指南 →',
                    'http://intro.sakizuki.site/index.html',
                    new_tab=True,
                ).classes('text-sm text-blue-600 hover:text-blue-800 font-medium')
                ui.html(
                    '觉得好用？给 '
                    '<a href="https://huggingface.co/spaces/SAkizuki/DanbooruSearch">'
                    'Space 点个 Like ❤️</a>，或到 '
                    '<a href="https://github.com/SuzumiyaAkizuki/DanbooruSearchOnline">'
                    'GitHub 点 Star ⭐</a>'
                ).classes('homepage-support-note')

            with ui.row().classes('w-full gap-3 items-stretch'):
                self.search_input = ui.textarea(
                    placeholder='输入自然语言描述或模糊概念，例如：一个穿着白色水手服的少女在雨中奔跑...'
                ).classes('flex-grow text-base').props('outlined rows=2')
                self.search_input.on('keydown.ctrl.enter', self.perform_search)

                with ui.column().classes('justify-center'):
                    self.search_btn = ui.button(
                        '', on_click=self.perform_search, icon='search'
                    ).classes('px-6 h-full min-h-16').props('unelevated color=dark')
                    with self.search_btn:
                        ui.label('搜索').classes('text-sm mt-1')
                    self.spinner = ui.spinner(size='2em').classes('hidden')

            self.search_params_row = ui.row().classes('w-full gap-6 items-center mt-3 flex-wrap')
            with self.search_params_row:
                with ui.row().classes('items-center gap-2'):
                    ui.label('搜索模式 (beta)').classes('text-sm text-gray-600')
                    self.input_search_mode = ui.select(
                        _SEARCH_MODE_OPTIONS, value='自定义',
                    ).classes('w-28').props('outlined dense')
                    self.input_search_mode.on('update:model-value', self._on_search_mode_change)
                    with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                        ui.label('选择模式自动填充对应参数；手动修改参数后自动变为「自定义」').style('font-size:14px;')

                with ui.row().classes('items-center gap-2'):
                    ui.label('Top K (语义相关)').classes('text-sm text-gray-600')
                    self.input_top_k = ui.number(value=10, min=1, max=200).classes('w-20') \
                        .props('outlined dense')
                    self.input_top_k.on('update:model-value', self._on_param_changed)

                with ui.row().classes('items-center gap-2'):
                    ui.label('结果上限').classes('text-sm text-gray-600')
                    self.input_limit = ui.number(value=80, min=10, max=500).classes('w-20') \
                        .props('outlined dense')
                    self.input_limit.on('update:model-value', self._on_param_changed)

                with ui.switch('显示 NSFW(成人) 内容', value=False).props('color=red') as _nsfw_sw:
                    if not nsfw_allowed():
                        with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                            ui.label('NSFW 内容在当前平台不可用').style('font-size:14px;')
                self.input_nsfw = _nsfw_sw
                if not nsfw_allowed():
                    self.input_nsfw.disable()
                else:
                    self.input_nsfw.on('update:model-value', self.on_nsfw_toggle)

                with ui.switch('智能分词', value=True).props('color=primary') as _seg_sw:
                    with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                        ui.label('关闭后系统将只匹配完整句子，适用于精准搜索整句。').style('font-size:14px;')
                self.input_segment = _seg_sw
                self.input_segment.on('update:model-value', self._on_param_changed)

            self.advanced_options = ui.expansion('高级选项', icon='tune').classes('w-full mt-2')
            with self.advanced_options:
                with ui.column().classes('w-full p-3 gap-4'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label('热度权重').classes('text-sm font-bold text-gray-700')
                        self.input_weight = ui.slider(
                            min=0.0, max=1.0, value=0.15, step=0.05,
                        ).classes('w-40')
                        ui.label().bind_text_from(
                            self.input_weight, 'value', lambda v: f"{v:.2f}",
                        ).classes('text-sm font-mono text-gray-700 w-8')
                        self.input_weight.on('update:model-value', self._on_param_changed)

                    with ui.row().classes('w-full gap-8 flex-wrap'):
                        with ui.column().classes('gap-2'):
                            ui.label('匹配层筛选').classes('font-bold text-sm text-gray-700')
                            display_map = {
                                '英文': '英文标签', '中文扩展词': '中文扩展词',
                                '释义': '维基释义', '中文核心词': '中文核心词',
                                'artist': 'artist',
                            }
                            for layer in ['英文', '中文扩展词', '释义', '中文核心词', 'artist']:
                                cb = ui.checkbox(
                                    display_map.get(layer, layer), value=True,
                                    on_change=lambda e, l=layer: self.selected_layers.__setitem__(l, e.value)
                                ).props('color=primary dense')
                                self._layer_checkboxes[layer] = cb

                        with ui.column().classes('gap-2'):
                            ui.label('类型筛选').classes('font-bold text-sm text-gray-700')
                            color_map = {'General': 'blue', 'Copyright': 'purple', 'Character': 'green'}
                            label_map = {
                                'General': '通用 (General)',
                                'Copyright': '作品 (Copyright)',
                                'Character': '角色 (Character)',
                            }
                            for cat in ['General', 'Copyright', 'Character']:
                                cb = ui.checkbox(
                                    label_map[cat], value=True,
                                    on_change=lambda e, c=cat: self.selected_cats.__setitem__(c, e.value)
                                ).props(f'color={color_map[cat]} dense')
                                self._cat_checkboxes[cat] = cb

                        with ui.column().classes('gap-2'):
                            ui.label('表格显示列').classes('font-bold text-sm text-gray-700')
                            self.sw_semantic = ui.switch('显示语义分', value=False)
                            self.sw_layer    = ui.switch('显示匹配层', value=False)
                            self.sw_source   = ui.switch('显示匹配来源', value=False)
                            self.sw_semantic.on('update:model-value', self._update_table_columns)
                            self.sw_layer.on('update:model-value', self._update_table_columns)
                            self.sw_source.on('update:model-value', self._update_table_columns)

                        with ui.column().classes('gap-2'):
                            ui.label('标签分组模式').classes('font-bold text-sm text-gray-700')
                            self.input_group_mode = ui.select(
                                ['off', 'expand', 'diverse'], value='off',
                            ).classes('w-40').props('outlined dense')
                            with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                                ui.label('off=关闭 | expand=同类召回增强 | diverse=多样性约束').style('font-size:14px;')
                            self.input_group_mode.on('update:model-value', self._on_param_changed)

                            self.input_max_per_group = ui.number(
                                value=2, min=1, max=10,
                            ).classes('w-20').props('outlined dense')
                            ui.label('每组最大标签数（diverse 模式）').classes('text-xs text-gray-500')
                            self.input_max_per_group.on('update:model-value', self._on_param_changed)

            with ui.element('div').classes('w-full border-t border-slate-100 pt-3 mt-1'):
                self.service_status_container = ui.column().classes('w-full gap-0')
                self._update_service_status()
                self._start_service_status_task()

    # ── 工作区工具 ────────────────────────────────────────────────────────

    def _build_workspace_toolbar(self):
        with ui.element('div').classes(
            'w-full bg-slate-50 border-b border-slate-200 px-4 py-2'
        ):
            with ui.row().classes('w-full items-center gap-2 flex-wrap'):
                ui.icon('workspaces', color='primary')
                ui.label('标签工作区').classes('section-heading mr-2')
                self.undo_btn = ui.button(
                    '撤销', icon='undo', on_click=self._undo_workspace,
                ).props('dense flat color=grey-7')
                self.redo_btn = ui.button(
                    '恢复', icon='redo', on_click=self._redo_workspace,
                ).props('dense flat color=grey-7')
                self.undo_btn.disable()
                self.redo_btn.disable()
                ui.separator().props('vertical').classes('h-7 mx-1')
                ui.button(
                    '历史', icon='history', on_click=self._open_history_dialog,
                ).props('dense flat color=primary')
                self.history_count_label = ui.label('0').classes('text-xs text-gray-500 -ml-2')
                ui.button(
                    '收藏', icon='star_outline', on_click=self._open_favorites_dialog,
                ).props('dense flat color=amber-8')
                self.favorites_count_label = ui.label('0').classes('text-xs text-gray-500 -ml-2')
                ui.button(
                    '保存收藏', icon='bookmark_add', on_click=self._open_save_favorite_dialog,
                ).props('dense flat color=teal-7')
                ui.button(
                    '导入 Prompt', icon='playlist_add', on_click=self._open_prompt_import_dialog,
                ).props('dense flat color=purple-7')
                ui.button(
                    '备份 / 迁移', icon='swap_horiz', on_click=self._open_backup_dialog,
                ).props('dense flat color=grey-7')
        self._update_workspace_counts()

    def _update_workspace_counts(self):
        if self.history_count_label is not None:
            self.history_count_label.text = str(len(self.search_history.get('items', [])))
        if self.favorites_count_label is not None:
            self.favorites_count_label.text = str(len(self.favorites.get('items', [])))

    def _current_search_settings(self) -> dict:
        return {
            'search_mode': self.input_search_mode.value if self.input_search_mode else '自定义',
            'top_k': int(self.input_top_k.value) if self.input_top_k else 10,
            'limit': int(self.input_limit.value) if self.input_limit else 80,
            'popularity_weight': float(self.input_weight.value) if self.input_weight else 0.15,
            'show_nsfw': bool(self.input_nsfw.value) if self.input_nsfw else False,
            'use_segmentation': bool(self.input_segment.value) if self.input_segment else True,
            'target_layers': [k for k, v in self.selected_layers.items() if v],
            'target_categories': [k for k, v in self.selected_cats.items() if v],
            'group_mode': self.input_group_mode.value if self.input_group_mode else 'off',
            'max_per_group': int(self.input_max_per_group.value) if self.input_max_per_group else 2,
        }

    def _apply_search_settings(self, settings: dict):
        if not isinstance(settings, dict):
            return
        self._applying_preset = True
        try:
            mode = settings.get('search_mode')
            if self.input_search_mode and mode in _SEARCH_MODE_OPTIONS:
                self.input_search_mode.set_value(mode)
            if self.input_top_k and isinstance(settings.get('top_k'), int):
                self.input_top_k.set_value(settings['top_k'])
            if self.input_limit and isinstance(settings.get('limit'), int):
                self.input_limit.set_value(settings['limit'])
            if self.input_weight and isinstance(settings.get('popularity_weight'), (int, float)):
                self.input_weight.set_value(settings['popularity_weight'])
            if self.input_segment and isinstance(settings.get('use_segmentation'), bool):
                self.input_segment.set_value(settings['use_segmentation'])
            if self.input_group_mode and settings.get('group_mode') in ('off', 'expand', 'diverse'):
                self.input_group_mode.set_value(settings['group_mode'])
            if self.input_max_per_group and isinstance(settings.get('max_per_group'), int):
                self.input_max_per_group.set_value(settings['max_per_group'])
            if nsfw_allowed() and self.input_nsfw and isinstance(settings.get('show_nsfw'), bool):
                self.input_nsfw.set_value(settings['show_nsfw'])

            layers = settings.get('target_layers')
            if isinstance(layers, list):
                selected = set(layers)
                for layer in self.selected_layers:
                    value = layer in selected
                    self.selected_layers[layer] = value
                    if layer in self._layer_checkboxes:
                        self._layer_checkboxes[layer].set_value(value)
            categories = settings.get('target_categories')
            if isinstance(categories, list):
                selected = set(categories)
                for category in self.selected_cats:
                    value = category in selected
                    self.selected_cats[category] = value
                    if category in self._cat_checkboxes:
                        self._cat_checkboxes[category].set_value(value)
        finally:
            self._applying_preset = False
        self._save_config()

    def _record_search_history(self, query: str):
        settings = self._current_search_settings()
        self.workspace_state = append_workspace_query(
            self.workspace_state,
            query,
            settings,
        )
        self._save_staged_tags()
        self.search_history = add_history_entry(
            self.search_history,
            query,
            settings,
            self.workspace_state,
        )
        self._save_history()
        self._update_workspace_counts()

    def _open_history_dialog(self):
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl max-h-[85vh]'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('搜索历史').classes('text-lg font-bold')
                with ui.row().classes('gap-2'):
                    if self.search_history.get('items'):
                        ui.button(
                            '清空全部', icon='delete_sweep',
                            on_click=lambda: self._confirm_clear_history(dialog),
                        ).props('flat dense color=red-7')
                    ui.button(icon='close', on_click=dialog.close).props('flat round dense')

            with ui.scroll_area().classes('w-full h-[65vh]'):
                items = self.search_history.get('items', [])
                if not items:
                    ui.label('暂无搜索历史').classes('text-sm text-gray-400 p-6')
                for item in items:
                    with ui.card().classes('w-full mb-2 p-3 border border-gray-200 shadow-none'):
                        with ui.row().classes('w-full items-start justify-between gap-3'):
                            with ui.column().classes('gap-1 flex-grow min-w-0'):
                                ui.label(item['query']).classes('font-medium text-gray-800 break-all')
                                selected_count = len(item['workspace'].get('selected', []))
                                ui.label(
                                    f"{_format_history_time(item.get('searched_at'))} · "
                                    f"工作区内有 {selected_count} 个标签"
                                ).classes('text-xs text-gray-400')
                                ui.label(
                                    _format_history_settings(item.get('settings'))
                                ).classes('text-xs text-gray-400')
                            with ui.row().classes('gap-1 flex-wrap justify-end'):
                                ui.button(
                                    '重新搜索', icon='search',
                                    on_click=lambda i=item, d=dialog: self._history_research(i, d),
                                ).props('flat dense color=primary')
                                ui.button(
                                    '恢复工作区', icon='restore',
                                    on_click=lambda i=item, d=dialog: self._history_restore(i, d),
                                ).props('flat dense color=teal-7')
                                ui.button(
                                    '追加查询', icon='playlist_add',
                                    on_click=lambda i=item, d=dialog: self._history_append(i, d),
                                ).props('flat dense color=purple-7')
                                ui.button(
                                    icon='delete_outline',
                                    on_click=lambda i=item, d=dialog: self._delete_history_entry(i, d),
                                ).props('flat round dense color=red-6')
        dialog.open()

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
        history_id = item.get('history_id')
        self.search_history['items'] = [
            entry for entry in self.search_history.get('items', [])
            if entry.get('history_id') != history_id
        ]
        self._save_history()
        self._update_workspace_counts()
        dialog.close()
        self._open_history_dialog()

    def _confirm_clear_history(self, parent_dialog):
        with ui.dialog() as confirm, ui.card():
            ui.label('确定清空全部搜索历史吗？收藏和当前工作区不会受到影响。')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('取消', on_click=confirm.close).props('flat')
                def clear():
                    self.search_history = empty_history()
                    self._save_history()
                    self._update_workspace_counts()
                    confirm.close()
                    parent_dialog.close()
                    ui.notify('搜索历史已清空', type='positive')
                ui.button('清空', on_click=clear).props('unelevated color=red-7')
        confirm.open()

    def _open_prompt_import_dialog(self):
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-3xl'):
            ui.label('导入 Prompt').classes('text-lg font-bold')
            ui.label(UI_TEXT['dialogs']['prompt_import_description']).classes(
                'text-sm text-gray-600'
            )
            prompt_input = ui.textarea(
                label='粘贴 Prompt',
                placeholder='1girl, (white_serafuku:1.2), {rain}, @artist_name',
            ).props('outlined autogrow maxlength=20000').classes('w-full min-h-48')
            import_btn = None

            async def submit_import():
                text = str(prompt_input.value or '').strip()
                if not text:
                    ui.notify('请先粘贴 Prompt', type='warning')
                    return
                import_btn.disable()
                try:
                    tagger = await DanbooruTagger.get_instance()
                    allow_nsfw = bool(
                        nsfw_allowed() and self.input_nsfw and self.input_nsfw.value
                    )
                    result = await asyncio.to_thread(
                        resolve_prompt_text,
                        text,
                        resolve_tag=tagger.resolve_tag_name,
                        resolve_artist=tagger.resolve_artist_name,
                        lookup_tag=tagger.get_tag_workspace_metadata,
                        allow_nsfw=allow_nsfw,
                    )
                    if result.parsed_count == 0:
                        ui.notify('没有解析到可导入内容', type='warning')
                        import_btn.enable()
                        return
                    added_count, duplicate_count = self._apply_prompt_import_result(result)
                    dialog.close()
                    self._show_prompt_import_summary(result, added_count, duplicate_count)
                except Exception as exc:
                    print(f'[UI] Prompt 导入异常: {exc}', flush=True)
                    ui.notify('Prompt 导入失败，请检查输入内容', type='negative')
                    import_btn.enable()

            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('取消', on_click=dialog.close).props('flat')
                import_btn = ui.button(
                    '解析并导入', icon='playlist_add', on_click=submit_import,
                ).props('unelevated color=purple-7')
        dialog.open()

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

    def _show_prompt_import_summary(
        self,
        result: PromptImportResult,
        added_count: int,
        duplicate_count: int,
    ):
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
                            f'{item.original}：{self._prompt_pending_reason(pending_record)}'
                        ).classes('text-sm text-orange-800 break-all')
            with ui.row().classes('w-full justify-end'):
                ui.button('知道了', on_click=dialog.close).props('unelevated color=primary')
        dialog.open()

    def _render_prompt_pending(self):
        if self.prompt_pending_container is None:
            return
        self.prompt_pending_container.clear()
        pending_items = [
            item for item in self.workspace_state.get('dismissed', [])
            if isinstance(item, dict) and item.get('kind') == 'prompt_import_pending'
        ]
        if not pending_items:
            return

        with self.prompt_pending_container:
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
                            ui.label(self._prompt_pending_reason(item)).classes(
                                'text-xs text-orange-700'
                            )
                        with ui.row().classes('gap-1 flex-wrap justify-end'):
                            if item.get('reason') not in {'alias_target_missing', 'nsfw_filtered'}:
                                for candidate in item.get('candidates', [])[:5]:
                                    ui.button(
                                        str(candidate),
                                        on_click=lambda i=item, c=str(candidate):
                                            self._accept_prompt_candidate(i, c),
                                    ).props('flat dense color=teal-7').classes('text-xs font-mono')
                            ui.button(
                                icon='close',
                                on_click=lambda i=item: self._remove_prompt_pending(i),
                            ).props('flat round dense color=grey-6')

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

    def _show_workspace_canonicalization(
        self,
        result: WorkspaceCanonicalizationResult,
        label: str,
    ):
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

    def _open_save_favorite_dialog(self):
        if not self._get_selected_tags():
            ui.notify('当前工作区没有可收藏的标签', type='warning')
            return
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
            ui.label('保存当前工作区为收藏').classes('text-lg font-bold')
            name_input = ui.input('收藏名称').props('outlined maxlength=80').classes('w-full')
            notes_input = ui.textarea('备注（可选）').props(
                'outlined autogrow maxlength=500'
            ).classes('w-full')

            def save():
                name = (name_input.value or '').strip()
                if not name:
                    ui.notify('请输入收藏名称', type='warning')
                    return
                if any(item['name'] == name for item in self.favorites.get('items', [])):
                    ui.notify('已存在同名收藏，请在收藏列表中使用“覆盖”', type='warning')
                    return
                favorite = favorite_from_workspace(
                    self.workspace_state,
                    name,
                    notes=(notes_input.value or '').strip(),
                )
                candidate = {
                    'schema_version': 1,
                    'items': [favorite] + self.favorites.get('items', [])[:199],
                }
                if not self._replace_favorites_safely(candidate):
                    return
                dialog.close()
                ui.notify(f'已保存收藏：{name}', type='positive')

            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('取消', on_click=dialog.close).props('flat')
                ui.button('保存', icon='bookmark_add', on_click=save).props(
                    'unelevated color=teal-7'
                )
        dialog.open()

    def _open_favorites_dialog(self):
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-5xl max-h-[88vh]'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('收藏').classes('text-lg font-bold')
                ui.button(icon='close', on_click=dialog.close).props('flat round dense')
            with ui.scroll_area().classes('w-full h-[70vh]'):
                items = self.favorites.get('items', [])
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
                                    on_click=lambda i=item, d=dialog: self._load_favorite(i, False, d),
                                ).props('flat dense color=teal-7')
                                ui.button(
                                    '合并', icon='merge',
                                    on_click=lambda i=item, d=dialog: self._load_favorite(i, True, d),
                                ).props('flat dense color=primary')
                                ui.button(
                                    '复制', icon='content_copy',
                                    on_click=lambda i=item: self._copy_favorite(i),
                                ).props('flat dense color=grey-7')
                                ui.button(
                                    '重命名', icon='edit',
                                    on_click=lambda i=item, d=dialog: self._rename_favorite(i, d),
                                ).props('flat dense color=grey-7')
                                ui.button(
                                    '覆盖', icon='save',
                                    on_click=lambda i=item, d=dialog: self._overwrite_favorite(i, d),
                                ).props('flat dense color=amber-8')
                                ui.button(
                                    '导出', icon='download',
                                    on_click=lambda i=item: self._export_favorite(i),
                                ).props('flat dense color=purple-7')
                                ui.button(
                                    icon='delete_outline',
                                    on_click=lambda i=item, d=dialog: self._confirm_delete_favorite(i, d),
                                ).props('flat round dense color=red-6')
        dialog.open()

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
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-md'):
            ui.label('重命名收藏').classes('font-bold')
            name_input = ui.input('新名称', value=favorite['name']).props(
                'outlined maxlength=80'
            ).classes('w-full')
            def rename():
                name = (name_input.value or '').strip()
                if not name:
                    ui.notify('请输入名称', type='warning')
                    return
                if any(
                    item['favorite_id'] != favorite['favorite_id'] and item['name'] == name
                    for item in self.favorites.get('items', [])
                ):
                    ui.notify('已存在同名收藏', type='warning')
                    return
                candidate = normalize_favorites(self.favorites)[0]
                for item in candidate['items']:
                    if item['favorite_id'] == favorite['favorite_id']:
                        item['name'] = name
                        item['updated_at'] = utc_now_iso()
                        break
                if not self._replace_favorites_safely(candidate):
                    return
                dialog.close()
                parent_dialog.close()
                self._open_favorites_dialog()
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('取消', on_click=dialog.close).props('flat')
                ui.button('保存', on_click=rename).props('unelevated color=primary')
        dialog.open()

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
        with ui.dialog() as dialog, ui.card():
            ui.label(f"确定删除收藏“{favorite['name']}”吗？")
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('取消', on_click=dialog.close).props('flat')
                def delete():
                    candidate = {
                        'schema_version': 1,
                        'items': [
                            item for item in self.favorites.get('items', [])
                            if item['favorite_id'] != favorite['favorite_id']
                        ],
                    }
                    if not self._replace_favorites_safely(candidate):
                        return
                    dialog.close()
                    parent_dialog.close()
                    ui.notify('收藏已删除', type='positive')
                ui.button('删除', on_click=delete).props('unelevated color=red-7')
        dialog.open()

    def _open_backup_dialog(self):
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-2xl'):
            ui.label('本地数据备份与迁移').classes('text-lg font-bold')
            ui.label(UI_TEXT['dialogs']['backup_description']).classes(
                'text-sm text-orange-700 bg-orange-50 rounded p-3'
            )
            ui.button(
                '导出完整 JSON', icon='download', on_click=self._export_backup,
            ).props('unelevated color=primary').classes('w-full')
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
            confirm_import_btn = None

            async def handle_upload(event):
                try:
                    raw = await event.file.text('utf-8')
                    _json.loads(raw)
                    pending_import['raw'] = raw
                    pending_import['name'] = event.file.name
                    pending_label.text = f'已选择：{event.file.name}；尚未执行导入'
                    confirm_import_btn.enable()
                except (UnicodeDecodeError, _json.JSONDecodeError) as exc:
                    pending_import['raw'] = None
                    pending_import['name'] = ''
                    pending_label.text = '文件读取失败，请重新选择有效的 JSON 文件。'
                    confirm_import_btn.disable()
                    ui.notify(f'文件读取失败：{exc}', type='negative', timeout=5000)
                except Exception as exc:
                    pending_import['raw'] = None
                    pending_import['name'] = ''
                    pending_label.text = '文件读取失败，请重新选择。'
                    confirm_import_btn.disable()
                    print(f'[UI] JSON 文件读取异常: {exc}', flush=True)
                    ui.notify('文件读取失败，请检查文件格式', type='negative')

            async def confirm_import():
                raw = pending_import.get('raw')
                if not isinstance(raw, str):
                    ui.notify('请先选择 JSON 文件', type='warning')
                    return
                confirm_import_btn.disable()
                try:
                    await self._import_backup_text(raw, import_mode.value)
                    dialog.close()
                except (WorkspaceDataError, ValueError) as exc:
                    ui.notify(f'导入失败：{exc}', type='negative', timeout=5000)
                    confirm_import_btn.enable()
                except Exception as exc:
                    print(f'[UI] JSON 导入异常: {exc}', flush=True)
                    ui.notify('导入失败，请检查文件格式', type='negative')
                    confirm_import_btn.enable()

            ui.upload(
                label='选择 JSON 文件',
                on_upload=handle_upload,
                auto_upload=True,
                max_file_size=12_000_000,
                on_rejected=lambda: ui.notify('文件过大，仅支持 12 MB 以内的 JSON', type='warning'),
            ).props('accept=.json').classes('w-full')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('关闭', on_click=dialog.close).props('flat')
                confirm_import_btn = ui.button(
                    '确认导入', icon='check', on_click=confirm_import,
                ).props('unelevated color=primary')
                confirm_import_btn.disable()
        dialog.open()

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
            incoming = {
                'schema_version': 1,
                'items': [parsed['favorite']],
            }
            normalized, warnings = normalize_favorites(incoming)
            candidate = merge_favorites(self.favorites, normalized)
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
        if mode == 'favorites_only':
            candidate = merge_favorites(self.favorites, backup['favorites'])
            if not self._replace_favorites_safely(candidate):
                return
            message = '收藏已合并导入'
        elif mode == 'overwrite':
            if not self._replace_favorites_safely(backup['favorites']):
                return
            self._push_undo_snapshot()
            self.search_history = backup['history']
            self._apply_config_state(backup['config'])
            self._apply_workspace_state(backup['workspace'])
            self._save_history()
            message = '本地数据已由备份覆盖'
        else:
            workspace = merge_workspaces(self.workspace_state, backup['workspace'])
            merged_history = merge_history(self.search_history, backup['history'])
            candidate = merge_favorites(self.favorites, backup['favorites'])
            if not self._replace_favorites_safely(candidate):
                return
            self._push_undo_snapshot()
            self.search_history = merged_history
            self._apply_workspace_state(workspace)
            self._save_history()
            message = '备份已合并；当前标签权重和配置保持不变'
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
        self.selection_bar_card = ui.element('div').classes('w-full bg-blue-50 p-4')
        with self.selection_bar_card:
            with ui.row().classes('w-full items-center justify-between'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('check_circle', color='primary')
                    ui.label('已选标签').classes('font-bold text-primary')
                    self.selection_count_label = ui.label('0').classes(
                        'bg-primary text-white px-2 rounded-full text-sm')
                    with ui.icon('info_outline', size='sm', color='grey').classes('cursor-help'):
                        with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                            ui.html(
                                '点击 <b>−</b> / <b>+</b> 可调整标签权重（步长 0.1，范围 0.1~1.9）。<br>'
                                '权重 1.0 时输出原始标签；其余输出 <code>(tag:1.2)</code> 格式。'
                            ).style('font-size:14px;line-height:1.6;')

                with ui.row().classes('items-center gap-2'):
                    with ui.button('没搜到？', icon='help_outline').props('dense flat color=grey-6').classes('text-sm') as _bad_btn:
                        with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                            ui.html('点击此处以反馈失败案例。<br>您的搜索词将被匿名收集用于优化引擎（不包含个人隐私）。').style('font-size:14px;line-height:1.5;')
                    self.bad_case_btn = _bad_btn
                    self.bad_case_btn.disable()
                    self.bad_case_btn.on_click(self.report_bad_case)
                    self.format_toggle_btn = ui.button(
                        'SDXL', icon='swap_horiz'
                    ).props('dense flat color=grey-7').classes('text-xs font-mono')
                    with self.format_toggle_btn:
                        with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                            ui.html(
                                '切换复制格式：<br>'
                                '<b>SDXL</b>：<code>(tag:1.2)</code><br>'
                                '<b>NAI</b>：<code>1.2::tag::</code><br>'
                                '<b>Anima</b>：<code>(tag:1.5)</code> 下划线→空格'
                            ).style('font-size:13px;line-height:1.7;')
                    self.format_toggle_btn.on_click(self._toggle_prompt_format)
                    clear_btn = ui.button('清空已选', icon='delete_sweep').props('dense flat color=red-7').classes('text-xs')
                    clear_btn.on_click(self._clear_all_staged)
                    copy_btn = ui.button('复制选中', icon='content_copy').props('dense unelevated color=primary')
                    copy_btn.on_click(self.copy_selection)

            # chip 容器：每个已选标签渲染为一个带加减按钮的 chip
            self.selected_chips_container = ui.element('div').classes(
                'w-full mt-2 min-h-10 p-1 rounded bg-white border border-blue-100'
            )
            self.prompt_pending_container = ui.column().classes('w-full gap-2 mt-2')

    def _render_selected_chips(self):
        """按稳定 Tag Group 规则渲染；复制顺序仍使用原始选择顺序。"""
        if self.selected_chips_container is None:
            return
        self.selected_chips_container.clear()
        tags = self._get_selected_tags()
        if not tags:
            with self.selected_chips_container:
                ui.label('暂无已选标签').classes('text-xs text-gray-400 italic p-2 self-center')
            return

        grouped: dict[str, list[str]] = {name: [] for name in WORKSPACE_GROUP_ORDER}
        for tag in tags:
            grouped[self._workspace_group_for_tag(tag)].append(tag)

        with self.selected_chips_container:
            step = 0.5 if self.prompt_format == 'anima' else 0.1
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
                            self._render_selected_tag_chip(tag, step)

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

    def _render_selected_tag_chip(self, tag: str, step: float):
        w = self.tag_weights.get(tag, 1.0)
        extra_cls = 'boosted' if w > 1.0 else ('reduced' if w < 1.0 else '')
        w_str = f'{w:.1f}'
        display_label = _format_selected_tag_label(tag, self._get_cn_name_for_tag(tag))
        with ui.element('div').classes(f'weight-chip {extra_cls}'):
            metadata = self._pending_selection_meta.get(tag, {})
            reason = selected_tag_reason(
                metadata.get('origin'),
                metadata.get('source'),
            )
            with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                ui.label(reason).style('font-size:13px;')
            with ui.element('button').classes('weight-btn').props(f'title="移除 {tag}"').on(
                'click', lambda t=tag: self._remove_selected_tag(t)
            ):
                ui.html('&times;')
            with ui.element('button').classes('weight-btn').on(
                'click', lambda t=tag, s=step: self._adjust_weight(t, -s)
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
                'click', lambda t=tag, s=step: self._adjust_weight(t, +s)
            )
            if self.prompt_format == 'anima':
                with plus_btn:
                    with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                        ui.html('Anima模型所需要的权重数值较大').style('font-size:12px;')
            with plus_btn:
                ui.html('&plus;')

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
        self._pending_selection_meta[tag] = {
            'origin': origin,
            'source': source,
        }

    def _push_undo_snapshot(self):
        snapshot = clone_workspace(self.workspace_state)
        signature = workspace_signature(snapshot)
        if self._undo_stack and workspace_signature(self._undo_stack[-1]) == signature:
            return
        self._undo_stack.append(snapshot)
        self._undo_stack = self._undo_stack[-30:]
        self._redo_stack.clear()
        self._update_undo_buttons()

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
        self.workspace_state = clone_workspace(workspace)
        selected = self.workspace_state['selected']
        tags = [item['tag'] for item in selected]
        tag_set = set(tags)
        self._selected_order = list(tags)
        self.tag_weights = {item['tag']: item.get('weight', 1.0) for item in selected}
        self._pending_selection_meta = {
            item['tag']: {
                'origin': item.get('origin', 'unknown'),
                'source': item.get('source', ''),
            }
            for item in selected
        }
        self._workspace_artist_tags = {
            item['tag'] for item in selected
            if item.get('origin') in _ARTIST_ORIGINS
        }

        table_tags = {row['tag'] for row in self.result_table.rows} if self.result_table else set()
        self.chip_extra_selected.clear()
        self.chip_extra_selected.update(tag for tag in tags if tag not in table_tags)
        if self.result_table is not None:
            self.result_table.selected = [
                row for row in self.result_table.rows if row.get('tag') in tag_set
            ]
        self._apply_prompt_format(self.workspace_state.get('prompt_format', 'sdxl'))
        self._render_selected_chips()
        self._render_prompt_pending()
        self._render_concept_coverage()
        if self.selection_count_label is not None:
            self.selection_count_label.text = str(len(tags))
        if self.results_section is not None:
            self.results_section.set_visibility(bool(tags) or bool(self.full_table_data))

        if persist:
            self._save_staged_tags()
            self._save_config()
        if refresh_recommendations:
            show_nsfw = bool(self.input_nsfw.value) if self.input_nsfw else False
            self._last_recommendation_seed_tags = []
            self._refresh_recommendations_if_seed_changed(tags, show_nsfw)

    def _undo_workspace(self):
        if not self._undo_stack:
            ui.notify('没有可撤销的操作', type='info', timeout=1500)
            return
        self._redo_stack.append(clone_workspace(self.workspace_state))
        target = self._undo_stack.pop()
        self._apply_workspace_state(target)
        self._update_undo_buttons()
        ui.notify('已撤销', type='positive', timeout=1500)

    def _redo_workspace(self):
        if not self._redo_stack:
            ui.notify('没有可恢复的操作', type='info', timeout=1500)
            return
        self._undo_stack.append(clone_workspace(self.workspace_state))
        self._undo_stack = self._undo_stack[-30:]
        target = self._redo_stack.pop()
        self._apply_workspace_state(target)
        self._update_undo_buttons()
        ui.notify('已恢复', type='positive', timeout=1500)

    def _schedule_workspace_persist(self):
        """去抖写入浏览器，避免每次快速勾选都发送完整工作区。"""
        if not self._storage_write_allowed('workspace'):
            return
        if self._workspace_save_task and not self._workspace_save_task.done():
            self._workspace_save_task.cancel()

        async def _persist():
            try:
                await asyncio.sleep(WORKSPACE_SAVE_DEBOUNCE_SECONDS)
            except asyncio.CancelledError:
                return
            try:
                data = dump_workspace(self.workspace_state)
            except WorkspaceDataError as exc:
                print(f'[UI] 工作区保存前校验失败: {exc}', flush=True)
                return

            if not self._storage_write_allowed('workspace'):
                return
            client = self.client
            try:
                client.run_javascript(
                    f"localStorage.setItem('{WORKSPACE_STORAGE_KEY}', {_json.dumps(data)});"
                )
            except RuntimeError:
                self._storage_session_dirty.add('workspace')
                return
            self._storage_session_dirty.discard('workspace')

        self._workspace_save_task = asyncio.ensure_future(_persist())

    def _save_staged_tags(self):
        """将实时选择同步到版本化 WorkspaceState，并去抖写入 localStorage。"""
        tags = self._get_selected_tags()
        self._selected_order = list(tags)
        cn_names = {t: self._get_cn_name_for_tag(t) for t in tags}
        self.workspace_state['prompt_format'] = self.prompt_format
        self.workspace_state = sync_selected_entries(
            self.workspace_state,
            tags,
            self.tag_weights,
            cn_names,
            self._pending_selection_meta,
        )
        self._workspace_artist_tags = {
            item['tag'] for item in self.workspace_state['selected']
            if item.get('origin') in _ARTIST_ORIGINS
        }
        self._schedule_workspace_persist()

    def _local_storage_keys(self) -> dict[str, str]:
        return {
            'workspace': WORKSPACE_STORAGE_KEY,
            'legacy': self._STAGED_LS_KEY,
            'config': _CONFIG_LS_KEY,
            'history': HISTORY_STORAGE_KEY,
            'favorites': FAVORITES_STORAGE_KEY,
        }

    async def _prepare_local_storage_restore(self, names: list[str]) -> dict:
        """Snapshot requested keys in-browser and compact legacy history in memory."""
        if not self._client_connected():
            raise RuntimeError('client is disconnected')
        keys = {name: self._local_storage_keys()[name] for name in names}
        keys_js = _json.dumps(keys, ensure_ascii=False)
        cache_key_js = _json.dumps(_LOCAL_STORAGE_RESTORE_CACHE)
        result = await self.client.run_javascript(
            f"""(() => {{
                const keys = {keys_js};
                const values = {{}};
                const manifest = {{}};
                for (const [name, key] of Object.entries(keys)) {{
                    let value = localStorage.getItem(key);
                    let prepared = false;
                    let originalLength = value === null ? null : value.length;
                    if (name === 'history' && value) {{
                        try {{
                            const data = JSON.parse(value);
                            if (data && typeof data === 'object' &&
                                (data.schema_version === 1 || data.schema_version === 2) &&
                                Array.isArray(data.items)) {{
                                let changed = data.schema_version !== 2;
                                const items = data.items.map((item) => {{
                                    if (!item || typeof item !== 'object' ||
                                        !item.workspace || typeof item.workspace !== 'object' ||
                                        typeof item.query !== 'string' || !item.query.trim() ||
                                        !item.settings || typeof item.settings !== 'object' ||
                                        Array.isArray(item.settings)) return item;
                                    const query = item.query.trim().slice(0, 4000);
                                    const searchedAt = typeof item.searched_at === 'string' && item.searched_at
                                        ? item.searched_at : new Date().toISOString();
                                    const compactQuery = {{
                                        query,
                                        searched_at: searchedAt,
                                        settings: item.settings,
                                    }};
                                    const oldQueries = item.workspace.queries;
                                    if (!Array.isArray(oldQueries) || oldQueries.length !== 1 ||
                                        !oldQueries[0] || oldQueries[0].query !== query) changed = true;
                                    const workspace = {{
                                        ...item.workspace,
                                        queries: [compactQuery],
                                        updated_at: searchedAt,
                                    }};
                                    return {{...item, workspace_id: workspace.workspace_id, workspace}};
                                }});
                                if (changed) {{
                                    value = JSON.stringify({{...data, schema_version: 2, items}});
                                    prepared = true;
                                }}
                            }}
                        }} catch (_) {{
                            // Python performs authoritative validation and corruption backup.
                        }}
                    }}
                    values[name] = value;
                    manifest[name] = {{
                        length: value === null ? null : value.length,
                        original_length: originalLength,
                        prepared,
                    }};
                }}
                window[{cache_key_js}] = values;
                return manifest;
            }})()""",
            timeout=5.0,
        )
        if not isinstance(result, dict):
            raise RuntimeError('localStorage manifest is invalid')
        return result

    async def _read_local_storage_value(
        self,
        name: str,
        key: str,
        length,
    ) -> str | None:
        """Read one prepared localStorage value in transport-safe chunks."""
        if length is None:
            return None
        if isinstance(length, bool) or not isinstance(length, (int, float)):
            raise RuntimeError(f'localStorage key {key!r} returned an invalid length')
        length = int(length)
        if length < 0 or length > _LOCAL_STORAGE_MAX_READ_CHARS:
            raise WorkspaceDataError(f'localStorage key {key!r} exceeds the read limit')
        if length == 0:
            return ''

        name_js = _json.dumps(name, ensure_ascii=False)
        key_js = _json.dumps(key, ensure_ascii=False)
        cache_key_js = _json.dumps(_LOCAL_STORAGE_RESTORE_CACHE)
        chunks: list[str] = []
        offset = 0
        while offset < length:
            if not self._client_connected():
                raise RuntimeError('client disconnected during localStorage restore')
            result = await self.client.run_javascript(
                f"""(() => {{
                    const cache = window[{cache_key_js}];
                    const value = cache && Object.prototype.hasOwnProperty.call(cache, {name_js})
                        ? cache[{name_js}] : localStorage.getItem({key_js});
                    if (value === null) return null;
                    let end = Math.min(value.length, {offset + _LOCAL_STORAGE_READ_CHUNK_CHARS});
                    if (end < value.length) {{
                        const lastCodeUnit = value.charCodeAt(end - 1);
                        if (lastCodeUnit >= 0xD800 && lastCodeUnit <= 0xDBFF) end += 1;
                    }}
                    return {{chunk: value.slice({offset}, end), next_offset: end}};
                }})()""",
                timeout=5.0,
            )
            if not isinstance(result, dict):
                raise RuntimeError(f'localStorage key {key!r} disappeared during restore')
            chunk = result.get('chunk')
            next_offset = result.get('next_offset')
            if not isinstance(chunk, str) or not isinstance(next_offset, (int, float)):
                raise RuntimeError(f'localStorage key {key!r} returned an invalid chunk')
            next_offset = int(next_offset)
            if next_offset <= offset or next_offset > length:
                raise RuntimeError(f'localStorage key {key!r} returned an invalid offset')
            chunks.append(chunk)
            offset = next_offset
        return ''.join(chunks)

    def _clear_local_storage_restore_cache(self):
        if not self._client_connected():
            return
        cache_key_js = _json.dumps(_LOCAL_STORAGE_RESTORE_CACHE)
        try:
            self.client.run_javascript(f'delete window[{cache_key_js}];')
        except RuntimeError:
            pass

    async def _backup_local_storage_key(self, source_key: str, backup_key: str) -> bool:
        """Copy an existing value inside the browser before replacing it."""
        if not self._client_connected():
            return False
        source_key_js = _json.dumps(source_key, ensure_ascii=False)
        backup_key_js = _json.dumps(backup_key, ensure_ascii=False)
        try:
            status = await self.client.run_javascript(
                f"""(() => {{
                    const source = localStorage.getItem({source_key_js});
                    if (source === null) return 'missing';
                    if (localStorage.getItem({backup_key_js}) !== null) return 'exists';
                    try {{
                        localStorage.setItem({backup_key_js}, source);
                        return 'created';
                    }} catch (error) {{
                        return `error:${{error && error.name ? error.name : 'unknown'}}`;
                    }}
                }})()""",
                timeout=5.0,
            )
        except Exception:
            return False
        return status in {'created', 'exists'}

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
        previous = self.favorites
        self.favorites = favorites
        if self._save_favorites():
            self._update_workspace_counts()
            return True
        self.favorites = previous
        self._storage_session_dirty.discard('favorites')
        ui.notify('收藏数据过大或无法写入，操作已取消', type='negative')
        return False

    async def _restore_staged_tags(self) -> tuple[dict[str, str], set[str], list[str]]:
        """Run one restore attempt and return failures, required writes and warnings."""
        unresolved = [
            name for name in _LOCAL_STORAGE_NAMES
            if self._storage_states.get(name) != 'ready'
        ]
        if not unresolved:
            return {}, set(), []

        failures: dict[str, str] = {}
        persist: set[str] = set()
        warnings: list[str] = []
        keys = self._local_storage_keys()
        try:
            manifest = await self._prepare_local_storage_restore(unresolved)
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            return {name: message for name in unresolved}, persist, warnings

        try:
            for name in unresolved:
                meta = manifest.get(name)
                if not isinstance(meta, dict):
                    failures[name] = 'missing manifest entry'
                    continue
                try:
                    raw = await self._read_local_storage_value(
                        name,
                        keys[name],
                        meta.get('length'),
                    )
                except Exception as exc:
                    failures[name] = str(exc) or type(exc).__name__
                    continue
                self._storage_raw_values[name] = raw
        finally:
            self._clear_local_storage_restore_cache()

        if 'legacy' in unresolved and 'legacy' not in failures:
            self._storage_states['legacy'] = 'ready'

        if 'config' in unresolved and 'config' not in failures:
            raw_config = self._storage_raw_values.get('config')
            config_dirty = 'config' in self._storage_session_dirty
            try:
                cfg = _json.loads(raw_config) if raw_config else {}
                if not isinstance(cfg, dict):
                    raise WorkspaceDataError('config must be a JSON object')
            except Exception as exc:
                if raw_config and not await self._backup_local_storage_key(
                    _CONFIG_LS_KEY,
                    f'{_CONFIG_LS_KEY}_corrupt_backup',
                ):
                    failures['config'] = f'corrupt config backup failed: {exc}'
                else:
                    warnings.append('config_corrupt')
                    persist.add('config')
            else:
                if cfg and cfg.get('version') != _CONFIG_VERSION:
                    warnings.append('config_schema_migrated')
                    persist.add('config')
                if not config_dirty:
                    self._storage_applying.add('config')
                    try:
                        self._apply_config_state(cfg)
                    finally:
                        self._storage_applying.discard('config')
                else:
                    persist.add('config')
            if 'config' not in failures:
                self._storage_states['config'] = 'ready'

        if 'workspace' in unresolved and 'workspace' not in failures:
            raw_workspace = self._storage_raw_values.get('workspace')
            if not raw_workspace and any(
                self._storage_states.get(name) != 'ready'
                for name in ('legacy', 'config')
            ):
                failures['workspace'] = 'legacy workspace inputs are not available'
            else:
                workspace_warnings: list[str] = []
                try:
                    if raw_workspace:
                        workspace, workspace_warnings = normalize_workspace(raw_workspace)
                    else:
                        workspace, workspace_warnings = migrate_legacy_workspace(
                            self._storage_raw_values.get('legacy'),
                            self._storage_raw_values.get('config'),
                        )
                        persist.add('workspace')
                except WorkspaceDataError as exc:
                    backed_up = await self._backup_local_storage_key(
                        WORKSPACE_STORAGE_KEY,
                        f'{WORKSPACE_STORAGE_KEY}_corrupt_backup',
                    )
                    if raw_workspace and not backed_up:
                        failures['workspace'] = f'corrupt workspace backup failed: {exc}'
                    else:
                        workspace, migration_warnings = migrate_legacy_workspace(
                            self._storage_raw_values.get('legacy'),
                            self._storage_raw_values.get('config'),
                        )
                        workspace_warnings = ['workspace_corrupt'] + migration_warnings
                        persist.add('workspace')
                if 'workspace' not in failures:
                    if 'workspace' in self._storage_session_dirty:
                        workspace = merge_workspaces(
                            self.workspace_state,
                            workspace,
                            origin='local_restore',
                            source='浏览器本地恢复',
                        )
                        persist.add('workspace')
                    self._storage_applying.add('workspace')
                    try:
                        self._apply_workspace_state(
                            workspace,
                            persist=False,
                            refresh_recommendations=False,
                        )
                    finally:
                        self._storage_applying.discard('workspace')
                    warnings.extend(workspace_warnings)
                    if workspace_warnings:
                        persist.add('workspace')
                    self._storage_states['workspace'] = 'ready'
                    if raw_workspace:
                        # A valid versioned workspace makes the legacy key irrelevant.
                        self._storage_states['legacy'] = 'ready'
                        failures.pop('legacy', None)

        if 'history' in unresolved and 'history' not in failures:
            raw_history = self._storage_raw_values.get('history')
            history_prepared = bool(
                isinstance(manifest.get('history'), dict)
                and manifest['history'].get('prepared')
            )
            try:
                history, history_warnings = normalize_history(raw_history)
            except WorkspaceDataError as exc:
                backed_up = await self._backup_local_storage_key(
                    HISTORY_STORAGE_KEY,
                    f'{HISTORY_STORAGE_KEY}_corrupt_backup',
                )
                if raw_history and not backed_up:
                    failures['history'] = f'corrupt history backup failed: {exc}'
                else:
                    history, history_warnings = empty_history(), ['history_corrupt']
                    persist.add('history')
            if 'history' not in failures:
                if history_prepared:
                    if not await self._backup_history_before_compaction():
                        failures['history'] = 'legacy history backup failed'
                    else:
                        history_warnings.extend([
                            'history_schema_migrated',
                            'history_workspace_queries_compacted',
                        ])
                        persist.add('history')
                if 'history' not in failures:
                    if 'history' in self._storage_session_dirty:
                        history = merge_history(self.search_history, history)
                        persist.add('history')
                    self.search_history = history
                    warnings.extend(history_warnings)
                    if history_warnings:
                        persist.add('history')
                    self._storage_states['history'] = 'ready'

        if 'favorites' in unresolved and 'favorites' not in failures:
            raw_favorites = self._storage_raw_values.get('favorites')
            try:
                favorites, favorite_warnings = normalize_favorites(raw_favorites)
            except WorkspaceDataError as exc:
                backed_up = await self._backup_local_storage_key(
                    FAVORITES_STORAGE_KEY,
                    f'{FAVORITES_STORAGE_KEY}_corrupt_backup',
                )
                if raw_favorites and not backed_up:
                    failures['favorites'] = f'corrupt favorites backup failed: {exc}'
                else:
                    favorites, favorite_warnings = empty_favorites(), ['favorites_corrupt']
                    persist.add('favorites')
            if 'favorites' not in failures:
                if 'favorites' in self._storage_session_dirty:
                    favorites = merge_favorites(self.favorites, favorites)
                    persist.add('favorites')
                self.favorites = favorites
                warnings.extend(favorite_warnings)
                if favorite_warnings:
                    persist.add('favorites')
                self._storage_states['favorites'] = 'ready'

        for name in failures:
            self._storage_states[name] = 'failed'
        self._update_undo_buttons()
        self._update_workspace_counts()
        return failures, persist, warnings

    def _persist_restored_storage(self, names: set[str]):
        if 'config' in names:
            self._save_config()
        if 'workspace' in names:
            self._save_staged_tags()
        if 'history' in names:
            self._save_history()
        if 'favorites' in names:
            self._save_favorites()

    def _flush_storage_session_changes(self):
        ready_dirty = {
            name for name in self._storage_session_dirty
            if self._storage_states.get(name) == 'ready'
        }
        self._persist_restored_storage(ready_dirty)

    async def _restore_local_storage_with_retries(self):
        self._storage_restore_started = True
        last_failures: dict[str, str] = {}
        all_warnings: list[str] = []
        was_cancelled = False
        try:
            for delay in _LOCAL_STORAGE_RESTORE_RETRY_DELAYS:
                if delay:
                    await asyncio.sleep(delay)
                if not self._client_alive():
                    return
                if not self._client_connected():
                    last_failures = {'connection': 'client is disconnected'}
                    continue
                self._storage_restoring = True
                try:
                    failures, persist, warnings = await self._restore_staged_tags()
                finally:
                    self._storage_restoring = False
                self._persist_restored_storage(persist)
                all_warnings.extend(warnings)
                last_failures = failures
                if not failures:
                    self._flush_storage_session_changes()
                    self._install_workspace_storage_listener()
                    self._storage_failure_notified = False
                    if all_warnings:
                        print(
                            f'[UI] 本地数据恢复提示: {sorted(set(all_warnings))}',
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
                    f'{name}={message}' for name, message in sorted(last_failures.items())
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
            self._storage_restoring = False
            if self._storage_restore_task is asyncio.current_task():
                self._storage_restore_task = None
                if was_cancelled and self._client_connected():
                    self._start_storage_restore_task()

    def _start_storage_restore_task(self):
        task = self._storage_restore_task
        if task is not None and not task.done():
            return task
        needs_work = any(
            state != 'ready' for state in self._storage_states.values()
        ) or bool(self._storage_session_dirty)
        if not needs_work:
            return None
        self._storage_restore_task = asyncio.create_task(
            self._restore_local_storage_with_retries()
        )
        return self._storage_restore_task

    def _pause_storage_restore(self):
        task = self._storage_restore_task
        if task is not None and not task.done():
            task.cancel()

    def _install_workspace_storage_listener(self):
        """其他标签页修改工作区时提示刷新，避免静默覆盖。"""
        if self._workspace_storage_listener_installed:
            return
        self._workspace_storage_listener_installed = True
        try:
            ui.run_javascript(f"""
                if (!window.__danbooruWorkspaceStorageListenerV1) {{
                    window.__danbooruWorkspaceStorageListenerV1 = true;
                    window.addEventListener('storage', (event) => {{
                        const watchedKeys = new Set([
                            '{WORKSPACE_STORAGE_KEY}',
                            '{HISTORY_STORAGE_KEY}',
                            '{FAVORITES_STORAGE_KEY}',
                        ]);
                        if (watchedKeys.has(event.key) && event.newValue !== event.oldValue) {{
                            const reload = window.confirm(
                                '工作区数据已在另一个标签页更新。是否重新加载当前页面以同步最新内容？'
                            );
                            if (reload) window.location.reload();
                        }}
                    }});
                }}
            """)
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
        self.two_col_container = ui.element('div').classes('w-full two-col-layout')
        with self.two_col_container:
            # ── 左栏：语义匹配结果（表格）──
            with ui.card().classes('col-left section-surface'):
                with ui.row().classes('items-center justify-between mb-2 w-full'):
                    ui.label('匹配标签结果').classes('section-heading')
                    ui.button('复制全部标签', icon='content_copy', on_click=self._copy_all_tags) \
                        .props('dense flat color=primary').classes('text-sm')

                self.result_table = ui.table(
                    columns=TABLE_COLUMNS,
                    rows=[],
                    pagination=0,
                    selection='multiple',
                    row_key='tag',
                ).props('flat separator=horizontal').classes(
                    'w-full result-table-flat'
                )
                self.result_table.on('selection', self._update_selection_display)
                self.result_table.on('link_click', self._mark_interaction)
                self.result_table.on('translation_feedback', self.report_translation_error)
                self.result_table.on('pagination', lambda _: self._save_config())

                # 自定义行模板：行背景色按分类，整行悬浮显示 wiki（NSFW模糊行除外）
                self.result_table.add_slot('body', r'''
                    <q-tr :props="props"
                          :class="props.row._nsfw_blocked ? 'nsfw-row-blocked' : ''"
                          :style="{
                              'background-color':
                                  props.row.layer === 'artist'       ? 'rgba(244,114,182,0.08)' :
                                  props.row.category === 'General'   ? 'rgba(59,130,246,0.06)' :
                                  props.row.category === 'Character' ? 'rgba(34,197,94,0.06)'  :
                                  props.row.category === 'Copyright' ? 'rgba(168,85,247,0.06)' : ''
                          }">
                        <q-td auto-width>
                            <q-checkbox v-model="props.selected"
                                :class="props.row._nsfw_blocked ? 'nsfw-checkbox-disabled' : ''"/>
                        </q-td>
                        <q-td v-for="col in props.cols" :key="col.name" :props="props">
                            <template v-if="col.name === 'tag' || col.name === 'cn_name'">
                                <div :class="props.row._nsfw_blocked ? 'nsfw-blur-cell' : ''">
                                    <template v-if="col.name === 'cn_name' && col.value && props.row.layer !== 'artist'">
                                        <span style="font-size:14px;display:inline-flex;align-items:center;gap:4px;">
                                            <span>{{ col.value.split(',')[0] }}</span>
                                            <q-btn icon="report_problem"
                                                size="sm"
                                                dense flat round
                                                color="grey-5"
                                                padding="xs"
                                                @click.stop.prevent="console.debug('[DanbooruSearch] translation_feedback click', props.row); $parent.$emit('translation_feedback', props.row)">
                                                <q-tooltip>反馈翻译错误</q-tooltip>
                                            </q-btn>
                                        </span>
                                    </template>
                                    <template v-else-if="col.name === 'tag'">
                                        <a :href="'https://danbooru.donmai.us/wiki_pages/'+col.value"
                                           target="_blank"
                                           class="text-primary hover:underline font-bold inline-flex items-center"
                                           style="text-decoration:none; font-family: Consolas, Monaco, Courier New, monospace;"
                                           @click.stop="$emit('link_click', col.value)">
                                            {{ col.value }}
                                            <q-icon name="open_in_new" size="xs" class="q-ml-xs opacity-50"/>
                                        </a>
                                    </template>
                                    <template v-else>{{ col.value }}</template>
                                </div>
                            </template>
                            <template v-else-if="col.name === 'nsfw'">
                                <div v-if="col.value === '1'" class="text-red-500">🔴</div>
                                <div v-else class="text-green-500">🟢</div>
                            </template>
                            <template v-else-if="col.name === 'final_score'">
                                <q-badge :color="col.value > 0.6 ? 'green' : (col.value > 0.5 ? 'teal' : 'orange')">
                                    {{ col.value }}
                                </q-badge>
                            </template>
                            <template v-else>{{ col.value }}</template>
                        </q-td>
                        <q-tooltip v-if="props.row.layer === 'artist' && props.row.artist_top_tags && props.row.artist_top_tags.length && !props.row._nsfw_blocked"
                            content-class="bg-black text-white shadow-4"
                            max-width="400px" :offset="[10,10]">
                            <div style="font-size:14px;line-height:1.5;max-width:380px;">
                                <b>{{ props.row.tag }}</b><br>这位画师经常画:<br>
                                <template v-for="tag in props.row.artist_top_tags.slice(0, 10)" :key="tag">
                                    &nbsp;&nbsp;· {{ tag }}<br>
                                </template>
                            </div>
                        </q-tooltip>
                        <q-tooltip v-else-if="(props.row.wiki || props.row.cn_name) && !props.row._nsfw_blocked"
                            content-class="bg-black text-white shadow-4"
                            max-width="500px" :offset="[10,10]">
                            <div style="font-size:14px;line-height:1.5;">
                                <span style="opacity:0.7;margin-right:4px;">{{
                                    props.row.category === 'General'   ? '[通用]' :
                                    props.row.category === 'Character' ? '[角色]' :
                                    props.row.category === 'Copyright' ? '[作品]' : ''
                                }}</span>{{ props.row.wiki }}
                                <div v-if="props.row.cn_name"
                                     style="margin-top:6px;opacity:0.85;">{{ props.row.cn_name }}</div>
                            </div>
                        </q-tooltip>
                    </q-tr>
                ''')

                # ── Group 同类扩展（左栏，表格下方）──
                ui.separator().classes('my-2')
                with ui.row().classes('items-center justify-between w-full mb-1'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label('同类标签').classes('font-bold text-sm text-gray-600')
                        with ui.icon('info_outline', size='xs', color='grey').classes('cursor-help'):
                            with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                                ui.label('基于标签分组数据，展示已选标签所属分组中的其他标签。勾选可加入已选。').style('font-size:14px;')
                    ui.button('根据已选刷新', icon='refresh', on_click=self._manual_refresh_group) \
                        .props('dense flat color=primary').classes('text-sm')
                self.group_expansion_container = ui.column().classes('w-full gap-0')
                with self.group_expansion_container:
                    ui.label('请先搜索并勾选标签…').classes('text-sm text-gray-400 italic p-4')

            # ── 右栏：推荐画师 + 关联推荐 ──
            with ui.card().classes('col-right section-surface'):
                # 推荐画师
                with ui.row().classes('items-center justify-between w-full mb-2'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label('推荐擅长画师(Beta)').classes('section-heading')
                        with ui.icon('info_outline', size='sm', color='grey').classes('cursor-help'):
                            with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                                ui.html(
                                    '基于标签-画师 NPMI 共现数据，根据您当前已选的标签，推荐擅长这些元素的画师。<br>悬停画师行可查看与该画师共现关联最强的标签。').style(
                                    'font-size:14px;line-height:1.5;')

                self.artist_rec_list = ui.column().classes(
                    'w-full gap-0 recommendation-grid'
                )
                with self.artist_rec_list:
                    ui.label('请先搜索并勾选标签…').classes('text-sm text-gray-400 italic p-4')
                self.artist_rec_pagination = ui.column().classes('w-full')

                ui.separator().classes('my-3')

                # 关联推荐
                with ui.row().classes('items-center justify-between w-full mb-2'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label('关联推荐').classes('section-heading')
                        with ui.icon('info_outline', size='sm', color='grey').classes('cursor-help'):
                            with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                                ui.html(
                                    '基于标签共现数据，发掘语义之外的相关性，为您推荐更多可能的标签。<br>勾选可加入或移出已选。如需根据最新选项更新推荐，请点击刷新按钮。').style(
                                    'font-size:14px;line-height:1.5;')

                    # 新增手动刷新按钮
                    ui.button('根据已选刷新', icon='refresh', on_click=self._manual_refresh_related) \
                        .props('dense flat color=primary').classes('text-sm')

                self.related_list_container = ui.column().classes(
                    'w-full gap-0 recommendation-grid'
                )
                with self.related_list_container:
                    ui.label('请先搜索并勾选标签…').classes('text-sm text-gray-400 italic p-4')
                self.related_pagination = ui.column().classes('w-full')

    # ══════════════════════════════════════════════════════════════════════
    # 渲染关联推荐列表
    # ══════════════════════════════════════════════════════════════════════

    def _set_related_page(self, page: int):
        """切换关联推荐页，只构建当前页的可见行。"""
        if self._related_page_count < 1:
            return
        self._related_page = max(1, min(page, self._related_page_count))
        self._render_related_page()
        if self._related_page_label is not None:
            self._related_page_label.text = (
                f'{self._related_page} / {self._related_page_count}'
            )
        if self._related_prev_button is not None:
            if self._related_page == 1:
                self._related_prev_button.disable()
            else:
                self._related_prev_button.enable()
        if self._related_next_button is not None:
            if self._related_page == self._related_page_count:
                self._related_next_button.disable()
            else:
                self._related_next_button.enable()

    def _render_related_page(self):
        """重建当前关联推荐页，节点数量固定不超过 10 条。"""
        self.related_list_container.clear()
        self._related_checkboxes.clear()

        if not self._related_results:
            with self.related_list_container:
                ui.label('暂无推荐').classes('text-sm text-gray-400 italic p-4')
            return

        selected_now = set(self._get_selected_tags())
        start = (self._related_page - 1) * RELATED_REC_PAGE_SIZE
        end = start + RELATED_REC_PAGE_SIZE
        page_results = self._related_results[start:end]

        with self.related_list_container:
            for r in page_results:
                tag = r.tag
                cn_first = r.cn_name.split(',')[0].strip() if r.cn_name else ''
                is_selected = tag in selected_now
                score_pct = f'+{r.cooc_score * 100:.0f}%'

                # 获取 wiki
                wiki_text = ''
                try:
                    tagger = DanbooruTagger._instance
                    if tagger and tagger.df is not None and tag in tagger._name_to_idx:
                        idx = tagger._name_to_idx[tag]
                        wiki_text = str(tagger.df.iloc[idx].get('wiki', ''))
                except Exception:
                    pass

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
                            on_change=lambda e, t=tag: self._on_related_checkbox_change(t, e.value)
                        ).props('dense').classes('flex-none')
                        self._related_checkboxes[tag] = cb

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
                                link.on('click', self._mark_interaction)
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

    def _render_related_list(self, related: list, show_nsfw: bool):
        """保存关联推荐快照，并仅渲染当前页。"""
        if self.related_list_container is None or self.related_pagination is None:
            return
        self.related_list_container.clear()
        self.related_pagination.clear()
        self._related_checkboxes.clear()
        self._related_page = 1
        self._related_page_label = None
        self._related_prev_button = None
        self._related_next_button = None
        self._related_results = [
            item for item in related
            if not (item.nsfw == '1' and not show_nsfw)
        ]
        self._related_show_nsfw = show_nsfw
        self._related_page_count = (
            len(self._related_results) + RELATED_REC_PAGE_SIZE - 1
        ) // RELATED_REC_PAGE_SIZE

        if not self._related_results:
            self._render_related_page()
            return

        if self._related_page_count > 1:
            with self.related_pagination:
                with ui.row().classes('w-full items-center justify-center gap-2 px-3 py-2'):
                    self._related_prev_button = ui.button(
                        '‹',
                        on_click=lambda: self._set_related_page(self._related_page - 1),
                    ).props('flat dense round color=grey-7')
                    self._related_page_label = ui.label().classes(
                        'text-xs text-gray-600 min-w-12 text-center'
                    )
                    self._related_next_button = ui.button(
                        '›',
                        on_click=lambda: self._set_related_page(self._related_page + 1),
                    ).props('flat dense round color=grey-7')

        self._set_related_page(1)

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
        self.current_filter_keyword = keyword if keyword else 'ALL'
        show_nsfw_val = self.input_nsfw.value
        if not keyword or keyword == 'ALL':
            filtered = self.full_table_data
        else:
            filtered = [r for r in self.full_table_data if r['source'] == keyword]

        self.result_table.rows = apply_nsfw_filter(filtered, show_nsfw_val)
        self._render_concept_coverage()

    def _render_concept_coverage(self):
        if self.coverage_container is None:
            return
        self.coverage_container.clear()
        if not self.current_query_str:
            return

        segments = list(dict.fromkeys(self.current_segments + self.current_keywords))
        if not segments:
            segments = [self.current_query_str]
        coverage = compute_concept_coverage(
            segments,
            self.full_table_data,
            self._get_selected_tags(),
        )
        if not coverage:
            return

        status_counts = {
            COVERED: sum(item.status == COVERED for item in coverage),
            CANDIDATE_UNSELECTED: sum(
                item.status == CANDIDATE_UNSELECTED for item in coverage
            ),
            UNCOVERED: sum(item.status == UNCOVERED for item in coverage),
        }

        def decorate_chip(chip, source: str) -> None:
            if source in self.current_cached_queries:
                chip.style(
                    'outline: 1px dashed rgba(100,116,139,0.45); outline-offset: 1px;'
                )
            if source == self.current_filter_keyword:
                chip.style('box-shadow: 0 0 0 2px #4a90e2;')

        with self.coverage_container:
            with ui.element('div').classes(
                'w-full query-insight-panel section-surface'
            ):
                with ui.row().classes('w-full items-center gap-2 flex-wrap'):
                    ui.icon('manage_search', color='primary', size='sm')
                    ui.label('查询理解').classes('section-heading')
                    ui.label(
                        f"已覆盖 {status_counts[COVERED]} · "
                        f"有候选 {status_counts[CANDIDATE_UNSELECTED]} · "
                        f"未覆盖 {status_counts[UNCOVERED]}"
                    ).classes('text-xs text-slate-500')
                    with ui.icon('info_outline', size='xs', color='grey').classes('cursor-help'):
                        with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                            ui.label(
                                '点击概念可筛选对应结果；红色未覆盖项点击后会发起补充搜索。'
                                '覆盖状态根据当前结果来源和工作区已选标签近似判断。'
                            ).style('font-size:13px;')

                with ui.row().classes('w-full gap-2 flex-wrap mt-2'):
                    all_chip = ui.chip(
                        '全部结果', on_click=lambda: self._filter_by_source('ALL')
                    )
                    if self.current_filter_keyword == 'ALL':
                        all_chip.props('color=primary text-color=white clickable')
                    else:
                        all_chip.props('color=grey-3 text-color=grey-9 clickable')

                    use_segmentation = self.input_segment.value if self.input_segment else True
                    if use_segmentation and self.current_query_str not in segments:
                        whole_chip = ui.chip(
                            '整句',
                            on_click=lambda: self._filter_by_source(self.current_query_str),
                        ).props('color=blue-grey-1 text-color=blue-grey-9 clickable')
                        decorate_chip(whole_chip, self.current_query_str)

                    for item in coverage:
                        if item.status == COVERED:
                            chip = ui.chip(
                                item.segment,
                                icon='check_circle',
                                on_click=lambda s=item.segment: self._filter_by_source(s),
                            ).props('color=green-1 text-color=green-9 clickable')
                            detail = (
                                f"已覆盖；已选择：{'、'.join(item.selected_tags)}。"
                                '点击筛选此概念的搜索结果。'
                            )
                        elif item.status == CANDIDATE_UNSELECTED:
                            chip = ui.chip(
                                item.segment,
                                icon='radio_button_unchecked',
                                on_click=lambda s=item.segment: self._filter_by_source(s),
                            ).props('color=amber-1 text-color=amber-9 clickable')
                            detail = (
                                f"有候选：{'、'.join(item.candidate_tags[:5])}。"
                                '点击筛选此概念的搜索结果。'
                            )
                        else:
                            chip = ui.chip(
                                item.segment,
                                icon='search',
                                on_click=lambda s=item.segment: self._search_uncovered_segment(s),
                            ).props('color=red-1 text-color=red-8 clickable')
                            detail = '点击后沿用当前搜索设置进行补充搜索；工作区标签保持不变。'
                        decorate_chip(chip, item.segment)
                        with chip:
                            with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                                ui.label(detail).style('font-size:13px;')

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
        table_tags = [row['tag'] for row in self.result_table.selected] if self.result_table else []
        seen = set(table_tags)
        extra_pool = set(self.chip_extra_selected)
        extra = [t for t in self._selected_order if t in extra_pool and t not in seen]
        seen.update(extra)
        extra.extend(sorted(t for t in extra_pool if t not in seen))
        return table_tags + extra

    def _get_recommendation_seed_tags(self, selected_tags: list[str]) -> list[str]:
        artist_tags = (
            set(self._current_artist_rec_tags)
            | set(self._artist_result_tags)
            | set(self._workspace_artist_tags)
        )
        if self.result_table is not None:
            for row in self.result_table.rows:
                if row.get('layer') != 'artist':
                    continue
                tag = row.get('tag')
                if tag:
                    artist_tags.add(tag)
        return [tag for tag in selected_tags if tag not in artist_tags]

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
        if related is None:
            related = []
        selected_now = set(self._get_selected_tags())
        old_related  = self.current_related
        new_tags  = {r.tag for r in related}
        preserved = [r for r in old_related if r.tag in selected_now and r.tag not in new_tags]
        merged = list(related) + preserved

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
        selected_tags = self._get_recommendation_seed_tags(selected_tags)
        requested_scopes = frozenset(scopes)
        pending = self._pending_recommendation_request
        if (
            pending is not None
            and pending['selected_tags'] == selected_tags
            and pending['show_nsfw'] == show_nsfw
        ):
            requested_scopes = pending['scopes'] | requested_scopes

        self._recommendation_generation += 1
        self._pending_recommendation_request = {
            'generation': self._recommendation_generation,
            'selected_tags': list(selected_tags),
            'show_nsfw': show_nsfw,
            'scopes': requested_scopes,
        }
        if self._recommendation_task is None or self._recommendation_task.done():
            self._recommendation_task = asyncio.ensure_future(
                self._recommendation_worker()
            )

    async def _recommendation_worker(self):
        """连续消费最新选择快照；过期结果只计算不渲染。"""
        try:
            while self._pending_recommendation_request is not None:
                await asyncio.sleep(RECOMMENDATION_DEBOUNCE_SECONDS)
                request = self._pending_recommendation_request
                self._pending_recommendation_request = None
                selected_tags = request['selected_tags']
                show_nsfw = request['show_nsfw']
                scopes = request['scopes']

                try:
                    if selected_tags:
                        tagger = await DanbooruTagger.get_instance()
                        result = await tagger.get_selection_recommendations_async(
                            selected_tags,
                            show_nsfw,
                            scopes,
                            related_limit=50,
                            artist_limit=ARTIST_REC_LIMIT,
                            artist_min_cooc=3,
                        )
                    else:
                        result = {
                            'related': [],
                            'groups': [],
                            'artists': [],
                            'artist_top_tags': {},
                        }
                except Exception as exc:
                    print(f'[UI] 推荐刷新失败: {exc}', flush=True)
                    continue

                if request['generation'] != self._recommendation_generation:
                    continue
                if not self._client_alive():
                    return

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
        finally:
            self._recommendation_task = None

    def _set_artist_rec_page(self, page: int):
        """切换推荐画师页，只构建当前页的可见行。"""
        if self._artist_rec_page_count < 1:
            return
        self._artist_rec_page = max(1, min(page, self._artist_rec_page_count))
        self._render_artist_rec_page()
        if self._artist_rec_page_label is not None:
            self._artist_rec_page_label.text = (
                f'{self._artist_rec_page} / {self._artist_rec_page_count}'
            )
        if self._artist_rec_prev_button is not None:
            if self._artist_rec_page == 1:
                self._artist_rec_prev_button.disable()
            else:
                self._artist_rec_prev_button.enable()
        if self._artist_rec_next_button is not None:
            if self._artist_rec_page == self._artist_rec_page_count:
                self._artist_rec_next_button.disable()
            else:
                self._artist_rec_next_button.enable()

    def _render_artist_rec_page(self):
        """重建当前画师页，节点数量固定不超过 ARTIST_REC_PAGE_SIZE。"""
        if self.artist_rec_list is None:
            return
        self.artist_rec_list.clear()
        self._artist_rec_checkboxes.clear()
        self._artist_rec_rows.clear()

        if not self._artist_rec_results:
            with self.artist_rec_list:
                ui.label('暂无推荐画师').classes('text-sm text-gray-400 italic p-4')
            return

        selected_now = set(self._get_selected_tags())
        start = (self._artist_rec_page - 1) * ARTIST_REC_PAGE_SIZE
        end = start + ARTIST_REC_PAGE_SIZE
        page_results = self._artist_rec_results[start:end]

        with self.artist_rec_list:
            for r in page_results:
                artist = r.artist
                is_selected = artist in selected_now
                # 归一化：除以命中标签数，cap 到 100%
                normalized = min(r.score / max(r.hit_count, 1), 1.0)
                score_pct = f'+{normalized * 100:.0f}%'
                reason = artist_candidate_reason(r.sources)
                post_str = f'{r.post_count:,}' if r.post_count else '—'

                # tooltip：画师擅长标签
                tag_list = self._artist_rec_top_tags.get(artist, [])
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
                    self._artist_rec_rows.append(row)
                    # tooltip
                    with ui.tooltip().props('content-class="bg-black text-white shadow-4" max-width="400px"'):
                        ui.html(tooltip_html).style('font-size:14px;line-height:1.5;max-width:380px;')

                    # Checkbox 单元格
                    with ui.element('div').classes(
                        'recommendation-cell flex-none justify-center px-2 py-2'
                    ):
                        cb = ui.checkbox(
                            '', value=is_selected,
                            on_change=lambda e, t=artist: self._on_artist_rec_checkbox_change(t, e.value)
                        ).props('dense')
                        self._artist_rec_checkboxes[artist] = cb

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

    def _render_artist_rec(self, artist_results, top_tags=None, show_nsfw: bool = True):
        """保存推荐快照，并仅渲染当前画师页。"""
        if self.artist_rec_list is None or self.artist_rec_pagination is None:
            return
        self.artist_rec_list.clear()
        self.artist_rec_pagination.clear()
        self._artist_rec_checkboxes.clear()
        self._artist_rec_rows.clear()
        self._artist_rec_page = 1
        self._artist_rec_page_label = None
        self._artist_rec_prev_button = None
        self._artist_rec_next_button = None
        self._artist_rec_results = list(artist_results[:ARTIST_REC_LIMIT])
        self._artist_rec_top_tags = dict(top_tags or {})
        self._artist_rec_show_nsfw = show_nsfw
        self._artist_rec_page_count = (
            len(self._artist_rec_results) + ARTIST_REC_PAGE_SIZE - 1
        ) // ARTIST_REC_PAGE_SIZE
        self._current_artist_rec_tags = {
            result.artist for result in self._artist_rec_results
        }
        self._artist_rec_sources = {
            result.artist: '、'.join(result.sources[:3])
            for result in self._artist_rec_results
        }

        if not self._artist_rec_results:
            self._render_artist_rec_page()
            return

        if self._artist_rec_page_count > 1:
            with self.artist_rec_pagination:
                with ui.row().classes('w-full items-center justify-center gap-2 px-3 py-2'):
                    self._artist_rec_prev_button = ui.button(
                        '‹',
                        on_click=lambda: self._set_artist_rec_page(self._artist_rec_page - 1),
                    ).props('flat dense round color=grey-7')
                    self._artist_rec_page_label = ui.label().classes('text-xs text-gray-600 min-w-12 text-center')
                    self._artist_rec_next_button = ui.button(
                        '›',
                        on_click=lambda: self._set_artist_rec_page(self._artist_rec_page + 1),
                    ).props('flat dense round color=grey-7')

        self._set_artist_rec_page(1)

    def _render_group_expansion(self, group_data: list, selected_tags: list[str], show_nsfw: bool):
        """渲染 Group 同类扩展区域。"""
        if self.group_expansion_container is None:
            return
        self.group_expansion_container.clear()
        self._group_checkboxes.clear()
        self._group_candidate_sources.clear()
        group_key = _group_names_key(group_data)
        if group_key != self._group_render_key:
            self._group_render_key = group_key
            self._group_render_limits.clear()
            self._group_expanded_names.clear()
            self._group_scroll_positions.clear()

        if not group_data:
            with self.group_expansion_container:
                ui.label('已选标签无分组信息').classes('text-sm text-gray-400 italic p-2')
            return

        # 行背景色按分类区分（与关联推荐一致）
        CAT_BG = {
            'General':   'background-color: rgba(59,130,246,0.06);',
            'Character': 'background-color: rgba(34,197,94,0.06);',
            'Copyright': 'background-color: rgba(168,85,247,0.06);',
        }
        CAT_LABEL = {'General': '通用', 'Character': '角色', 'Copyright': '作品'}

        selected_now = set(self._get_selected_tags())

        with self.group_expansion_container:
            for group_info in group_data:
                group_name = group_info['group']
                group_cn = group_info.get('group_cn_name', group_name.replace('tag_group:', ''))
                group_sources = list(group_info.get('sources') or [])
                group_reason = tag_group_candidate_reason(group_cn, group_sources)
                group_source_detail = group_cn
                if group_sources:
                    group_source_detail += f"；触发标签：{'、'.join(group_sources[:3])}"
                tags = group_info['tags']
                visible_limit = self._group_render_limits.get(group_name, GROUP_RENDER_TAG_LIMIT)
                visible_tags, hidden_count = _limit_group_render_tags(tags, visible_limit)
                scroll_id = _group_scroll_dom_id(group_name)

                expansion = ui.expansion(
                    f'{group_cn} ({len(tags)} 个标签)',
                    icon='label',
                    value=_should_group_start_expanded(group_name, self._group_expanded_names),
                ).classes('w-full').props('dense')
                expansion.on(
                    'update:model-value',
                    lambda e, g=group_name: self._on_group_expansion_change(g, e),
                )
                with expansion:
                    with ui.element('div').props(
                        f'id="{scroll_id}" data-danbooru-group-scroll="1"'
                    ).classes('w-full grid grid-cols-2 gap-1 p-1').style('max-height: 600px; overflow-y: auto;'):
                        for t in visible_tags:
                            tag = t['tag']
                            self._group_candidate_sources.setdefault(tag, group_source_detail)
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
                                    on_change=lambda e, t=tag: self._on_group_checkbox_change(t, e.value),
                                ).props('dense').classes('flex-none')
                                self._group_checkboxes[tag] = cb

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
                                await self._load_more_group_tags(g, total, gd, st, sn)

                            ui.button(
                                f'加载更多（剩余 {hidden_count} 个）',
                                icon='expand_more',
                                on_click=_load_more,
                            ).props('dense flat color=primary').classes('col-span-2 text-xs')
        self._restore_group_scroll_positions()

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
        cols = list(TABLE_COLUMNS)
        if self.sw_semantic and self.sw_semantic.value:
            cols.append(OPTIONAL_COLS['semantic'])
        if self.sw_layer and self.sw_layer.value:
            cols.append(OPTIONAL_COLS['layer'])
        if self.sw_source and self.sw_source.value:
            cols.append(OPTIONAL_COLS['source'])
        self.result_table.columns = cols

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

        with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
            ui.label('反馈搜索问题').classes('text-base font-bold text-gray-800')
            ui.label(f'当前搜索词：{query}').classes('text-sm text-gray-600')
            ui.label(UI_TEXT['dialogs']['search_feedback_privacy']).classes(
                'text-xs text-slate-500 bg-slate-50 rounded p-2'
            )
            detail_input = ui.textarea(
                label='具体问题（可选）',
                placeholder='例如：结果偏题、缺少某个关键标签、召回了不相关角色/作品...',
            ).props('outlined autogrow maxlength=500 counter').classes('w-full')

            async def submit_feedback():
                detail = (detail_input.value or '').strip()

                submit_btn.disable()
                try:
                    await telemetry.add_feedback(
                        feedback_type='search_bad_case',
                        query=query,
                        search_settings=self._feedback_settings(),
                        app_version=_get_git_commit(),
                        platform=PLATFORM,
                        details=detail,
                    )
                    if self.bad_case_btn is not None:
                        self.bad_case_btn.disable()
                    dialog.close()
                    ui.notify('感谢反馈！我们会持续优化。', type='positive', timeout=3000)
                except Exception as e:
                    print(f'[UI] bad_case 记录异常: {e}')
                    submit_btn.enable()
                    ui.notify('记录失败，请稍后再试。', type='warning', timeout=3000)

            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('取消', on_click=dialog.close).props('flat color=grey-7')
                submit_btn = ui.button('提交反馈', on_click=submit_feedback).props('unelevated color=primary')
        dialog.open()

    def report_translation_error(self, e):
        from platform_utils import PLATFORM
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

        query = self.current_query_str.strip()
        current_cn_first = current_cn_name.split(',', 1)[0].strip()
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
            ui.label('反馈翻译错误').classes('text-base font-bold text-gray-800')
            ui.label(f'词条：{tag}').classes('text-sm font-mono text-gray-700')
            ui.label(f'当前中文名：{current_cn_first or current_cn_name or "（空）"}').classes('text-sm text-gray-600')
            ui.label(UI_TEXT['dialogs']['translation_feedback_privacy']).classes(
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

            async def submit_feedback():
                suggested = (suggested_input.value or '').strip()
                detail = (detail_input.value or '').strip()

                submit_btn.disable()
                try:
                    await telemetry.add_feedback(
                        feedback_type='translation_error',
                        query=query,
                        search_settings=self._feedback_settings(),
                        app_version=_get_git_commit(),
                        platform=PLATFORM,
                        details=detail,
                        tag=tag,
                        current_cn_name=current_cn_name,
                        suggested_cn_name=suggested,
                        category=str(row.get('category') or ''),
                    )
                    dialog.close()
                    ui.notify('感谢反馈！这条翻译问题已记录。', type='positive', timeout=3000)
                except Exception as e:
                    print(f'[UI] translation_error 记录异常: {e}')
                    submit_btn.enable()
                    ui.notify('记录失败，请稍后再试。', type='warning', timeout=3000)

            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('取消', on_click=dialog.close).props('flat color=grey-7')
                submit_btn = ui.button('提交反馈', on_click=submit_feedback).props('unelevated color=primary')
        dialog.open()

# ── 页面路由 ───────────────────────────────────────────────────────────────────

@ui.page('/')
async def main_page():
    client = ui.context.client
    client_id = client.id
    app_ui = DanbooruSearchUI()

    def mark_connected(*_):
        _mark_ui_session_active(client_id)
        app_ui._start_service_status_task()
        # 页面先完成构建；localStorage 恢复只在 Socket.IO 真正连接后后台执行。
        app_ui._start_storage_restore_task()

    def mark_disconnected(*_):
        _mark_ui_session_inactive(client_id)
        app_ui._pause_storage_restore()

    def mark_deleted(*_):
        _mark_ui_session_inactive(client_id)
        app_ui._dispose()

    client.on_connect(mark_connected)
    client.on_disconnect(mark_disconnected)
    on_delete = getattr(client, 'on_delete', None)
    if callable(on_delete):
        on_delete(mark_deleted)

    app_ui.build_page()

    async def silent_visit_update():
        try:
            await counter.increment_visit()
            await telemetry.increment("ui_visit")
            app_ui._update_footer_text()
        except Exception:
            pass
    asyncio.create_task(silent_visit_update())

# ── 入口 ───────────────────────────────────────────────────────────────────────

if __name__ in {'__main__', '__mp_main__'}:
    host, port = get_host_port()

    @app.on_startup
    def _warmup():
        async def background_init_tasks():
            await asyncio.sleep(5)
            print("[UI] 开始预热计数器与引擎", flush=True)
            await counter.init()
            await telemetry.init()
            cold_start_started_at = time.perf_counter()
            await telemetry.increment("engine_cold_start_attempt")
            try:
                await DanbooruTagger.get_instance()
            except Exception:
                await telemetry.increment("engine_cold_start_failure")
                await telemetry.record_timing(
                    "engine_cold_start",
                    (time.perf_counter() - cold_start_started_at) * 1000,
                )
                raise
            await telemetry.increment("engine_cold_start_success")
            await telemetry.record_timing(
                "engine_cold_start",
                (time.perf_counter() - cold_start_started_at) * 1000,
            )
            print("[UI] 后台预热全部完成！", flush=True)
        asyncio.create_task(background_init_tasks())

    @app.on_shutdown
    def _shutdown():
        async def force_sync_all():
            await asyncio.gather(counter.force_sync(), telemetry.force_sync())

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(force_sync_all())
            else:
                asyncio.run(force_sync_all())
        except Exception as e:
            print(f"[UI] 关机同步失败: {e}")

    app.mount('/api', api_app)

    mcp_app = mcp.streamable_http_app()
    app.mount('/mcp', mcp_app)
    _mcp_lifespan_ctx = None
    @app.on_startup
    async def _start_mcp():
        global _mcp_lifespan_ctx
        _mcp_lifespan_ctx = mcp_app.router.lifespan_context(mcp_app)
        await _mcp_lifespan_ctx.__aenter__()
    @app.on_shutdown
    async def _stop_mcp():
        global _mcp_lifespan_ctx
        if _mcp_lifespan_ctx is not None:
            await _mcp_lifespan_ctx.__aexit__(None, None, None)


    @app.get('/googlebd34b54f8562aa06.html')
    def google_verification():
        return PlainTextResponse('google-site-verification: googlebd34b54f8562aa06.html')

    @app.get('/robots.txt')
    def robots_txt():
        content = (
            'User-agent: *\n'
            'Allow: /$\n'
            'Disallow: /api/\n'
            'Disallow: /_nicegui/\n'
            'Disallow: /socket.io/\n'
        )
        return PlainTextResponse(content)


    @app.get('/robots.txt')
    def robots_txt():
        content = (
            'User-agent: *\n'
            'Allow: /$\n'
            'Disallow: /api/\n'
            'Disallow: /_nicegui/\n'
            'Disallow: /socket.io/\n'
        )
        return PlainTextResponse(content)

    @app.head('/')
    async def head_root():
        return PlainTextResponse('')


    ui.run(
        host=host,
        port=port,
        title='Danbooru Tags Searcher',
        reload=not is_cloud(),
        show=not is_cloud(),
        reconnect_timeout=120,
    )
