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
import os
import re
import json as _json
import subprocess
import traceback
from dataclasses import asdict
from fastapi.responses import PlainTextResponse

def _excepthook(exc_type, exc_value, exc_tb):
    print("[UI] FATAL ERROR ON STARTUP:", flush=True)
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _excepthook

from nicegui import ui, app, run
from core import counter
from api_fastapi import app as api_app
from core.engine import DanbooruTagger
from core.models import RelatedTag, SearchRequest
from platform_utils import is_cloud, get_host_port, nsfw_allowed
from mcp_server import mcp

import logging
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
]

OPTIONAL_COLS = {
    'semantic': {'name': 'semantic_score', 'label': '语义分',   'field': 'semantic_score', 'sortable': True},
    'layer':    {'name': 'layer',          'label': '匹配层',   'field': 'layer'},
    'source':   {'name': 'source',         'label': '匹配来源', 'field': 'source'},
}

# localStorage key 与配置版本，版本变更时自动丢弃旧配置
_CONFIG_LS_KEY = 'danbooru_search_config'
_CONFIG_VERSION = 6

SPONSOR_IMAGE_URL = "https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202501120027592.png"
SPONSOR_TOOLCHAIN_URL = "http://intro.sakizuki.site/index.html"
SPONSOR_NOTICE_TEXT = "喜欢的话，可以请作者喝杯咖啡"
SPONSOR_TITLE = "谢谢你愿意支持"
SPONSOR_BODY = (
    "如果你喜欢这个小工具，可以自愿赞赏一点维护成本。"
    "它会继续免费开放；赞赏不会解锁任何额外功能、不会提高额度、不会影响搜索结果，"
    "量力而行就好，未成年人请勿赞赏。"
)


def _resolve_group_render_limit(default: int = 80) -> int:
    raw = os.environ.get('DANBOORU_GROUP_RENDER_LIMIT')
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


GROUP_RENDER_TAG_LIMIT = _resolve_group_render_limit()

# 搜索模式预设
_SEARCH_MODE_PRESETS: dict[str, dict] = {
    '精确查词': {'top_k': 20, 'limit': 10, 'popularity_weight': 0.15, 'use_segmentation': False, 'group_mode': 'off', 'max_per_group': 2},
    '概念扩展': {'top_k': 80, 'limit': 80, 'popularity_weight': 0.15, 'use_segmentation': True,  'group_mode': 'expand', 'max_per_group': 2},
    '描述查词': {'top_k': 20, 'limit': 20, 'popularity_weight': 0.15, 'use_segmentation': False, 'group_mode': 'off', 'max_per_group': 2},
    '完整场景': {'top_k': 5,  'limit': 80, 'popularity_weight': 0.15, 'use_segmentation': True,  'group_mode': 'diverse', 'max_per_group': 2},
}
_SEARCH_MODE_OPTIONS = ['自定义'] + list(_SEARCH_MODE_PRESETS.keys())


# ── 辅助函数 ───────────────────────────────────────────────────────────────────

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
        self.current_search_interacted = True

        self.full_table_data: list[dict] = []
        self.current_segments: list[str] = []   # 从句级原始片段，用于区分 chip 颜色
        self.current_filter_keyword: str = 'ALL'  # 当前选中的分词筛选 keyword（NSFW 切换时复用）
        self.current_query_str: str = ""
        self.full_tags_str: str = ""
        self.full_tags_str_sfw: str = ""

        self.result_table = None           # 左栏表格
        self.related_list_container = None  # 右栏关联推荐列表
        self.group_expansion_container = None  # 左栏 Group 同类扩展（表格下方）
        self.client = None
        self._group_render_limits: dict[str, int] = {}
        self._group_expanded_names: set[str] = set()
        self._group_scroll_positions: dict[str, int] = {}
        self._group_render_key: tuple[str, ...] = ()
        self.results_section = None        # 整个结果区域（搜索前隐藏）
        self.selection_count_label = None
        self.selected_display = None       # 已废弃 textarea，保留兼容
        self.selected_chips_container = None  # 已选标签 chip 容器
        self.current_related: list = []
        self.chip_extra_selected: set = set()
        # 去抖任务句柄（取消旧任务避免 CPU 洪峰）
        self._debounce_related_task = None  # type: asyncio.Task | None
        self._debounce_group_task = None    # type: asyncio.Task | None
        self._debounce_artist_task = None   # type: asyncio.Task | None

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

        self.mcp_notice = None
        self.notice_expansion = None
        self.sponsor_dialog = None

        # 表格显示选项开关
        self.sw_semantic = None
        self.sw_layer = None
        self.sw_source = None

        # 关联推荐的 checkbox 引用
        self._related_checkboxes: dict[str, ui.checkbox] = {}
        # 同类标签的 checkbox 引用
        self._group_checkboxes: dict[str, ui.checkbox] = {}
        # 推荐画师的 checkbox 引用
        self._artist_rec_checkboxes: dict[str, ui.checkbox] = {}
        # 当前推荐画师的标签名集合（用于 Anima 模式复制时加 @ 前缀）
        self._current_artist_rec_tags: set[str] = set()
        self._artist_result_tags: set[str] = set()
        self._last_recommendation_seed_tags: list[str] = []

        # 高级选项中各层/类型的 checkbox 引用，用于 restore 时同步控件状态
        self._layer_checkboxes: dict[str, ui.checkbox] = {}
        self._cat_checkboxes: dict[str, ui.checkbox] = {}

    def _update_footer_text(self):
        if self.search_count_label is not None:
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

    def _build_sponsor_dialog(self):
        with ui.dialog() as self.sponsor_dialog, ui.card().classes('w-full max-w-sm'):
            with ui.column().classes('w-full items-center gap-2 text-center'):
                ui.label(SPONSOR_TITLE).classes('text-base font-bold text-gray-800')
                ui.label(SPONSOR_BODY).classes('text-sm text-gray-600 leading-relaxed')
                ui.image(SPONSOR_IMAGE_URL).classes('w-60 max-w-full rounded border border-gray-200')
                ui.label('微信赞赏码').classes('text-xs text-gray-400')
                ui.link(
                    '如果你想继续折腾，也可以看看基于这个小工具的更多工具链',
                    SPONSOR_TOOLCHAIN_URL,
                    new_tab=True,
                ).classes('text-xs text-blue-500 hover:text-blue-700 hover:underline')
            with ui.row().classes('w-full justify-end'):
                ui.button('关闭', on_click=self.sponsor_dialog.close).props('flat color=grey-7')

    def _mark_interaction(self, e=None):
        if not self.current_search_interacted:
            self.current_search_interacted = True

            async def silent_success_update():
                try:
                    await counter.increment_success()
                except Exception:
                    pass
            asyncio.create_task(silent_success_update())

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

    def _save_config(self):
        """将当前控件状态序列化并写入 localStorage。"""
        cfg = {
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
            'notice_expanded': bool(self.notice_expansion.value) if self.notice_expansion else True,
            'mcp_notice_dismissed': not bool(self.mcp_notice.visible) if self.mcp_notice else False,
            'search_mode': self.input_search_mode.value if self.input_search_mode else '自定义',
            'group_mode': self.input_group_mode.value if self.input_group_mode else 'off',
            'max_per_group': int(self.input_max_per_group.value) if self.input_max_per_group else 2,
        }
        js = _json.dumps(cfg, ensure_ascii=False)
        ui.run_javascript(f"localStorage.setItem('{_CONFIG_LS_KEY}', {_json.dumps(js)});")

    async def _restore_config(self):
        """从 localStorage 读取配置并恢复控件状态。"""
        try:
            if getattr(ui.context.client, '_deleted', False):
                return
            raw = await ui.run_javascript(
                f"localStorage.getItem('{_CONFIG_LS_KEY}');",
                timeout=5.0,
            )
        except Exception:
            return

        if not raw:
            return

        try:
            cfg = _json.loads(raw)
        except Exception:
            return

        if cfg.get('version') != _CONFIG_VERSION:
            # 版本不符，丢弃旧配置
            ui.run_javascript(f"localStorage.removeItem('{_CONFIG_LS_KEY}');")
            return

        # 恢复搜索模式（会触发预设填充，但 _applying_preset 防止联动覆盖）
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

        # NSFW：仅在平台允许时恢复
        if nsfw_allowed() and self.input_nsfw and 'show_nsfw' in cfg:
            self.input_nsfw.set_value(cfg['show_nsfw'])

        if 'selected_layers' in cfg:
            for layer, val in cfg['selected_layers'].items():
                if layer in self.selected_layers:
                    self.selected_layers[layer] = bool(val)
                    if layer in self._layer_checkboxes:
                        self._layer_checkboxes[layer].set_value(bool(val))

        if 'selected_cats' in cfg:
            for cat, val in cfg['selected_cats'].items():
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

        if 'prompt_format' in cfg and cfg['prompt_format'] in ('sdxl', 'nai', 'anima'):
            self.prompt_format = cfg['prompt_format']
            if self.format_toggle_btn:
                if self.prompt_format == 'nai':
                    self.format_toggle_btn.text = 'NAI'
                    self.format_toggle_btn.props('color=purple-7')
                elif self.prompt_format == 'anima':
                    self.format_toggle_btn.text = 'Anima'
                    self.format_toggle_btn.props('color=teal-7')
                else:
                    self.format_toggle_btn.text = 'SDXL'
                    self.format_toggle_btn.props('color=grey-7')

        if 'rows_per_page' in cfg:
            self._set_rows_per_page(cfg['rows_per_page'])

        if self.search_input and cfg.get('search_query'):
            self.search_input.set_value(cfg['search_query'])

        if self.notice_expansion and 'notice_expanded' in cfg:
            self.notice_expansion.set_value(cfg['notice_expanded'])

        if self.mcp_notice and cfg.get('mcp_notice_dismissed'):
            self.mcp_notice.set_visibility(False)

        # 若高级选项列有变更，同步更新表格列
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

        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-4'):

            # ── 初始化提示 ──
            self.init_banner = ui.card().classes(
                'w-full bg-blue-50 border-l-4 border-blue-400'
            )
            with self.init_banner:
                with ui.row().classes('items-center gap-3 p-2'):
                    ui.spinner(size='sm')
                    ui.label('引擎初始化中，请稍候…约需 5~10 分钟').classes('text-sm text-blue-700')
                from platform_utils import PLATFORM
                _alt_url = (
                    'https://www.modelscope.cn/studios/SAkizuki/DanbooruSearchOnline'
                    if PLATFORM == 'hf' else
                    'https://huggingface.co/spaces/SAkizuki/DanbooruSearch'
                )
                ui.html(
                    f'初始化期间，您可以使用'
                    f'<a href="{_alt_url}" target="_blank" rel="noopener noreferrer" '
                    f'class="text-blue-600 hover:text-blue-800 underline font-bold">备用服务</a>'
                    f'，或先查看'
                    f'<a href="http://intro.sakizuki.site/index.html" '
                    f'target="_blank" rel="noopener noreferrer" '
                    f'class="text-blue-600 hover:text-blue-800 underline font-bold">使用指南</a>'
                ).classes('text-xs text-blue-600 px-6 pb-3')
            self.init_banner.set_visibility(not DanbooruTagger.is_ready())
            if not DanbooruTagger.is_ready():
                asyncio.ensure_future(self._hide_banner_when_ready())

            # ── 0. 公告栏（同类标签 + MCP）──
            self._build_group_notice()

            # ── 1. 注意事项 ──
            self._build_notice()

            # ── 2. 搜索卡片 ──
            self._build_search_card()

            # ── 3~5. 结果区域（搜索前隐藏）──
            self.results_section = ui.column().classes('w-full gap-4')
            self.results_section.set_visibility(False)

            with self.results_section:
                # ── 3. 已选标签栏 ──
                self._build_selection_bar()

                # ── 4. 分词筛选 chips ──
                self.keywords_container = ui.row().classes('gap-2 items-center flex-wrap')

                # ── 5. 两栏结果 ──
                self._build_results_columns()

            # ── 6. 底部 ──
            with ui.element('div').classes('w-full text-center py-4 mt-2'):
                self.search_count_label = ui.html('正在加载数据...').classes('text-xs text-gray-400')
                self._update_footer_text()
                ui.button(SPONSOR_NOTICE_TEXT, on_click=self.sponsor_dialog.open) \
                    .props('flat dense no-caps color=grey-6') \
                    .classes('text-xs mt-1')

    # ── 公告栏（标签组 + MCP）─────────────────────────────────────────────

    def _build_group_notice(self):
        self.mcp_notice = ui.card().classes(
            'w-full bg-green-50 border-l-4 border-green-500 p-0 overflow-hidden'
        )
        with self.mcp_notice:
            with ui.column().classes('px-4 py-3 w-full gap-2'):
                # ── 画师查找公告 ──
                with ui.row().classes('items-center justify-between w-full'):
                    with ui.row().classes('items-center gap-1'):
                        ui.label('🧪 新功能：推荐擅长画师（beta）').classes('text-sm font-bold text-green-800')
                    ui.button(icon='close').props('flat dense round color=grey-6') \
                        .on_click(self._dismiss_mcp_notice)
                ui.html(
                    '基于标签共现数据，根据已选标签查找对应的擅长画师。'
                    '鼠标悬停画师行可查看该画师最常画的标签'
                ).classes('text-xs text-green-900')
                ui.separator().classes('my-1')
                # ── 标签组扩展 ──
                ui.html(
                    '【标签组扩展】 勾选标签后，搜索结果下方会出现<b>同类标签</b>区域，'
                    '展示已选标签所属分组中的其他标签，勾选即可加入已选。'
                ).classes('text-xs text-green-900')
                ui.separator().classes('my-1')
                # ── MCP 服务 ──
                ui.html(
                    '【MCP 服务】 支持通过 MCP 协议接入 AI Agent（如 Claude Desktop）。'
                    '免配置托管版体验：'
                    '<a href="https://huggingface.co/spaces/SAkizuki/WenQiuYue" '
                    'target="_blank" rel="noopener noreferrer" '
                    'class="text-green-700 font-bold underline">问秋月 Space</a>，'
                    '<span class="text-gray-500 ml-1">API 额度有限，仅供体验。</span>'
                    '&nbsp;'
                    '<a href="https://github.com/SuzumiyaAkizuki/DanbooruSearchOnline#mcp-接口" '
                    'target="_blank" rel="noopener noreferrer" '
                    'class="text-green-700 underline">接入文档 →</a>'
                ).classes('text-xs text-green-900')

    def _dismiss_mcp_notice(self):
        if self.mcp_notice:
            self.mcp_notice.set_visibility(False)
        self._save_config()

    # ── 注意事项 ──────────────────────────────────────────────────────────

    def _build_notice(self):
        with ui.card().classes('w-full bg-orange-50 border-l-4 border-orange-500 p-0 overflow-hidden'):
            with ui.expansion(value=True).classes('w-full') as notice_expansion:
                self.notice_expansion = notice_expansion
                notice_expansion.on('update:model-value', lambda _: self._save_config())
                notice_expansion.add_slot('header', '''
                    <div class="flex items-center gap-2 px-4 py-2 w-full flex-wrap">
                        <span class="text-base font-bold text-orange-800">⚠️ 注意事项 / Note</span>
                        <span v-if="!props.expanded" class="text-sm text-gray-600 ml-1">
                             如果觉得好用，请点击顶部给本 Space 点个
                            <strong>Like ❤️</strong>，或前往 GitHub 点个 <strong>Star ⭐</strong>！
                        </span>
                    </div>
                ''')
                ui.markdown("""
- **AI 辅助**：基于语义匹配，结果未必绝对准确(Results may contain errors)
- **内容警告**：查找结果可能包含 NSFW 内容 (May include NSFW content)
- **检索限制**：仅支持中/英双语查找 ，更推荐中文(CN/EN only,CN is preferred)
- **标签范围**：仅显示特征、角色与作品标签，且频数须 ≥ 100 (General, Character & Copyright only, Freq ≥ 100)
- **集成与接口**：[ComfyUI 插件](https://github.com/SuzumiyaAkizuki/ComfyUI-DanbooruSearcher) · [API 文档](/api/docs) · [MCP 接入](https://github.com/SuzumiyaAkizuki/DanbooruSearchOnline#mcp-接口)
- **支持作者**：如果觉得好用，欢迎点击顶部给本 Space 点个 **Like ❤️**，或前往 [GitHub](https://github.com/SuzumiyaAkizuki/DanbooruSearchOnline) 点个 **Star ⭐**！也可以自愿赞赏一点维护成本；不影响任何功能使用。
- **🚀 首次使用？[点击查看使用指南](http://intro.sakizuki.site/index.html)**，了解五种搜索模式与进阶技巧
""").classes('text-sm text-gray-800 px-4 pb-3')

    # ── 搜索卡片 ─────────────────────────────────────────────────────────

    def _build_search_card(self):
        with ui.card().classes('w-full'):
            with ui.row().classes('items-center gap-2 mb-2'):
                ui.icon('search', size='2em', color='primary')
                ui.label('Danbooru 标签模糊搜索').classes('text-2xl font-bold text-gray-800')
            ui.label(
                '基于语义匹配的标签搜索引擎，支持多维匹配与共现关联推荐。'
            ).classes('text-sm text-gray-500 -mt-1 mb-1')
            ui.html(
                '<a href="http://intro.sakizuki.site/index.html" '
                'target="_blank" rel="noopener noreferrer" '
                'class="text-blue-600 hover:text-blue-800 underline font-medium">'
                '查看工具链介绍 / 使用指南 →</a>'
            ).classes('text-sm mb-3')

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

                with ui.row().classes('items-center gap-2'):
                    ui.label('热度权重').classes('text-sm text-gray-600')
                    self.input_weight = ui.slider(min=0.0, max=1.0, value=0.15, step=0.05).classes('w-32')
                    ui.label().bind_text_from(self.input_weight, 'value', lambda v: f"{v:.2f}") \
                        .classes('text-sm font-mono text-gray-700 w-8')
                    self.input_weight.on('update:model-value', self._on_param_changed)

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

    # ── 已选标签栏 ────────────────────────────────────────────────────────

    def _build_selection_bar(self):
        self.selection_bar_card = ui.card().classes('w-full bg-blue-50 border border-blue-200')
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
                'w-full mt-2 min-h-10 p-1 rounded bg-white border border-blue-100 flex flex-wrap'
            )

    def _render_selected_chips(self):
        """重新渲染已选标签的 chip 列表。"""
        if self.selected_chips_container is None:
            return
        self.selected_chips_container.clear()
        tags = self._get_selected_tags()
        if not tags:
            with self.selected_chips_container:
                ui.label('暂无已选标签').classes('text-xs text-gray-400 italic p-2 self-center')
            return
        with self.selected_chips_container:
            step = 0.5 if self.prompt_format == 'anima' else 0.1
            for tag in tags:
                w = self.tag_weights.get(tag, 1.0)
                extra_cls = 'boosted' if w > 1.0 else ('reduced' if w < 1.0 else '')
                w_str = f'{w:.1f}'
                display_label = _format_selected_tag_label(tag, self._get_cn_name_for_tag(tag))
                with ui.element('div').classes(f'weight-chip {extra_cls}'):
                    # 删除按钮（×）
                    with ui.element('button').classes('weight-btn').props(f'title="移除 {tag}"').on(
                        'click', lambda t=tag: self._remove_selected_tag(t)
                    ):
                        ui.html('&times;')
                    # 减号
                    with ui.element('button').classes('weight-btn').on(
                        'click', lambda t=tag, s=step: self._adjust_weight(t, -s)
                    ):
                        ui.html('&minus;')
                    # 标签名
                    ui.label(display_label).style(
                        'font-family:Consolas,Monaco,monospace;font-size:12px;'
                        'color:#2c5282;max-width:240px;overflow:hidden;'
                        'text-overflow:ellipsis;white-space:nowrap;'
                    )
                    # 权重值（仅非 1.0 时显示）
                    if w != 1.0:
                        ui.label(w_str).classes('weight-label').style('color:#e65100;font-weight:bold;')
                    # 加号
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

    _STAGED_LS_KEY = 'danbooru_staged_tags'

    def _save_staged_tags(self):
        """将已选标签及其权重保存到 localStorage。"""
        tags = self._get_selected_tags()
        weights = {t: self.tag_weights.get(t, 1.0) for t in tags}
        data = _json.dumps({'tags': tags, 'weights': weights}, ensure_ascii=False)
        try:
            if getattr(ui.context.client, '_deleted', False):
                return
            ui.run_javascript(f"localStorage.setItem('{self._STAGED_LS_KEY}', {_json.dumps(data)});")
        except RuntimeError:
            pass  # 事件上下文已销毁（UI 重建中），数据仍在内存里，下次保存时会同步

    async def _restore_staged_tags(self):
        """从 localStorage 恢复已选标签。"""
        try:
            if getattr(ui.context.client, '_deleted', False):
                return
            raw = await ui.run_javascript(
                f"localStorage.getItem('{self._STAGED_LS_KEY}');",
                timeout=5.0,
            )
        except Exception:
            return
        if not raw:
            return
        try:
            data = _json.loads(raw)
        except Exception:
            return
        tags = data.get('tags', [])
        weights = data.get('weights', {})
        if not tags:
            return
        self.chip_extra_selected.update(tags)
        for t in tags:
            self.tag_weights[t] = weights.get(t, 1.0)
        self._render_selected_chips()
        if self.selection_count_label is not None:
            self.selection_count_label.text = str(len(tags))

    def _clear_all_staged(self):
        """清空所有已选标签。"""
        self._mark_interaction()
        self.chip_extra_selected.clear()
        self.tag_weights.clear()
        if self.result_table is not None:
            self.result_table.selected = []
        self._artist_rec_checkboxes.clear()
        self._current_artist_rec_tags.clear()
        self._artist_result_tags.clear()
        self._last_recommendation_seed_tags = []
        self._render_selected_chips()
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
            with ui.card().classes('col-left'):
                with ui.row().classes('items-center justify-between mb-2 w-full'):
                    ui.label('匹配标签结果').classes('font-bold text-lg text-gray-800')
                    ui.button('复制全部标签', icon='content_copy', on_click=self._copy_all_tags) \
                        .props('dense flat color=primary').classes('text-sm')

                self.result_table = ui.table(
                    columns=TABLE_COLUMNS,
                    rows=[],
                    pagination=0,
                    selection='multiple',
                    row_key='tag',
                ).classes('w-full')
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
            with ui.card().classes('col-right'):
                # 推荐画师
                with ui.row().classes('items-center justify-between w-full mb-2'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label('推荐擅长画师(Beta)').classes('font-bold text-lg text-gray-800')
                        with ui.icon('info_outline', size='sm', color='grey').classes('cursor-help'):
                            with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                                ui.html(
                                    '基于标签-画师 NPMI 共现数据，根据您当前已选的标签，推荐擅长这些元素的画师。<br>悬停画师行可查看与该画师共现关联最强的标签。').style(
                                    'font-size:14px;line-height:1.5;')

                self.artist_rec_list = ui.column().classes('w-full gap-0').style('max-height: 420px; overflow-y: auto;')
                with self.artist_rec_list:
                    ui.label('请先搜索并勾选标签…').classes('text-sm text-gray-400 italic p-4')

                ui.separator().classes('my-3')

                # 关联推荐
                with ui.row().classes('items-center justify-between w-full mb-2'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label('关联推荐').classes('font-bold text-lg text-gray-800')
                        with ui.icon('info_outline', size='sm', color='grey').classes('cursor-help'):
                            with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                                ui.html(
                                    '基于标签共现数据，发掘语义之外的相关性，为您推荐更多可能的标签。<br>勾选可加入或移出已选。如需根据最新选项更新推荐，请点击刷新按钮。').style(
                                    'font-size:14px;line-height:1.5;')

                    # 新增手动刷新按钮
                    ui.button('根据已选刷新', icon='refresh', on_click=self._manual_refresh_related) \
                        .props('dense flat color=primary').classes('text-sm')

                self.related_list_container = ui.column().classes('w-full gap-0')
                with self.related_list_container:
                    ui.label('请先搜索并勾选标签…').classes('text-sm text-gray-400 italic p-4')

    # ══════════════════════════════════════════════════════════════════════
    # 渲染关联推荐列表
    # ══════════════════════════════════════════════════════════════════════

    def _render_related_list(self, related: list, show_nsfw: bool):
        self.related_list_container.clear()
        self._related_checkboxes.clear()

        filtered = [r for r in related if not (r.nsfw == '1' and not show_nsfw)]
        if not filtered:
            with self.related_list_container:
                ui.label('暂无推荐').classes('text-sm text-gray-400 italic p-4')
            return

        selected_now = set(self._get_selected_tags())

        with self.related_list_container:
            for r in filtered:
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
                    'w-full items-center gap-2 px-3 py-2 related-item border-b border-gray-100'
                ).style(row_bg):
                    # 整行 wiki tooltip
                    if tooltip_html:
                        with ui.tooltip().props('content-class="bg-black text-white shadow-4" max-width="500px"'):
                            ui.html(tooltip_html).style('font-size:14px;line-height:1.5;max-width:480px;')

                    # Checkbox
                    cb = ui.checkbox(
                        '', value=is_selected,
                        on_change=lambda e, t=tag: self._on_related_checkbox_change(t, e.value)
                    ).props('dense')
                    self._related_checkboxes[tag] = cb

                    # 标签名（可点击跳转）+ 中文名
                    with ui.column().classes('flex-grow gap-0 min-w-0'):
                        with ui.row().classes('items-center gap-1'):
                            link = ui.link(
                                tag,
                                f'https://danbooru.donmai.us/wiki_pages/{tag}',
                                new_tab=True
                            ).classes('tag-link text-primary font-bold text-xs')
                            link.on('click', self._mark_interaction)
                            if r.sources and r.sources[0].startswith('tag_group:'):
                                group_display = r.sources[0].replace('tag_group:', '')
                                ui.label(group_display).classes(
                                    'text-xs text-orange-500 font-bold bg-orange-50 px-1 rounded'
                                )

                        if cn_first:
                            ui.label(cn_first).classes('text-xs text-gray-500 truncate')

                    # 关联分数
                    score_color = 'green' if r.cooc_score > 0.6 else ('teal' if r.cooc_score > 0.3 else 'grey')
                    ui.label(score_pct).classes(f'text-sm font-bold text-{score_color}-600 whitespace-nowrap')

    # ══════════════════════════════════════════════════════════════════════
    # 交互逻辑
    # ══════════════════════════════════════════════════════════════════════

    async def _hide_banner_when_ready(self):
        while not DanbooruTagger.is_ready():
            await asyncio.sleep(1)
        if self.init_banner:
            self.init_banner.set_visibility(False)

    def _client_alive(self) -> bool:
        try:
            _ = self.search_btn.client
            return True
        except RuntimeError:
            return False

    # ── 分词筛选 ──────────────────────────────────────────────────────────

    def _filter_by_source(self, keyword: str):
        self.current_filter_keyword = keyword if keyword else 'ALL'
        show_nsfw_val = self.input_nsfw.value
        if not keyword or keyword == 'ALL':
            filtered = self.full_table_data
        else:
            filtered = [r for r in self.full_table_data if r['source'] == keyword]

        self.result_table.rows = apply_nsfw_filter(filtered, show_nsfw_val)

        for child in self.keywords_container.default_slot.children:
            if isinstance(child, ui.chip):
                selected = (
                    (keyword == 'ALL' and child.text == '全部')
                    or (keyword == self.current_query_str and child.text == '整句')
                    or (child.text == keyword)
                )
                is_segment = child.text in self.current_segments
                if selected:
                    chip_color, text_color = 'primary', 'white'
                elif is_segment:
                    chip_color, text_color = 'blue-1', 'blue-8'
                else:
                    chip_color, text_color = 'grey-4', 'black'
                child.props(f'color={chip_color} text-color={text_color}')

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

            self.results_section.set_visibility(True)

            _saved_rpp = self._get_rows_per_page()
            self.result_table.rows = apply_nsfw_filter(table_data, show_nsfw_val)
            self._set_rows_per_page(_saved_rpp)
            all_selected = self._get_selected_tags()
            self.chip_extra_selected.clear()
            self.chip_extra_selected.update(all_selected)
            self.result_table.selected = []
            self._render_selected_chips()
            self._update_selection_display(None)
            self._save_staged_tags()

            self._refresh_related([], show_nsfw_val)
            self._last_recommendation_seed_tags = []

            # 分词筛选 chips
            self.current_filter_keyword = 'ALL'
            self.keywords_container.clear()
            cached_set = set(response.cached_queries) if response.cached_queries else set()
            with self.keywords_container:
                ui.label('分词筛选:').classes('text-sm text-gray-500 font-bold mr-2')
                ui.chip('全部', on_click=lambda: self._filter_by_source('ALL')) \
                    .props('color=primary text-color=white clickable')
                use_seg = self.input_segment.value if self.input_segment else True
                if use_seg:
                    whole = ui.chip('整句',
                            on_click=lambda: self._filter_by_source(self.current_query_str))
                    whole.props('color=grey-4 text-color=black clickable')
                    if self.current_query_str in cached_set:
                        whole.style('outline: 1px dashed rgba(128,128,128,0.3); outline-offset: 1px;')
                    for seg in response.segments:
                        sc = ui.chip(seg,
                                on_click=lambda s=seg: self._filter_by_source(s))
                        sc.props('color=blue-1 text-color=blue-8 clickable')
                        if seg in cached_set:
                            sc.style('outline: 1px dashed rgba(128,128,128,0.3); outline-offset: 1px;')
                    for kw in response.keywords:
                        kc = ui.chip(kw,
                                on_click=lambda k=kw: self._filter_by_source(k))
                        kc.props('color=grey-4 text-color=black clickable')
                        if kw in cached_set:
                            kc.style('outline: 1px dashed rgba(128,128,128,0.3); outline-offset: 1px;')
                else:
                    ui.label('(分词已关闭)').classes('text-xs text-gray-400')

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
        extra = [t for t in self.chip_extra_selected if t not in seen]
        return table_tags + extra

    def _get_recommendation_seed_tags(self, selected_tags: list[str]) -> list[str]:
        artist_tags = set(self._current_artist_rec_tags) | set(self._artist_result_tags)
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

    def _set_selected_tags(self, tags: list[str], skip_refresh: bool = False):
        tag_set = set(tags)
        table_tag_set = {row['tag'] for row in self.result_table.rows} if self.result_table else set()
        self.chip_extra_selected.clear()
        self.chip_extra_selected.update(t for t in tag_set if t not in table_tag_set)
        # clean up weights for deselected tags
        for t in list(self.tag_weights):
            if t not in tag_set:
                del self.tag_weights[t]

        if self.result_table is not None:
            self.result_table.selected = [row for row in self.result_table.rows if row.get('tag') in tag_set]

        # 同步推荐画师 checkbox
        for t, cb in self._artist_rec_checkboxes.items():
            cb.set_value(t in tag_set)

        all_tags = self._get_selected_tags()
        if self.selection_count_label is not None:
            self.selection_count_label.text = str(len(all_tags))
        self._save_staged_tags()
        self._render_selected_chips()
        # 显式刷新关联推荐和 Group 区域（不依赖 table.on('selection') 事件，
        # 因为在 chip 点击回调上下文中该事件可能不可靠）。
        # 从关联推荐/同类标签勾选时跳过，由各自动态刷新或手动按钮触发。
        if not skip_refresh:
            show_nsfw_val = self.input_nsfw.value
            self._refresh_recommendations_if_seed_changed(all_tags, show_nsfw_val)
            if not all_tags:
                self.chip_extra_selected.clear()

    def _update_selection_display(self, _e):
        if self.result_table is None:
            return
        self._mark_interaction()

        all_tags = self._get_selected_tags()
        # clean up weights for deselected tags
        tag_set = set(all_tags)
        for t in list(self.tag_weights):
            if t not in tag_set:
                del self.tag_weights[t]
        # init weight for newly selected tags
        for t in all_tags:
            self.tag_weights.setdefault(t, 1.0)

        if self.selection_count_label is not None:
            self.selection_count_label.text = str(len(all_tags))
        self._render_selected_chips()

        # 同步推荐画师 checkbox
        for t, cb in self._artist_rec_checkboxes.items():
            cb.set_value(t in tag_set)

        show_nsfw_val = self.input_nsfw.value
        self._refresh_recommendations_if_seed_changed(all_tags, show_nsfw_val)
        if not all_tags:
            self.chip_extra_selected.clear()
        self._save_staged_tags()

    def _on_related_checkbox_change(self, tag: str, checked: bool):
        self._mark_interaction()
        current = self._get_selected_tags()
        if checked:
            if tag not in current:
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
        """仅刷新关联推荐列表（300ms 去抖，避免快速勾选产生 CPU 洪峰）。"""
        selected_tags = self._get_recommendation_seed_tags(selected_tags)
        # 取消上次未执行的刷新
        if self._debounce_related_task and not self._debounce_related_task.done():
            self._debounce_related_task.cancel()
        async def _do():
            await asyncio.sleep(0.3)
            if not selected_tags:
                self._refresh_related([], show_nsfw)
                return
            tagger = await DanbooruTagger.get_instance()
            related = await tagger.get_related_async(
                selected_tags,
                set(selected_tags),
                50,
                show_nsfw,
            )
            self._refresh_related(related, show_nsfw)
        self._debounce_related_task = asyncio.ensure_future(_do())

    def _refresh_group_from_selection(self, selected_tags: list[str], show_nsfw: bool):
        """仅刷新同类扩展区域（300ms 去抖，避免快速勾选产生 CPU 洪峰）。"""
        selected_tags = self._get_recommendation_seed_tags(selected_tags)
        if self._debounce_group_task and not self._debounce_group_task.done():
            self._debounce_group_task.cancel()
        async def _do():
            await asyncio.sleep(0.3)
            if not selected_tags:
                if self.group_expansion_container is not None:
                    self.group_expansion_container.clear()
                    with self.group_expansion_container:
                        ui.label('请先搜索并勾选标签…').classes('text-sm text-gray-400 italic p-4')
                return
            tagger = await DanbooruTagger.get_instance()
            group_data = await tagger.get_group_candidates_async(
                selected_tags,
                show_nsfw,
            )
            await self._capture_group_scroll_positions()
            self._render_group_expansion(group_data, selected_tags, show_nsfw)
        self._debounce_group_task = asyncio.ensure_future(_do())

    def _refresh_artist_from_selection(self, selected_tags: list[str], show_nsfw: bool = True):
        """根据已选标签刷新画师推荐（300ms 去抖）。"""
        selected_tags = self._get_recommendation_seed_tags(selected_tags)
        if self._debounce_artist_task and not self._debounce_artist_task.done():
            self._debounce_artist_task.cancel()
        async def _do():
            await asyncio.sleep(0.3)
            if len(selected_tags) < 1:
                self._render_artist_rec([], {}, show_nsfw)
                return
            tagger = await DanbooruTagger.get_instance()
            artist_results = await tagger.search_artists_by_tags_async(
                selected_tags, limit=30, min_cooc=3,
            )
            top_tags = {}
            if artist_results:
                names = [r.artist for r in artist_results[:10]]
                top_tags = tagger.get_artist_top_tags(names, show_nsfw=show_nsfw)
            self._render_artist_rec(artist_results, top_tags, show_nsfw)
        self._debounce_artist_task = asyncio.ensure_future(_do())

    def _render_artist_rec(self, artist_results, top_tags=None, show_nsfw: bool = True):
        """渲染推荐画师列表（对标关联推荐样式）。"""
        if self.artist_rec_list is None:
            return
        self.artist_rec_list.clear()
        self._artist_rec_checkboxes.clear()
        self._current_artist_rec_tags.clear()

        if not artist_results:
            with self.artist_rec_list:
                ui.label('暂无推荐画师').classes('text-sm text-gray-400 italic p-4')
            return

        top_tags = top_tags or {}
        selected_now = set(self._get_selected_tags())

        with self.artist_rec_list:
            for r in artist_results[:10]:
                artist = r.artist
                self._current_artist_rec_tags.add(artist)
                is_selected = artist in selected_now
                # 归一化：除以命中标签数，cap 到 100%
                normalized = min(r.score / max(r.hit_count, 1), 1.0)
                score_pct = f'+{normalized * 100:.0f}%'
                sources_str = '、'.join(r.sources[:3]) if r.sources else '—'
                post_str = f'{r.post_count:,}' if r.post_count else '—'

                # tooltip：画师擅长标签
                tag_list = top_tags.get(artist, [])
                tooltip_html = f'<div><b>{artist}</b><br>这位画师经常画:<br>'
                if tag_list:
                    for t in tag_list[:10]:
                        tooltip_html += f'  · {t}<br>'
                else:
                    tooltip_html += '  (无数据)'
                tooltip_html += '</div>'

                with ui.row().classes(
                    'w-full items-center gap-2 px-3 py-2 related-item border-b border-gray-100'
                ).style('background: rgba(244,114,182,0.04);'):
                    # tooltip
                    with ui.tooltip().props('content-class="bg-black text-white shadow-4" max-width="400px"'):
                        ui.html(tooltip_html).style('font-size:14px;line-height:1.5;max-width:380px;')

                    # Checkbox
                    cb = ui.checkbox(
                        '', value=is_selected,
                        on_change=lambda e, t=artist: self._on_artist_rec_checkbox_change(t, e.value)
                    ).props('dense')
                    self._artist_rec_checkboxes[artist] = cb

                    # 画师名 + 信息
                    with ui.column().classes('flex-grow gap-0 min-w-0'):
                        ui.link(
                            artist,
                            f'https://danbooru.donmai.us/posts?tags={artist}',
                            new_tab=True,
                        ).classes('text-primary font-bold text-xs')
                        ui.label(f'{sources_str} · 作品 {post_str}').classes('text-xs text-gray-500')

                    # 分值
                    score_color = 'green' if normalized > 0.6 else ('teal' if normalized > 0.3 else 'grey')
                    ui.label(score_pct).classes(f'text-sm font-bold text-{score_color}-600 whitespace-nowrap')

    def _render_group_expansion(self, group_data: list, selected_tags: list[str], show_nsfw: bool):
        """渲染 Group 同类扩展区域。"""
        if self.group_expansion_container is None:
            return
        self.group_expansion_container.clear()
        self._group_checkboxes.clear()
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
                                'w-full items-center gap-1.5 px-2 py-1.5 rounded related-item'
                            ).style(row_bg):
                                if tooltip_html:
                                    with ui.tooltip().props('content-class="bg-black text-white shadow-4" max-width="500px"'):
                                        ui.html(tooltip_html).style('font-size:14px;line-height:1.5;max-width:480px;')

                                # 复选框
                                cb = ui.checkbox(
                                    '', value=is_selected,
                                    on_change=lambda e, t=tag: self._on_group_checkbox_change(t, e.value),
                                ).props('dense')
                                self._group_checkboxes[tag] = cb

                                # 标签名 + 中文名（与关联推荐对齐方式一致）
                                with ui.column().classes('flex-grow gap-0 min-w-0 overflow-hidden'):
                                    link = ui.link(
                                        tag,
                                        f'https://danbooru.donmai.us/wiki_pages/{tag}',
                                        new_tab=True,
                                    ).classes('tag-link text-primary font-bold text-xs truncate')
                                    if cn_first:
                                        ui.label(cn_first).classes('text-xs text-gray-500 truncate')

                                # 热度
                                count = t['post_count']
                                if count > 0:
                                    if count >= 10000:
                                        count_str = f'{count/1000:.0f}k'
                                    elif count >= 1000:
                                        count_str = f'{count/1000:.1f}k'
                                    else:
                                        count_str = str(count)
                                    ui.label(count_str).classes('text-sm font-bold text-grey-600 whitespace-nowrap')
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

    def on_nsfw_toggle(self, e):
        show_nsfw_val = self.input_nsfw.value

        # 复用当前分词筛选：同时套用新 NSFW 状态并保持 chip 选中态
        self._filter_by_source(self.current_filter_keyword)
        if not show_nsfw_val:
            self.result_table.selected = [r for r in self.result_table.selected if r.get('nsfw') != '1']
        self._update_selection_display(None)

    # ── 复制 / 反馈 ──────────────────────────────────────────────────────

    def _toggle_prompt_format(self):
        if self.prompt_format == 'sdxl':
            self.prompt_format = 'nai'
            if self.format_toggle_btn:
                self.format_toggle_btn.text = 'NAI'
                self.format_toggle_btn.props('color=purple-7')
        elif self.prompt_format == 'nai':
            self.prompt_format = 'anima'
            if self.format_toggle_btn:
                self.format_toggle_btn.text = 'Anima'
                self.format_toggle_btn.props('color=teal-7')
        else:
            self.prompt_format = 'sdxl'
            if self.format_toggle_btn:
                self.format_toggle_btn.text = 'SDXL'
                self.format_toggle_btn.props('color=grey-7')
        self._render_selected_chips()

    def copy_selection(self):
        self._mark_interaction()
        tags = self._get_selected_tags()
        parts = []
        artist_tags = set(self._current_artist_rec_tags) | set(self._artist_result_tags)
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

        async def silent_copy_update():
            try:
                await counter.increment_copy()
            except Exception:
                pass
        asyncio.create_task(silent_copy_update())

    def _copy_all_tags(self):
        self._mark_interaction()
        show_nsfw_val = self.input_nsfw.value
        tags_str = self.full_tags_str if show_nsfw_val else self.full_tags_str_sfw
        if tags_str:
            tags_str = tags_str.replace('(', '\\(').replace(')', '\\)')
            ui.clipboard.write(tags_str)
            ui.notify('已复制全部标签!', type='positive')
        else:
            ui.notify('暂无标签可复制', type='warning')

    def _feedback_settings(self) -> dict:
        return {
            'top_k': int(self.input_top_k.value) if self.input_top_k else None,
            'segmentation': self.input_segment.value if self.input_segment else None,
            'nsfw': self.input_nsfw.value if self.input_nsfw else None,
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
            detail_input = ui.textarea(
                label='具体问题（可选）',
                placeholder='例如：结果偏题、缺少某个关键标签、召回了不相关角色/作品...',
            ).props('outlined autogrow maxlength=500 counter').classes('w-full')

            async def submit_feedback():
                detail = (detail_input.value or '').strip()

                submit_btn.disable()
                try:
                    await counter.add_bad_case(
                        query,
                        platform=PLATFORM,
                        settings=self._feedback_settings(),
                        feedback_type='search_bad_case',
                        detail=detail,
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
                    await counter.add_bad_case(
                        query,
                        platform=PLATFORM,
                        settings=self._feedback_settings(),
                        feedback_type='translation_error',
                        detail=detail,
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
    app_ui = DanbooruSearchUI()
    app_ui.build_page()

    async def silent_visit_update():
        try:
            await counter.increment_visit()
            app_ui._update_footer_text()
        except Exception:
            pass
    asyncio.create_task(silent_visit_update())

    # 恢复用户配置（在页面渲染完成后执行）
    await app_ui._restore_config()
    # 恢复已选标签（备选区）
    await app_ui._restore_staged_tags()


# ── 入口 ───────────────────────────────────────────────────────────────────────

if __name__ in {'__main__', '__mp_main__'}:
    host, port = get_host_port()

    @app.on_startup
    def _warmup():
        async def background_init_tasks():
            await asyncio.sleep(5)
            print("[UI] 开始预热计数器与引擎", flush=True)
            await counter.init()
            await DanbooruTagger.get_instance()
            print("[UI] 后台预热全部完成！", flush=True)
        asyncio.create_task(background_init_tasks())

    @app.on_shutdown
    def _shutdown():
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(counter.force_sync())
            else:
                asyncio.run(counter.force_sync())
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
