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
from api_fastapi import app as api_app
from core.engine import DanbooruTagger
from core.models import RelatedTag, SearchRequest
from platform_utils import is_cloud, get_host_port, nsfw_allowed

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

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
_CONFIG_VERSION = 4



# ── 辅助函数 ───────────────────────────────────────────────────────────────────

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
    sdxl: (tag:1.2)  权重 1.0 时输出 tag
    nai:  1.2::tag:: 权重 1.0 时输出 tag
    """
    if weight == 1.0:
        return tag
    if fmt == 'nai':
        return f'{weight:.1f}::{tag}::'
    return f'({tag}:{weight:.1f})'


# ── UI 类 ─────────────────────────────────────────────────────────────────────

class DanbooruSearchUI:
    def __init__(self):
        self.search_count_label = None
        self.current_search_interacted = True

        self.full_table_data: list[dict] = []
        self.current_segments: list[str] = []   # 从句级原始片段，用于区分 chip 颜色
        self.current_query_str: str = ""
        self.full_tags_str: str = ""
        self.full_tags_str_sfw: str = ""

        self.result_table = None           # 左栏表格
        self.related_list_container = None  # 右栏关联推荐列表
        self.results_section = None        # 整个结果区域（搜索前隐藏）
        self.selection_count_label = None
        self.selected_display = None       # 已废弃 textarea，保留兼容
        self.selected_chips_container = None  # 已选标签 chip 容器
        self.current_related: list = []
        self.chip_extra_selected: set = set()

        # tag -> prompt 权重，范围 [0.1, 1.9]，默认 1.0
        self.tag_weights: dict[str, float] = {}
        # 复制格式：'sdxl' 或 'nai'
        self.prompt_format: str = 'sdxl'
        self.format_toggle_btn = None

        self.init_banner = None
        self.input_top_k = None
        self.input_limit = None
        self.input_weight = None
        self.input_nsfw = None
        self.input_segment = None
        self.search_input = None
        self.keywords_container = None
        self.spinner = None
        self.search_btn = None

        self.selected_layers = {'英文': True, '中文扩展词': True, '释义': True, '中文核心词': True}
        self.selected_cats = {'General': True, 'Copyright': True, 'Character': True}

        self.notice_expansion = None

        # 表格显示选项开关
        self.sw_semantic = None
        self.sw_layer = None
        self.sw_source = None

        # 关联推荐的 checkbox 引用
        self._related_checkboxes: dict[str, ui.checkbox] = {}

        # 高级选项中各层/类型的 checkbox 引用，用于 restore 时同步控件状态
        self._layer_checkboxes: dict[str, ui.checkbox] = {}
        self._cat_checkboxes: dict[str, ui.checkbox] = {}

    def _update_footer_text(self):
        if self.search_count_label is not None:
            try:
                commit = _get_git_commit()
                self.search_count_label.content = (
                    f'<span class="font-mono text-gray-300">版本号: {commit}</span>'
                    f'<br>'
                    f'<a href="/api/docs" '
                    f'target="_blank" rel="noopener noreferrer" '
                    f'class="text-blue-400 hover:text-blue-600 hover:underline">使用 API 服务</a>'
                )
            except AttributeError:
                pass

    def _mark_interaction(self, e=None):
        if not self.current_search_interacted:
            self.current_search_interacted = True

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

        if self.input_top_k and 'top_k' in cfg:
            self.input_top_k.set_value(cfg['top_k'])
        if self.input_limit and 'limit' in cfg:
            self.input_limit.set_value(cfg['limit'])
        if self.input_weight and 'popularity_weight' in cfg:
            self.input_weight.set_value(cfg['popularity_weight'])
        if self.input_segment and 'use_segmentation' in cfg:
            self.input_segment.set_value(cfg['use_segmentation'])

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

        if 'prompt_format' in cfg and cfg['prompt_format'] in ('sdxl', 'nai'):
            self.prompt_format = cfg['prompt_format']
            if self.format_toggle_btn:
                if self.prompt_format == 'nai':
                    self.format_toggle_btn.text = 'NAI'
                    self.format_toggle_btn.props('color=purple-7')
                else:
                    self.format_toggle_btn.text = 'SDXL'
                    self.format_toggle_btn.props('color=grey-7')

        if 'rows_per_page' in cfg:
            self._set_rows_per_page(cfg['rows_per_page'])

        if self.search_input and cfg.get('search_query'):
            self.search_input.set_value(cfg['search_query'])

        if self.notice_expansion and 'notice_expanded' in cfg:
            self.notice_expansion.set_value(cfg['notice_expanded'])

        # 若高级选项列有变更，同步更新表格列
        self._update_table_columns()

    # ══════════════════════════════════════════════════════════════════════
    # 页面构建
    # ══════════════════════════════════════════════════════════════════════

    def build_page(self):
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

        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-4'):

            # ── 初始化提示 ──
            self.init_banner = ui.card().classes(
                'w-full bg-blue-50 border-l-4 border-blue-400'
            )
            with self.init_banner:
                with ui.row().classes('items-center gap-3 p-2'):
                    ui.spinner(size='sm')
                    ui.label('引擎初始化中，请稍候…首次加载约需 15 秒').classes('text-sm text-blue-700')
            self.init_banner.set_visibility(not DanbooruTagger.is_ready())
            if not DanbooruTagger.is_ready():
                asyncio.ensure_future(self._hide_banner_when_ready())

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
- **支持作者**：如果觉得好用，欢迎点击顶部给本 Space 点个 **Like ❤️**，或前往 [GitHub](https://github.com/SuzumiyaAkizuki/DanbooruSearchOnline) 点个 **Star ⭐**！
- **🚀 首次使用？[点击查看使用指南](https://github.com/SuzumiyaAkizuki/DanbooruSearchOnline)**，了解五种搜索模式与进阶技巧
""").classes('text-sm text-gray-800 px-4 pb-3')

    # ── 搜索卡片 ─────────────────────────────────────────────────────────

    def _build_search_card(self):
        with ui.card().classes('w-full'):
            with ui.row().classes('items-center gap-2 mb-2'):
                ui.icon('search', size='2em', color='primary')
                ui.label('Danbooru 标签模糊搜索').classes('text-2xl font-bold text-gray-800')
            ui.label('基于语义匹配的标签搜索引擎，支持多维匹配与共现关联推荐。').classes(
                'text-sm text-gray-500 -mt-1 mb-3'
            )

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

            with ui.row().classes('w-full gap-6 items-center mt-3 flex-wrap'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('Top K (语义相关)').classes('text-sm text-gray-600')
                    self.input_top_k = ui.number(value=10, min=1, max=200).classes('w-20') \
                        .props('outlined dense')

                with ui.row().classes('items-center gap-2'):
                    ui.label('结果上限').classes('text-sm text-gray-600')
                    self.input_limit = ui.number(value=80, min=10, max=500).classes('w-20') \
                        .props('outlined dense')

                with ui.row().classes('items-center gap-2'):
                    ui.label('热度权重').classes('text-sm text-gray-600')
                    self.input_weight = ui.slider(min=0.0, max=1.0, value=0.15, step=0.05).classes('w-32')
                    ui.label().bind_text_from(self.input_weight, 'value', lambda v: f"{v:.2f}") \
                        .classes('text-sm font-mono text-gray-700 w-8')

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

            with ui.expansion('高级选项', icon='tune').classes('w-full mt-2'):
                with ui.column().classes('w-full p-3 gap-4'):
                    with ui.row().classes('w-full gap-8 flex-wrap'):
                        with ui.column().classes('gap-2'):
                            ui.label('匹配层筛选').classes('font-bold text-sm text-gray-700')
                            display_map = {
                                '英文': '英文标签', '中文扩展词': '中文扩展词',
                                '释义': '维基释义', '中文核心词': '中文核心词',
                            }
                            for layer in ['英文', '中文扩展词', '释义', '中文核心词']:
                                cb = ui.checkbox(
                                    display_map.get(layer, layer), value=True,
                                    on_change=lambda e, l=layer: self.selected_layers.__setitem__(l, e.value)
                                ).props('color=primary dense')
                                self._layer_checkboxes[layer] = cb

                        with ui.column().classes('gap-2'):
                            ui.label('类型筛选').classes('font-bold text-sm text-gray-700')
                            color_map = {'General': 'blue', 'Copyright': 'pink', 'Character': 'green'}
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

    # ── 已选标签栏 ────────────────────────────────────────────────────────

    def _build_selection_bar(self):
        with ui.card().classes('w-full bg-blue-50 border border-blue-200'):
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
                    self.format_toggle_btn = ui.button(
                        'SDXL', icon='swap_horiz'
                    ).props('dense flat color=grey-7').classes('text-xs font-mono')
                    with self.format_toggle_btn:
                        with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                            ui.html(
                                '切换复制格式：<br>'
                                '<b>SDXL</b>：<code>(tag:1.2)</code><br>'
                                '<b>NAI</b>：<code>1.2::tag::</code>'
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
            for tag in tags:
                w = self.tag_weights.get(tag, 1.0)
                extra_cls = 'boosted' if w > 1.0 else ('reduced' if w < 1.0 else '')
                w_str = f'{w:.1f}'
                with ui.element('div').classes(f'weight-chip {extra_cls}'):
                    # 删除按钮（×）
                    with ui.element('button').classes('weight-btn').props(f'title="移除 {tag}"').on(
                        'click', lambda t=tag: self._remove_selected_tag(t)
                    ):
                        ui.html('&times;')
                    # 减号
                    with ui.element('button').classes('weight-btn').on(
                        'click', lambda t=tag: self._adjust_weight(t, -0.1)
                    ):
                        ui.html('&minus;')
                    # 标签名
                    ui.label(tag).style(
                        'font-family:Consolas,Monaco,monospace;font-size:12px;'
                        'color:#2c5282;max-width:150px;overflow:hidden;'
                        'text-overflow:ellipsis;white-space:nowrap;'
                    )
                    # 权重值（仅非 1.0 时显示）
                    if w != 1.0:
                        ui.label(w_str).classes('weight-label').style('color:#e65100;font-weight:bold;')
                    # 加号
                    with ui.element('button').classes('weight-btn').on(
                        'click', lambda t=tag: self._adjust_weight(t, +0.1)
                    ):
                        ui.html('&plus;')

    def _adjust_weight(self, tag: str, delta: float):
        """调整单个标签权重，钳位到 [0.1, 1.9]。"""
        current = self.tag_weights.get(tag, 1.0)
        new_w = round(current + delta, 1)
        if new_w < 0.1:
            ui.notify('权重范围为 0.1 ~ 1.9，已到达最小值', type='warning', timeout=2000)
            return
        if new_w > 1.9:
            ui.notify('权重范围为 0.1 ~ 1.9，已到达最大值', type='warning', timeout=2000)
            return
        self.tag_weights[tag] = new_w
        self._save_staged_tags()
        self._render_selected_chips()

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
        self._render_selected_chips()
        if self.selection_count_label is not None:
            self.selection_count_label.text = '0'
        show_nsfw_val = self.input_nsfw.value
        self._refresh_related([], show_nsfw_val)
        self._save_staged_tags()
        ui.notify('已清空所有已选标签', type='warning')

        # ── 两栏结果（CSS 强制并排）──────────────────────────────────────────

    def _build_results_columns(self):
        with ui.element('div').classes('w-full two-col-layout'):
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
                self.result_table.on('pagination', lambda _: self._save_config())

                # 自定义行模板：行背景色按分类，整行悬浮显示 wiki（NSFW模糊行除外）
                self.result_table.add_slot('body', r'''
                    <q-tr :props="props"
                          :class="props.row._nsfw_blocked ? 'nsfw-row-blocked' : ''"
                          :style="{
                              'background-color':
                                  props.row.category === 'General'   ? 'rgba(59,130,246,0.06)' :
                                  props.row.category === 'Character' ? 'rgba(34,197,94,0.06)'  :
                                  props.row.category === 'Copyright' ? 'rgba(236,72,153,0.06)' : ''
                          }">
                        <q-td auto-width>
                            <q-checkbox v-model="props.selected"
                                :class="props.row._nsfw_blocked ? 'nsfw-checkbox-disabled' : ''"/>
                        </q-td>
                        <q-td v-for="col in props.cols" :key="col.name" :props="props">
                            <template v-if="col.name === 'tag' || col.name === 'cn_name'">
                                <div :class="props.row._nsfw_blocked ? 'nsfw-blur-cell' : ''">
                                    <template v-if="col.name === 'cn_name' && col.value">
                                        <span style="font-size:14px">
                                            {{ col.value.split(',')[0] }}
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
                        <q-tooltip v-if="props.row.wiki && !props.row._nsfw_blocked"
                            content-class="bg-black text-white shadow-4"
                            max-width="500px" :offset="[10,10]">
                            <div style="font-size:14px;line-height:1.5;">
                                <span style="opacity:0.7;margin-right:4px;">{{
                                    props.row.category === 'General'   ? '[通用]' :
                                    props.row.category === 'Character' ? '[角色]' :
                                    props.row.category === 'Copyright' ? '[作品]' : ''
                                }}</span>{{ props.row.wiki }}
                            </div>
                        </q-tooltip>
                    </q-tr>
                ''')

            # ── 右栏：关联推荐 ──
            with ui.card().classes('col-right'):
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

                sources_str = '、'.join(r.sources) if r.sources else '—'
                CAT_LABEL = {'General': '通用', 'Character': '角色', 'Copyright': '作品'}
                cat_label = CAT_LABEL.get(r.category, '')
                tooltip_html = ''
                if wiki_text:
                    prefix = f'<span style="opacity:0.7;margin-right:4px;">[{cat_label}] </span>' if cat_label else ''
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
                    'Copyright': 'background-color: rgba(236,72,153,0.06);',   # 淡红/粉
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
                        link = ui.link(
                            tag,
                            f'https://danbooru.donmai.us/wiki_pages/{tag}',
                            new_tab=True
                        ).classes('tag-link text-primary font-bold text-xs')
                        link.on('click', self._mark_interaction)

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

        # 搜索前保存配置
        self._save_config()

        self.current_query_str = query
        self.search_btn.disable()
        self.spinner.classes(remove='hidden')
        ui.notify('正在搜索...', type='info')

        target_layers_list = [k for k, v in self.selected_layers.items() if v]
        target_cats_list   = [k for k, v in self.selected_cats.items()   if v]

        if not target_layers_list:
            ui.notify('请至少选择一个匹配层！', type='warning')
            self.search_btn.enable()
            self.spinner.classes(add='hidden')
            return

        try:
            tagger = await DanbooruTagger.get_instance()

            # NSFW 保护模式：开 = 不显示 NSFW
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
            )
            response = await run.io_bound(tagger.search, request)

            if not self._client_alive():
                return

            table_data = [result_to_row(r, show_nsfw_val) for r in response.results]
            self.full_table_data = table_data
            self.full_tags_str = response.tags_all
            self.full_tags_str_sfw = response.tags_sfw
            self.current_segments = list(response.segments) if response.segments else []

            # 显示结果区域
            self.results_section.set_visibility(True)

            _saved_rpp = self._get_rows_per_page()
            self.result_table.rows = apply_nsfw_filter(table_data, show_nsfw_val)
            self._set_rows_per_page(_saved_rpp)
            # 搜索时保留已选标签（跨搜索积累）
            all_selected = self._get_selected_tags()
            self.chip_extra_selected.clear()
            self.chip_extra_selected.update(all_selected)
            self.result_table.selected = []
            self._render_selected_chips()
            self._update_selection_display(None)
            self._save_staged_tags()

            # 清空关联推荐
            self._refresh_related([], show_nsfw_val)

            # 分词筛选 chips
            self.keywords_container.clear()
            with self.keywords_container:
                ui.label('分词筛选:').classes('text-sm text-gray-500 font-bold mr-2')
                ui.chip('全部', on_click=lambda: self._filter_by_source('ALL')) \
                    .props('color=primary text-color=white clickable')
                use_seg = self.input_segment.value if self.input_segment else True
                if use_seg:
                    ui.chip('整句',
                            on_click=lambda: self._filter_by_source(self.current_query_str)) \
                        .props('color=grey-4 text-color=black clickable')
                    # 从句级原始片段（分隔符切分后未 jieba 的长片段，区别于关键词）
                    for seg in response.segments:
                        ui.chip(seg,
                                on_click=lambda s=seg: self._filter_by_source(s)) \
                            .props('color=blue-1 text-color=blue-8 clickable')
                    for kw in response.keywords:
                        ui.chip(kw,
                                on_click=lambda k=kw: self._filter_by_source(k)) \
                            .props('color=grey-4 text-color=black clickable')
                else:
                    ui.label('(分词已关闭)').classes('text-xs text-gray-400')

            ui.notify(f'找到 {len(table_data)} 个标签', type='positive')
            self.current_search_interacted = False

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

    def _set_selected_tags(self, tags: list[str]):
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

        all_tags = self._get_selected_tags()
        if self.selection_count_label is not None:
            self.selection_count_label.text = str(len(all_tags))
        # 先保存再重建 UI，避免 clear() 销毁事件上下文后 run_javascript 失败
        self._save_staged_tags()
        self._render_selected_chips()

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

        show_nsfw_val = self.input_nsfw.value
        if all_tags:
            self._refresh_related_from_selection(all_tags, show_nsfw_val)
        else:
            self.chip_extra_selected.clear()
            self._refresh_related([], show_nsfw_val)
        self._save_staged_tags()

    def _on_related_checkbox_change(self, tag: str, checked: bool):
        self._mark_interaction()
        current = self._get_selected_tags()
        if checked:
            if tag not in current:
                current.append(tag)
                self.tag_weights.setdefault(tag, 1.0)
                self._set_selected_tags(current)
                ui.notify(f'已添加 {tag}', type='positive', timeout=1500)
        else:
            if tag in current:
                current.remove(tag)
                self.tag_weights.pop(tag, None)
                self._set_selected_tags(current)
                ui.notify(f'已移除 {tag}', type='warning', timeout=1500)

    def _manual_refresh_related(self):
        """手动触发关联推荐列表的刷新"""
        self._mark_interaction()
        show_nsfw_val = self.input_nsfw.value
        all_tags = self._get_selected_tags()

        if all_tags:
            self._refresh_related_from_selection(all_tags, show_nsfw_val)
            ui.notify('已触发关联推荐更新', type='info', timeout=1500)
        else:
            self.chip_extra_selected.clear()
            self._refresh_related([], show_nsfw_val)
            ui.notify('已清空关联推荐', type='info', timeout=1500)

    # ── 关联推荐 ──────────────────────────────────────────────────────────

    def _refresh_related(self, related: list, show_nsfw: bool):
        selected_now = set(self._get_selected_tags())
        old_related  = self.current_related
        new_tags  = {r.tag for r in related}
        preserved = [r for r in old_related if r.tag in selected_now and r.tag not in new_tags]
        merged = list(related) + preserved

        self.current_related = merged
        if self.related_list_container is not None:
            self._render_related_list(merged, show_nsfw)

    def _refresh_related_from_selection(self, selected_tags: list[str], show_nsfw: bool):
        async def _do():
            tagger  = await DanbooruTagger.get_instance()
            related = await run.io_bound(
                tagger.get_related,
                selected_tags,
                set(selected_tags),
                50,
                show_nsfw,
            )
            self._refresh_related(related, show_nsfw)
        asyncio.ensure_future(_do())

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

    # ── NSFW 切换 ─────────────────────────────────────────────────────────

    def on_nsfw_toggle(self, e):
        show_nsfw_val = self.input_nsfw.value

        self.result_table.rows = apply_nsfw_filter(self.full_table_data, show_nsfw_val)
        if not show_nsfw_val:
            self.result_table.selected = [r for r in self.result_table.selected if r.get('nsfw') != '1']
        self._update_selection_display(None)

    # ── 复制 ────────────────────────────────────────────────────────────

    def _toggle_prompt_format(self):
        if self.prompt_format == 'sdxl':
            self.prompt_format = 'nai'
            if self.format_toggle_btn:
                self.format_toggle_btn.text = 'NAI'
                self.format_toggle_btn.props('color=purple-7')
        else:
            self.prompt_format = 'sdxl'
            if self.format_toggle_btn:
                self.format_toggle_btn.text = 'SDXL'
                self.format_toggle_btn.props('color=grey-7')

    def copy_selection(self):
        self._mark_interaction()
        tags = self._get_selected_tags()
        prompt = ', '.join(
            _format_tag_with_weight(t, self.tag_weights.get(t, 1.0), self.prompt_format)
            for t in tags
        )
        ui.clipboard.write(prompt)
        fmt_label = 'NAI' if self.prompt_format == 'nai' else 'SDXL'
        ui.notify(f'已复制选中标签（{fmt_label} 格式）!', type='positive')

    def _copy_all_tags(self):
        self._mark_interaction()
        show_nsfw_val = self.input_nsfw.value
        tags_str = self.full_tags_str if show_nsfw_val else self.full_tags_str_sfw
        if tags_str:
            ui.clipboard.write(tags_str)
            ui.notify('已复制全部标签!', type='positive')
        else:
            ui.notify('暂无标签可复制', type='warning')

# ── 页面路由 ───────────────────────────────────────────────────────────────────

@ui.page('/')
async def main_page():
    app_ui = DanbooruSearchUI()
    app_ui.build_page()

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
            print("[UI] 开始预热引擎", flush=True)
            await DanbooruTagger.get_instance()
            print("[UI] 后台预热完成！", flush=True)
        asyncio.create_task(background_init_tasks())

    app.mount('/api', api_app)

    @app.get('/googlebd34b54f8562aa06.html')
    def google_verification():
        return PlainTextResponse('google-site-verification: googlebd34b54f8562aa06.html')

    @app.get('/robots.txt')
    def robots_txt():
        content = (
            "User-agent: *\n"
            "Allow: /$\n"
            "Disallow: /api/\n"
            "Disallow: /_nicegui/\n"
            "Disallow: /socket.io/\n"
        )
        return PlainTextResponse(content)

    @app.head('/')
    async def head_root():
        return PlainTextResponse("")

    ui.run(
        host=host,
        port=port,
        title='Danbooru Tags Searcher',
        reload=not is_cloud(),
        show=not is_cloud(),
        reconnect_timeout=120,
    )