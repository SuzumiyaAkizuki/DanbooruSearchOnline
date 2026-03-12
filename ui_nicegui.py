"""
ui_nicegui.py
─────────────
NiceGUI 前端层。

▸ 只负责渲染 / 交互。
▸ 调用 core.engine.DanbooruTagger，通过 core.models 的数据结构通信。
▸ 不包含任何算法逻辑。
"""
import sys
sys.stdout.reconfigure(line_buffering=True) # 强制 Python 实时输出日志
print("==== [Step 1] 脚本开始执行 ====", flush=True)
import asyncio
import os
import json as _json
import traceback
from dataclasses import asdict

# ── 全局异常捕获：确保启动崩溃时有完整堆栈输出到日志 ──────────
def _excepthook(exc_type, exc_value, exc_tb):
    print("=" * 60, flush=True)
    print("FATAL ERROR ON STARTUP:", flush=True)
    traceback.print_exception(exc_type, exc_value, exc_tb)
    print("=" * 60, flush=True)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _excepthook

try:
    from nicegui import ui, app, run
    from core import counter
    from api_fastapi import app as api_app
    from core.engine import DanbooruTagger
    from core.models import RelatedTag, SearchRequest
except Exception:
    traceback.print_exc()
    raise


def is_running_on_huggingface_space() -> bool:
    return os.environ.get("SPACE_ID") is not None


# 表格列定义

BASE_COLUMNS = [
    {'name': 'tag',         'label': '匹配标签', 'field': 'tag',         'align': 'left', 'sortable': True},
    {'name': 'cn_name',     'label': '含义',     'field': 'cn_name',     'align': 'left'},
    {'name': 'category',    'label': '类型',     'field': 'category',    'align': 'left', 'sortable': True},
    {'name': 'nsfw',        'label': '分级',     'field': 'nsfw',        'align': 'center', 'sortable': True},
    {'name': 'final_score', 'label': '综合分',   'field': 'final_score', 'sortable': True},
    {'name': 'count',       'label': '热度',     'field': 'count',       'sortable': True},
]

OPTIONAL_COLS = {
    'semantic': {'name': 'semantic_score', 'label': '语义分',   'field': 'semantic_score', 'sortable': True},
    'layer':    {'name': 'layer',          'label': '匹配层',   'field': 'layer'},
    'source':   {'name': 'source',         'label': '匹配来源', 'field': 'source'},
}

# TagResult 2 dict
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


# ── 面向对象重构的 UI 类 ───────────────────────────────────

class DanbooruSearchUI:
    def __init__(self):
        # 页面状态 (替代原先的 list[None] 做法)
        self.search_count_label = None
        self.full_table_data: list[dict] = []
        self.current_query_str: str = ""
        self.full_tags_str: str = ""
        self.full_tags_str_sfw: str = ""

        self.result_table = None
        self.all_result_area = None
        self.selection_count_label = None
        self.selected_display = None
        self.related_container = None
        self.current_related: list = []
        self.chip_extra_selected: set = set()

        # UI 控件引用
        self.init_banner = None
        self.input_top_k = None
        self.input_limit = None
        self.input_weight = None
        self.input_nsfw = None
        self.input_segment = None
        self.sw_semantic = None
        self.sw_layer = None
        self.sw_source = None
        self.search_input = None
        self.keywords_container = None
        self.spinner = None
        self.search_btn = None

        self.selected_layers = {'英文': True, '中文扩展词': True, '释义': True, '中文核心词': True}
        self.selected_cats = {'General': True, 'Copyright': True, 'Character': True}

    def build_page(self):
        ui.colors(primary='#4A90E2', secondary='#5E6C84', accent='#FF6B6B')
        ui.add_head_html('''
            <meta name="description" content="基于语义匹配的 Danbooru 标签搜索引擎，支持中英双语描述、多维匹配、智能分词与共现关联推荐。">
            <meta name="keywords" content="Danbooru, AI绘画, Stable Diffusion, 提示词, 标签搜索,RAG, Prompt, NovelAI">

            <style>
                .nsfw-blur-cell      { filter: blur(8px); opacity: 0.5; transition: all 0.3s ease;
                                       pointer-events: none !important; user-select: none !important; }
                .nsfw-checkbox-disabled { pointer-events: none !important; opacity: 0.3 !important; }
                .nsfw-row-blocked    { cursor: not-allowed !important; }
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

        # 引擎预热提示
        self.init_banner = ui.card().classes(
            'w-full max-w-6xl mx-auto bg-blue-50 border-l-4 border-blue-400 mb-2'
        )
        with self.init_banner:
            with ui.row().classes('items-center gap-3 p-2'):
                ui.spinner(size='sm')
                ui.label('引擎初始化中，请稍候…首次加载约需 15 秒').classes('text-sm text-blue-700')
        self.init_banner.set_visibility(not DanbooruTagger.is_ready())

        if not DanbooruTagger.is_ready():
            asyncio.ensure_future(self._hide_banner_when_ready())

        with ui.card().classes('w-full max-w-6xl mx-auto bg-orange-50 border-l-4 border-orange-500 mb-2'):
            with ui.column().classes('gap-1'):
                ui.label('⚠️ 注意事项 / Note').classes('text-lg font-bold text-orange-800')
                ui.markdown("""
- **AI 辅助**：基于语义匹配，结果未必绝对准确 (Results may contain errors)
- **内容警告**：查找结果可能会包括 NSFW 内容 (May include NSFW content)
- **检索限制**：仅支持中/英双语查找(CN/EN only)
- **标签类型**：仅显示特征、角色与作品标签，且仅显示 Danbooru 频数 ≥100 的标签 (General, Character, and Copyright only, Freq>=100)
- **使用指南**：[DanbooruSearchOnline](https://github.com/SuzumiyaAkizuki/DanbooruSearchOnline)
- **ComfyUI 插件**：[ComfyUI-DanbooruSearcher](https://github.com/SuzumiyaAkizuki/ComfyUI-DanbooruSearcher)
- **支持作者**：如果觉得好用，请点击顶部给本 Space 点个 **Like ❤️**，或前往 GitHub 点个 **Star ⭐**！
""").classes('text-sm text-gray-800 ml-4')

        with ui.column().classes('w-full max-w-6xl mx-auto p-4 gap-6'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('search', size='2em', color='primary')
                ui.label('Danbooru 标签模糊搜索').classes('text-2xl font-bold text-gray-800')

            # 基础控制面板
            with ui.card().classes('w-full'):
                with ui.grid(columns=4).classes('w-full gap-8 items-center'):
                    self.input_top_k = ui.number('Top K (语义相关)', value=5, min=1, max=50) \
                        .props('outlined dense suffix="个"').classes('w-full')
                    self.input_limit = ui.number('结果上限', value=80, min=10, max=500) \
                        .props('outlined dense suffix="个"').classes('w-full')
                    with ui.column().classes('gap-0'):
                        with ui.row().classes('w-full justify-between'):
                            ui.label('热度权重').classes('text-xs text-gray-500')
                            self.input_weight = ui.slider(min=0.0, max=1.0, value=0.15, step=0.05).classes('w-full')
                            ui.label().bind_text_from(self.input_weight, 'value', lambda v: f"{v:.2f}")
                    self.input_nsfw = ui.switch('显示 NSFW', value=False).props('color=red').classes('w-full')
                    self.input_nsfw.on('update:model-value', self.on_nsfw_toggle)

            # 高级设置
            with ui.expansion('高级设置 (Advanced Settings)', icon='tune').classes('w-full bg-gray-50 border rounded-lg'):
                with ui.column().classes('w-full p-4 gap-4'):
                    self.input_segment = ui.switch('启用智能分词 (Segmentation)', value=True).props('color=primary')
                    ui.label('关闭后系统将只匹配完整句子，适用于精准搜索整句。').classes('text-xs text-gray-500 -mt-2 ml-10')
                    ui.separator()

                    ui.label('匹配层筛选 (Target Layers):').classes('font-bold text-gray-700')
                    layer_options = ['英文', '中文扩展词', '释义', '中文核心词']
                    with ui.row().classes('gap-4'):
                        for layer in layer_options:
                            ui.checkbox(layer, value=True,
                                        on_change=lambda e, l=layer: self.selected_layers.__setitem__(l, e.value))

                    ui.separator()
                    ui.label('标签类型筛选 (Categories):').classes('font-bold text-gray-700')
                    cat_options = ['General', 'Copyright', 'Character']
                    color_map = {'General': 'blue', 'Copyright': 'pink', 'Character': 'green'}
                    with ui.row().classes('gap-4 flex-wrap'):
                        for cat in cat_options:
                            ui.checkbox(cat, value=True,
                                        on_change=lambda e, c=cat: self.selected_cats.__setitem__(c, e.value)) \
                                .props(f'color={color_map.get(cat, "primary")}')

                    ui.separator()
                    ui.label('表格显示选项 (Display Options):').classes('font-bold text-gray-700')
                    self.sw_semantic = ui.switch('显示语义分', value=False)
                    self.sw_layer    = ui.switch('显示匹配层', value=False)
                    self.sw_source   = ui.switch('显示匹配来源', value=False)
                    self.sw_semantic.on('update:model-value', self.update_table_columns)
                    self.sw_layer.on('update:model-value', self.update_table_columns)
                    self.sw_source.on('update:model-value', self.update_table_columns)

            # 搜索输入
            with ui.card().classes('w-full p-0 overflow-hidden'):
                with ui.column().classes('w-full p-6 gap-4'):
                    ui.label('画面描述').classes('text-lg font-bold text-gray-700')
                    self.search_input = ui.textarea(
                        placeholder='例如：一个穿着白色水手服的女孩在雨中奔跑'
                    ).classes('w-full text-lg').props('outlined rows=3')

                    self.keywords_container = ui.row().classes('gap-2 items-center')

                    with ui.row().classes('w-full justify-end items-center gap-4'):
                        self.spinner = ui.spinner(size='2em').classes('hidden')
                        self.search_btn = ui.button('开始搜索', on_click=self.perform_search, icon='search')
                        self.search_btn.classes('px-8 py-2 text-lg').props('unelevated color=primary')

                    self.search_input.on('keydown.ctrl.enter', self.perform_search)

            # 结果区
            with ui.row().classes('w-full gap-6'):
                with ui.card().classes('w-1/3 flex-grow'):
                    ui.label('推荐 Prompt (全部)').classes('font-bold text-gray-600')
                    self.all_result_area = ui.textarea().classes('w-full h-full bg-gray-50') \
                        .props('readonly outlined input-class=text-sm')

                with ui.column().classes('w-2/3 flex-grow'):
                    with ui.card().classes('w-full bg-blue-50 border-blue-200 border'):
                        with ui.row().classes('w-full items-center justify-between'):
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('check_circle', color='primary')
                                ui.label('已选标签:').classes('font-bold text-primary')
                                self.selection_count_label = ui.label('0').classes(
                                    'bg-primary text-white px-2 rounded-full text-sm')
                            copy_btn = ui.button('复制选中', icon='content_copy').props('dense unelevated color=primary')
                            copy_btn.on_click(self.copy_selection)

                        self.selected_display = ui.textarea().classes('w-full mt-2') \
                            .props('outlined dense rows=2 readonly bg-white')

                    # 关联推荐折叠面板
                    with ui.expansion(
                        '关联推荐',
                        icon='auto_awesome',
                        value=True,
                    ).classes('w-full bg-purple-50 border border-purple-200 rounded-lg mt-2'):
                        with ui.column().classes('w-full p-3 gap-2'):
                            ui.label(
                                '基于标签共现数据，为您推荐更多可能的标签。勾选结果行后自动更新；点击标签可加入或移出已选。'
                            ).classes('text-xs text-gray-500')
                            self.related_container = ui.row().classes('gap-2 flex-wrap items-center min-h-8')
                            with self.related_container:
                                ui.label('请先搜索…').classes('text-xs text-gray-400 italic')

                    self.result_table = ui.table(
                        columns=BASE_COLUMNS,
                        rows=[],
                        pagination=10,
                        selection='multiple',
                        row_key='tag',
                    ).classes('w-full')
                    self.result_table.on('selection', self._update_selection_display)

                    # Vue slots
                    self.result_table.add_slot('body', '''
                        <q-tr :props="props" :class="props.row._nsfw_blocked ? 'nsfw-row-blocked' : ''">
                            <q-td auto-width>
                                <q-checkbox v-model="props.selected"
                                    :class="props.row._nsfw_blocked ? 'nsfw-checkbox-disabled' : ''"/>
                            </q-td>
                            <q-td v-for="col in props.cols" :key="col.name" :props="props">
                                <template v-if="col.name === 'tag' || col.name === 'cn_name'">
                                    <div :class="props.row._nsfw_blocked ? 'nsfw-blur-cell' : ''">
                                        <template v-if="col.name === 'cn_name' && col.value">
                                            <q-badge v-for="(item, index) in col.value.split(',')" :key="index"
                                                :color="index === 0 ? 'black' : 'grey'" outline
                                                style="font-size:14px" class="q-mr-xs q-mb-xs cursor-help">
                                                {{ item }}
                                            </q-badge>
                                            <q-tooltip v-if="props.row.wiki"
                                                content-class="bg-black text-white shadow-4"
                                                max-width="500px" :offset="[10,10]">
                                                <div style="font-size:14px;line-height:1.5;">{{ props.row.wiki }}</div>
                                            </q-tooltip>
                                        </template>
                                        <template v-else-if="col.name === 'tag'">
                                            <a :href="'https://danbooru.donmai.us/wiki_pages/'+col.value"
                                               target="_blank"
                                               class="text-primary hover:underline font-bold inline-flex items-center"
                                               style="text-decoration:none;" @click.stop>
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
                                <template v-else-if="col.name === 'category'">
                                    <q-badge :color="
                                        col.value === 'General'   ? 'blue'  :
                                        (col.value === 'Character' ? 'green' :
                                        (col.value === 'Copyright' ? 'pink'  : 'red'))" outline>
                                        {{ col.value }}
                                    </q-badge>
                                </template>
                                <template v-else>{{ col.value }}</template>
                            </q-td>
                        </q-tr>
                    ''')

        # 底部计数页脚
        with ui.element('div').classes('w-full text-center py-4 mt-2'):
            self.search_count_label = ui.label(f'累计搜索 {counter.get():,} 次').classes('text-xs text-gray-400')


    # ── 交互与业务逻辑 ─────────────────────────────────────────

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

    def filter_table_by_source(self, keyword: str):
        filtered = (self.full_table_data if (not keyword or keyword == 'ALL')
                    else [r for r in self.full_table_data if r['source'] == keyword])
        self.result_table.rows = apply_nsfw_filter(filtered, self.input_nsfw.value)

        for child in self.keywords_container.default_slot.children:
            if isinstance(child, ui.chip):
                selected = (
                    (keyword == 'ALL' and child.text == '全部')
                    or (keyword == self.current_query_str and child.text == '整句')
                    or (child.text == keyword)
                )
                child.props(
                    f'color={"primary" if selected else "grey-4"} '
                    f'text-color={"white" if selected else "black"}'
                )

    async def perform_search(self):
        query = self.search_input.value.strip()
        if not query:
            return

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
            request = SearchRequest(
                query=query,
                top_k=int(self.input_top_k.value),
                limit=int(self.input_limit.value),
                popularity_weight=float(self.input_weight.value),
                show_nsfw=self.input_nsfw.value,
                use_segmentation=self.input_segment.value,
                target_layers=target_layers_list,
                target_categories=target_cats_list,
            )
            response = await run.io_bound(tagger.search, request)

            async def silent_counter_update():
                try:
                    new_count = await counter.increment()
                    if self.search_count_label is not None:
                        self.search_count_label.text = f'累计搜索 {new_count:,} 次'
                except Exception as e:
                    print(f"[Counter Error] 后台静默更新计数器失败: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

            asyncio.create_task(silent_counter_update())

            await ui.run_javascript(
                f"typeof gtag !== 'undefined' && gtag('event', 'search', {{'search_term': {_json.dumps(query)}}});"
            )

            if not self._client_alive():
                return

            table_data = [result_to_row(r, self.input_nsfw.value) for r in response.results]
            self.full_table_data = table_data
            self.full_tags_str = response.tags_all
            self.full_tags_str_sfw = response.tags_sfw

            self.all_result_area.value = (
                response.tags_sfw if not self.input_nsfw.value else response.tags_all
            )
            self.result_table.rows = apply_nsfw_filter(table_data, self.input_nsfw.value)
            self.result_table.selected = []
            self._update_selection_display(None)

            self._refresh_related([], self.input_nsfw.value)

            self.keywords_container.clear()
            with self.keywords_container:
                ui.label('分词筛选:').classes('text-sm text-gray-500 font-bold mr-2')
                ui.chip('全部', on_click=lambda: self.filter_table_by_source('ALL')) \
                    .props('color=primary text-color=white clickable')
                if self.input_segment.value:
                    ui.chip('整句',
                            on_click=lambda: self.filter_table_by_source(self.current_query_str)) \
                        .props('color=grey-4 text-color=black clickable')
                    for kw in response.keywords:
                        ui.chip(kw,
                                on_click=lambda k=kw: self.filter_table_by_source(k)) \
                            .props('color=grey-4 text-color=black clickable')
                else:
                    ui.label('(分词已关闭)').classes('text-xs text-gray-400')

            ui.notify(f'找到 {len(table_data)} 个标签', type='positive')

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

    def copy_selection(self):
        ui.clipboard.write(self.selected_display.value)
        ui.notify('已复制选中标签!', type='positive')

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

        if self.result_table is not None:
            self.result_table.selected = [row for row in self.result_table.rows if row.get('tag') in tag_set]

        all_tags = self._get_selected_tags()
        if self.selected_display is not None:
            self.selected_display.value = ', '.join(all_tags)
        if self.selection_count_label is not None:
            self.selection_count_label.text = str(len(all_tags))

    def _update_selection_display(self, _e):
        if self.result_table is None:
            return
        all_tags = self._get_selected_tags()
        self.selected_display.value = ", ".join(all_tags)
        self.selection_count_label.text = str(len(all_tags))
        if all_tags:
            self._refresh_related_from_selection(all_tags, self.input_nsfw.value)
        else:
            self.chip_extra_selected.clear()
            self._refresh_related([], self.input_nsfw.value)

    def _render_related_chips(self, related: list, show_nsfw: bool):
        CAT_CHIP_COLORS = {
            'General': 'blue', 'Character': 'green',
            'Copyright': 'pink', 'Artist': 'orange',
        }
        filtered = [r for r in related if not (r.nsfw == '1' and not show_nsfw)]
        if not filtered:
            ui.label('暂无推荐').classes('text-xs text-gray-400 italic')
            return
        selected_now = set(self._get_selected_tags())
        for r in filtered:
            color = CAT_CHIP_COLORS.get(r.category, 'grey')
            is_selected = r.tag in selected_now
            label = r.tag + (' 🔴' if r.nsfw == '1' else '')
            sources_str = '、'.join(r.sources) if r.sources else '—'
            tooltip = (
                f"{r.cn_name}\n"
                f"共现: {r.cooc_count:,}  相关度: {r.cooc_score:.2f}\n"
                f"来自选中: {sources_str}"
            ) if r.cn_name else f"共现: {r.cooc_count:,}\n来自选中: {sources_str}"
            props = f'color={color} clickable' if is_selected else f'color={color} outline clickable'
            with ui.chip(label).props(props) as chip:
                ui.tooltip(tooltip).classes('text-sm whitespace-pre')

            def _on_click(tag=r.tag):
                current = self._get_selected_tags()
                if tag in current:
                    current.remove(tag)
                    self._set_selected_tags(current)
                    ui.notify(f'已移除 {tag}', type='warning', timeout=1500)
                else:
                    current.append(tag)
                    self._set_selected_tags(current)
                    ui.notify(f'已添加 {tag}', type='positive', timeout=1500)
                self._refresh_related(self.current_related, show_nsfw)
            chip.on('click', _on_click)

    def _refresh_related(self, related: list, show_nsfw: bool):
        selected_now = set(self._get_selected_tags())
        old_related  = self.current_related
        new_tags  = {r.tag for r in related}
        preserved = [r for r in old_related if r.tag in selected_now and r.tag not in new_tags]
        merged = list(related) + preserved

        self.current_related = merged
        if self.related_container is None:
            return
        self.related_container.clear()
        with self.related_container:
            self._render_related_chips(merged, show_nsfw)

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

    def update_table_columns(self, e=None):
        cols = list(BASE_COLUMNS)
        if self.sw_semantic.value: cols.append(OPTIONAL_COLS['semantic'])
        if self.sw_layer.value:    cols.append(OPTIONAL_COLS['layer'])
        if self.sw_source.value:   cols.append(OPTIONAL_COLS['source'])
        self.result_table.columns = cols

    def handle_nsfw_change(self, val: bool):
        self.result_table.rows = apply_nsfw_filter(self.full_table_data, val)
        if not val:
            self.result_table.selected = [r for r in self.result_table.selected if r.get('nsfw') != '1']
        if self.full_tags_str or self.full_tags_str_sfw:
            self.all_result_area.value = self.full_tags_str if val else self.full_tags_str_sfw
        self._update_selection_display(None)

    def on_nsfw_toggle(self, e):
        args = e.args
        val = args[0] if isinstance(args, list) else args.get('value', args) if isinstance(args, dict) else args
        self.handle_nsfw_change(bool(val))


# 页面

@ui.page('/')
async def main_page():
    # 使用面向对象模式，确保每个独立访问者的状态互相隔离
    app_ui = DanbooruSearchUI()
    app_ui.build_page()


# 入口
if __name__ in {'__main__', '__mp_main__'}:
    host = '0.0.0.0' if is_running_on_huggingface_space() else '127.0.0.1'
    port = 7860 if is_running_on_huggingface_space() else 8888

    # 程序启动时立即在后台预热引擎，不等用户第一次搜索
    @app.on_startup
    def _warmup():
        async def background_init_tasks():
            # ⚠️ 给 Uvicorn 留出 5 秒钟绑定端口并响应 HF 的健康检查！
            await asyncio.sleep(5)
            print("==== [System] 开始预热计数器与引擎 ====", flush=True)

            await counter.init()
            await DanbooruTagger.get_instance()

            print("==== [System] 后台预热全部完成！ ====", flush=True)

        asyncio.create_task(background_init_tasks())

    # 把 FastAPI 子应用挂载到 /api，与 UI 共用同一端口
    app.mount('/api', api_app)

    ui.run(
        host=host,
        port=port,
        title='Danbooru Tags Searcher',
        reload=not is_running_on_huggingface_space(),
        show=not is_running_on_huggingface_space(),
        reconnect_timeout=120,  # 给引擎冷启动足够的时间（秒）
    )