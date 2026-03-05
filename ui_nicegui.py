"""
ui_nicegui.py
─────────────
NiceGUI 前端层。

▸ 只负责渲染 / 交互。
▸ 调用 core.engine.DanbooruTagger，通过 core.models 的数据结构通信。
▸ 不包含任何算法逻辑。
"""

import asyncio
import os
from dataclasses import asdict
from nicegui import ui, app, run

from core.engine import DanbooruTagger
from core.models import RelatedTag, SearchRequest



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


# 页面

@ui.page('/')
async def main_page():
    ui.colors(primary='#4A90E2', secondary='#5E6C84', accent='#FF6B6B')

    # 引擎预热提示
    init_banner = ui.card().classes(
        'w-full max-w-6xl mx-auto bg-blue-50 border-l-4 border-blue-400 mb-2'
    )
    with init_banner:
        with ui.row().classes('items-center gap-3 p-2'):
            ui.spinner(size='sm')
            ui.label('引擎初始化中，请稍候…首次加载约需 15 秒').classes('text-sm text-blue-700')
    init_banner.set_visibility(not DanbooruTagger.is_ready())

    async def _hide_banner_when_ready():
        while not DanbooruTagger.is_ready():
            await asyncio.sleep(1)
        init_banner.set_visibility(False)

    if not DanbooruTagger.is_ready():
        asyncio.ensure_future(_hide_banner_when_ready())

    ui.add_head_html('''
        <style>
            .nsfw-blur-cell      { filter: blur(8px); opacity: 0.5; transition: all 0.3s ease;
                                   pointer-events: none !important; user-select: none !important; }
            .nsfw-checkbox-disabled { pointer-events: none !important; opacity: 0.3 !important; }
            .nsfw-row-blocked    { cursor: not-allowed !important; }
        </style>
    ''')

    # 页面状态
    full_table_data: list[dict] = []
    current_query_str: str = ""
    full_tags_str:     list[str] = [""]
    full_tags_str_sfw: list[str] = [""]

    with ui.card().classes('w-full max-w-6xl mx-auto bg-orange-50 border-l-4 border-orange-500 mb-2'):
        with ui.column().classes('gap-1'):
            ui.label('⚠️ 注意事项 / Note').classes('text-lg font-bold text-orange-800')
            ui.markdown("""
- **本网站为AI工具，其结果未必正确无误** (Results may contain errors)
- **查找结果可能会包括 NSFW 内容** (Results may include NSFW content)
- **仅支持汉语、英语查找** (Only supports Chinese/English)
- **仅显示Danbooru频数超过100的标签** (Frequency >= 100)
- **仅显示特征、角色、作品标签**  (General Character and Copyright tags only)
- **Comfy UI 插件地址** ： https://github.com/SuzumiyaAkizuki/ComfyUI-DanbooruSearcher
""").classes('text-sm text-gray-800 ml-4')

    with ui.column().classes('w-full max-w-6xl mx-auto p-4 gap-6'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('search', size='2em', color='primary')
            ui.label('Danbooru 标签模糊搜索').classes('text-2xl font-bold text-gray-800')

        # 基础控制面板
        with ui.card().classes('w-full'):
            with ui.grid(columns=4).classes('w-full gap-8 items-center'):
                input_top_k = ui.number('Top K (语义相关)', value=5, min=1, max=50) \
                    .props('outlined dense suffix="个"').classes('w-full')
                input_limit = ui.number('结果上限', value=80, min=10, max=500) \
                    .props('outlined dense suffix="个"').classes('w-full')
                with ui.column().classes('gap-0'):
                    with ui.row().classes('w-full justify-between'):
                        ui.label('热度权重').classes('text-xs text-gray-500')
                        input_weight = ui.slider(min=0.0, max=1.0, value=0.15, step=0.05).classes('w-full')
                        ui.label().bind_text_from(input_weight, 'value', lambda v: f"{v:.2f}")
                input_nsfw = ui.switch('显示 NSFW', value=False).props('color=red').classes('w-full')

        # 高级设置
        with ui.expansion('高级设置 (Advanced Settings)', icon='tune').classes('w-full bg-gray-50 border rounded-lg'):
            with ui.column().classes('w-full p-4 gap-4'):
                input_segment = ui.switch('启用智能分词 (Segmentation)', value=True).props('color=primary')
                ui.label('关闭后系统将只匹配完整句子，适用于精准搜索整句。').classes('text-xs text-gray-500 -mt-2 ml-10')
                ui.separator()

                ui.label('匹配层筛选 (Target Layers):').classes('font-bold text-gray-700')
                layer_options = ['英文', '中文扩展词', '释义', '中文核心词']
                selected_layers = {l: True for l in layer_options}
                with ui.row().classes('gap-4'):
                    for layer in layer_options:
                        ui.checkbox(layer, value=True,
                                    on_change=lambda e, l=layer: selected_layers.__setitem__(l, e.value))

                ui.separator()
                ui.label('标签类型筛选 (Categories):').classes('font-bold text-gray-700')
                cat_options = ['General', 'Copyright', 'Character']
                selected_cats = {c: True for c in cat_options}
                color_map = {'General': 'blue', 'Copyright': 'pink', 'Character': 'green'}
                with ui.row().classes('gap-4 flex-wrap'):
                    for cat in cat_options:
                        ui.checkbox(cat, value=True,
                                    on_change=lambda e, c=cat: selected_cats.__setitem__(c, e.value)) \
                            .props(f'color={color_map.get(cat, "primary")}')

                ui.separator()
                ui.label('表格显示选项 (Display Options):').classes('font-bold text-gray-700')
                sw_semantic = ui.switch('显示语义分', value=False)
                sw_layer    = ui.switch('显示匹配层', value=False)
                sw_source   = ui.switch('显示匹配来源', value=False)

        # 搜索输入
        with ui.card().classes('w-full p-0 overflow-hidden'):
            with ui.column().classes('w-full p-6 gap-4'):
                ui.label('画面描述').classes('text-lg font-bold text-gray-700')
                search_input = ui.textarea(
                    placeholder='例如：一个穿着白色水手服的女孩在雨中奔跑'
                ).classes('w-full text-lg').props('outlined rows=3')

                keywords_container = ui.row().classes('gap-2 items-center')
                spinner = ui.spinner(size='2em').classes('hidden')

                # 结果区
                result_table_ref:     list = [None]
                all_result_area_ref:  list = [None]
                selection_count_ref:  list = [None]
                selected_display_ref: list = [None]
                related_container_ref: list = [None]  # 关联推荐
                current_related_ref:  list = [[] ]   # 当前正在展示的 related 列表

                def filter_table_by_source(keyword: str):
                    nonlocal full_table_data, current_query_str
                    filtered = (full_table_data if (not keyword or keyword == 'ALL')
                                else [r for r in full_table_data if r['source'] == keyword])
                    result_table_ref[0].rows = apply_nsfw_filter(filtered, input_nsfw.value)

                    for child in keywords_container.default_slot.children:
                        if isinstance(child, ui.chip):
                            selected = (
                                (keyword == 'ALL' and child.text == '全部')
                                or (keyword == current_query_str and child.text == '整句')
                                or (child.text == keyword)
                            )
                            child.props(
                                f'color={"primary" if selected else "grey-4"} '
                                f'text-color={"white" if selected else "black"}'
                            )

                def _client_alive() -> bool:
                    """检查当前 NiceGUI 客户端连接是否仍然存活。"""
                    try:
                        _ = search_btn.client
                        return True
                    except RuntimeError:
                        return False

                async def perform_search():
                    nonlocal full_table_data, current_query_str

                    query = search_input.value.strip()
                    if not query:
                        return

                    current_query_str = query
                    search_btn.disable()
                    spinner.classes(remove='hidden')
                    ui.notify('正在搜索...', type='info')

                    target_layers_list = [k for k, v in selected_layers.items() if v]
                    target_cats_list   = [k for k, v in selected_cats.items()   if v]

                    if not target_layers_list:
                        ui.notify('请至少选择一个匹配层！', type='warning')
                        search_btn.enable()
                        spinner.classes(add='hidden')
                        return

                    try:
                        tagger = await DanbooruTagger.get_instance()

                        # ▸ 构造 SearchRequest，传给 Engine
                        request = SearchRequest(
                            query=query,
                            top_k=int(input_top_k.value),
                            limit=int(input_limit.value),
                            popularity_weight=float(input_weight.value),
                            show_nsfw=input_nsfw.value,
                            use_segmentation=input_segment.value,
                            target_layers=target_layers_list,
                            target_categories=target_cats_list,
                        )
                        response = await run.io_bound(tagger.search, request)

                        if not _client_alive():
                            return

                        table_data = [result_to_row(r, input_nsfw.value) for r in response.results]
                        full_table_data      = table_data
                        full_tags_str[0]     = response.tags_all
                        full_tags_str_sfw[0] = response.tags_sfw

                        all_result_area_ref[0].value = (
                            response.tags_sfw if not input_nsfw.value else response.tags_all
                        )
                        result_table_ref[0].rows     = apply_nsfw_filter(table_data, input_nsfw.value)
                        result_table_ref[0].selected = []
                        _update_selection_display(None)

                        _refresh_related([], input_nsfw.value)

                        # 分词
                        keywords_container.clear()
                        with keywords_container:
                            ui.label('分词筛选:').classes('text-sm text-gray-500 font-bold mr-2')
                            ui.chip('全部', on_click=lambda: filter_table_by_source('ALL')) \
                                .props('color=primary text-color=white clickable')
                            if input_segment.value:
                                ui.chip('整句',
                                        on_click=lambda: filter_table_by_source(current_query_str)) \
                                    .props('color=grey-4 text-color=black clickable')
                                for kw in response.keywords:
                                    ui.chip(kw,
                                            on_click=lambda k=kw: filter_table_by_source(k)) \
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
                            search_btn.enable()
                            spinner.classes(add='hidden')
                        except RuntimeError:
                            pass

                with ui.row().classes('w-full justify-end items-center gap-4'):
                    spinner
                    search_btn = ui.button('开始搜索', on_click=perform_search, icon='search')
                    search_btn.classes('px-8 py-2 text-lg').props('unelevated color=primary')

                search_input.on('keydown.ctrl.enter', perform_search)

        # 结果区
        with ui.row().classes('w-full gap-6'):
            with ui.card().classes('w-1/3 flex-grow'):
                ui.label('推荐 Prompt (全部)').classes('font-bold text-gray-600')
                all_result_area = ui.textarea().classes('w-full h-full bg-gray-50') \
                    .props('readonly outlined input-class=text-sm')
                all_result_area_ref[0] = all_result_area

            with ui.column().classes('w-2/3 flex-grow'):
                with ui.card().classes('w-full bg-blue-50 border-blue-200 border'):
                    with ui.row().classes('w-full items-center justify-between'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('check_circle', color='primary')
                            ui.label('已选标签:').classes('font-bold text-primary')
                            selection_count_label = ui.label('0').classes(
                                'bg-primary text-white px-2 rounded-full text-sm')
                            selection_count_ref[0] = selection_count_label
                        copy_btn = ui.button('复制选中', icon='content_copy').props('dense unelevated color=primary')

                    selected_display = ui.textarea().classes('w-full mt-2') \
                        .props('outlined dense rows=2 readonly bg-white')
                    selected_display_ref[0] = selected_display

                    def copy_selection():
                        ui.clipboard.write(selected_display.value)
                        ui.notify('已复制选中标签!', type='positive')

                    copy_btn.on_click(copy_selection)

                def _update_selection_display(_e):
                    if result_table_ref[0] is None:
                        return
                    tags = [row['tag'] for row in result_table_ref[0].selected]
                    selected_display_ref[0].value = ", ".join(tags)
                    selection_count_ref[0].text   = str(len(tags))
                    if tags:
                        _refresh_related_from_selection(tags, input_nsfw.value)
                    else:
                        _refresh_related([], input_nsfw.value)

                # 关联推荐辅助函数
                CAT_CHIP_COLORS = {
                    'General': 'blue', 'Character': 'green',
                    'Copyright': 'pink', 'Artist': 'orange',
                }

                def _get_selected_tags() -> list[str]:
                    disp = selected_display_ref[0]
                    if disp is None:
                        return []
                    return [t.strip() for t in disp.value.split(',') if t.strip()]

                def _set_selected_tags(tags: list[str]):
                    disp = selected_display_ref[0]
                    cnt  = selection_count_ref[0]
                    if disp is not None:
                        disp.value = ', '.join(tags)
                    if cnt is not None:
                        cnt.text = str(len(tags))
                    tbl = result_table_ref[0]
                    if tbl is not None:
                        tag_set = set(tags)
                        tbl.selected = [row for row in tbl.rows if row.get('tag') in tag_set]

                def _render_related_chips(related: list, show_nsfw: bool):
                    """在 related_container 内渲染推荐 chips（调用前 container 已 clear）。"""
                    filtered = [r for r in related if not (r.nsfw == '1' and not show_nsfw)]
                    if not filtered:
                        ui.label('暂无推荐').classes('text-xs text-gray-400 italic')
                        return
                    selected_now = set(_get_selected_tags())
                    for r in filtered:
                        color       = CAT_CHIP_COLORS.get(r.category, 'grey')
                        is_selected = r.tag in selected_now
                        label       = r.tag + (' 🔴' if r.nsfw == '1' else '')
                        sources_str = '、'.join(r.sources) if r.sources else '—'
                        tooltip = (
                            f"{r.cn_name}\n"
                            f"共现: {r.cooc_count:,}  相关度: {r.cooc_score:.2f}\n"
                            f"来自选中: {sources_str}"
                        ) if r.cn_name else f"共现: {r.cooc_count:,}\n来自选中: {sources_str}"
                        # 已选用实心，未选用镂空
                        props = f'color={color} clickable' if is_selected else f'color={color} outline clickable'
                        with ui.chip(label).props(props) as chip:
                            ui.tooltip(tooltip).classes('text-sm whitespace-pre')

                        def _on_click(tag=r.tag):
                            current = _get_selected_tags()
                            if tag in current:
                                current.remove(tag)
                                _set_selected_tags(current)
                                ui.notify(f'已移除 {tag}', type='warning', timeout=1500)
                            else:
                                current.append(tag)
                                _set_selected_tags(current)
                                ui.notify(f'已添加 {tag}', type='positive', timeout=1500)
                            # 重绘 chips 同步高亮状态
                            _refresh_related(current_related_ref[0], show_nsfw)
                        chip.on('click', _on_click)

                def _refresh_related(related: list, show_nsfw: bool):
                    current_related_ref[0] = related  # 记录当前展示内容供 chip 点击后重绘
                    c = related_container_ref[0]
                    if c is None:
                        return
                    c.clear()
                    with c:
                        _render_related_chips(related, show_nsfw)

                def _refresh_related_from_selection(selected_tags: list[str], show_nsfw: bool):
                    """用已选 tag 异步重算关联推荐。"""
                    async def _do():
                        tagger  = await DanbooruTagger.get_instance()
                        related = await run.io_bound(
                            tagger.get_related,
                            selected_tags,
                            set(selected_tags),
                            50,
                            show_nsfw,
                        )
                        _refresh_related(related, show_nsfw)

                    asyncio.ensure_future(_do())

                # ── 关联推荐折叠面板 ──────────────────────────────────
                with ui.expansion(
                    '关联推荐',
                    icon='auto_awesome',
                    value=True,
                ).classes('w-full bg-purple-50 border border-purple-200 rounded-lg mt-2'):
                    with ui.column().classes('w-full p-3 gap-2'):
                        ui.label(
                            '基于标签共现数据，为您推荐更多可能的标签。勾选结果行后自动更新；点击标签可加入或移出已选。'
                        ).classes('text-xs text-gray-500')
                        related_row = ui.row().classes('gap-2 flex-wrap items-center min-h-8')
                        related_container_ref[0] = related_row
                        with related_row:
                            ui.label('请先搜索…').classes('text-xs text-gray-400 italic')

                result_table = ui.table(
                    columns=BASE_COLUMNS,
                    rows=[],
                    pagination=10,
                    selection='multiple',
                    row_key='tag',
                ).classes('w-full')
                result_table_ref[0] = result_table

                result_table.on('selection', _update_selection_display)

                # 动态列更新
                def update_table_columns():
                    cols = list(BASE_COLUMNS)
                    if sw_semantic.value: cols.append(OPTIONAL_COLS['semantic'])
                    if sw_layer.value:    cols.append(OPTIONAL_COLS['layer'])
                    if sw_source.value:   cols.append(OPTIONAL_COLS['source'])
                    result_table.columns = cols

                sw_semantic.on('update:model-value', update_table_columns)
                sw_layer.on('update:model-value',    update_table_columns)
                sw_source.on('update:model-value',   update_table_columns)

                # NSFW 切换
                def handle_nsfw_change(val: bool):
                    result_table.rows = apply_nsfw_filter(full_table_data, val)
                    if not val:
                        result_table.selected = [r for r in result_table.selected if r.get('nsfw') != '1']
                    if full_tags_str[0] or full_tags_str_sfw[0]:
                        all_result_area.value = full_tags_str[0] if val else full_tags_str_sfw[0]
                    _update_selection_display(None)

                def on_nsfw_toggle(e):
                    args = e.args
                    val = args[0] if isinstance(args, list) else args.get('value', args) if isinstance(args, dict) else args
                    handle_nsfw_change(bool(val))

                input_nsfw.on('update:model-value', on_nsfw_toggle)

                # Vue slots
                result_table.add_slot('body', '''
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



# 入口

if __name__ in {'__main__', '__mp_main__'}:
    host = '0.0.0.0' if is_running_on_huggingface_space() else '127.0.0.1'
    port = 7860       if is_running_on_huggingface_space() else 1145

    # 程序启动时立即在后台预热引擎，不等用户第一次搜索
    @app.on_startup
    async def _warmup():
        await DanbooruTagger.get_instance()

    ui.run(
        host=host,
        port=port,
        title='Danbooru Tags Searcher',
        reload=True,
        show=True,
        reconnect_timeout=120,  # 给引擎冷启动足够的时间（秒）
    )