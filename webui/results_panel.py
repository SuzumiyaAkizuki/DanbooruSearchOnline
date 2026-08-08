"""结果区域的 NiceGUI 布局、查询理解和表格显示逻辑。"""

from typing import Any

from nicegui import ui

from core.workspace_insights import (
    CANDIDATE_UNSELECTED,
    COVERED,
    UNCOVERED,
    compute_concept_coverage,
)
from webui.constants import OPTIONAL_COLS, TABLE_COLUMNS
from webui.helpers import apply_nsfw_filter


RESULT_TABLE_BODY = r'''
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
'''


def build_results_columns(controller: Any) -> None:
    """构建搜索结果与三个推荐区的静态容器。"""
    controller.two_col_container = ui.element('div').classes('w-full two-col-layout')
    with controller.two_col_container:
        with ui.card().classes('col-left section-surface'):
            with ui.row().classes('items-center justify-between mb-2 w-full'):
                ui.label('匹配标签结果').classes('section-heading')
                ui.button('复制全部标签', icon='content_copy', on_click=controller._copy_all_tags) \
                    .props('dense flat color=primary').classes('text-sm')

            controller.result_table = ui.table(
                columns=TABLE_COLUMNS,
                rows=[],
                pagination=0,
                selection='multiple',
                row_key='tag',
            ).props('flat separator=horizontal').classes('w-full result-table-flat')
            controller.result_table.on('selection', controller._update_selection_display)
            controller.result_table.on('link_click', controller._mark_interaction)
            controller.result_table.on('translation_feedback', controller.report_translation_error)
            controller.result_table.on('pagination', lambda _: controller._save_config())
            controller.result_table.add_slot('body', RESULT_TABLE_BODY)

            ui.separator().classes('my-2')
            with ui.row().classes('items-center justify-between w-full mb-1'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('同类标签').classes('font-bold text-sm text-gray-600')
                    with ui.icon('info_outline', size='xs', color='grey').classes('cursor-help'):
                        with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                            ui.label('基于标签分组数据，展示已选标签所属分组中的其他标签。勾选可加入已选。').style('font-size:14px;')
                ui.button('根据已选刷新', icon='refresh', on_click=controller._manual_refresh_group) \
                    .props('dense flat color=primary').classes('text-sm')
            controller.group_expansion_container = ui.column().classes('w-full gap-0')
            controller.group_expansion_container.props('id="danbooru-group-expansion"')
            with controller.group_expansion_container:
                ui.label('请先搜索并勾选标签…').classes('text-sm text-gray-400 italic p-4')

        with ui.card().classes('col-right section-surface'):
            with ui.row().classes('items-center justify-between w-full mb-2'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('推荐擅长画师(Beta)').classes('section-heading')
                    with ui.icon('info_outline', size='sm', color='grey').classes('cursor-help'):
                        with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                            ui.html(
                                '基于标签-画师 NPMI 共现数据，根据您当前已选的标签，推荐擅长这些元素的画师。<br>悬停画师行可查看与该画师共现关联最强的标签。'
                            ).style('font-size:14px;line-height:1.5;')
            controller.artist_rec_list = ui.column().classes(
                'w-full gap-0 recommendation-grid'
            ).props('id="danbooru-artist-recommendations"')
            with controller.artist_rec_list:
                ui.label('请先搜索并勾选标签…').classes('text-sm text-gray-400 italic p-4')
            controller.artist_rec_pagination = ui.column().classes('w-full')

            ui.separator().classes('my-3')
            with ui.row().classes('items-center justify-between w-full mb-2'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('关联推荐').classes('section-heading')
                    with ui.icon('info_outline', size='sm', color='grey').classes('cursor-help'):
                        with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                            ui.html(
                                '基于标签共现数据，发掘语义之外的相关性，为您推荐更多可能的标签。<br>勾选可加入或移出已选。如需根据最新选项更新推荐，请点击刷新按钮。'
                            ).style('font-size:14px;line-height:1.5;')
                ui.button('根据已选刷新', icon='refresh', on_click=controller._manual_refresh_related) \
                    .props('dense flat color=primary').classes('text-sm')
            controller.related_list_container = ui.column().classes(
                'w-full gap-0 recommendation-grid'
            ).props('id="danbooru-related-recommendations"')
            with controller.related_list_container:
                ui.label('请先搜索并勾选标签…').classes('text-sm text-gray-400 italic p-4')
            controller.related_pagination = ui.column().classes('w-full')


def filter_by_source(controller: Any, keyword: str) -> None:
    """按查询概念过滤表格，并重建查询理解卡片。"""
    controller.current_filter_keyword = keyword if keyword else 'ALL'
    show_nsfw = controller.input_nsfw.value
    if not keyword or keyword == 'ALL':
        filtered = controller.full_table_data
    else:
        filtered = [row for row in controller.full_table_data if row['source'] == keyword]
    controller.result_table.rows = apply_nsfw_filter(filtered, show_nsfw)
    controller._render_concept_coverage()


def render_concept_coverage(controller: Any) -> None:
    """渲染查询概念覆盖状态与概念筛选入口。"""
    if controller.coverage_container is None:
        return
    controller.coverage_container.clear()
    if not controller.current_query_str:
        return

    segments = list(dict.fromkeys(controller.current_segments + controller.current_keywords))
    if not segments:
        segments = [controller.current_query_str]
    coverage = compute_concept_coverage(
        segments, controller.full_table_data, controller._get_selected_tags()
    )
    if not coverage:
        return

    status_counts = {
        COVERED: sum(item.status == COVERED for item in coverage),
        CANDIDATE_UNSELECTED: sum(item.status == CANDIDATE_UNSELECTED for item in coverage),
        UNCOVERED: sum(item.status == UNCOVERED for item in coverage),
    }

    def decorate_chip(chip: Any, source: str) -> None:
        if source in controller.current_cached_queries:
            chip.style('outline: 1px dashed rgba(100,116,139,0.45); outline-offset: 1px;')
        if source == controller.current_filter_keyword:
            chip.style('box-shadow: 0 0 0 2px #4a90e2;')

    with controller.coverage_container:
        with ui.element('div').classes('w-full query-insight-panel section-surface'):
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
                all_chip = ui.chip('全部结果', on_click=lambda: controller._filter_by_source('ALL'))
                if controller.current_filter_keyword == 'ALL':
                    all_chip.props('color=primary text-color=white clickable')
                else:
                    all_chip.props('color=grey-3 text-color=grey-9 clickable')

                use_segmentation = controller.input_segment.value if controller.input_segment else True
                if use_segmentation and controller.current_query_str not in segments:
                    whole_chip = ui.chip(
                        '整句',
                        on_click=lambda: controller._filter_by_source(controller.current_query_str),
                    ).props('color=blue-grey-1 text-color=blue-grey-9 clickable')
                    decorate_chip(whole_chip, controller.current_query_str)

                for item in coverage:
                    if item.status == COVERED:
                        chip = ui.chip(
                            item.segment,
                            icon='check_circle',
                            on_click=lambda s=item.segment: controller._filter_by_source(s),
                        ).props('color=green-1 text-color=green-9 clickable')
                        detail = f"已覆盖；已选择：{'、'.join(item.selected_tags)}。点击筛选此概念的搜索结果。"
                    elif item.status == CANDIDATE_UNSELECTED:
                        chip = ui.chip(
                            item.segment,
                            icon='radio_button_unchecked',
                            on_click=lambda s=item.segment: controller._filter_by_source(s),
                        ).props('color=amber-1 text-color=amber-9 clickable')
                        detail = f"有候选：{'、'.join(item.candidate_tags[:5])}。点击筛选此概念的搜索结果。"
                    else:
                        chip = ui.chip(
                            item.segment,
                            icon='search',
                            on_click=lambda s=item.segment: controller._search_uncovered_segment(s),
                        ).props('color=red-1 text-color=red-8 clickable')
                        detail = '点击后沿用当前搜索设置进行补充搜索；工作区标签保持不变。'
                    decorate_chip(chip, item.segment)
                    with chip:
                        with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                            ui.label(detail).style('font-size:13px;')


def update_table_columns(controller: Any, _event: Any = None) -> None:
    """按显示开关更新结果表的可选列。"""
    columns = list(TABLE_COLUMNS)
    if controller.sw_semantic and controller.sw_semantic.value:
        columns.append(OPTIONAL_COLS['semantic'])
    if controller.sw_layer and controller.sw_layer.value:
        columns.append(OPTIONAL_COLS['layer'])
    if controller.sw_source and controller.sw_source.value:
        columns.append(OPTIONAL_COLS['source'])
    controller.result_table.columns = columns
