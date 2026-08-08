"""搜索卡与搜索参数区域的 NiceGUI 视图。"""

from typing import Any

from nicegui import ui

from platform_utils import nsfw_allowed
from webui.constants import SEARCH_MODE_OPTIONS


def build_search_panel(controller: Any) -> None:
    with ui.card().classes('w-full product-search-card'):
        with ui.row().classes('w-full items-start justify-between gap-3 mb-1'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('search', size='2em', color='primary')
                ui.label('Danbooru 标签模糊搜索').classes('text-2xl font-bold text-gray-800')
            ui.button(
                '帮助 / 关于', icon='help_outline', on_click=controller.help_dialog.open,
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
            controller.search_input = ui.textarea(
                placeholder='输入自然语言描述或模糊概念，例如：一个穿着白色水手服的少女在雨中奔跑...'
            ).classes('flex-grow text-base motion-search-input').props('outlined rows=2')
            controller.search_input.on('keydown.ctrl.enter', controller.perform_search)

            with ui.column().classes('justify-center'):
                controller.search_btn = ui.button(
                    '', on_click=controller.perform_search, icon='search'
                ).classes('px-6 h-full min-h-16 motion-search-button').props('unelevated color=dark')
                with controller.search_btn:
                    ui.label('搜索').classes('text-sm mt-1')
                controller.spinner = ui.spinner(size='2em').classes(
                    'hidden motion-search-spinner'
                )

        controller.search_params_row = ui.row().classes(
            'w-full gap-6 items-center mt-3 flex-wrap'
        )
        with controller.search_params_row:
            with ui.row().classes('items-center gap-2'):
                ui.label('搜索模式 (beta)').classes('text-sm text-gray-600')
                controller.input_search_mode = ui.select(
                    SEARCH_MODE_OPTIONS, value='自定义',
                ).classes('w-28').props('outlined dense')
                controller.input_search_mode.on(
                    'update:model-value', controller._on_search_mode_change
                )
                with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                    ui.label('选择模式自动填充对应参数；手动修改参数后自动变为「自定义」').style('font-size:14px;')

            with ui.row().classes('items-center gap-2'):
                ui.label('Top K (语义相关)').classes('text-sm text-gray-600')
                controller.input_top_k = ui.number(value=10, min=1, max=200).classes('w-20') \
                    .props('outlined dense')
                controller.input_top_k.on('update:model-value', controller._on_param_changed)

            with ui.row().classes('items-center gap-2'):
                ui.label('结果上限').classes('text-sm text-gray-600')
                controller.input_limit = ui.number(value=80, min=10, max=500).classes('w-20') \
                    .props('outlined dense')
                controller.input_limit.on('update:model-value', controller._on_param_changed)

            with ui.switch('显示 NSFW(成人) 内容', value=False).props('color=red') as nsfw_switch:
                if not nsfw_allowed():
                    with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                        ui.label('NSFW 内容在当前平台不可用').style('font-size:14px;')
            controller.input_nsfw = nsfw_switch
            if not nsfw_allowed():
                controller.input_nsfw.disable()
            else:
                controller.input_nsfw.on('update:model-value', controller.on_nsfw_toggle)

            with ui.switch('智能分词', value=True).props('color=primary') as segment_switch:
                with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                    ui.label('关闭后系统将只匹配完整句子，适用于精准搜索整句。').style('font-size:14px;')
            controller.input_segment = segment_switch
            controller.input_segment.on('update:model-value', controller._on_param_changed)

        controller.advanced_options = ui.expansion('高级选项', icon='tune').classes('w-full mt-2')
        with controller.advanced_options:
            with ui.column().classes('w-full p-3 gap-4'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('热度权重').classes('text-sm font-bold text-gray-700')
                    controller.input_weight = ui.slider(
                        min=0.0, max=1.0, value=0.15, step=0.05,
                    ).classes('w-40')
                    ui.label().bind_text_from(
                        controller.input_weight, 'value', lambda value: f'{value:.2f}',
                    ).classes('text-sm font-mono text-gray-700 w-8')
                    controller.input_weight.on('update:model-value', controller._on_param_changed)

                with ui.row().classes('w-full gap-8 flex-wrap'):
                    with ui.column().classes('gap-2'):
                        ui.label('匹配层筛选').classes('font-bold text-sm text-gray-700')
                        display_map = {
                            '英文': '英文标签', '中文扩展词': '中文扩展词',
                            '释义': '维基释义', '中文核心词': '中文核心词',
                            'artist': 'artist',
                        }
                        for layer in ['英文', '中文扩展词', '释义', '中文核心词', 'artist']:
                            checkbox = ui.checkbox(
                                display_map.get(layer, layer), value=True,
                                on_change=lambda event, current=layer:
                                    controller.selected_layers.__setitem__(current, event.value),
                            ).props('color=primary dense')
                            controller._layer_checkboxes[layer] = checkbox

                    with ui.column().classes('gap-2'):
                        ui.label('类型筛选').classes('font-bold text-sm text-gray-700')
                        color_map = {'General': 'blue', 'Copyright': 'purple', 'Character': 'green'}
                        label_map = {
                            'General': '通用 (General)',
                            'Copyright': '作品 (Copyright)',
                            'Character': '角色 (Character)',
                        }
                        for category in ['General', 'Copyright', 'Character']:
                            checkbox = ui.checkbox(
                                label_map[category], value=True,
                                on_change=lambda event, current=category:
                                    controller.selected_cats.__setitem__(current, event.value),
                            ).props(f'color={color_map[category]} dense')
                            controller._cat_checkboxes[category] = checkbox

                    with ui.column().classes('gap-2'):
                        ui.label('表格显示列').classes('font-bold text-sm text-gray-700')
                        controller.sw_semantic = ui.switch('显示语义分', value=False)
                        controller.sw_layer = ui.switch('显示匹配层', value=False)
                        controller.sw_source = ui.switch('显示匹配来源', value=False)
                        controller.sw_semantic.on('update:model-value', controller._update_table_columns)
                        controller.sw_layer.on('update:model-value', controller._update_table_columns)
                        controller.sw_source.on('update:model-value', controller._update_table_columns)

                    with ui.column().classes('gap-2'):
                        ui.label('标签分组模式').classes('font-bold text-sm text-gray-700')
                        controller.input_group_mode = ui.select(
                            ['off', 'expand', 'diverse'], value='off',
                        ).classes('w-40').props('outlined dense')
                        with ui.tooltip().props('content-class="bg-black text-white shadow-4"'):
                            ui.label('off=关闭 | expand=同类召回增强 | diverse=多样性约束').style('font-size:14px;')
                        controller.input_group_mode.on(
                            'update:model-value', controller._on_param_changed
                        )

                        controller.input_max_per_group = ui.number(
                            value=2, min=1, max=10,
                        ).classes('w-20').props('outlined dense')
                        ui.label('每组最大标签数（diverse 模式）').classes('text-xs text-gray-500')
                        controller.input_max_per_group.on(
                            'update:model-value', controller._on_param_changed
                        )

        with ui.element('div').classes('w-full border-t border-slate-100 pt-3 mt-1'):
            controller.service_status_container = ui.column().classes('w-full gap-0')
            controller._update_service_status()
            controller._start_service_status_task()
