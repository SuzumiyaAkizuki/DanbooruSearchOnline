"""主页装配视图；只创建控件并连接控制器回调。"""

from typing import Any

from nicegui import ui


def build_page(controller: Any, *, motion_style: str, sponsor_notice_text: str) -> None:
    controller.client = ui.context.client
    ui.colors(primary='#4A90E2', secondary='#5E6C84', accent='#FF6B6B')
    ui.add_head_html(motion_style)
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
            .related-item {
                transition: box-shadow var(--motion-fast) var(--motion-ease-out);
            }
            @media (hover: hover) {
                .related-item:hover {
                    box-shadow: inset 0 0 0 9999px rgba(15, 23, 42, 0.045);
                }
            }
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

    controller._build_sponsor_dialog()
    controller._build_help_dialog()

    with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-3'):

        # ── 1. 搜索主路径 ──
        controller._build_search_card()

        # ── 2. 紧凑版本公告 ──
        controller._build_release_announcement()

        # ── 3. 工作区工具和已选标签（无需搜索即可恢复）──
        controller.workspace_card = ui.card().classes(
            'w-full p-0 gap-0 overflow-hidden section-surface'
        )
        with controller.workspace_card:
            controller._build_workspace_toolbar()
            controller._build_selection_bar()

        # ── 4~5. 搜索结果区域（搜索前隐藏）──
        controller.results_section = ui.column().classes('w-full gap-4').props(
            'id="danbooru-results-section"'
        )
        controller.results_section.set_visibility(False)

        with controller.results_section:
            # ── 4. 查询理解：来源筛选 + 概念覆盖 ──
            controller.coverage_container = ui.column().classes('w-full gap-2').props(
                'id="danbooru-coverage-section"'
            )

            # ── 5. 两栏结果 ──
            controller._build_results_columns()

        # ── 6. 页脚 ──
        with ui.element('div').classes('w-full text-center py-4 mt-2'):
            controller.search_count_label = ui.html('正在加载数据...').classes('text-xs text-gray-400')
            controller._update_footer_text()
            ui.button(sponsor_notice_text, on_click=controller.sponsor_dialog.open) \
                .props('flat dense no-caps color=grey-6') \
                .classes('text-xs mt-1')


def render_service_status(controller: Any, status: dict) -> None:
    """根据控制器整理好的服务快照更新状态区域。"""
    controller.service_status_container.clear()
    with controller.service_status_container:
        if not status['ready']:
            with ui.row().classes(
                'w-full items-center gap-2 service-state-panel loading'
            ):
                ui.spinner(size='18px', color='primary')
                ui.label('引擎初始化中，请稍候…约需 5~10 分钟').classes('font-medium')
            return

        with ui.row().classes(
            f'w-full items-center gap-2 service-state-panel '
            f'{"busy" if status["busy"] else "ready"}'
        ):
            ui.icon(
                'schedule' if status['busy'] else 'check_circle',
                size='18px',
                color='warning' if status['busy'] else 'positive',
            )
            parts = [
                '服务繁忙' if status['busy'] else '服务可用',
                f'{status["online_sessions"]} 个在线页面',
            ]
            if status['active'] > 0:
                parts.append(f'正在处理 {status["active"]} 个任务')
            if status['waiting'] > 0:
                parts.append(f'等待 {status["waiting"]} 个')
            ui.label(' · '.join(parts)).classes('font-medium')


def build_release_announcement(controller: Any) -> None:
    controller.announcement_banner = ui.element('div').classes(
        'w-full release-notice section-surface px-3 py-2'
    )
    with controller.announcement_banner:
        with ui.row().classes('w-full items-center justify-between gap-2'):
            with ui.row().classes('items-center gap-2 min-w-0 flex-wrap'):
                ui.icon('new_releases', size='18px', color='primary')
                ui.label(
                    '新版已加入标签工作区、Prompt 导入、Alias 纠错和分渠道统计。'
                ).classes('text-sm text-slate-700')
                ui.button(
                    '查看详情', on_click=controller.help_dialog.open,
                ).props('flat dense no-caps color=primary').classes('text-xs')
            ui.button(
                icon='close', on_click=controller._dismiss_release_announcement,
            ).props('flat dense round color=grey-6')
