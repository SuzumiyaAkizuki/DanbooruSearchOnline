"""NiceGUI 页面路由与客户端生命周期绑定。"""

import asyncio
from collections.abc import Callable

from nicegui import ui


def register_main_page(
    create_page_controller: Callable[[], object],
    mark_active: Callable[[str], None],
    mark_inactive: Callable[[str], None],
    increment_visit: Callable[[], object],
    increment_telemetry: Callable[[str], object],
) -> None:
    """注册主页；控制器仍由入口注入，避免路由层依赖具体实现。"""

    @ui.page('/')
    async def main_page():
        client = ui.context.client
        client_id = client.id
        app_ui = create_page_controller()

        def mark_connected(*_):
            mark_active(client_id)
            app_ui._start_service_status_task()
            app_ui._start_storage_restore_task()

        def mark_disconnected(*_):
            mark_inactive(client_id)
            app_ui._pause_storage_restore()

        def mark_deleted(*_):
            mark_inactive(client_id)
            app_ui._dispose()

        client.on_connect(mark_connected)
        client.on_disconnect(mark_disconnected)
        on_delete = getattr(client, 'on_delete', None)
        if callable(on_delete):
            on_delete(mark_deleted)

        app_ui.build_page()

        async def silent_visit_update():
            try:
                await increment_visit()
                await increment_telemetry('ui_visit')
                app_ui._update_footer_text()
            except Exception:
                pass

        asyncio.create_task(silent_visit_update())
