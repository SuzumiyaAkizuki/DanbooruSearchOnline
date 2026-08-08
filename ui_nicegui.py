"""DanbooruSearch NiceGUI 启动入口与兼容门面。"""

import asyncio
import sys
import time
import traceback

from fastapi.responses import PlainTextResponse
from nicegui import app, ui

from api_fastapi import app as api_app
from core import counter, telemetry
from core.engine import DanbooruTagger
from mcp_server import mcp
from platform_utils import get_host_port, is_cloud
from webui import controller as _controller
from webui.routes import register_main_page


sys.stdout.reconfigure(line_buffering=True)
print('[UI] 脚本开始执行', flush=True)


def _excepthook(exc_type, exc_value, exc_tb):
    print('[UI] FATAL ERROR ON STARTUP:', flush=True)
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _excepthook


# 保留既有导入方式：外部代码仍可从 ui_nicegui 获取控制器和旧辅助符号。
DanbooruSearchUI = _controller.DanbooruSearchUI


def __getattr__(name: str):
    return getattr(_controller, name)


register_main_page(
    DanbooruSearchUI,
    _controller._mark_ui_session_active,
    _controller._mark_ui_session_inactive,
    counter.increment_visit,
    telemetry.increment,
)


if __name__ in {'__main__', '__mp_main__'}:
    host, port = get_host_port()

    @app.on_startup
    def _warmup():
        async def background_init_tasks():
            await asyncio.sleep(5)
            print('[UI] 开始预热计数器与引擎', flush=True)
            await counter.init()
            await telemetry.init()
            cold_start_started_at = time.perf_counter()
            await telemetry.increment('engine_cold_start_attempt')
            try:
                await DanbooruTagger.get_instance()
            except Exception:
                await telemetry.increment('engine_cold_start_failure')
                await telemetry.record_timing(
                    'engine_cold_start',
                    (time.perf_counter() - cold_start_started_at) * 1000,
                )
                raise
            await telemetry.increment('engine_cold_start_success')
            await telemetry.record_timing(
                'engine_cold_start',
                (time.perf_counter() - cold_start_started_at) * 1000,
            )
            print('[UI] 后台预热全部完成！', flush=True)

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
        except Exception as exc:
            print(f'[UI] 关机同步失败: {exc}')

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
