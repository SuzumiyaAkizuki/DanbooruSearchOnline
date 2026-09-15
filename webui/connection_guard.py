"""处理 NiceGUI 轮询请求在 Engine.IO 读取期间断开的情况。"""

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _PollingDisconnected(Exception):
    """请求已断开，不能再交给 Engine.IO 的 ASGI 转换器。"""


class NiceGUIPollingDisconnectGuard:
    """只保护 NiceGUI HTTP 轮询；其他请求及异常原样传递。"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get('path', '')
        root_path = scope.get('root_path', '')
        if root_path and path.startswith(root_path + '/'):
            path = path[len(root_path):]
        if scope['type'] != 'http' or not path.startswith('/_nicegui_ws/'):
            await self.app(scope, receive, send)
            return

        async def receive_connected() -> Message:
            message = await receive()
            if message['type'] == 'http.disconnect':
                raise _PollingDisconnected
            return message

        try:
            await self.app(scope, receive_connected, send)
        except _PollingDisconnected:
            # 对端已关闭连接，无需构造 HTTP 响应。
            return
