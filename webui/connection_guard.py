"""处理 NiceGUI 轮询断连和超限，保留其他异常。"""

import logging
from time import perf_counter
from urllib.parse import parse_qs
from uuid import uuid4

from engineio.payload import Payload
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class _PollingDisconnected(Exception):
    """请求已断开，不能再交给 Engine.IO 的 ASGI 转换器。"""


class _PollingPayloadTooLarge(Exception):
    """在 Engine.IO 吞掉包数量超限异常并返回 OK 之前终止请求。"""


def _is_closed_engineio_session(exc: KeyError) -> bool:
    if exc.args != ('Session is disconnected',):
        return False
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        if (frame.f_globals.get('__name__') == 'engineio.base_server'
                and frame.f_code.co_name == '_get_socket'):
            return True
        tb = tb.tb_next
    return False


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

        query = parse_qs(scope.get('query_string', b'').decode('latin-1'))
        polling = query.get('transport', ['polling']) == ['polling']
        # JSONP 使用另一种编码；这里只统计 NiceGUI 使用的 EIO4 原生轮询。
        inspect_packets = (polling and scope.get('method') == 'POST'
                           and query.get('EIO') == ['4'] and 'j' not in query)
        started = perf_counter()
        request_id = uuid4().hex[:12]
        response_started = False
        body_bytes = separators = 0

        async def send_tracked(message: Message) -> None:
            nonlocal response_started
            if message['type'] == 'http.response.start':
                response_started = True
            await send(message)

        async def receive_connected() -> Message:
            nonlocal body_bytes, separators
            message = await receive()
            if message['type'] == 'http.disconnect':
                raise _PollingDisconnected
            if inspect_packets and message['type'] == 'http.request':
                body = message.get('body', b'')
                body_bytes += len(body)
                separators += body.count(b'\x1e')
                if body_bytes and separators + 1 > Payload.max_decode_packets:
                    raise _PollingPayloadTooLarge
            return message

        try:
            await self.app(scope, receive_connected, send_tracked)
        except _PollingDisconnected:
            # 对端已关闭连接，无需构造 HTTP 响应。
            return
        except _PollingPayloadTooLarge:
            if response_started:
                raise
            logger.warning(
                'NiceGUI polling rejected request=%s reason=packet_limit '
                'packets_seen=%d limit=%d bytes_seen=%d read_ms=%.1f',
                request_id, separators + 1, Payload.max_decode_packets,
                body_bytes, (perf_counter() - started) * 1000,
            )
            await PlainTextResponse('Too many packets in payload', status_code=400)(scope, receive, send)
        except KeyError as exc:
            if (not polling or scope.get('method') not in {'GET', 'POST'}
                    or response_started or not _is_closed_engineio_session(exc)):
                raise
            logger.info('NiceGUI polling rejected request=%s reason=session_disconnected', request_id)
            await PlainTextResponse('Session is disconnected', status_code=400)(scope, receive, send)
