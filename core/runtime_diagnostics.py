"""异常诊断只保留类型和栈位置，不序列化请求内容或异常中的输入值。"""

import logging
import re
import sys
import traceback
from contextvars import ContextVar
from functools import wraps
from uuid import uuid4


def exception_details(exc: BaseException) -> str:
    # 不使用 format_exception：ValidationError 的异常文本可能包含完整输入。
    parts: list[str] = []
    seen: set[int] = set()

    def append(error: BaseException, relation: str = '') -> None:
        if id(error) in seen or len(seen) >= 10:
            return
        seen.add(id(error))
        frames = traceback.extract_tb(error.__traceback__)
        stack = '\n'.join(f'  {f.filename}:{f.lineno} in {f.name}' for f in frames)
        parts.append(f'{relation}{type(error).__name__}\n{stack or "  <traceback unavailable>"}')
        cause = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
        if cause is not None:
            append(cause, 'caused by: ')
        if isinstance(error, BaseExceptionGroup):
            for child in error.exceptions:
                append(child, 'group member: ')

    append(exc)
    return '\n'.join(parts)


_mcp_context: ContextVar[tuple[str, BaseException | None] | None] = ContextVar(
    'mcp_diagnostic_context', default=None,
)


class _MCPDiagnosticFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = _mcp_context.get()
        message = record.getMessage()
        if context and message.startswith('Received exception from stream:'):
            correlation, exc = context
            if exc is not None:
                record.msg = 'MCP stream failure %s exception=%s'
                record.args = (correlation, exception_details(exc))
                record.exc_info = record.exc_text = None
        elif message.startswith('Failed to validate notification:') and 'notifications/cancelled' in message:
            exc = sys.exception()
            request_id = re.search(r"['\"]requestId['\"]:\s*(\d+)\b", message)
            record.msg = 'MCP cancellation failure event=%s request_id=%s exception=%s'
            record.args = (
                uuid4().hex[:12], request_id[1] if request_id else 'unavailable',
                exception_details(exc) if exc else 'unknown (traceback unavailable)',
            )
            record.exc_info = record.exc_text = None
        return True


def install_mcp_diagnostics(server) -> None:
    """包装固定版本 SDK 的消息入口，保留其处理、响应和抛错行为。"""
    original = server._handle_message
    if getattr(original, '_danbooru_diagnostics', False):
        return

    for name in ('mcp.server.lowlevel.server', ''):
        logger = logging.getLogger(name)
        if not any(isinstance(f, _MCPDiagnosticFilter) for f in logger.filters):
            logger.addFilter(_MCPDiagnosticFilter())

    @wraps(original)
    async def handle_message(message, session, lifespan_context, raise_exceptions=False):
        request = getattr(getattr(message, 'request', None), 'root', None)
        request_id = getattr(message, 'request_id', None)
        # 字符串 requestId 可由客户端任意指定，不写入日志。
        numeric_id = request_id if type(request_id) is int else 'unavailable'
        correlation = (f'event={uuid4().hex[:12]} request_id={numeric_id} '
                       f'kind={type(request).__name__ if request is not None else type(message).__name__}')
        stream_exception = message if isinstance(message, Exception) else None
        token = _mcp_context.set((correlation, stream_exception))
        try:
            return await original(message, session, lifespan_context, raise_exceptions)
        except Exception as exc:
            # stream 对象已由 SDK 记录；只补记处理/响应阶段的新异常。
            if exc is not stream_exception:
                logging.getLogger(__name__).error(
                    'MCP message handling failure %s exception=%s', correlation, exception_details(exc),
                )
            raise
        finally:
            _mcp_context.reset(token)

    handle_message._danbooru_diagnostics = True
    server._handle_message = handle_message
