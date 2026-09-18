"""低开销 UI 性能采样：只保留固定指标的聚合值，不记录用户内容。"""

import asyncio
import json
import math
from contextlib import contextmanager
from time import perf_counter


METRICS = frozenset({
    'selected_render', 'history_render', 'favorites_render',
    'browser_roundtrip', 'event_loop_lag',
    'recommendation_wait', 'recommendation_compute',
})
_samples: dict[str, dict] = {}
_probe_failures = 0
_monitor_task: asyncio.Task | None = None


def record(name: str, duration_ms: float) -> None:
    """仅从事件循环调用；固定内存占用，不保存逐次样本。"""
    if name not in METRICS or not math.isfinite(duration_ms) or duration_ms < 0:
        return
    sample = _samples.setdefault(name, {'count': 0, 'sum_ms': 0.0, 'max_ms': 0.0, 'over_200ms': 0})
    sample['count'] += 1
    sample['sum_ms'] += duration_ms
    sample['max_ms'] = max(sample['max_ms'], duration_ms)
    sample['over_200ms'] += duration_ms > 200


@contextmanager
def measure(name: str):
    started = perf_counter()
    try:
        yield
    finally:
        record(name, (perf_counter() - started) * 1000)


def flush() -> None:
    global _probe_failures
    if not _samples and not _probe_failures:
        return
    metrics = {
        name: {
            'count': value['count'],
            'avg_ms': round(value['sum_ms'] / value['count'], 1),
            'max_ms': round(value['max_ms'], 1),
            'over_200ms': value['over_200ms'],
        }
        for name, value in sorted(_samples.items())
    }
    print('[UI PERF] ' + json.dumps({
        'metrics': metrics, 'browser_probe_failures': _probe_failures,
    }), flush=True)
    _samples.clear()
    _probe_failures = 0


async def _monitor() -> None:
    last_flush = perf_counter()
    while True:
        expected = perf_counter() + 1.0
        await asyncio.sleep(1.0)
        now = perf_counter()
        record('event_loop_lag', max(0.0, now - expected) * 1000)
        if now - last_flush >= 60:
            flush()
            last_flush = now


def start_monitor() -> None:
    global _monitor_task
    if _monitor_task is None or _monitor_task.done():
        _monitor_task = asyncio.create_task(_monitor())


async def stop_monitor() -> None:
    global _monitor_task
    if _monitor_task is not None:
        _monitor_task.cancel()
        try:
            await _monitor_task
        except asyncio.CancelledError:
            pass
        _monitor_task = None
    flush()


async def probe_browser(client) -> None:
    """测量服务器到可见页面再返回的往返；不是用户点击到绘制的耗时。"""
    global _probe_failures
    started = perf_counter()
    try:
        visible = await client.run_javascript('return document.visibilityState === "visible"', timeout=3.0)
    except (TimeoutError, RuntimeError):
        _probe_failures += 1
        return
    if visible is True:
        record('browser_roundtrip', (perf_counter() - started) * 1000)
