"""低开销 UI 性能采样：只保留固定指标的聚合值，不记录用户内容。"""

import asyncio
import gc
import json
import logging
import math
import os
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from time import perf_counter, process_time
from uuid import UUID, uuid4

from platform_utils import PLATFORM, get_counter_cfg, read_bytes, upload_bytes


METRICS = frozenset({
    'selected_render', 'history_render', 'favorites_render',
    'browser_roundtrip', 'event_loop_lag',
    'recommendation_wait', 'recommendation_compute',
})
_samples: dict[str, dict] = {}
_probe_failures = 0
_monitor_task: asyncio.Task | None = None
_save_task: asyncio.Task | None = None
_history: deque[dict] = deque(maxlen=180)
_storage_failed = False
SYNC_INTERVAL_SECONDS = 300
REMOTE_WINDOW_LIMIT = 1440
_gc_lock = RLock()
_gc_started: dict[int, float] = {}
_gc_samples: dict[int, dict] = {}
_runtime_started: tuple[float, float] | None = None


def _gc_callback(phase: str, info: dict) -> None:
    """GC 可能在线程池触发；只在锁内聚合，不操作事件循环或输出日志。"""
    generation = info.get('generation')
    if generation not in (0, 1, 2):
        return
    now = perf_counter()
    with _gc_lock:
        if phase == 'start':
            _gc_started[generation] = now
        elif phase == 'stop':
            started = _gc_started.pop(generation, None)
            if started is not None:
                duration = max(0.0, now - started) * 1000
                sample = _gc_samples.setdefault(generation, {
                    'count': 0, 'sum_ms': 0.0, 'max_ms': 0.0, 'over_200ms': 0,
                })
                sample['count'] += 1
                sample['sum_ms'] += duration
                sample['max_ms'] = max(sample['max_ms'], duration)
                sample['over_200ms'] += duration > 200


def _summarize(sample: dict) -> dict:
    return {
        'count': sample['count'],
        'avg_ms': round(sample['sum_ms'] / sample['count'], 1),
        'max_ms': round(sample['max_ms'], 1),
        'over_200ms': sample['over_200ms'],
    }


def _take_runtime() -> dict | None:
    global _gc_samples, _runtime_started
    if _runtime_started is None:
        return None
    now, cpu = perf_counter(), process_time()
    wall = max(0.0, now - _runtime_started[0])
    used_cpu = max(0.0, cpu - _runtime_started[1])
    _runtime_started = (now, cpu)
    with _gc_lock:
        samples, _gc_samples = _gc_samples, {}
    return {
        # 100% 表示消耗一个逻辑核；8 核满载可接近 800%，不是整机占用率。
        'process_cpu_percent': round(100 * used_cpu / wall, 1) if wall else 0.0,
        'wall_ms': round(wall * 1000, 1),
        'logical_cpus': os.cpu_count() or 1,
        'gc': {str(gen): _summarize(sample) for gen, sample in sorted(samples.items())},
    }


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
    if not _samples and not _probe_failures and _runtime_started is None:
        return
    metrics = {
        name: _summarize(value)
        for name, value in sorted(_samples.items())
    }
    report = {
        'window_id': uuid4().hex,
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'metrics': metrics, 'browser_probe_failures': _probe_failures,
    }
    runtime = _take_runtime()
    if runtime is not None:
        report['runtime'] = runtime
    _history.append(report)
    if os.environ.get('DANBOORU_UI_PERF_LOG', '').strip() == '1':
        print('[UI PERF] ' + json.dumps(report), flush=True)
    _samples.clear()
    _probe_failures = 0


def get_snapshot() -> dict:
    """返回最近 180 个统计窗口的独立副本，不暴露可修改的内部状态。"""
    return {'schema_version': 1, 'platform': PLATFORM, 'windows': deepcopy(list(_history))}


def _parse_windows(raw: bytes | None) -> list[dict]:
    """读取失败或格式异常时拒绝覆盖已有对象，不把损坏文件当作空数据。"""
    if raw is None:
        return []
    payload = json.loads(raw)
    if (not isinstance(payload, dict) or payload.get('schema_version') != 1
            or payload.get('platform') != PLATFORM or not isinstance(payload.get('windows'), list)):
        raise ValueError('invalid UI performance snapshot')
    result = []
    for item in payload['windows']:
        if not isinstance(item, dict):
            raise ValueError('invalid performance window')
        window_id = UUID(item['window_id']).hex
        recorded_at = datetime.fromisoformat(item['recorded_at'])
        if recorded_at.tzinfo is None:
            raise ValueError('window timestamp requires timezone')
        failures = item['browser_probe_failures']
        metrics = item['metrics']
        if type(failures) is not int or failures < 0 or not isinstance(metrics, dict):
            raise ValueError('invalid performance counts')
        clean_metrics = {}
        for name, metric in metrics.items():
            if name not in METRICS or not isinstance(metric, dict):
                raise ValueError('invalid performance metric')
            count, over = metric['count'], metric['over_200ms']
            if type(count) is not int or count <= 0 or type(over) is not int or not 0 <= over <= count:
                raise ValueError('invalid metric counts')
            for key in ('avg_ms', 'max_ms'):
                value = metric[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('invalid metric duration')
            clean_metrics[name] = {key: metric[key] for key in ('count', 'avg_ms', 'max_ms', 'over_200ms')}
        window = {
            'window_id': window_id, 'recorded_at': recorded_at.astimezone(timezone.utc).isoformat(),
            'metrics': clean_metrics, 'browser_probe_failures': failures,
        }
        if 'runtime' in item:
            window['runtime'] = _parse_runtime(item['runtime'])
        result.append(window)
    return result


def _parse_runtime(runtime: dict) -> dict:
    """可选扩展字段；旧窗口仍可读，写回时仅保留固定数值字段。"""
    if not isinstance(runtime, dict):
        raise ValueError('invalid runtime metrics')
    for key in ('process_cpu_percent', 'wall_ms'):
        value = runtime[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid runtime value')
    cpus = runtime['logical_cpus']
    if type(cpus) is not int or cpus < 1 or not isinstance(runtime['gc'], dict):
        raise ValueError('invalid runtime counts')
    clean_gc = {}
    for generation, sample in runtime['gc'].items():
        if generation not in ('0', '1', '2') or not isinstance(sample, dict):
            raise ValueError('invalid GC generation')
        count, over = sample['count'], sample['over_200ms']
        if type(count) is not int or count <= 0 or type(over) is not int or not 0 <= over <= count:
            raise ValueError('invalid GC counts')
        for key in ('avg_ms', 'max_ms'):
            value = sample[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid GC duration')
        clean_gc[generation] = {key: sample[key] for key in ('count', 'avg_ms', 'max_ms', 'over_200ms')}
    return {key: runtime[key] for key in ('process_cpu_percent', 'wall_ms', 'logical_cpus')} | {'gc': clean_gc}


def _write_snapshot(snapshot: dict) -> None:
    """在线程中读、合并、写 OSS；窗口 ID 保证重试不会重复计数。"""
    cfg = get_counter_cfg()
    if not cfg.available:
        return
    filename = f'ui_performance_v1_{PLATFORM}.json'
    remote = _parse_windows(read_bytes(filename, cfg))
    windows = {item['window_id']: item for item in remote}
    windows.update({item['window_id']: item for item in snapshot['windows']})
    merged = sorted(windows.values(), key=lambda item: item['recorded_at'])[-REMOTE_WINDOW_LIMIT:]
    content = json.dumps({**snapshot, 'windows': merged}, ensure_ascii=False).encode('utf-8')
    if not upload_bytes(content, filename, cfg, retries=3, retry_delay=1.0):
        raise OSError('UI performance upload failed')


async def _save_snapshot() -> None:
    global _storage_failed
    snapshot = get_snapshot()
    if not snapshot['windows']:
        return
    try:
        await asyncio.to_thread(_write_snapshot, snapshot)
    except Exception as exc:
        # 保存失败只在首次失败时提醒，后续窗口继续尝试；不影响 UI。
        if not _storage_failed:
            logging.getLogger(__name__).warning('UI performance snapshot save failed: %s', type(exc).__name__)
        _storage_failed = True
    else:
        _storage_failed = False


async def _monitor() -> None:
    global _save_task
    last_flush = perf_counter()
    last_save_started = last_flush
    while True:
        expected = perf_counter() + 1.0
        await asyncio.sleep(1.0)
        now = perf_counter()
        record('event_loop_lag', max(0.0, now - expected) * 1000)
        if now - last_flush >= 60:
            flush()
            # 每五分钟批量上传；最多一个任务，不堆积线程或阻塞采样。
            if now - last_save_started >= SYNC_INTERVAL_SECONDS and (_save_task is None or _save_task.done()):
                _save_task = asyncio.create_task(_save_snapshot())
                last_save_started = now
            last_flush = now


def start_monitor() -> None:
    global _monitor_task, _runtime_started
    if _monitor_task is None or _monitor_task.done():
        _monitor_task = asyncio.create_task(_monitor())
        _runtime_started = (perf_counter(), process_time())
        with _gc_lock:
            _gc_started.clear()
            _gc_samples.clear()
        if _gc_callback not in gc.callbacks:
            gc.callbacks.append(_gc_callback)


async def stop_monitor() -> None:
    global _monitor_task, _save_task, _runtime_started
    if _gc_callback in gc.callbacks:
        gc.callbacks.remove(_gc_callback)
    if _monitor_task is not None:
        _monitor_task.cancel()
        try:
            await _monitor_task
        except asyncio.CancelledError:
            pass
        _monitor_task = None
    flush()
    _runtime_started = None
    with _gc_lock:
        _gc_started.clear()
        _gc_samples.clear()
    if _save_task is not None:
        await _save_task
        _save_task = None
    await _save_snapshot()


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
