"""
core/counter.py
──────────────
极简持久化计数器（核心业务指标：总搜索、总访问、复制次数、成功交互、词频聚合、bad_case）。

数据持久化到阿里云 OSS，通过 platform_utils 屏蔽细节。
无任何本地落盘操作。

count.json 结构：
  {
    "total":        int,
    "visits":       int,
    "copies":       int,
    "successes":    int,
    "hot_keywords": {word: count, ...},
    "bad_cases":    [{q, t}, ...],
    "history":      [{"date": "YYYY-MM-DD", "total": int}, ...]   # 每日快照，最近 180 天
  }
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone, timedelta, date
from typing import Optional

from platform_utils import get_counter_cfg, read_bytes, upload_bytes, CounterConfig

# ── 核心状态变量 ────────────────────────────────
_memory_count: int = 0
_dirty_count: int = 0

_memory_visits: int = 0
_dirty_visits: int = 0
BASE_VISITS: int = 0

_memory_copies: int = 0
_dirty_copies: int = 0
BASE_COPIES: int = 0

_memory_mcp: int = 0
_dirty_mcp: int = 0

_memory_successes: int = 0
_dirty_successes: int = 0

_memory_keywords: Counter = Counter()
_dirty_keywords: Counter = Counter()
MAX_KEYWORDS_LIMIT = 200

_memory_bad_cases: list[dict] = []
_dirty_bad_cases: list[dict] = []
MAX_BAD_CASES = 50

_memory_history: list[dict] = []   # [{"date": "YYYY-MM-DD", "total": int}, ...]
MAX_HISTORY_DAYS = 180

_last_sync: float = 0.0
_sync_lock: Optional[asyncio.Lock] = None

SYNC_INTERVAL = 1800       # 每 30 分钟同步一次
SYNC_THRESHOLD = 500       # 或各项增量之和达到 200 次时同步


def _get_sync_lock() -> asyncio.Lock:
    global _sync_lock
    if _sync_lock is None:
        _sync_lock = asyncio.Lock()
    return _sync_lock


# ── 历史快照工具 ──────────────────────────────────────────────────────────────

def _today_str() -> str:
    return datetime.now(timezone(timedelta(hours=8))).date().isoformat()


def _merge_history(
    remote: list[dict],
    current_total: int,
    memory: list[dict] | None = None,
) -> list[dict]:
    """
    将当天的 total 写入（或更新）历史列表，保留最近 MAX_HISTORY_DAYS 天。
    同一天只保留最新一条（total 取较大值）。

    remote : 从 OSS 读回的历史（可能因缓存导致不完整）
    memory : 内存中已加载的历史（init() 时从 OSS 完整读取，用于兜底）
    """
    today = _today_str()
    by_date: dict[str, int] = {}
    # 先用内存数据打底，再用远端数据覆盖（远端是最终写入态，优先级更高）
    for r in (memory or []):
        by_date[r['date']] = r['total']
    for r in remote:
        by_date[r['date']] = max(by_date.get(r['date'], 0), r['total'])
    # 当天记录取较大值，避免同步时序问题导致数据回退
    by_date[today] = max(by_date.get(today, 0), current_total)

    sorted_records = sorted(by_date.items(), reverse=True)[:MAX_HISTORY_DAYS]
    # 返回时按时间正序，方便绘图
    return [{'date': d, 'total': t} for d, t in sorted(sorted_records)]


# ── 远端 IO ───────────────────────────────────────────────────────────────────

_COUNTER_FILE = 'count.json'


class CounterDataError(ValueError):
    """远端对象存在但不可安全合并，禁止使用零值覆盖。"""


def _parse_remote_data(raw: bytes | None) -> tuple[int, int, int, int, int, dict, list, list]:
    """解析远端 JSON bytes，返回 (total, visits, copies, mcp, successes, keywords, bad_cases, history)。"""
    if raw is None:
        return 0, BASE_VISITS, BASE_COPIES, 0, 0, {}, [], []
    try:
        data = json.loads(raw.decode('utf-8'))
        if not isinstance(data, dict):
            raise ValueError()
        r_total     = int(data.get('total',     0))
        r_visits    = int(data.get('visits',    BASE_VISITS))
        r_copies    = int(data.get('copies',    BASE_COPIES))
        r_mcp       = int(data.get('mcp',       0))
        r_successes = int(data.get('successes', int(r_total * 0.75)))
        r_keywords  = data.get('hot_keywords', {})
        r_bad_cases = data.get('bad_cases', [])
        r_history   = data.get('history', [])
        if any(value < 0 for value in (r_total, r_visits, r_copies, r_mcp, r_successes)):
            raise ValueError()
        if not isinstance(r_keywords, dict) or any(
            not isinstance(key, str) or type(value) is not int or value < 0
            for key, value in r_keywords.items()
        ):
            raise ValueError()
        if not isinstance(r_bad_cases, list) or any(not isinstance(item, dict) for item in r_bad_cases):
            raise ValueError()
        if not isinstance(r_history, list):
            raise ValueError()
        for item in r_history:
            if not isinstance(item, dict) or type(item.get('total')) is not int or item['total'] < 0:
                raise ValueError()
            date.fromisoformat(item['date'])
        return r_total, r_visits, r_copies, r_mcp, r_successes, r_keywords, r_bad_cases, r_history
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        raise CounterDataError('count.json 数据格式无效') from exc


def _read_remote() -> tuple[int, int, int, int, int, dict, list, list]:
    cfg = get_counter_cfg()
    if not cfg.available:
        return 0, BASE_VISITS, BASE_COPIES, 0, 0, {}, [], []

    try:
        raw = read_bytes(_COUNTER_FILE, cfg)
        return _parse_remote_data(raw)
    except Exception as e:
        print(f'[Counter] 启动读取远端数据异常: {type(e).__name__}')
        return 0, BASE_VISITS, BASE_COPIES, 0, 0, {}, [], []


def _sync_remote_task(
    adds_count: int,
    adds_visits: int,
    adds_copies: int,
    adds_mcp: int,
    adds_successes: int,
    adds_keywords: dict,
    adds_bad_cases: list,
    memory_history_snapshot: list,
) -> tuple[bool, int, int, int, int, int, dict, list, list]:
    cfg: CounterConfig = get_counter_cfg()
    if not cfg.available:
        return False, 0, 0, 0, 0, 0, {}, [], []

    try:
        raw = read_bytes(_COUNTER_FILE, cfg)
        r_total, r_visits, r_copies, r_mcp, r_successes, r_keywords, r_bad_cases, r_history = (
            _parse_remote_data(raw)
        )
    except Exception as e:
        print(f'[Counter] 远端数据不可合并，中止本次同步以保护数据: {type(e).__name__}')
        return False, 0, 0, 0, 0, 0, {}, [], []

    n_total     = r_total     + adds_count
    n_visits    = r_visits    + adds_visits
    n_copies    = r_copies    + adds_copies
    n_mcp       = r_mcp       + adds_mcp
    n_successes = r_successes + adds_successes

    merged_keywords = Counter(r_keywords)
    for word, count in adds_keywords.items():
        merged_keywords[word] += count
    top_keywords = dict(merged_keywords.most_common(MAX_KEYWORDS_LIMIT))

    merged_bad_cases = (adds_bad_cases + r_bad_cases)[:MAX_BAD_CASES]

    n_history = _merge_history(r_history, n_total, memory=memory_history_snapshot)

    content = json.dumps({
        'total':        n_total,
        'visits':       n_visits,
        'copies':       n_copies,
        'mcp':          n_mcp,
        'successes':    n_successes,
        'hot_keywords': top_keywords,
        'bad_cases':    merged_bad_cases,
        'history':      n_history,
    }, ensure_ascii=False, indent=2).encode('utf-8')

    commit_msg = (
        f'Sync: 搜索:{n_total} | MCP:{n_mcp} | 成功:{n_successes} | '
        f'复制:{n_copies} | 访问:{n_visits} | bad_cases:{len(merged_bad_cases)}'
    )

    ok = upload_bytes(content, _COUNTER_FILE, cfg, commit_msg, retries=3, retry_delay=1.0)
    if ok:
        print(
            f'[Counter] 同步成功！搜索:{n_total}, MCP:{n_mcp}, 成功交互:{n_successes}, '
            f'复制:{n_copies}, bad_cases:{len(merged_bad_cases)}'
        )
        return True, n_total, n_visits, n_copies, n_mcp, n_successes, top_keywords, merged_bad_cases, n_history

    return False, 0, 0, 0, 0, 0, {}, [], []


# ── 后台同步协程 ──────────────────────────────────────────────────────────────

async def _perform_sync():
    global _dirty_count, _dirty_visits, _dirty_copies, _dirty_mcp, _dirty_successes
    global _dirty_keywords, _dirty_bad_cases, _last_sync
    global _memory_count, _memory_visits, _memory_copies, _memory_mcp, _memory_successes
    global _memory_keywords, _memory_bad_cases, _memory_history

    lock = _get_sync_lock()
    if lock.locked():
        return

    async with lock:
        has_dirty = (
            _dirty_count + _dirty_visits + _dirty_copies + _dirty_mcp + _dirty_successes
            + len(_dirty_keywords) + len(_dirty_bad_cases)
        ) > 0
        if not has_dirty:
            return

        c_adds, v_adds, cp_adds, m_adds, s_adds = (
            _dirty_count, _dirty_visits, _dirty_copies, _dirty_mcp, _dirty_successes
        )
        k_adds  = dict(_dirty_keywords)
        bc_adds = list(_dirty_bad_cases)
        history_snapshot = list(_memory_history)

        _dirty_count     = 0
        _dirty_visits    = 0
        _dirty_copies    = 0
        _dirty_mcp       = 0
        _dirty_successes = 0
        _dirty_keywords.clear()
        _dirty_bad_cases.clear()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, _sync_remote_task,
            c_adds, v_adds, cp_adds, m_adds, s_adds, k_adds, bc_adds, history_snapshot,
        )
        success, l_total, l_visits, l_copies, l_mcp, l_successes, l_keywords, l_bad_cases, l_history = result

        if success:
            _last_sync = time.time()
            _memory_count     = max(_memory_count,     l_total)
            _memory_visits    = max(_memory_visits,    l_visits)
            _memory_copies    = max(_memory_copies,    l_copies)
            _memory_mcp       = max(_memory_mcp,       l_mcp)
            _memory_successes = max(_memory_successes, l_successes)
            _memory_keywords.clear()
            _memory_keywords.update(l_keywords)
            _memory_bad_cases.clear()
            _memory_bad_cases.extend(l_bad_cases)
            _memory_history.clear()
            _memory_history.extend(l_history)
        else:
            _dirty_count     += c_adds
            _dirty_visits    += v_adds
            _dirty_copies    += cp_adds
            _dirty_mcp       += m_adds
            _dirty_successes += s_adds
            _dirty_keywords.update(k_adds)
            _dirty_bad_cases.extend(bc_adds)


# ── 公共 API ──────────────────────────────────────────────────────────────────

async def init():
    """启动时从 OSS 拉取最新计数，初始化内存状态。"""
    global _memory_count, _memory_visits, _memory_copies, _memory_mcp, _memory_successes
    global _memory_keywords, _memory_bad_cases, _memory_history, _last_sync

    cfg = get_counter_cfg()
    if not cfg.available:
        print(f'[Counter] 计数器未配置（platform={cfg.platform}），仅使用内存计数。')
        return

    loop = asyncio.get_running_loop()
    (
        _memory_count, _memory_visits, _memory_copies, _memory_mcp, _memory_successes,
        r_keys, r_bad_cases, r_history,
    ) = await loop.run_in_executor(None, _read_remote)

    _memory_keywords.update(r_keys)
    _memory_bad_cases.clear()
    _memory_bad_cases.extend(r_bad_cases)
    _memory_history.clear()
    _memory_history.extend(r_history)
    _last_sync = time.time()
    print(
        f'[Counter] 初始化完成（{cfg.platform}）：搜索={_memory_count}, MCP={_memory_mcp}, '
        f'访问={_memory_visits}, 复制={_memory_copies}, 历史={len(_memory_history)}天'
    )


def _check_sync():
    cfg = get_counter_cfg()
    if not cfg.available:
        return
    if (time.time() - _last_sync > SYNC_INTERVAL) or (
        _dirty_count + _dirty_visits + _dirty_copies + _dirty_mcp + _dirty_successes
        + len(_dirty_keywords) + len(_dirty_bad_cases)
    ) >= SYNC_THRESHOLD:
        asyncio.create_task(_perform_sync())


async def increment() -> int:
    global _memory_count, _dirty_count
    _memory_count += 1; _dirty_count += 1
    _check_sync()
    return _memory_count

async def increment_visit() -> int:
    global _memory_visits, _dirty_visits
    _memory_visits += 1; _dirty_visits += 1
    _check_sync()
    return _memory_visits

async def increment_copy() -> int:
    global _memory_copies, _dirty_copies
    _memory_copies += 1; _dirty_copies += 1
    _check_sync()
    return _memory_copies

async def increment_mcp() -> int:
    global _memory_mcp, _dirty_mcp
    _memory_mcp += 1; _dirty_mcp += 1
    _check_sync()
    return _memory_mcp

async def increment_success() -> int:
    global _memory_successes, _dirty_successes
    _memory_successes += 1; _dirty_successes += 1
    _check_sync()
    return _memory_successes

async def add_keywords(words: list[str]):
    global _dirty_keywords, _memory_keywords
    valid_words = [w for w in words if len(w) > 1 and w not in {'一个', '的', '在', '了', '是', '有'}]
    for word in valid_words:
        _memory_keywords[word] += 1
        _dirty_keywords[word] += 1
    _check_sync()

async def add_bad_case(
    query: str,
    platform: str = '',
    settings: dict | None = None,
    feedback_type: str = 'search_bad_case',
    detail: str = '',
    tag: str = '',
    current_cn_name: str = '',
    suggested_cn_name: str = '',
    category: str = '',
) -> None:
    global _dirty_bad_cases, _memory_bad_cases
    entry = {
        'type': feedback_type,
        'q': query,
        't': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    optional_fields = {
        'detail': detail,
        'tag': tag,
        'current_cn_name': current_cn_name,
        'suggested_cn_name': suggested_cn_name,
        'category': category,
    }
    for key, value in optional_fields.items():
        if value:
            entry[key] = value
    if platform:
        entry['platform'] = platform
    if settings:
        entry['settings'] = settings
    print(f'[Counter] bad_case 上报: {json.dumps(entry, ensure_ascii=False)}')
    _memory_bad_cases.insert(0, entry)
    if len(_memory_bad_cases) > MAX_BAD_CASES:
        _memory_bad_cases[:] = _memory_bad_cases[:MAX_BAD_CASES]
    _dirty_bad_cases.insert(0, entry)
    _check_sync()


def get() -> int:           return _memory_count
def get_visits() -> int:    return _memory_visits
def get_copies() -> int:    return _memory_copies
def get_mcp() -> int:       return _memory_mcp
def get_successes() -> int: return _memory_successes

def get_hot_keywords(top_n: int = 10) -> dict:
    return dict(_memory_keywords.most_common(top_n))

def get_bad_cases() -> list[dict]:
    return list(_memory_bad_cases)

def get_history() -> list[dict]:
    """返回每日搜索量快照，格式 [{"date": "YYYY-MM-DD", "total": int}, ...]，按日期正序。"""
    return list(_memory_history)

async def force_sync():
    await _perform_sync()
