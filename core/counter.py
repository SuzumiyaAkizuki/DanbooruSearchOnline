"""
counter.py
──────────
持久化计数器（支持三轨制：搜索次数 + 复制次数 + 页面访问量）。
存储后端：HuggingFace Dataset (私有)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Optional

# ── 内存缓存与状态 ────────────────────────────────
_memory_count: int = 0      # 当前总搜索
_dirty_count: int = 0       # 尚未同步的搜索增量

_memory_visits: int = 0     # 当前总访问量
_dirty_visits: int = 0      # 尚未同步的访问增量
BASE_VISITS: int = 0     # 旧版 JSON 没有 visits 时的默认起点

_memory_copies: int = 0     # 当前总复制次数
_dirty_copies: int = 0      # 尚未同步的复制增量
BASE_COPIES: int = 1832     # 基于搜索转化率(约22%)推算的初始复制基准值

_last_sync: float = 0.0     # 上次同步时间戳
_sync_lock: Optional[asyncio.Lock] = None

SYNC_INTERVAL = 1800        # 最小同步间隔（建议30分钟: 1800秒）
SYNC_THRESHOLD = 50         # 三者增量之和达到 50 次时强制同步

def _get_sync_lock() -> asyncio.Lock:
    global _sync_lock
    if _sync_lock is None:
        _sync_lock = asyncio.Lock()
    return _sync_lock

def _get_config():
    token = os.environ.get("HF_TOKEN")
    username = os.environ.get("HF_USERNAME") or os.environ.get("SPACE_AUTHOR_NAME")
    repo_id = os.environ.get("COUNTER_REPO") or (
        f"{username}/DanbooruSearchStats" if username else None
    )
    return repo_id, token

# ── 远端 IO 操作 ──────────────────────────────────

def _read_remote() -> tuple[int, int, int]:
    """初始化时读取 count.json，返回 (total, visits, copies)"""
    repo_id, token = _get_config()
    if not repo_id or not token:
        return 0, BASE_VISITS, BASE_COPIES

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="count.json",
            token=token,
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return (
                int(data.get("total", 0)),
                int(data.get("visits", BASE_VISITS)),
                int(data.get("copies", BASE_COPIES))
            )
    except Exception as e:
        print(f"[Counter] 读取远端失败 (默认基准值启动): {e}")
        return 0, BASE_VISITS, BASE_COPIES

def _sync_remote_task(adds_count: int, adds_visits: int, adds_copies: int) -> tuple[bool, int, int, int]:
    """后台同步任务，返回 (是否成功, 最新云端搜索, 最新云端访问, 最新云端复制)"""
    repo_id, token = _get_config()
    if not repo_id or not token:
        return False, 0, 0, 0

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    api = HfApi(token=token)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 1. 强制拉取云端最新数据
            try:
                path = hf_hub_download(
                    repo_id=repo_id, repo_type="dataset", filename="count.json",
                    force_download=True, token=token
                )
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    remote_total = int(data.get("total", 0))
                    remote_visits = int(data.get("visits", BASE_VISITS))
                    remote_copies = int(data.get("copies", BASE_COPIES))
            except Exception:
                remote_total, remote_visits, remote_copies = 0, BASE_VISITS, BASE_COPIES

            # 2. 计算最新总数
            new_total = remote_total + adds_count
            new_visits = remote_visits + adds_visits
            new_copies = remote_copies + adds_copies

            # 3. 构造新的三轨 JSON 写回云端
            content = json.dumps({
                "total": new_total,
                "visits": new_visits,
                "copies": new_copies
            }, ensure_ascii=False)

            api.upload_file(
                path_or_fileobj=content.encode("utf-8"),
                path_in_repo="count.json",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message=f"搜索：{new_total}, 复制：{new_copies},访问：{new_visits}"
            )
            print(f"[Counter] 同步成功！云端: 搜索：{new_total}, 复制：{new_copies},访问：{new_visits}")
            return True, new_total, new_visits, new_copies

        except HfHubHTTPError as e:
            if "412 Precondition Failed" in str(e):
                print(f"[Counter] 遇到 412 冲突，重试 ({attempt+1}/{max_retries})...")
                time.sleep(1)
            else:
                print(f"[Counter] HubError: {e}")
                break
        except Exception as e:
            print(f"[Counter] 同步未知错误: {e}")
            break

    return False, 0, 0, 0

async def _perform_sync():
    """包装函数：控制异步锁并派发同步任务"""
    global _dirty_count, _dirty_visits, _dirty_copies, _last_sync
    global _memory_count, _memory_visits, _memory_copies

    lock = _get_sync_lock()
    if lock.locked(): return

    async with lock:
        if _dirty_count == 0 and _dirty_visits == 0 and _dirty_copies == 0: return

        c_adds = _dirty_count
        v_adds = _dirty_visits
        cp_adds = _dirty_copies

        loop = asyncio.get_running_loop()
        success, l_total, l_visits, l_copies = await loop.run_in_executor(
            None, _sync_remote_task, c_adds, v_adds, cp_adds
        )

        if success:
            _dirty_count -= c_adds
            _dirty_visits -= v_adds
            _dirty_copies -= cp_adds
            _last_sync = time.time()

            _memory_count = max(_memory_count, l_total)
            _memory_visits = max(_memory_visits, l_visits)
            _memory_copies = max(_memory_copies, l_copies)

# ── 公共 API ──────────────────────────────────────

async def init():
    global _memory_count, _memory_visits, _memory_copies, _last_sync
    repo_id, token = _get_config()

    if not repo_id or not token:
        print(f"[Counter] 未配置环境变量，纯内存模式。")
        _memory_visits, _memory_copies = BASE_VISITS, BASE_COPIES
        return

    loop = asyncio.get_running_loop()
    r_total, r_visits, r_copies = await loop.run_in_executor(None, _read_remote)

    _memory_count = r_total
    _memory_visits = r_visits
    _memory_copies = r_copies
    _last_sync = time.time()
    print(f"[Counter] 启动成功: 搜索：{r_total} | 复制：{r_copies} | 访问：{r_visits}")

def _check_and_trigger_sync():
    repo_id, token = _get_config()
    now = time.time()
    total_dirty = _dirty_count + _dirty_visits + _dirty_copies
    should_sync = repo_id and token and (
        (now - _last_sync > SYNC_INTERVAL) or (total_dirty >= SYNC_THRESHOLD)
    )
    if should_sync:
        asyncio.create_task(_perform_sync())

async def increment() -> int:
    global _memory_count, _dirty_count
    _memory_count += 1
    _dirty_count += 1
    _check_and_trigger_sync()
    return _memory_count

async def increment_visit() -> int:
    global _memory_visits, _dirty_visits
    _memory_visits += 1
    _dirty_visits += 1
    _check_and_trigger_sync()
    return _memory_visits

async def increment_copy() -> int:
    global _memory_copies, _dirty_copies
    _memory_copies += 1
    _dirty_copies += 1
    _check_and_trigger_sync()
    return _memory_copies

def get() -> int: return _memory_count
def get_visits() -> int: return _memory_visits
def get_copies() -> int: return _memory_copies

async def force_sync():
    global _dirty_count, _dirty_visits, _dirty_copies
    if _dirty_count > 0 or _dirty_visits > 0 or _dirty_copies > 0:
        print(f"[Counter] 程序异常关闭，正在保存： {_dirty_count}搜索, {_dirty_copies}复制, {_dirty_visits}访问...")
        await _perform_sync()