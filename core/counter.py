"""
counter.py
──────────
搜索次数持久化计数器。
存储后端：HuggingFace Dataset (私有)
特色：支持环境变量配置免密钥、内存积攒、并发 412 冲突重试、彻底无阻塞后台同步。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Optional

# ── 内存缓存与状态 ────────────────────────────────
_memory_count: int = 0      # 当前总计数（展示给前端用）
_dirty: int = 0             # 尚未同步到远端的增量
_last_sync: float = 0.0     # 上次同步时间戳
_sync_lock: Optional[asyncio.Lock] = None

SYNC_INTERVAL = 30          # 最小同步间隔（秒）
SYNC_THRESHOLD = 10         # 累计达到 10 次搜索强制同步

def _get_sync_lock() -> asyncio.Lock:
    """动态获取同步锁，确保在正确的 Event Loop 中初始化"""
    global _sync_lock
    if _sync_lock is None:
        _sync_lock = asyncio.Lock()
    return _sync_lock

def _get_config():
    """从环境变量获取配置，无密钥时返回 None"""
    token = os.environ.get("HF_TOKEN")
    username = os.environ.get("HF_USERNAME") or os.environ.get("SPACE_AUTHOR_NAME")
    # 优先使用显式指定的 REPO，否则根据用户名拼接
    repo_id = os.environ.get("COUNTER_REPO") or (
        f"{username}/DanbooruSearchStats" if username else None
    )
    return repo_id, token

# ── 远端 IO 操作 (同步函数，将在 executor 中运行) ─────────────────

def _read_remote() -> int:
    """初始化时从 HF Dataset 读取 count.json"""
    repo_id, token = _get_config()
    if not repo_id or not token:
        return 0

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
            return int(data.get("total", 0))
    except Exception as e:
        print(f"[Counter] 读取远端失败 (可能是首次运行，文件不存在): {e}")
        return 0

def _sync_remote_task(adds_to_sync: int) -> tuple[bool, int]:
    """
    后台执行的同步任务（带 412 防护与重试机制）。
    返回: (是否成功, 最新的云端总数)
    """
    repo_id, token = _get_config()
    if not repo_id or not token:
        return False, 0

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    api = HfApi(token=token)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 1. 强制拉取云端最新数据（防冲突核心，force_download=True）
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename="count.json",
                    force_download=True, # 必须强制刷新，防止拿旧指针去提交
                    token=token
                )
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    remote_total = int(data.get("total", 0))
            except Exception:
                # 找不到文件说明是空的，从 0 开始
                remote_total = 0

            # 2. 计算出真正的全球最新总数
            new_total = remote_total + adds_to_sync

            # 3. 构造新的 JSON 写回云端
            content = json.dumps({"total": new_total}, ensure_ascii=False)
            api.upload_file(
                path_or_fileobj=content.encode("utf-8"),
                path_in_repo="count.json",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message=f"Update search count to {new_total}"
            )
            print(f"[Counter] 同步成功！当前全球云端总计数: {new_total}")
            return True, new_total

        except HfHubHTTPError as e:
            if "412 Precondition Failed" in str(e):
                print(f"[Counter] 遇到 412 Git并发冲突，正在重试 ({attempt+1}/{max_retries})...")
                time.sleep(1) # 在 Executor 里运行，用 time.sleep 绝对安全
            else:
                print(f"[Counter] 同步至远端失败 (HubError): {e}")
                break
        except Exception as e:
            print(f"[Counter] 同步发生未知错误: {e}")
            break

    return False, 0


async def _perform_sync():
    """控制异步锁与内存状态同步的包装函数"""
    global _dirty, _last_sync, _memory_count

    lock = _get_sync_lock()
    if lock.locked():
        return # 如果已经在同步了，直接跳过，等下一波

    async with lock:
        if _dirty == 0:
            return

        adds_to_sync = _dirty
        loop = asyncio.get_running_loop()

        # 抛给线程池执行，坚决不阻塞主 Event Loop
        success, latest_cloud_total = await loop.run_in_executor(None, _sync_remote_task, adds_to_sync)

        if success:
            # 扣除成功提交的次数
            _dirty -= adds_to_sync
            _last_sync = time.time()

            # 一个额外的小优化：同步成功后，把本地数字校准为全球最新数字
            # 这可以防止多个 HF Space 实例同时运行时，本地数字落后的问题
            _memory_count = max(_memory_count, latest_cloud_total)


# ── 公共 API ──────────────────────────────────────

async def init():
    """引擎启动时初始化，加载远端基准值"""
    global _memory_count, _last_sync
    repo_id, token = _get_config()

    if not repo_id or not token:
        print("[Counter] 未配置 HF 环境变量，已切换为纯内存模式（重启归零）。")
        return

    loop = asyncio.get_running_loop()
    remote_val = await loop.run_in_executor(None, _read_remote)

    _memory_count = remote_val
    _last_sync = time.time()
    print(f"[Counter] 初始化成功，当前基准计数: {remote_val}")

async def increment() -> int:
    """增加计数，并自动在后台派发同步任务"""
    global _memory_count, _dirty, _last_sync

    # 修改内存速度极快，不需要跨线程锁，直接累加
    _memory_count += 1
    _dirty += 1
    current = _memory_count

    repo_id, token = _get_config()
    now = time.time()

    # 策略判断
    should_sync = repo_id and token and (
        (now - _last_sync > SYNC_INTERVAL) or (_dirty >= SYNC_THRESHOLD)
    )

    if should_sync:
        # 【核心改动】：使用 create_task 直接抛给后台静默运行
        # 绝不使用 await 阻塞当前的 increment 函数，保证前端秒开！
        asyncio.create_task(_perform_sync())

    return current

def get() -> int:
    """快速获取当前内存中的数值"""
    return _memory_count

async def force_sync():
    """提供给 app.on_shutdown 调用的“临终托孤”抢救函数"""
    global _dirty
    if _dirty > 0:
        print(f"[Counter] 收到服务器关闭信号，正在抢救最后 {_dirty} 条未提交计数...")
        await _perform_sync()