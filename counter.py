"""
counter.py
──────────
搜索次数持久化计数器。

存储后端：HuggingFace Dataset（私有）
  - 数据集：<HF_USERNAME>/DanbooruSearchStats
  - 文件：count.json，内容格式：{"total": 12345}

环境变量：
  HF_TOKEN        HuggingFace Access Token（需要 write 权限）
  HF_USERNAME     HuggingFace 用户名（用于定位 Dataset）
  COUNTER_REPO    可选，覆盖默认的 Dataset repo id

若环境变量未配置（本地开发），退化为内存计数，重启归零。
"""

from __future__ import annotations

import asyncio
import json
import os
import time

# ── 内存缓存（避免每次搜索都读写远程）────────────────────────────────
_memory_count: int = 7382       # 内存中的计数（包含本次启动后的增量）
_remote_base:  int = 7382       # 上次从远端读到的基准值
_dirty:        int = 0          # 未同步到远端的增量
_last_sync:  float = 0.0        # 上次同步时间戳
_lock = asyncio.Lock()

SYNC_INTERVAL = 30              # 最少每 30 秒同步一次远端


def _repo_id() -> str | None:
    custom = os.environ.get("COUNTER_REPO")
    if custom:
        return custom
    username = os.environ.get("HF_USERNAME") or os.environ.get("SPACE_AUTHOR_NAME")
    if username:
        return f"{username}/DanbooruSearchStats"
    return None


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def _read_remote() -> int:
    """从 HF Dataset 读取当前计数，失败返回 0。"""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=_repo_id(), repo_type="dataset",
            filename="count.json", token=_hf_token(),
        )
        with open(path, encoding="utf-8") as f:
            return int(json.load(f).get("total", 0))
    except Exception as e:
        print(f"[Counter] 读取远端计数失败（将使用本地值）: {e}")
        return _remote_base


def _write_remote(total: int) -> bool:
    """将计数写回 HF Dataset，成功返回 True。"""
    try:
        from huggingface_hub import HfApi
        content = json.dumps({"total": total}, ensure_ascii=False)
        HfApi().upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo="count.json",
            repo_id=_repo_id(), repo_type="dataset",
            token=_hf_token(),
            commit_message=f"update count → {total}",
        )
        return True
    except Exception as e:
        print(f"[Counter] 写入远端计数失败: {e}")
        return False


async def init():
    """启动时从远端读取基准值，在引擎预热时调用一次。"""
    global _remote_base, _memory_count
    if not _repo_id() or not _hf_token():
        print("[Counter] 未配置 HF_TOKEN / HF_USERNAME，使用内存计数（重启归零）。")
        return
    loop = asyncio.get_event_loop()
    base = await loop.run_in_executor(None, _read_remote)
    async with _lock:
        _remote_base = base
        _memory_count = base
    print(f"[Counter] 初始化完成，当前累计搜索次数: {base}")


async def increment() -> int:
    """搜索一次，计数 +1，按需同步远端，返回最新计数。"""
    global _memory_count, _dirty, _last_sync, _remote_base
    async with _lock:
        _memory_count += 1
        _dirty += 1
        current = _memory_count
        should_sync = (
            _repo_id() and _hf_token()
            and (time.time() - _last_sync > SYNC_INTERVAL or _dirty >= 10)
        )

    if should_sync:
        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(None, _write_remote, current)
        if ok:
            async with _lock:
                _remote_base = current
                _dirty = 0
                _last_sync = time.time()

    return current


def get() -> int:
    """同步获取当前内存计数（不触发远端读写）。"""
    return _memory_count
