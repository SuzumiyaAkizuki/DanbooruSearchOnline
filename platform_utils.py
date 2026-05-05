"""
platform_utils.py

本地版平台工具函数与配置加载。

对外暴露：
  配置项（从 config.json 读取，含默认值）：
    CFG_MODEL_DIR         — 本地模型目录名
    CFG_PORT              — UI 监听端口
    CFG_CSV_FILE          — 标签数据库路径
    CFG_COOC_FILE         — 共现矩阵路径（引擎会自动查找同名 .parquet）
    CFG_ATOMIC_CJK_MAX_LEN — 中文分词原子长度阈值

  函数：
    is_cloud()          : bool（始终返回 False）
    get_host_port()     : tuple[str, int]
    nsfw_allowed()      : bool
    resolve_model_path(): 解析模型本地路径，不存在时自动从 HF Hub 下载
    ensure_data_files() : 从 GitHub 下载最新的数据文件
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


#  配置加载

_CONFIG_FILE = 'config.json'

_DEFAULTS = {
    'model_dir':          'my_model_bge_m3',
    'port':               11111,
    'csv_file':           'origin_database/tags_enhanced.csv',
    'cooc_file':          'origin_database/cooccurrence_clean.csv',
    'atomic_cjk_max_len': 7,
}


def _load_config() -> dict:
    """从 config.json 读取配置，与默认值合并。"""
    cfg = dict(_DEFAULTS)
    path = Path(_CONFIG_FILE)
    if path.is_file():
        try:
            user_cfg = json.loads(path.read_text(encoding='utf-8'))
            cfg.update(user_cfg)
            print(f'[PlatformUtils] 已加载配置: {path}')
        except (json.JSONDecodeError, OSError) as e:
            print(f'[PlatformUtils] 配置文件读取失败，使用默认值: {e}')
    else:
        print(f'[PlatformUtils] 未找到 {_CONFIG_FILE}，使用默认配置')
    return cfg


_cfg = _load_config()

# 对外暴露的配置项
CFG_MODEL_DIR: str          = _cfg['model_dir']
CFG_PORT: int               = int(_cfg['port'])
CFG_CSV_FILE: str           = _cfg['csv_file']
CFG_COOC_FILE: str          = _cfg['cooc_file']
CFG_ATOMIC_CJK_MAX_LEN: int = int(_cfg['atomic_cjk_max_len'])


#  平台常量（本地版固定）

PLATFORM = 'local'


def is_cloud() -> bool:
    """本地版始终返回 False。"""
    return False


def get_host_port() -> tuple[str, int]:
    """返回 NiceGUI 应使用的 (host, port)。"""
    return '127.0.0.1', CFG_PORT


def nsfw_allowed() -> bool:
    """
    返回当前是否允许用户开启 NSFW 显示。
    如需禁用，可设置环境变量 DISABLE_NSFW=1。
    """
    if os.environ.get('DISABLE_NSFW', '0') == '1':
        return False
    return True


#  模型路径解析

HF_MODEL_ID = 'BAAI/bge-m3'


def resolve_model_path(prefer_local: Optional[str] = None) -> str:
    """
    按优先级解析模型路径：
      1. 本地目录（prefer_local 或 CFG_MODEL_DIR）
      2. 从 HuggingFace Hub 下载（首次运行）
    返回可直接传给 SentenceTransformer 的路径或 model_id 字符串。
    """
    local = prefer_local or CFG_MODEL_DIR
    if os.path.exists(local):
        print(f'[PlatformUtils] 使用本地模型: {local}')
        return local

    print(f'[PlatformUtils] 本地模型目录不存在，将从 HuggingFace Hub 下载: {HF_MODEL_ID}')
    return HF_MODEL_ID


#  数据文件自动更新

_GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/SuzumiyaAkizuki/DanbooruSearchOnline/main'

_ETAG_CACHE_FILE = '.etag_cache.json'


def _get_data_files() -> list[str]:
    """根据配置生成需要下载的数据文件列表（共现矩阵自动转换为 .parquet）。"""
    cooc_path = Path(CFG_COOC_FILE)
    cooc_parquet = str(cooc_path.with_suffix('.parquet'))
    return [CFG_CSV_FILE, cooc_parquet]


def _load_etag_cache(cache_path: Path) -> dict[str, str]:
    """从磁盘加载 ETag 缓存。"""
    if cache_path.is_file():
        try:
            return json.loads(cache_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_etag_cache(cache_path: Path, cache: dict[str, str]) -> None:
    """将 ETag 缓存写入磁盘。"""
    try:
        cache_path.write_text(json.dumps(cache, indent=2), encoding='utf-8')
    except OSError as e:
        print(f'[PlatformUtils] 保存 ETag 缓存失败: {e}')


def _download_with_etag(url: str, dest: Path, etag: Optional[str]) -> Optional[str]:
    """
    下载单个文件，支持 ETag 条件请求。
    返回新的 ETag（如果文件有更新），否则返回 None 表示无需更新。
    """
    req = urllib.request.Request(url)
    if etag:
        req.add_header('If-None-Match', etag)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            # 200 = 有更新，下载
            new_etag = resp.headers.get('ETag')
            data = resp.read()

            # 先写到临时文件，成功后再原子替换
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix='.tmp') as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            tmp_path.replace(dest)

            size_mb = len(data) / (1024 * 1024)
            print(f'[PlatformUtils] 已下载: {dest.name} ({size_mb:.1f} MB)')
            return new_etag

    except urllib.error.HTTPError as e:
        if e.code == 304:
            # 304 Not Modified，文件无变化
            return None
        print(f'[PlatformUtils] 下载失败 {dest.name}: HTTP {e.code}')
        return None
    except Exception as e:
        print(f'[PlatformUtils] 下载失败 {dest.name}: {e}')
        return None


def ensure_data_files(data_dir: str = 'origin_database') -> None:
    """
    启动时检查并下载最新的数据文件。
    使用 ETag 缓存避免重复下载未变更的文件。

    - 文件不存在 → 全量下载
    - 文件存在但有更新 → 增量更新
    - 文件存在且无变化 → 跳过
    - 网络不可用 → 跳过（使用本地已有文件）
    """
    cache_path = Path(data_dir) / _ETAG_CACHE_FILE
    etag_cache = _load_etag_cache(cache_path)
    updated = False

    for rel_path in _get_data_files():
        url = f'{_GITHUB_RAW_BASE}/{rel_path}'
        dest = Path(rel_path)

        # 文件不存在时清除旧 ETag，强制重新下载
        if not dest.is_file():
            etag_cache.pop(rel_path, None)

        old_etag = etag_cache.get(rel_path)
        new_etag = _download_with_etag(url, dest, old_etag)

        if new_etag is not None:
            etag_cache[rel_path] = new_etag
            updated = True

    if updated:
        _save_etag_cache(cache_path, etag_cache)
