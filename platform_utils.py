"""
platform_utils.py

本地版平台工具函数。

对外暴露：
  is_cloud()        : bool（始终返回 False）
  get_host_port()   : tuple[str, int]
  nsfw_allowed()    : bool
  resolve_model_path(): 解析模型本地路径，不存在时自动从 HF Hub 下载
"""

from __future__ import annotations

import os
from typing import Optional


#  平台常量（本地版固定）

PLATFORM = 'local'


def is_cloud() -> bool:
    """本地版始终返回 False。"""
    return False


def get_host_port() -> tuple[str, int]:
    """返回 NiceGUI 应使用的 (host, port)。"""
    return '127.0.0.1', 11111


def nsfw_allowed() -> bool:
    """
    返回当前是否允许用户开启 NSFW 显示。
    如需禁用，可设置环境变量 DISABLE_NSFW=1。
    """
    if os.environ.get('DISABLE_NSFW', '0') == '1':
        return False
    return True


#  模型路径解析

LOCAL_MODEL_PATH = 'my_model_bge_m3'
HF_MODEL_ID      = 'BAAI/bge-m3'


def resolve_model_path(prefer_local: Optional[str] = None) -> str:
    """
    按优先级解析模型路径：
      1. 本地目录（prefer_local 或 LOCAL_MODEL_PATH）
      2. 从 HuggingFace Hub 下载（首次运行）
    返回可直接传给 SentenceTransformer 的路径或 model_id 字符串。
    """
    local = prefer_local or LOCAL_MODEL_PATH
    if os.path.exists(local):
        print(f'[PlatformUtils] 使用本地模型: {local}')
        return local

    print(f'[PlatformUtils] 本地模型目录不存在，将从 HuggingFace Hub 下载: {HF_MODEL_ID}')
    return HF_MODEL_ID
