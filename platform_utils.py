"""
platform_utils.py

统一的平台检测与 Hub 操作封装。

支持平台：
  - HuggingFace Space
  - ModelScope 创空间（魔搭）
  - 本地开发环境

对外暴露：
  PLATFORM          : Literal['hf', 'ms', 'local']
  is_cloud()        : bool
  get_host_port()   : tuple[str, int]
  download_file()   : 下载单个文件，返回本地路径

环境变量约定：
  
   HuggingFace Space（由 HF 自动注入）                                  
     SPACE_ID          Space 唯一标识，存在即代表在 HF 环境            
     SPACE_AUTHOR_NAME 作者名                                           
                                                                        
   用户手动配置（HF Secrets）：                                         
     HF_TOKEN          HF 访问令牌（仅用于 download_file，非计数器）    

   ModelScope 创空间（由魔搭自动注入）                                   
     MODELSCOPE_ENVIRONMENT  存在即代表在魔搭环境（值通常为 "studio"）  
     STUDIO_ID               创空间 ID（备用检测）                      
                                                                        
   魔搭平台数据文件说明：
     数据文件（CSV / parquet / safetensors）直接放在创空间 studio repo
     中，容器启动时会自动同步到工作目录，download_file() 在 MS 平台
     直接返回本地路径，无需额外配置 Model repo。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

#  平台检测 

def _detect_platform() -> Literal['hf', 'ms', 'local']:
    if os.environ.get('SPACE_ID'):
        return 'hf'
    if os.environ.get('MODELSCOPE_ENVIRONMENT') or os.environ.get('STUDIO_ID'):
        return 'ms'
    return 'local'


PLATFORM: Literal['hf', 'ms', 'local'] = _detect_platform()


def is_cloud() -> bool:
    """是否运行在任意云端平台。"""
    return PLATFORM in ('hf', 'ms')


def get_host_port() -> tuple[str, int]:
    """
    返回 NiceGUI 应使用的 (host, port)。
    HF 和魔搭创空间都使用 0.0.0.0:7860；本地使用 127.0.0.1:1111。
    """
    if is_cloud():
        return '0.0.0.0', 7860
    return '127.0.0.1', 11111


def nsfw_allowed() -> bool:
    """
    返回当前平台是否允许用户开启 NSFW 显示。
    魔搭（MS）平台禁用 NSFW，其余平台默认允许。
    如需在任意平台强制禁用，可设置环境变量 DISABLE_NSFW=1。
    """
    if os.environ.get('DISABLE_NSFW', '0') == '1':
        return False
    return PLATFORM != 'ms'


#  文件下载（引擎数据文件）

# 魔搭创空间工作目录，studio repo 的文件会被同步到此处
_MS_WORKDIR = Path('/home/user/app')


#  HF Storage Bucket 挂载检测

# HF Storage Buckets 挂载到 Space 时，会映射到容器内的一个本地路径
# （通常为 /data），文件可直接以本地路径读取，无需 hf_hub_download。
_HF_BUCKET_MOUNT = Path('/data')


def get_hf_bucket_path(relative: str) -> Optional[Path]:
    """
    如果 HF Storage Bucket 已挂载且目标文件存在，返回本地绝对路径。
    否则返回 None（调用方应 fallback 到 hf_hub_download）。
    """
    candidate = _HF_BUCKET_MOUNT / relative
    if candidate.exists():
        return candidate
    return None


def download_file(
    filename: str,
    *,
    # HF 专用参数
    hf_repo_id:   Optional[str] = None,
    hf_repo_type: str           = 'space',
    hf_token:     Optional[str] = None,
    # MS 专用参数（保留签名兼容性，魔搭平台已不再使用）
    ms_repo_id:   Optional[str] = None,
    ms_token:     Optional[str] = None,
    ms_cache_dir: str           = '/tmp/ms_cache',
) -> str:
    """
    下载单个引擎数据文件，返回本地绝对路径字符串。

    HF 平台：
        优先从挂载的 Storage Bucket（/data）读取本地文件（零延迟）。
        若 Bucket 未挂载或文件不存在，回退到从 Space repo 下载
        （hf_repo_id 默认读取环境变量 SPACE_ID）。

    MS 平台：
        文件已随 studio repo 部署到容器本地，直接返回工作目录下的路径，
        无需配置任何额外 repo。若文件不存在则抛出 FileNotFoundError。

    本地：
        直接返回原始路径。
    """
    if PLATFORM == 'hf':
        # 优先从挂载的 Storage Bucket 读取（本地路径，零延迟）
        bucket_path = get_hf_bucket_path(filename)
        if bucket_path is not None:
            print(f'[PlatformUtils] 从 Storage Bucket 读取: {bucket_path}')
            return str(bucket_path)

        # Bucket 未挂载或文件不存在，回退到从 Space repo 下载
        from huggingface_hub import hf_hub_download
        repo_id = hf_repo_id or os.environ.get('SPACE_ID')
        if not repo_id:
            raise RuntimeError('[PlatformUtils] HF 平台未找到 SPACE_ID，无法下载文件。')
        return hf_hub_download(
            repo_id=repo_id,
            repo_type=hf_repo_type,
            filename=filename,
            token=hf_token or os.environ.get('HF_TOKEN'),
        )

    if PLATFORM == 'ms':
        local_path = _MS_WORKDIR / filename
        if not local_path.is_file():
            raise FileNotFoundError(
                f'[PlatformUtils] 魔搭平台本地文件不存在: {local_path}\n'
                f'请确认已将 {filename} 提交到创空间 studio repo 中。'
            )
        print(f'[PlatformUtils] MS 本地文件: {local_path}')
        return str(local_path)

    # 本地环境：直接返回原始路径（由调用方保证文件存在）
    return filename


#  模型路径解析 

LOCAL_MODEL_PATH = 'my_model_bge_m3'
HF_MODEL_ID      = 'BAAI/bge-m3'
MS_MODEL_ID      = 'BAAI/bge-m3'   # 魔搭上同名，走国内节点


def resolve_model_path(prefer_local: Optional[str] = None) -> str:
    """
    按优先级解析模型路径：
      1. 本地目录（prefer_local 或 LOCAL_MODEL_PATH）
      2. 当前平台的 Hub Model ID（首次会自动下载缓存）
    返回可直接传给 SentenceTransformer 的路径或 model_id 字符串。
    """
    local = prefer_local or LOCAL_MODEL_PATH
    if os.path.exists(local):
        print(f'[PlatformUtils] 使用本地模型: {local}')
        return local

    if PLATFORM == 'ms':
        print(f'[PlatformUtils] 魔搭环境，使用 ModelScope Hub 模型: {MS_MODEL_ID}')
        try:
            from modelscope import snapshot_download
            cached = snapshot_download(MS_MODEL_ID, cache_dir='/tmp/ms_model')
            print(f'[PlatformUtils] 模型已缓存至: {cached}')
            return cached
        except Exception as e:
            print(f'[PlatformUtils] ModelScope snapshot_download 失败，回退到 HF ID: {e}')

    print(f'[PlatformUtils] 使用 HuggingFace Hub 模型: {HF_MODEL_ID}')
    return HF_MODEL_ID