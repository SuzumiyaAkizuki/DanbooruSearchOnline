"""NiceGUI 前端共享常量，避免入口文件承载配置定义。"""

import os

from core.ui_text import load_ui_text
from core.workspace import (
    ARTIST_SELECTION_ORIGINS,
    FAVORITES_STORAGE_KEY,
    HISTORY_STORAGE_KEY,
    LEGACY_STAGED_STORAGE_KEY,
    WORKSPACE_STORAGE_KEY,
)


TABLE_COLUMNS = [
    {'name': 'tag', 'label': '匹配标签', 'field': 'tag', 'align': 'left', 'sortable': True},
    {'name': 'cn_name', 'label': '含义', 'field': 'cn_name', 'align': 'left'},
    {'name': 'nsfw', 'label': '分级', 'field': 'nsfw', 'align': 'center', 'sortable': True},
    {'name': 'final_score', 'label': '综合分', 'field': 'final_score', 'sortable': True},
    {'name': 'count', 'label': '热度', 'field': 'count', 'sortable': True},
    {'name': 'reason', 'label': '推荐原因', 'field': 'reason', 'align': 'left'},
]

OPTIONAL_COLS = {
    'semantic': {'name': 'semantic_score', 'label': '语义分', 'field': 'semantic_score', 'sortable': True},
    'layer': {'name': 'layer', 'label': '匹配层', 'field': 'layer'},
    'source': {'name': 'source', 'label': '匹配来源', 'field': 'source'},
}

CONFIG_LS_KEY = 'danbooru_search_config'
CONFIG_VERSION = 7
ANNOUNCEMENT_VERSION = 'p0-workspace-2026-07'
LOCAL_STORAGE_READ_CHUNK_CHARS = 200_000
LOCAL_STORAGE_MAX_READ_CHARS = 4_000_000
HISTORY_PRE_COMPACTION_BACKUP_KEY = f'{HISTORY_STORAGE_KEY}_pre_compaction_backup'
LOCAL_STORAGE_RESTORE_CACHE = '__danbooruLocalStorageRestoreV2'
LOCAL_STORAGE_RESTORE_RETRY_DELAYS = (0.0, 1.0, 3.0)
LOCAL_STORAGE_NAMES = ('config', 'workspace', 'history', 'favorites', 'legacy')

SPONSOR_IMAGE_URL = 'https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202501120027592.png'
SPONSOR_TOOLCHAIN_URL = 'http://intro.sakizuki.site/index.html'
SPONSOR_NOTICE_TEXT = '喜欢的话，可以请作者喝杯咖啡'
SPONSOR_TITLE = '谢谢你愿意支持'
UI_TEXT = load_ui_text()


def resolve_group_render_limit(default: int = 80) -> int:
    raw = os.environ.get('DANBOORU_GROUP_RENDER_LIMIT')
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


GROUP_RENDER_TAG_LIMIT = resolve_group_render_limit()
ARTIST_REC_LIMIT = 64
ARTIST_REC_PAGE_SIZE = 8
RELATED_REC_PAGE_SIZE = 10
RECOMMENDATION_DEBOUNCE_SECONDS = 0.1
WORKSPACE_SAVE_DEBOUNCE_SECONDS = 0.3

SEARCH_MODE_PRESETS: dict[str, dict] = {
    '精确查词': {'top_k': 20, 'limit': 10, 'popularity_weight': 0.15, 'use_segmentation': False, 'group_mode': 'off', 'max_per_group': 2},
    '概念扩展': {'top_k': 80, 'limit': 80, 'popularity_weight': 0.15, 'use_segmentation': True, 'group_mode': 'expand', 'max_per_group': 2},
    '描述查词': {'top_k': 20, 'limit': 20, 'popularity_weight': 0.15, 'use_segmentation': False, 'group_mode': 'off', 'max_per_group': 2},
    '完整场景': {'top_k': 5, 'limit': 80, 'popularity_weight': 0.15, 'use_segmentation': True, 'group_mode': 'diverse', 'max_per_group': 2},
}
SEARCH_MODE_OPTIONS = ['自定义'] + list(SEARCH_MODE_PRESETS)
ARTIST_ORIGINS = set(ARTIST_SELECTION_ORIGINS)
